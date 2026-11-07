"""Straight-magnet tracking, export and independent frame/offset maps."""

import math
import re
import subprocess
import tempfile
from pathlib import Path

from _diagnostics import run

# isort: split
# _diagnostics sets numerical thread limits before loading _elegant and NumPy.
import _elegant as native
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

P0 = 2300.0
native.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-native-process deadline.


def close(observed, expected, tolerance):
    np.testing.assert_allclose(observed, expected, rtol=0, atol=tolerance)


def fault(observed, expected, faulty, tolerance):
    """Recognize the physical answer or this defect; unrelated results raise."""
    if np.allclose(observed, expected, rtol=0, atol=tolerance):
        return False
    close(observed, faulty, tolerance)
    return True


def lattice(kind, length, strength, tilt, dx=0.0, dy=0.0, method=0):
    order = 1 if kind in ("QUAD", "KQUAD") else 2
    extra = ""
    if kind.startswith("K"):
        extra = ",N_SLICES=64,INTEGRATION_ORDER=6,SYNCH_RAD=0,ISR=0"
    if order == 1:
        extra += f",MALIGN_METHOD={method},EDGE1_EFFECTS=0,EDGE2_EFFECTS=0"
    return (
        f"Q: {kind},L={length:.17g},K{order}={strength:.17g},TILT={tilt:.17g},"
        f"DX={dx:.17g},DY={dy:.17g}{extra}\nBL: LINE=(Q)\n"
    )


def track(source, rays):
    count = len(rays)
    beam = {name: rays[:, i] for i, name in enumerate(("x", "xp", "y", "yp"))}
    beam.update(p=np.full(count, P0), particleID=np.arange(1, count + 1))
    commands = (
        f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0},"
        "output=out.sdds,default_order=2,random_number_seed=104729 &end\n"
        "&run_control n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
    )
    result = run(
        {"lattice.lte": source, "run.ele": commands},
        "serial",
        {"beam.sdds": beam},
        ("out.sdds",),
    )
    data = native.particles(result["out.sdds"])
    if not np.array_equal(data["particleID"], beam["particleID"]):
        raise RuntimeError("Particles were lost or particle IDs changed")
    return np.column_stack([data[name] for name in ("x", "xp", "y", "yp")])


def export(source, fmt, *, ender=None):
    with tempfile.TemporaryDirectory(prefix="elegant-misalignment-repro-") as tmp:
        work = Path(tmp)
        (work / "input.lte").write_text(source + "\nUSE,BL\n")
        args = ["elegantto", "input.lte", "output.txt", f"-{fmt}"]
        if ender is not None:
            (work / "ender.txt").write_text(ender)
            args.append("-ender=ender.txt")
        result = subprocess.run(
            args, cwd=work, capture_output=True, text=True, check=False, timeout=290
        )
        log = result.stdout + result.stderr
        if result.returncode or re.search(r"warning|error|unsupported", log, re.IGNORECASE):
            raise RuntimeError(f"elegantto -{fmt} failed or issued a diagnostic:\n{log}")
        return (work / "output.txt").read_text()


def coefficient(text, name):
    matches = re.findall(rf"\b{name}\s*=\s*([+\-0-9.eE]+)", text, re.IGNORECASE)
    if len(matches) != 1 or not math.isfinite(float(matches[0])):
        raise RuntimeError(f"Expected one finite {name}: {text}")
    return float(matches[0])


def quad_map(k, length, tilt):
    # H=(px²+py²)/2 + k*cos(2t)*(x²-y²)/2 + k*sin(2t)*x*y.
    # The laboratory Hamiltonian Hessian gives M=exp(L*J*Hessian(H)).
    generator = np.zeros((4, 4))
    generator[0, 1] = generator[2, 3] = 1.0
    generator[1, 0] = -k * math.cos(2 * tilt)
    generator[1, 2] = generator[3, 0] = -k * math.sin(2 * tilt)
    generator[3, 2] = k * math.cos(2 * tilt)
    return expm(generator * length)


def exact_quad_offset(k, length, dx, dy):
    # At delta=0 the normalized canonical longitudinal momentum is
    # ps=sqrt(1-px²-py²); the transverse Lorentz force uses the translated body.
    def rhs(_, u):
        x, px, y, py = u
        ps = math.sqrt(1 - px * px - py * py)
        return [px / ps, -k * (x - dx), py / ps, k * (y - dy)]

    solution = solve_ivp(
        rhs,
        (0.0, length),
        np.zeros(4),
        method="DOP853",
        rtol=2.3e-14,
        atol=2e-18,
        max_step=length / 32,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    x, px, y, py = solution.y[:, -1]
    ps = math.sqrt(1 - px * px - py * py)
    return np.array([x, px / ps, y, py / ps])
