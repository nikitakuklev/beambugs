"""Coordinate, lattice and derivative helpers for division and inverse-map probes.

The native runner uses one fresh process per case. Its inputs use slopes and SDDS
absolute momentum; the returned q uses (x, xprime, y, yprime, beta*c*t, delta).
"""

import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NPROC",
    "PJRT_NPROC",
):
    os.environ[_name] = "1"

import _elegant as native
import numpy as np

C = 299792458.0
MASS_MEV = 0.51099906  # SDDS constants.h, used by the executed elegant build.
native.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-probe native deadline.


def beta(p):
    return p / np.hypot(p, 1)


def track(
    _label,
    lattice,
    rays,
    *,
    p0=250.0,
    reverse=False,
    divisions=1,
    before="",
    matrix=False,
    mode="serial",
    default_order=1,
):
    """Track fixed rays, checking survival, IDs and finiteness before comparing maps."""
    q = np.array(rays, dtype=float, ndmin=2)
    p = p0 * (1 + q[:, 5])
    beam = dict(zip(("x", "xp", "y", "yp"), q[:, :4].T, strict=True))
    beam.update(t=q[:, 4] / (C * beta(p)), p=p, particleID=np.arange(1, len(q) + 1))
    commands = before + (
        "&run_setup lattice=lattice.lte, use_beamline=BL, "
        f"p_central={p0:.17g}, back_tracking={int(reverse)}, "
        f"element_divisions={divisions}, default_order={default_order}, output=out.sdds &end\n"
        "&run_control n_steps=1 &end\n"
    )
    if matrix:
        commands += "&matrix_output SDDS_output=matrix.sdds, SDDS_output_order=1, output_at_each_step=1 &end\n"
    commands += "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
    result = native.run(
        {"lattice.lte": lattice, "run.ele": commands},
        mode,
        {"beam.sdds": beam},
        outputs=("out.sdds", "matrix.sdds") if matrix else ("out.sdds",),
    )
    data = native.particles(result["out.sdds"])
    if not np.array_equal(data["particleID"], beam["particleID"]):
        raise RuntimeError("Particle IDs changed or particles were lost")
    raw = np.column_stack([data[name] for name in ("x", "xp", "y", "yp", "t", "p")])
    if raw.shape != q.shape:
        raise RuntimeError("Unexpected output shape")
    out = raw.copy()
    out[:, 4] = C * raw[:, 4] * beta(raw[:, 5])
    out[:, 5] = raw[:, 5] / p0 - 1
    answer = {"q": out, "raw": raw}
    if matrix:
        mat = result["matrix.sdds"]
        answer["R"] = np.array([[mat[0, f"R{i}{j}"][-1] for j in range(1, 7)] for i in range(1, 7)])
        answer["C"] = np.array([mat[0, f"C{i}"][-1] for i in range(1, 7)])
        if not np.isfinite(answer["R"]).all() or not np.isfinite(answer["C"]).all():
            raise RuntimeError("Nonfinite matrix output")
    return answer


def canonical(q):
    """Exact normalized transverse momentum, not the SDDS slopes."""
    q = np.array(q, ndmin=2)
    return (1 + q[:, 5, None]) * q[:, [1, 3]] / np.sqrt(1 + q[:, 1, None] ** 2 + q[:, 3, None] ** 2)


def close(a, b, atol, label):
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    if not np.isfinite(error) or error > atol:
        raise RuntimeError(f"{label}: error={error:.17g}, tolerance={atol:g}")
    return error


def discrepancy(actual, expected, faulty, atol):
    """Accept the physical result or the established faulty law; reject unrelated failures."""
    if np.allclose(actual, expected, rtol=0, atol=atol):
        return False
    close(actual, faulty, atol, "Unexpected discrepancy")
    return True


def straight(kind, *, length=0.6, hk=0.00017, vk=-0.00011, slices=30, strength=0.0, order=4):
    pars = (
        f"K1={strength:.17g}"
        if kind == "QUAD"
        else (
            f"K{2 if kind == 'KSEXT' else 1}={strength:.17g},N_SLICES={slices},INTEGRATION_ORDER={order}"
        )
    )
    return f"Q: {kind},L={length:.17g},{pars},HKICK={hk:.17g},VKICK={vk:.17g}\nBL: LINE=(Q)\n"


def bend(*, length=0.6, angle=0.045, hk=0.00017, vk=-0.00011, slices=66, expand=0):
    return (
        f"Q: CSBEND,L={length:.17g},ANGLE={angle:.17g},XKICK={hk:.17g},YKICK={vk:.17g},"
        f"N_SLICES={slices},INTEGRATION_ORDER=4,EXPAND_HAMILTONIAN={expand},"
        "EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,SYNCH_RAD=0,ISR=0\nBL: LINE=(Q)\n"
    )


def linear_rays(h):
    rays = [np.zeros(6)]
    for j in range(4):
        for multiplier in (-2, -1, 1, 2):
            q = np.zeros(6)
            q[j] = multiplier * h
            rays.append(q)
    return np.array(rays)


def derivative(out, h):
    # Four-point stencils cancel the cubic terms in the linear fringe derivative.
    return np.column_stack(
        [
            (
                out[1 + 4 * j, :4]
                - 8 * out[2 + 4 * j, :4]
                + 8 * out[3 + 4 * j, :4]
                - out[4 + 4 * j, :4]
            )
            / (12 * h)
            for j in range(4)
        ]
    )
