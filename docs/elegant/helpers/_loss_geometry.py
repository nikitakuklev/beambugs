"""Fixed-ray loss fixtures and independent Cartesian geometry for loss diagnostics."""

import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

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
import pysdds

native.TIMEOUT = 290  # ty: ignore[invalid-assignment]


def close(actual, expected, atol=3e-14):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise RuntimeError(f"Invalid comparison: {actual}, {expected}")
    if np.any(np.abs(a - b) > atol):
        raise RuntimeError(f"Observed {actual}; expected {expected}, tolerance {atol}")


def fault(actual, expected, faulty, atol=3e-14):
    if np.all(np.isfinite(actual)) and np.all(np.abs(np.asarray(actual) - expected) <= atol):
        return False
    close(actual, faulty, atol)
    return True


def world(x, z, heading=0, origin=(0.41, -0.26)):
    return [
        origin[0] + x * math.cos(heading) + z * math.sin(heading),
        origin[1] - x * math.sin(heading) + z * math.cos(heading),
    ]


def circle(s, x0, xp0, *, length=0.9, angle=0.24):
    """Circle/radial-plane intersection, returning Cartesian x/z, tangent and local x."""
    rho = length / angle
    alpha0 = math.atan(xp0)
    cx, cz = x0 - rho * math.cos(alpha0), rho * math.sin(alpha0)
    phi = s / rho
    wx, wz = cx + rho, cz
    dot = wx * math.cos(phi) + wz * math.sin(phi)
    radius = dot + math.copysign(math.sqrt(rho * rho - wx * wx - wz * wz + dot * dot), rho)
    x, z = radius * math.cos(phi) - rho, radius * math.sin(phi)
    alpha = math.atan2(-(z - cz) / rho, (x - cx) / rho)
    return x, z, alpha, radius - rho


def projection(lost, *, heading, origin, length=1.0, angle=0.15):
    s, x, xp = (float(lost[k][0]) for k in ("s", "x", "xp"))
    rho, phi = length / angle, angle * s / length
    # The local x lies along the radial normal to the reference circle.
    xz = world(
        rho * (math.cos(phi) - 1) + x * math.cos(phi), (rho + x) * math.sin(phi), heading, origin
    )
    return [*xz, heading - phi + math.atan(xp)]


def same_physics(first, second):
    for a, b in zip(first, second, strict=True):
        for name in ("x", "xp", "y", "yp", "s", "t", "p", "particleID"):
            if name in a or name in b:
                if name not in a or name not in b or not np.array_equal(a[name], b[name]):
                    raise RuntimeError(f"Global-output switch changed {name}")


def ids(lost, out, expected_lost=(1,), expected_out=(2,)):
    close(lost["particleID"], expected_lost, 0)
    close(out["particleID"], expected_out, 0)


def shown(values):
    return "(" + ", ".join(f"{v:.12g}" for v in values) + ")"


def track(
    lattice,
    rays,
    *,
    heading=0,
    origin=(0.41, -0.26),
    p0=1700,
    global_coords=True,
    files=None,
    commands="",
    memcheck=False,
):
    """Fresh native process; optional Memcheck reports only the identified loss-output error."""
    beam = {k: np.asarray(rays, dtype=float)[:, j] for j, k in enumerate(("x", "xp", "y", "yp"))}
    beam.update(p=np.full(len(rays), p0), particleID=np.arange(1, len(rays) + 1))
    inputs = {
        "line.lte": lattice,
        "run.ele": (
            "&run_setup lattice=line.lte,use_beamline=BL,output=out.sdds,losses=lost.sdds,"
            f"p_central={p0},losses_include_global_coordinates={int(global_coords)},"
            "random_number_seed=78231 &end\n"
            f"&floor_coordinates filename=floor.sdds,X0={origin[0]},Z0={origin[1]},"
            f"theta0={heading:.17g} &end\n" + commands + "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
        ),
        **(files or {}),
    }
    if not memcheck:
        result = native.run(inputs, beams={"beam.sdds": beam}, outputs=("lost.sdds", "out.sdds"))
        return native.particles(result["lost.sdds"]), native.particles(result["out.sdds"]), 0

    executable = shutil.which("elegant")
    if not executable or not shutil.which("valgrind") or "RPN_DEFNS" not in os.environ:
        raise RuntimeError("Put elegant and valgrind on PATH and set RPN_DEFNS")
    with tempfile.TemporaryDirectory(prefix="elegant-loss-memcheck-") as tmp:
        work = Path(tmp)
        for name, contents in inputs.items():
            (work / name).write_text(contents)
        native.write_beam(work / "beam.sdds", beam)
        done = subprocess.run(
            [
                "valgrind",
                "--vgdb=no",
                "--track-origins=yes",
                "--error-exitcode=97",
                "--log-file=memcheck.log",
                executable,
                "run.ele",
            ],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=290,
            check=False,
        )
        log = (work / "memcheck.log").read_text()
        match = re.search(r"ERROR SUMMARY: (\d+) errors from (\d+) contexts", log)
        if match is None:
            raise RuntimeError(f"Missing Memcheck summary:\n{log}")
        errors, contexts = map(int, match.groups())
        if (done.returncode, errors, contexts) not in ((0, 0, 0), (97, 1, 1)):
            raise RuntimeError(
                f"Unexpected native/Memcheck failure:\n{done.stdout}{done.stderr}\n{log}"
            )
        if errors and (
            "Syscall param write(buf) points to uninitialised byte(s)" not in log
            or "dump_lost_particles" not in log
        ):
            raise RuntimeError(f"Unexpected Memcheck context:\n{log}")
        lost, out = (
            native.particles(pysdds.read(work / name)) for name in ("lost.sdds", "out.sdds")
        )
        return lost, out, errors


def obstruction(xmin=-1, xmax=1, zmin=0.2, zmax=0.5):
    return (
        "SDDS1\n&parameter name=ZCenter,type=double,units=m,fixed_value=0 &end\n"
        "&parameter name=XCenter,type=double,units=m,fixed_value=0 &end\n"
        "&parameter name=Superperiodicity,type=long,fixed_value=1 &end\n"
        "&column name=Z,type=double,units=m &end\n&column name=X,type=double,units=m &end\n"
        f"&data mode=ascii &end\n5\n{zmin} {xmin}\n{zmax} {xmin}\n{zmax} {xmax}\n"
        f"{zmin} {xmax}\n{zmin} {xmin}\n"
    )
