"""Shared setup for the CSRCSBEND and CSRDRIFT reproducers: Elegant's physical constants,
deterministic electron beams, one-run tracking with the wake files Elegant exports, the analytic
steady-state CSR wake of a Gaussian bunch, and a serial Valgrind runner for the memory defects.

Lengths are in m, momenta p in units of m_e c, and the longitudinal coordinate z = c t of a
particle grows toward the tail of the bunch, as in Elegant's wake output column `s`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from _elegant import RunFailure, particles, run, write_beam
from scipy.integrate import quad
from scipy.special import ndtri

# SDDS constants.h, the values Elegant computes with: the comparisons below are then not limited
# by revisions of the CODATA constants.
C = 2.99792458e8
E_CHARGE = 1.60217733e-19
M_E = 9.1093897e-31
EPSILON_0 = 1e7 / (4 * np.pi * C**2)
R_E = E_CHARGE**2 / (4 * np.pi * EPSILON_0 * M_E * C**2)
TIMEOUT = 600  # seconds per Valgrind run


def gaussian_beam(count: int, sigma_ct: float, p0: float, chirp: float = 0.0, x0: float = 0.0):
    """Gaussian quantiles in z = c t with rms sigma_ct, all at p0 (1 + chirp z) and x = x0.

    Quantiles (i + 1/2)/count place the particles deterministically and symmetrically about z = 0.
    """
    z = sigma_ct * ndtri((np.arange(count) + 0.5) / count) if count > 1 else np.zeros(1)
    return {
        "x": np.full(count, x0),
        "t": z / C,
        "p": p0 * (1 + chirp * z),
        "particleID": np.arange(1, count + 1),
    }


def commands(p0: float, steps: int = 1, passes: int = 1, seed: int = 0) -> str:
    """Command file: track beam.sdds through BL, final coordinates in out.sdds."""
    random = f",random_number_seed={seed}" if seed else ""
    return (
        "&global_settings echo_namelists=0 &end\n"
        f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0!r},output=out.sdds{random} &end\n"
        f"&run_control n_steps={steps},n_passes={passes} &end\n"
        "&sdds_beam input=beam.sdds,reuse_bunch=1 &end\n"
        "&track &end\n"
    )


def track(elements: str, line: str, beam, p0: float, outputs=("out.sdds",), **kw):
    """Track `beam` through BL: LINE=(line) defined after `elements`; returns run()'s dict."""
    lattice = elements.rstrip("\n") + f"\nBL: LINE=({line})\n"
    return run(
        {"lattice.lte": lattice, "run.ele": commands(p0, **kw)},
        beams={"beam.sdds": beam},
        outputs=outputs,
    )


def status(elements: str, line: str, beam, p0: float, **kw) -> int:
    """Exit status of a plain serial run (0 on success): the memory defects end in a signal."""
    try:
        track(elements, line, beam, p0, outputs=(), **kw)
    except RunFailure as failure:
        return failure.status
    return 0


def page(sdds, index: int = 0) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Columns and parameters of one page of an SDDS file."""
    columns = {c.name: np.asarray(sdds[index, c.name]) for c in sdds.columns}
    parameters = {p.name: p.data[index] for p in sdds.parameters}
    return columns, parameters


def final(result: dict) -> dict[str, np.ndarray]:
    """Final particle coordinates of a run, ordered by particle ID."""
    return particles(result["out.sdds"])


def steady_state_wake(z: np.ndarray, sigma: float, k: float) -> np.ndarray:
    """Steady-state CSR energy change of a Gaussian bunch, per particle, in units of m_e c^2.

    Saldin, Schneidmiller and Yurkov (NIM A 398, 1997) and Stupakov and Emma (LCLS-TN-01-12, 2001):
        Delta gamma(z) = k int_0^inf lambda'(z + u) u^(-1/3) du,
    lambda the line density normalized to 1 and k = 2 N r_e L / (3 rho^2)^(1/3) for N electrons
    in a bend of length L and radius rho. With z = a sigma and u = sigma t^3 the integrable
    endpoint singularity disappears:
        Delta gamma = k sigma^(-4/3) int_0^inf 3 t lambda_1'(a + t^3) dt,
    lambda_1(a) = exp(-a^2/2)/sqrt(2 pi), lambda_1'(a) = -a lambda_1(a); the integrand vanishes
    beyond a + t^3 = 12.
    """

    def one(a: float) -> float:
        upper = max(0.0, 12.0 - a) ** (1 / 3)
        integrand = lambda t: -3 * t * (a + t**3) * np.exp(-0.5 * (a + t**3) ** 2)
        return quad(integrand, 0, upper, epsabs=1e-13, epsrel=1e-12, limit=200)[0]

    values = np.array([one(a) for a in np.asarray(z) / sigma])
    return k * sigma ** (-4 / 3) * values / np.sqrt(2 * np.pi)


def valgrind(elements: str, line: str, beam, p0: float, **kw) -> tuple[int, str]:
    """Serial Elegant under `valgrind --track-origins=yes`: (exit status, Valgrind and run log)."""
    if shutil.which("valgrind") is None or shutil.which("elegant") is None:
        sys.exit("valgrind and elegant must be on PATH.")
    if "RPN_DEFNS" not in os.environ:
        sys.exit("Set RPN_DEFNS to the defns.rpn file of SDDS (see the elegant installation).")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="elegant-repro-") as tmp:
        work = Path(tmp)
        (work / "lattice.lte").write_text(elements.rstrip("\n") + f"\nBL: LINE=({line})\n")
        (work / "run.ele").write_text(commands(p0, **kw))
        write_beam(work / "beam.sdds", beam)
        try:
            done = subprocess.run(
                ["valgrind", "--error-limit=no", "--track-origins=yes", "elegant", "run.ele"],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            sys.exit(f"valgrind elegant exceeded {TIMEOUT} s")
    return done.returncode, done.stdout + done.stderr
