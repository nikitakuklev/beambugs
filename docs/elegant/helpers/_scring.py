"""Shared setup for the SCMULT space-charge reproducers: a closed four-bend ring, a gridded beam and
the Gaussian-beam kick computed from the Poisson integral, independently of Elegant's code.

One turn around a ring of four 1 m sector bends (angle pi/2, K1 = -0.5 m^-2), each preceded by a
drift of length L. `insert_sceffects` places one SCMULT after the first drift, and a WATCH right
after it records every particle's slope kick before the first bend.
"""

from __future__ import annotations

import numpy as np
from _elegant import particles, run
from scipy.integrate import quad
from scipy.special import ndtri

# SDDS constants.h, as Elegant uses them.
C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
P = 10.0  # momentum, units of m_e c (electrons)
GAMMA = np.sqrt(1 + P**2)
BETA = P / GAMMA
CHARGE = 1e-9  # C
BEND_LENGTH = 1.0  # m, four bends make the ring


def levels(n: int, uniform: bool = False) -> np.ndarray:
    """n symmetric levels of unit rms: Gaussian quantiles, or equally spaced for a uniform beam."""
    q = (np.arange(n) + 0.5) / n
    q = q - 0.5 if uniform else ndtri(q)
    return q / np.std(q)


def beam(
    sx: float, sy: float, times: np.ndarray, n: int = 8, copies: int = 1
) -> dict[str, np.ndarray]:
    """Particles on the full grid of n levels in x and y and the given arrival times, zero slopes.

    The rms sizes are exactly sx and sy, which is what SCMULT reads from the particles.
    """
    q = levels(n)
    x, y, t = (a.ravel() for a in np.meshgrid(sx * q, sy * q, times, indexing="ij"))
    x, y, t = (np.tile(a, copies) for a in (x, y, t))
    return {"x": x, "y": y, "t": t, "p": np.full(len(x), P), "particleID": np.arange(1, len(x) + 1)}


def circumference(length: float) -> float:
    return 4 * (BEND_LENGTH + length)


def track(
    mode: str,
    length: float,
    particles_in: dict[str, np.ndarray],
    *,
    horizontal: int = 1,
    vertical: int = 1,
    nonlinear: int = 1,
    uniform: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Slope kicks (xp, yp) of every particle after SCMULT, in input order, and the effective
    integrating lengths of the two planes.

    The effective length of a drift of constant beam size is L (beta_in + beta_out) / (2 beta_out),
    from the trapezoid weighting that SCMULT applies to the matched beta functions at the entrance
    and exit of the drift; the beta functions are read from the matched Twiss output.
    """
    lattice = (
        f"Q: CHARGE,TOTAL={CHARGE!r}\n"
        f"D: DRIF,L={length!r}\n"
        'W: WATCH,FILENAME="watch.sdds",MODE="coordinates"\n'
        f"A: DRIF,L={length!r}\n"
        f"B: SBEND,L={BEND_LENGTH!r},ANGLE={np.pi / 2!r},K1=-0.5\n"
        "BL: LINE=(Q,D,W,B,A,B,A,B,A,B)\n"
    )
    commands = (
        "&global_settings echo_namelists=0 &end\n"
        f'&insert_sceffects name="D",skip=1,horizontal={horizontal},vertical={vertical},'
        f"nonlinear={nonlinear},uniform_distribution={uniform} &end\n"
        f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},output=out.sdds &end\n"
        "&run_control n_steps=1,n_passes=1 &end\n"
        '&twiss_output filename="twiss.sdds",matched=1 &end\n'
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    result = run(
        {"lattice.lte": lattice, "run.ele": commands},
        mode,
        {"beam.sdds": particles_in},
        outputs=("watch.sdds", "twiss.sdds"),
    )
    twiss = result["twiss.sdds"]
    names = list(twiss[0, "ElementName"])
    if not names.index("D") < names.index("MYSC") < names.index("W"):
        raise RuntimeError(f"unexpected element order {names}")
    leff = np.array(
        [
            length * (b[0] + b[names.index("D")]) / (2 * b[names.index("MYSC")])
            for b in (twiss[0, "betax"], twiss[0, "betay"])
        ]
    )
    if not (np.all(np.isfinite(leff)) and np.all(leff > 0)):
        raise RuntimeError(f"{mode}: invalid effective lengths {leff}")
    out = particles(result["watch.sdds"])
    if not np.array_equal(out["particleID"], particles_in["particleID"]):
        raise RuntimeError(f"{mode}: particles lost before the kick")
    # The initial slopes are zero, so the slopes after SCMULT are the kicks.
    return np.column_stack([out["xp"], out["yp"]]), leff


def poisson_field(x: np.ndarray, y: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Transverse field (Fx, Fy), in 1/m, of a Gaussian charge density with rms sizes sx and sy.

    Heat-kernel solution of the 2D Poisson equation, written with positive real integrands:
      Fx = x/2 int_0^inf exp[-x^2/(2(sx^2+t)) - y^2/(2(sy^2+t))] / [(sx^2+t)^(3/2) (sy^2+t)^(1/2)] dt
    and Fy with the planes exchanged. Round beams give Gauss's law,
    F = (1 - exp[-r^2/(2 sigma^2)]) (x, y)/r^2, and the origin gives the linear force,
    Fx = x/[sx (sx + sy)].
    """
    r2 = (sy / sx) ** 2  # t = sx^2 u makes the integrals dimensionless

    def integral(a: float, b: float, plane: int) -> float:
        def integrand(u: float) -> float:
            common = np.exp(-(a**2) / (2 * (1 + u)) - b**2 / (2 * (r2 + u)))
            common /= 2 * np.sqrt((1 + u) * (r2 + u))
            return common * (a / (1 + u) if plane == 0 else b / (r2 + u)) / sx

        return quad(integrand, 0, np.inf, epsabs=0, epsrel=1e-13)[0]

    pairs, inverse = np.unique(np.column_stack([x, y]), axis=0, return_inverse=True)
    unique = np.array(
        [[integral(px / sx, py / sx, 0), integral(px / sx, py / sx, 1)] for px, py in pairs]
    )
    return unique[inverse.reshape(-1)]


def kick(
    particles_in: dict[str, np.ndarray],
    leff: np.ndarray,
    *,
    nonlinear: bool,
    uniform: bool = False,
) -> np.ndarray:
    """Physical slope kick (xp, yp) of every particle, shape (n, 2).

    dx' = K L lambda F, with K = 2 r_e / (beta^2 gamma^3) and L the effective length of the plane;
    K multiplies the particle number per metre lambda, not a charge density.
    beta^2 gamma^3 = P^2 sqrt(1 + P^2). The longitudinal density lambda = N f(z), in particles per
    metre with N = CHARGE/e, is Gaussian, f = exp(-z^2/(2 sz^2))/(sqrt(2 pi) sz), or uniform over the
    interval of the same rms, f = 1/(sqrt(12) sz). The field F is the Gaussian transverse field
    (nonlinear) or its linear part near the axis.
    """
    sx, sy = np.std(particles_in["x"]), np.std(particles_in["y"])
    x = particles_in["x"] - np.mean(particles_in["x"])
    y = particles_in["y"] - np.mean(particles_in["y"])
    z = BETA * C * (particles_in["t"] - np.mean(particles_in["t"]))
    sz = np.std(z)
    shape = np.full_like(z, 1 / (np.sqrt(12) * sz)) if uniform else _gauss(z, sz)
    strength = 2 * (CHARGE / QE) * RE / (P**2 * np.sqrt(1 + P**2))
    if nonlinear:
        field = poisson_field(x, y, sx, sy)
    else:
        field = np.column_stack([x / (sx * (sx + sy)), y / (sy * (sx + sy))])
    return strength * shape[:, None] * leff[None, :] * field


def _gauss(z: np.ndarray, sz: float) -> np.ndarray:
    return np.exp(-(z**2) / (2 * sz**2)) / (np.sqrt(2 * np.pi) * sz)


def fmt(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"


def require_finite(label: str, *arrays: np.ndarray) -> None:
    """Fail the run when an array holds a NaN or infinity; NaN would otherwise pass as a mismatch."""
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise RuntimeError(f"{label}: nonfinite values")


def compare(
    label: str, observed: np.ndarray, expected: np.ndarray, rtol: float, atol: float = 1e-18
) -> bool:
    """Print the first particle's (xp, yp) and return True when any particle disagrees.

    Arrays of different shape or with a nonfinite entry make the comparison invalid: the run fails
    instead of returning a verdict.
    """
    observed, expected = np.asarray(observed), np.asarray(expected)
    if observed.shape != expected.shape:
        raise RuntimeError(f"{label}: shapes {observed.shape} and {expected.shape} differ")
    require_finite(label, observed, expected)
    print(f"{label}: observed={fmt(observed[0])} expected={fmt(expected[0])}")
    return not np.allclose(observed, expected, rtol=rtol, atol=atol)
