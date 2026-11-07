"""Shared native fixtures and stable transport oracles for precision reproducers."""

from __future__ import annotations

import math
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

import _elegant
import mpmath as mp
import numpy as np
from _elegant import particles

_elegant.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-native-process deadline.
mp.mp.dps = 70
MASS = 0.51099906  # MeV/c²: the executed Elegant/SDDS constant set.
C = 299792458.0
LENGTH = 0.4
P0 = 3e9 / (MASS * 1e6)
PIN = P0 * 1.01
D = mp.mpf(PIN) / mp.mpf(P0)


def native(element, rays, p0=P0, *, matrix=False):
    """Run one element with fixed rays, rejecting loss or malformed output."""
    keys = ("x", "xp", "y", "yp", "t", "p")
    beam = {
        key: np.array([ray.get(key, p0 if key == "p" else 0.0) for ray in rays]) for key in keys
    }
    commands = (
        f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0:.17g},"
        "output=out.sdds,default_order=1,random_number_seed=271828 &end\n"
    )
    if matrix:
        commands += "&matrix_output SDDS_output=matrix.sdds,individual_matrices=1 &end\n"
    commands += (
        "&run_control n_steps=1,n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
    )
    result = _elegant.run(
        {"lattice.lte": "E: " + element + "\nBL: LINE=(E)\n", "run.ele": commands},
        beams={"beam.sdds": beam},
        outputs=("out.sdds", "matrix.sdds") if matrix else ("out.sdds",),
    )
    data = result["out.sdds"]
    if data.n_pages != 1:
        raise RuntimeError("Expected one output page")
    out = particles(data)
    if len(out["p"]) != len(rays):
        raise RuntimeError("Unexpected particle loss")
    r12 = None
    if matrix:
        data = result["matrix.sdds"]
        row = list(data[0, "ElementName"]).index("E")
        r12 = float(data[0, "R12"][row])
        if not math.isfinite(r12):
            raise RuntimeError("Nonfinite analytic R12")
    return out, r12


def verdict(defect):
    """Print the final verdict in the shared format of `_elegant.verdict`."""
    _elegant.verdict(defect)


def require_close(observed, expected, tolerance):
    """Reject a failed control instead of counting it as a reproduced defect."""
    if abs(mp.mpf(float(observed)) - expected) > tolerance:
        raise RuntimeError(f"Failed control: {observed}, expected {expected}, atol={tolerance}")


def position_line(label, observed, expected, method, *, coordinate="x"):
    """Print one native position and its independent expectation, in metres."""
    error = float(mp.mpf(float(observed)) - expected)
    print(
        f"{label}: {coordinate}={observed:.17g} expected={float(expected):.17g} m; "
        f"error={error:+.6e} m ({method})"
    )
    return abs(error)


def corrector(kick, *, length=LENGTH, x=0.001, xp=0.0001):
    """Integrate a planar uniform field between parallel end planes."""
    xx, slope = mp.mpf(x), mp.mpf(xp)
    if not length:
        return xx, mp.tan(mp.atan(slope) + mp.mpf(kick) / D)
    u0 = slope / mp.sqrt(1 + slope * slope)
    kappa = mp.sin(mp.mpf(kick)) / (mp.mpf(length) * D)

    # The transverse direction cosine grows linearly with z; x'=u/sqrt(1-u²).
    def u(z):
        return u0 + kappa * z

    position = xx + mp.quad(lambda z: u(z) / mp.sqrt(1 - u(z) ** 2), [0, length])
    final_u = u(mp.mpf(length))
    return position, final_u / mp.sqrt(1 - final_u * final_u)


def rf_transport(p0, voltage, phase, frequency, slope):
    """Momentum-profile transport, with the sloped ray's midpoint RF phase."""
    gi = mp.sqrt(1 + mp.mpf(p0) ** 2)
    beta = mp.mpf(p0) / gi
    # The explicit reference time is the axial midpoint arrival time.
    dt = mp.mpf(LENGTH) / (2 * C * beta) * (mp.sqrt(1 + mp.mpf(slope) ** 2) - 1)
    phi = mp.mpf(phase) * mp.pi / 180 + 2 * mp.pi * frequency * dt
    dg = mp.mpf(voltage) / (mp.mpf(MASS) * 1000000) * mp.sin(phi)
    pf = mp.sqrt((gi + dg) ** 2 - 1)
    return mp.quad(lambda u: mp.mpf(LENGTH) * p0 / (p0 + (pf - p0) * u), [0, 1])


def rf_analytic(p0, voltage):
    """Energy-profile quadrature for the default analytic matrix at crest."""
    gi = mp.sqrt(1 + mp.mpf(p0) ** 2)
    dg = mp.mpf(voltage) / (mp.mpf(MASS) * 1000000)
    return mp.quad(lambda u: mp.mpf(LENGTH) * p0 / mp.sqrt((gi + dg * u) ** 2 - 1), [0, 1])


def rf_srs(p0, voltage):
    """Crest solution of x_tt+x/8=0, where t=log(gamma/gamma_i)."""
    gi = mp.sqrt(1 + mp.mpf(p0) ** 2)
    dg = mp.mpf(voltage) / (mp.mpf(MASS) * 1000000)
    if not dg:
        return mp.mpf(LENGTH)
    t = mp.log1p(dg / gi)
    return mp.mpf(LENGTH) * gi / dg * mp.sqrt(8) * mp.sin(t / mp.sqrt(8))


def bend_split(angle, slices, length, x, xp, d, order=4):
    """Exact split flows from the planar curvilinear Hamiltonian.

    H_D = -(1+h*x)*sqrt(d^2-q^2), H_K = h*x+h^2*x^2/2.
    q rotates during H_D. Rationalizing its position update avoids subtracting
    near-unity ratios; the path update follows the conserved drift invariant.
    """
    h = mp.mpf(angle) / mp.mpf(length)
    x = mp.mpf(x)
    q = d * mp.mpf(xp) / mp.sqrt(1 + mp.mpf(xp) ** 2)
    distance = mp.mpf(0)
    b = 1.25992104989487316477
    ds = [
        mp.mpf(v)
        for v in (0.5 / (2 - b), (1 - b) / (2 - b) / 2, (1 - b) / (2 - b) / 2, 0.5 / (2 - b))
    ]
    ks = [mp.mpf(v) for v in (1 / (2 - b), -b / (2 - b), 1 / (2 - b), 0.0)]
    if order == 2:
        ds = list(map(mp.mpf, (0.5, 0.5)))
        ks = list(map(mp.mpf, (1.0, 0.0)))
    elif order == 6:
        ds = list(
            map(
                mp.mpf,
                (
                    0.39225680523878,
                    0.5100434119184585,
                    -0.47105338540975655,
                    0.0687531682525181,
                    0.0687531682525181,
                    -0.47105338540975655,
                    0.5100434119184585,
                    0.39225680523878,
                ),
            )
        )
        ks = list(
            map(
                mp.mpf,
                (
                    0.784513610477560,
                    0.235573213359357,
                    -1.17767998417887,
                    1.3151863206839063,
                    -1.17767998417887,
                    0.235573213359357,
                    0.784513610477560,
                    0.0,
                ),
            )
        )
    elif order != 4:
        raise ValueError(order)
    step = mp.mpf(float(length / slices))
    for _ in range(slices):
        for df, kf in zip(ds, ks, strict=True):
            a = h * step * df
            longitudinal = mp.sqrt(d * d - q * q)
            sn, cs = mp.sin(a), mp.cos(a)
            lf = longitudinal * cs - q * sn
            # (1-cos a)/h is evaluated by 2 sin²(a/2)/h.
            numer = longitudinal * 2 * mp.sin(a / 2) ** 2 + q * sn
            distance += (1 / h + x) * d * sn / lf
            x += (1 / h + x) * numer / lf
            q = q * cs + longitudinal * sn
            q -= h * step * kf * (1 + h * x)
    return x, q / mp.sqrt(d * d - q * q), distance
