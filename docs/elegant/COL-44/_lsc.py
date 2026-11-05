"""Shared setup for the longitudinal space-charge reproducers: Elegant's physical constants,
deterministic round electron beams, one-run tracking, Elegant's periodic FFT kick rebuilt in NumPy,
and the open-boundary (isolated-bunch) kick of the same uniform-disk model.

Lengths are in m, momenta p = beta*gamma in units of m_e c, energies in eV, and z = c t grows
toward the tail of the bunch. Every model here uses p, as Elegant's impedance does (its Bessel
argument is k r_b / p), so the comparisons isolate the boundary condition and the reference
momentum rather than the p-versus-gamma choice.
"""

from __future__ import annotations

import numpy as np
from _elegant import particles, run
from scipy.integrate import quad
from scipy.special import erf, k1, ndtri

# SDDS constants.h, the values Elegant computes with.
C = 2.99792458e8
EPSILON_0 = 1e7 / (4 * np.pi * C**2)
Z0 = 4e-7 * np.pi * C
ME_EV = 0.51099906e6
RADIUS_FACTOR = 1.7  # Elegant's default: r_b = 1.7 (sigma_x + sigma_y)/2


def round_beam(count: int, sigma_ct: float, sigma_r: float, p: float) -> dict:
    """Gaussian quantiles in z = c t with rms sigma_ct, all at momentum p, zero slopes.

    Transverse positions lie on a golden-angle circle rescaled to rms sigma_r exactly in x and y,
    so Elegant's effective radius is RADIUS_FACTOR * sigma_r. Quantiles (i + 1/2)/count make the
    bunch symmetric about z = 0.
    """
    z = sigma_ct * ndtri((np.arange(count) + 0.5) / count)
    phi = np.arange(count) * np.pi * (np.sqrt(5) - 1)
    x, y = np.cos(phi), np.sin(phi)
    x = (x - x.mean()) / x.std() * sigma_r
    y = (y - y.mean()) / y.std() * sigma_r
    return {
        "x": x,
        "y": y,
        "t": z / C,
        "p": np.full(count, p),
        "particleID": np.arange(1, count + 1),
    }


def track(elements: str, line: str, beam: dict, p0: float) -> dict[str, np.ndarray]:
    """Serial run of `beam` through BL: LINE=(line); final coordinates ordered by particle ID."""
    commands = (
        "&global_settings echo_namelists=0 &end\n"
        f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0!r},output=out.sdds &end\n"
        "&run_control n_steps=1,n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    lattice = elements.rstrip("\n") + f"\nBL: LINE=({line})\n"
    result = run({"lattice.lte": lattice, "run.ele": commands}, beams={"beam.sdds": beam})
    out = particles(result["out.sdds"])
    if len(out["p"]) != len(beam["p"]):
        raise SystemExit("particles were lost")
    return out


def energy(p: np.ndarray | float) -> np.ndarray:
    """Total energy in eV of electrons with momentum p (m_e c)."""
    return ME_EV * np.sqrt(1 + np.asarray(p) ** 2)


def periodic_kick(z: np.ndarray, p: float, rb: float, charge: float, length: float, bins: int):
    """Elegant's LSC energy kick (eV) at each particle, rebuilt in NumPy, and the grid period (m).

    As in lsc.c: bin width h = (z_max - z_min)/(bins - 3), centers at z_min + i h, nearest-center
    deposition, voltage V = IFFT[-i Z(k) FFT(counts)] q_m c/h over `bins` points with
    Z(k) = Z0 length/(pi k rb^2) [1 - a K1(a)], a = k rb/p, k = 2 pi f/c, zero at DC and Nyquist,
    and linear interpolation between bin centers; the kick is -V. A discrete transform of length
    `bins` is periodic, so this is the field of a train of copies spaced bins*h apart.
    """
    lower = z.min()
    h = (z.max() - lower) / (bins - 3)
    counts = np.bincount(np.floor((z - lower) / h + 0.5).astype(int), minlength=bins)
    k = 2 * np.pi * np.fft.rfftfreq(bins, h)
    impedance = np.zeros(len(k))
    a = k[1:] * rb / p
    impedance[1:] = Z0 * length / (np.pi * rb**2 * k[1:]) * (1 - a * k1(a))
    impedance[-1] = 0
    voltage = np.fft.irfft(-1j * impedance * np.fft.rfft(counts), bins) * charge / len(z) * C / h
    return -np.interp(z, lower + h * np.arange(bins), voltage), bins * h


def open_shape(q: float, big_r: float) -> float:
    """Dimensionless on-axis field h(q, R) of an isolated Gaussian bunch of uniform disks.

    A disk of radius r_b carrying charge dQ, seen in its rest frame from a distance gamma*u on
    the axis, has the longitudinal field dQ/(2 pi eps0 r_b^2) [sign(u) - gamma u /
    sqrt(gamma^2 u^2 + r_b^2)]; E_z is unchanged by the boost. Summing over a Gaussian line
    density of rms sigma, with q = z/sigma and R = r_b/(gamma sigma):
        h(q, R) = erf(q/sqrt 2) - int phi(y) (q - y)/sqrt((q - y)^2 + R^2) dy,
    phi the unit normal density. Its Fourier transform is the impedance Z(k) above, with no
    periodic images.
    """
    integrand = lambda y: (
        np.exp(-y * y / 2) / np.sqrt(2 * np.pi) * (q - y) / np.sqrt((q - y) ** 2 + big_r**2)
    )
    smooth = quad(integrand, -14, 14, points=[q], epsabs=1e-14, epsrel=1e-13, limit=400)[0]
    return float(erf(q / np.sqrt(2)) - smooth)


def open_kick(q: float, sigma: float, p: float, rb: float, charge: float, length: float) -> float:
    """Energy kick (eV) of an electron at z = q sigma from an isolated Gaussian bunch of total
    charge `charge` over `length`: -charge length/(2 pi eps0 rb^2) h(q, rb/(p sigma))."""
    return -charge * length / (2 * np.pi * EPSILON_0 * rb**2) * open_shape(q, rb / (p * sigma))


def at(z: np.ndarray, kick: np.ndarray, z0: float) -> float:
    """Kick at z0, interpolated between the sorted particle positions (quantile beams)."""
    order = np.argsort(z)
    return float(np.interp(z0, z[order], kick[order]))
