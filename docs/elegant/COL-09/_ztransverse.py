"""Shared setup for the broadband ZTRANSVERSE reproducers: decks, analytic impedance, Fourier oracle.

The beam is twelve electron macroparticles (p = 50 m_e c, 2 nC) in two adjacent time bins of 50 ps, six particles
each, so that the transverse source moment of each bin is six times its offset. `kicks` tracks it
through a ZTRANSVERSE element, either the built-in broadband resonator or a table holding the
documented impedance of the same resonator on the same frequency grid; `oracle` is the kick of the
analytic impedance, evaluated as a finite Fourier series without Elegant's FFT.
"""

from __future__ import annotations

import numpy as np
from _elegant import particles, run

ELECTRON_EV = 0.51099906e6  # SDDS constants.h me_mev, in eV
RS, Q, FR = 1e6, 1.0, 1e9  # shunt impedance Ohm/m, quality factor, resonant frequency Hz
DT = 50e-12  # bin width, s
CHARGE, COUNT, P = 2e-9, 12, 50.0
TIMES = np.repeat([0.0, DT], COUNT // 2)  # s; group 1 in one bin, group 2 in the next


def impedance(r):
    """Documented broadband impedance Z(r) = (-i/r) Rs / [1 + i Q (r - 1/r)], r = f/fr, Ohm/m.

    Z(0) = Rs/Q is the limit r -> 0 of the formula (the bracket tends to -iQ/r).
    """
    r = np.asarray(r, dtype=float)
    safe = np.where(r == 0, 1.0, r)
    z = (-1j / safe) * RS / (1 + 1j * Q * (safe - 1 / safe))
    return np.where(r == 0, RS / Q, z)


def sdds_table(n_bins):
    """ASCII SDDS impedance table on the FFT frequency grid f_k = k/(N dt), k = 0..N/2.

    The loader doubles the DC row (it assumes a folded FFT of the impedance), so the table holds
    Rs/(2Q) there; it reads the real part alone at the Nyquist row.
    """
    k = np.arange(n_bins // 2 + 1)
    f = k / (n_bins * DT)
    z = impedance(f / FR)
    re = z.real.copy()
    re[0] /= 2
    head = (
        "SDDS1\n"
        "&column name=f, type=double, units=Hz &end\n"
        "&column name=re, type=double, units=Ohms/m &end\n"
        "&column name=im, type=double, units=Ohms/m &end\n"
        f"&data mode=ascii &end\n! page number 1\n{len(k)}\n"
    )
    rows = zip(f.tolist(), re.tolist(), z.imag.tolist())
    return head + "".join(f"{a!r} {b!r} {c!r}\n" for a, b, c in rows)


def show(mode, label, observed, expected):
    """Print one case with 16 significant digits."""

    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"

    print(f"{mode} {label}: observed={fmt(observed)} expected={fmt(expected)}")


def kicks(mode, x, n_bins, broadband):
    """Horizontal slope kicks of the two particle groups, from a run with offsets `x` (m)."""
    if broadband:
        element = (
            f"ZTRANSVERSE,BROAD_BAND=1,RS={RS!r},Q={Q!r},FREQ={FR!r},"
            f"N_BINS={n_bins},BIN_SIZE={DT!r},YFACTOR=0,SMOOTHING=0,INTERPOLATE=0"
        )
    else:
        element = (
            'ZTRANSVERSE,INPUTFILE="impedance.sdds",FREQCOLUMN="f",ZXREAL="re",ZXIMAG="im",'
            "YFACTOR=0,SMOOTHING=0,INTERPOLATE=0"
        )
    lattice = f"Q0: CHARGE,TOTAL={CHARGE!r}\nZ: {element}\nBL: LINE=(Q0,Z)\n"
    commands = (
        "&global_settings echo_namelists=0 &end\n"
        f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},output=out.sdds &end\n"
        "&run_control n_steps=1,n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    files = {"lattice.lte": lattice, "run.ele": commands}
    if not broadband:
        files["impedance.sdds"] = sdds_table(n_bins)
    beam = {"x": x, "t": TIMES, "p": np.full(COUNT, P), "particleID": np.arange(1, COUNT + 1)}
    result = run(files, mode, {"beam.sdds": beam})
    if "adjust" in result["log"].lower():
        raise RuntimeError("ZTRANSVERSE changed the requested bin settings")
    out = particles(result["out.sdds"])
    if len(out["xp"]) != COUNT:
        raise RuntimeError(f"{mode}: particles lost")
    return out["xp"].reshape(2, -1).mean(axis=1)


def oracle(x, n_bins, nyquist_re=None):
    """Kick of the analytic impedance, as slopes of the two groups.

    A real periodic signal of period T = N dt has the Green function (Fourier series)
        W[d] = [Z_0 + Re Z_{N/2} (-1)^d + 2 Re sum_{k=1}^{N/2-1} Z_k exp(2 pi i k d / N)] / T,
    with Z_k = Z(f_k), f_k = k/T, for a delay of d bins. The slope kick of a particle in bin i is
        dx' = q sum_j x_j W[b_i - b_j] / (p0 m c^2 / e),
    q the macroparticle charge in C, x_j the offsets (a source moment of the bins), p0 m c^2/e
    the beam momentum in volts. `nyquist_re` replaces Re Z_{N/2} (to evaluate a given endpoint).
    """
    period = n_bins * DT
    k = np.arange(1, n_bins // 2)
    z = impedance(np.arange(n_bins // 2 + 1) / period / FR)
    nyquist = z[-1].real if nyquist_re is None else nyquist_re
    q = CHARGE / COUNT
    source = x.reshape(2, -1).sum(axis=1)  # m, summed offset of each group
    bins = np.array([0, 1])
    delay = bins[:, None] - bins[None, :]  # b_i - b_j
    series = np.array(
        [
            z[0].real
            + nyquist * (-1.0) ** d
            + 2 * np.sum((z[k] * np.exp(2j * np.pi * k * d / n_bins)).real)
            for d in delay.ravel()
        ]
    ).reshape(2, 2)
    return q * (series @ source) / period / (P * ELECTRON_EV)
