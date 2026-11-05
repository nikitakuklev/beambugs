"""SCT-11: Approximate CBSCAT recoil depends on the reference momentum

With the approximate recoil of Elegant's CBSCAT element (CROSS_SECTION="thomson", EXACT_RECOIL=0),
the momentum an electron loses in a laser Compton scatter depends on the reference momentum P₀ the
user chose, although the physics does not. The code computes the fractional change ΔP/P of the
electron's momentum P and adds it to δ = P/P₀ − 1, which requires ΔP/P₀. For 12 mₑc electrons and a
30 µm laser, the same 454 scattered electrons lose a mean 2.303×10⁻⁵ mₑc at P₀ = 12 mₑc, 1.152×10⁻⁵
mₑc at P₀ = 6 mₑc and 4.607×10⁻⁵ mₑc at P₀ = 24 mₑc.

The script scatters 16000 electrons of momentum 12 mₑc off a head-on laser with the Thomson cross
section, with the same random seed so that the same 454 electrons scatter, changing only the
reference momentum p_central from 6 to 24 mₑc. The mean and maximum losses are compared with the
Thomson angular-distribution expectation, and losses are compared event by event with the run whose
reference equals the electrons' momentum. The same runs with the exact-recoil option, which does not
use the faulty expression, are the control.
"""

import math
import sys

import numpy as np
from _elegant import modes, run, verdict

# SDDS constants.h, as Elegant uses them.
C, E_CHARGE, H, RE, ELECTRON_MEV = (
    299792458.0,
    1.60217733e-19,
    6.6260755e-34,
    2.81794092e-15,
    0.51099906,
)
P = 12.0  # electron momentum, units of m_e c
COUNT, SPOT = 16000, 1.5e-3
SIGMA_T = 8 * math.pi / 3 * RE**2
REFERENCES = (6.0, 11.88, 12.0, 12.12, 24.0)
SIGMAS = 4  # standard deviations of the sample mean allowed
INVARIANCE_ATOL = 1e-12  # m_e c; the exact-recoil runs agree within 3e-14

GAMMA = math.sqrt(1 + P * P)
BETA = P / GAMMA


def losses(mode, wavelength, exact, reference):
    """Momentum loss (m_e c) of the scattered electrons, ordered as the electrons are."""
    photons = 0.03 * 2 * math.pi * SPOT**2 / SIGMA_T  # mean 0.03 Thomson scatters per electron
    lattice = (
        f"K: CBSCAT, LASER_WAVELENGTH={wavelength!r}, N_PHOTONS={photons!r}, SIGMA_X={SPOT!r}, "
        f'SIGMA_Y={SPOT!r}, LASER_PULSE_LENGTH=8e-6, N_STEPS=81, CROSS_SECTION="thomson", '
        f"EXACT_RECOIL={exact}\nBL: LINE=(K)\n"
    )
    commands = (
        "&global_settings inhibit_seed_permutation=1 &end\n"
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={reference!r}, "
        "output=out.sdds, random_number_seed=1357911 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )
    columns = {name: np.zeros(COUNT) for name in ("x", "xp", "y", "yp", "t")} | {
        "p": np.full(COUNT, P),
        "particleID": np.arange(1, COUNT + 1),
    }
    result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})
    out = result["out.sdds"]
    if out.n_pages != 1 or len(out[0, "p"]) != COUNT:
        sys.exit(f"{mode}: the output is not one page of {COUNT} particles")
    order = np.argsort(out[0, "particleID"])
    if not np.array_equal(out[0, "particleID"][order], np.arange(1, COUNT + 1)):
        sys.exit(f"{mode}: the output particle IDs differ from the input")
    # NaNs would propagate into the mean and maximum, and comparisons with them are false.
    if not all(np.all(np.isfinite(out[0, name])) for name in ("p", "xp", "yp")):
        sys.exit(f"{mode}: the output holds nonfinite p, xp or yp")
    scattered = np.hypot(out[0, "xp"][order], out[0, "yp"][order]) > 0
    if not scattered.any():
        sys.exit(f"{mode}: no electron scattered")
    return P - out[0, "p"][order][scattered], scattered


defect = False
print("momentum loss of the scattered electrons in m_e c: mean, maximum / (2A), residual")
for mode in modes():
    for wavelength in (3e-5, 3e-3):
        # Head-on (cos theta_0 = -1) photon-momentum conservation, in units of m_e c:
        # Delta P = [gamma^2 E_h (cos theta_0 - beta) - gamma E' cos theta]/(m_e c^2) with
        # E' = gamma E_h (1 - beta cos theta_0), so the loss is -Delta P = A (1 + cos theta) with
        # A = gamma^2 E_h (1 + beta)/(m_e c^2). cos theta follows the 1 + cos^2 theta law, whose
        # mean is 0 and variance 2/5: the mean loss is A, and no loss exceeds 2A.
        energy = H * C / wavelength / E_CHARGE / 1e6  # E_h, MeV
        amplitude = GAMMA**2 * energy * (1 + BETA) / ELECTRON_MEV
        for exact in (0, 1):
            runs = {ref: losses(mode, wavelength, exact, ref) for ref in REFERENCES}
            base, base_mask = runs[P]
            for reference, (loss, mask) in runs.items():
                if not np.array_equal(mask, base_mask):
                    sys.exit(f"{mode}: the reference changed which electrons scatter")
                tolerance = SIGMAS * amplitude * math.sqrt(0.4 / len(loss))
                residual = float(np.max(np.abs(loss - base)))
                observed = (float(loss.mean()), float(loss.max() / (2 * amplitude)), residual)
                bad = (
                    abs(observed[0] - amplitude) > tolerance
                    or observed[1] > 1 + 1e-9
                    or residual > INVARIANCE_ATOL
                )
                label = f"{mode} wavelength={wavelength:g} exact={exact} reference={reference:g}"
                print(
                    f"{label} ({len(loss)} events): "
                    f"observed=[{observed[0]:.5g}, {observed[1]:.5g}, {observed[2]:.3g}] "
                    f"expected=[{amplitude:.5g}+-{tolerance:.2g}, <=1, 0]"
                )
                if (exact == 1 or reference == P) and bad:
                    sys.exit(f"{mode}: a control disagrees: the comparison is not valid")
                defect |= bad
print(f"comparison: {SIGMAS} standard deviations of the mean, residual atol={INVARIANCE_ATOL:g}")
verdict(defect)
