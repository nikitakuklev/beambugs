"""SCT-15: MATTER converts between mass and atom density with the proton mass

With PRESSURE, TEMPERATURE and MULTIPLICITY, MATTER computes gas density using the proton mass
m_p instead of the atomic mass unit u. ENERGY_DECAY and bremsstrahlung rates are consequently
0.7293% too large, and single-scattering and recoil rates, which also use a rounded u, are
0.7316% too large; with RHO supplied directly, the latter rates are 0.002344% too large. For
nitrogen at 100 Pa and 300 K over 10 m, the fractional energy loss is 2.978×10⁻⁵ instead of
2.956×10⁻⁵, as an independent ideal-gas atom-density calculation shows. Forced multiple
scattering converts RHO to atom density with m_p, making its angles 0.3626% too small; pressure
input gives correct angles.

The script tracks 200 on-axis electrons at 1000 mₑc through 10 m of nitrogen with ENERGY_DECAY=1
and FORCE_MCS=1, describing the same gas three ways with one seed: by PRESSURE, TEMPERATURE and
MULTIPLICITY, by its true mass density RHO = n·A·u, and by RHO = n·A·m_p. It compares each
attenuation exponent with the ideal-gas prediction, in which A and every mass unit cancel, and
compares the forced multiple-scattering kicks of the runs, which draw the same random numbers. The
true-density energy loss and the pressure-input kicks are the controls.
"""

import numpy as np
from _elegant import modes, verdict
from _matter_model import ALPHA_FS, K_BOLTZMANN, M_PROTON, RE, U_KG, track

SEED, P0, COUNT = 97531, 1000.0, 200  # P0 in m_e c
Z, A, MULT = 7, 14.0067, 2  # nitrogen: atomic number, atomic mass in AMU, atoms per molecule
PRESSURE, TEMPERATURE, LENGTH = 100.0, 300.0, 10.0  # Pa, K, m
N_ATOM = MULT * PRESSURE / (K_BOLTZMANN * TEMPERATURE)  # ideal-gas atom density, 1/m^3

# Radiation depth from the atom density directly (radiationLength, matter.c:315-349, with
# rho N_A/(A 1e-3) = n): L/X0 = 4 alpha r_e^2 n L [Z^2 (Lrad - f) + Z Lrad'], Tsai's large-Z
# logarithms for Z = 7. No mass unit enters.
a2 = (ALPHA_FS * Z) ** 2
f_z = a2 * (1 / (1 + a2) + 0.20206 + a2 * (-0.0369 + a2 * (0.0083 - a2 * 0.002)))
lrad, lradp = np.log(184.15 * Z ** (-1 / 3)), np.log(1194 * Z ** (-2 / 3))
TAU = 4 * ALPHA_FS * RE**2 * N_ATOM * LENGTH * (Z * Z * (lrad - f_z) + Z * lradp)

GAS = f"L={LENGTH!r},Z={Z},A={A!r},ENERGY_DECAY=1,FORCE_MCS=1"
CASES = (  # label, material parameters
    ("pressure input", f"PRESSURE={PRESSURE!r},TEMPERATURE={TEMPERATURE!r},MULTIPLICITY={MULT}"),
    ("RHO=n*A*u (true mass density)", f"RHO={N_ATOM * A * U_KG!r}"),
    ("RHO=n*A*m_p", f"RHO={N_ATOM * A * M_PROTON!r}"),
)

defect = False
gamma_in = np.sqrt(1 + P0**2)
for mode in modes():
    outs = {}
    for label, material in CASES:
        out = outs[label] = track(mode, f"{GAS},{material}", P0, SEED, COUNT)
        # gamma_out = gamma_in exp(-L/X0) (matter.c:113, 265); recover the exponent stably.
        g_out = np.sqrt(1 + out["p"] ** 2)
        tau = np.log1p((P0 - out["p"]) * (P0 + out["p"]) / (gamma_in + g_out) / g_out)
        print(f"{mode} {label}: L/X0 over ideal-gas prediction observed={np.mean(tau) / TAU:.10f} "
              f"(spread {np.ptp(tau) / TAU:.1e}) expected=1")  # fmt: skip
        if label.startswith("pressure"):
            defect |= abs(np.mean(tau) / TAU - 1) > 1e-6
        elif "true" in label and abs(np.mean(tau) / TAU - 1) > 1e-9:
            raise SystemExit(f"{mode}: the true-density control misses the radiation length")
    print(f"  proton mass over atomic mass unit m_p/u = {M_PROTON / U_KG:.10f}")
    # The same gas with the same seed must receive the same angular kicks. The forced-MCS width
    # uses n = rho/(m_p A) (matter.c:92, 144), so a RHO input gives kicks scaled by sqrt(u/m_p).
    ref = outs["pressure input"]
    for label in ("RHO=n*A*u (true mass density)", "RHO=n*A*m_p"):
        ratio = outs[label]["xp"] / ref["xp"]
        print(f"{mode} {label}: kick over pressure-input kick observed=[{ratio.min():.10f}, "
              f"{ratio.max():.10f}] expected=1")  # fmt: skip
        if "true" in label:
            defect |= bool(np.any(abs(ratio - 1) > 1e-9))
    print(f"  sqrt(u/m_p) = {np.sqrt(U_KG / M_PROTON):.10f}")
print("comparison: L/X0 atol 1e-6 relative (control 1e-9), kick ratios atol 1e-9")
verdict(defect)
