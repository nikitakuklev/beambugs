"""SCT-13: MATTER recoil probability per slice exceeds one and saturates

In MATTER with ELECTRON_RECOIL and NUCLEAR_BREMSSTRAHLUNG enabled, the material is sliced so that
only the elastic-scattering probability per slice stays below PLIMIT; the recoil probability per
slice has no bound. When it exceeds one, every electron receives exactly one recoil per slice
instead of a Poisson-distributed number. For electrons of momentum 20 mₑc crossing 1 m of dilute
hydrogen with 2.500 expected recoils each, the model's own rates leave exp(−2.500) = 8.208% of
electrons without energy loss; Elegant leaves none at the default PLIMIT, and the expected fraction
once PLIMIT forces 377 slices. At the default PLIMIT, saturation needs momenta below 66.55 mₑc in
hydrogen.

The script tracks 20000 on-axis electrons through one hydrogen MATTER element whose density makes the
recoil rate of the source 2.5 events per electron, once with the default PLIMIT (one slice) and once
with PLIMIT=1e-4 (377 slices), and counts the electrons that lose no energy. Both are compared with
the zero-event probability of the source's own integrated recoil and bremsstrahlung rates; the
fine-slice run is the control, compared with the exact finite-slice product.
"""

import numpy as np
from _elegant import modes, verdict
from _matter_model import AMU_SOURCE, PI, RE, Y0, gamma_loss, model, track

COUNT, SEED = 20000, 246813
P0, LENGTH, Z, A = 20.0, 1.0, 1, 1.0  # m_e c, m; hydrogen
TARGET_RECOIL = 2.5  # mean recoils per electron over the element
# Recoil rate of matter.c:128, L n 2 pi Z r_e^2 / P0 (1/y0 - 1) with n = rho/(AMU A), solved for rho.
RHO = TARGET_RECOIL / (LENGTH / (AMU_SOURCE * A) * 2 * PI * Z * RE**2 / P0 * (1 / Y0 - 1))
THRESHOLD = 1e-10  # gamma loss; the smallest recoil is y0 = 1e-8, unchanged electrons give < 3e-13

defect = False
for mode in modes():
    for plimit in (0.05, 1e-4):
        rates = model(Z, A, RHO, LENGTH, P0, plimit)
        n = rates.sections
        p_recoil, p_brems = rates.recoil / n, rates.brems / n
        # Physical law of the model's rates: independent Poisson recoil and photon counts with means
        # rates.recoil and rates.brems, so P(no loss) = exp(-(recoil + brems)).
        poisson = np.exp(-(rates.recoil + rates.brems))
        # The source's n Bernoulli slices give (1 - p_R)^n (1 - p_B)^n, valid only when p_R <= 1.
        sliced = (1 - p_recoil) ** n * (1 - p_brems) ** n if p_recoil <= 1 else None
        element = (
            f"L={LENGTH!r},Z={Z},A={A!r},RHO={RHO!r},PLIMIT={plimit!r},"
            "ENERGY_DECAY=0,NUCLEAR_BREMSSTRAHLUNG=1,ELECTRON_RECOIL=1"
        )
        out = track(mode, element, P0, SEED, COUNT)
        unchanged = int(np.sum(gamma_loss(P0, out["p"]) <= THRESHOLD))
        expected = COUNT * poisson
        sigma = np.sqrt(COUNT * poisson * (1 - poisson))
        print(
            f"{mode} PLIMIT={plimit:g} (slices={n}, recoil probability per slice {p_recoil:.6g}), "
            f"electrons without energy loss: observed={unchanged} expected={expected:.2f}±{sigma:.2f} "
            "(Poisson), finite-slice product "
            + (f"{COUNT * sliced:.2f}" if sliced is not None else "undefined (probability > 1)")
        )
        if plimit == 0.05:
            defect |= p_recoil > 1 and abs(unchanged - expected) > 5 * sigma
        elif sliced is None or abs(unchanged - COUNT * sliced) > 5 * np.sqrt(
            COUNT * sliced * (1 - sliced)
        ):
            raise SystemExit(f"{mode}: the fine-slice control misses the finite-slice prediction")
print(f"recoil and photon means per electron: {rates.recoil:.6g}, {rates.brems:.6g}; 5 sigma")
verdict(defect)
