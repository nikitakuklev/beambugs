---
title: "COL-01: LSpaceChargeCalc and LImpedance apply the Fourier kick at z + z_c + Δz/2 instead of z"
hide:
  - toc
---

# COL-01 — `LSpaceChargeCalc` and `LImpedance` apply the Fourier kick at z + z_c + Δz/2 instead of z

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Collective effects |
| Priority | P1 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [LSpaceChargeCalc.cc#L115-L119](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/LSpaceChargeCalc.cc#L115-L119)<br>[LImpedance.cc#L102-L106](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Impedances/LImpedance.cc#L102-L106)<br>[LSpaceChargeCalc.cc#L175-L187](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/LSpaceChargeCalc.cc#L175-L187)<br>[LImpedance.cc#L160-L174](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Impedances/LImpedance.cc#L160-L174)<br>[LSpaceChargeCalc.cc#L247-L261](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/LSpaceChargeCalc.cc#L247-L261)<br>[LImpedance.cc#L194-L210](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Impedances/LImpedance.cc#L194-L210)<br>[Grid1D::getGridZ](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/Grid1D.cc#L132-L135) |

`LSpaceChargeCalc` (default Fourier branch) and `LImpedance` bin the bunch on a grid of length C centred at the midpoint z_c of the bunch extrema, but reconstruct the kick with the phase 2πz/C + π, as if bin j were centred at −C/2 + jΔz, with Δz = C/N_b for N_b bins. A particle at z therefore receives the kick that belongs at z + z_c + Δz/2, and the kick changes when the bunch is shifted rigidly. In a 250 m ring with 128 bins, the space-charge kick on a Gaussian bunch with rms length 12.5 m deviates from the analytic kick by up to 12.84% of its peak when the bunch is centred at z = 0 and by 76.26% when centred at z = 5 m.

Run `python COL-01.py`.

??? example "Reproducer: COL-01.py"

    [Download COL-01.py](COL-01.py)

    ```python
    """COL-01: `LSpaceChargeCalc` and `LImpedance` apply the Fourier kick at z + z_c + Δz/2 instead of z

    `LSpaceChargeCalc` (default Fourier branch) and `LImpedance` bin the bunch on a grid of length C centred at the midpoint z_c of the bunch extrema, but reconstruct the kick with the phase 2πz/C + π, as if bin j were centred at −C/2 + jΔz, with Δz = C/N_b for N_b bins. A particle at z therefore receives the kick that belongs at z + z_c + Δz/2, and the kick changes when the bunch is shifted rigidly. In a 250 m ring with 128 bins, the space-charge kick on a Gaussian bunch with rms length 12.5 m deviates from the analytic kick by up to 12.84% of its peak when the bunch is centred at z = 0 and by 76.26% when centred at z = 5 m.

    The script builds a deterministic Gaussian bunch (quantile sampling, 20000 macroparticles), tracks copies centred at 0, 1, 5 and 20 m through `LSpaceChargeCalc` and through `LImpedance` loaded with the space-charge impedance Z/n, and compares each kick with the analytic Gaussian kick at the particle's own position. The control compares the same kicks with the analytic kick evaluated at z + z_c + Δz/2, which agrees to the grid's resolution in every case and isolates the position error.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.core.impedances import LImpedance
    from orbit.core.spacecharge import LSpaceChargeCalc
    from orbit.utils.consts import mass_proton
    from scipy.special import ndtri

    C, NBINS, N, SIG, NPHYS, B_A = 250.0, 128, 20000, 12.5, 2e14, 3.0
    DZ = C / NBINS
    U = SIG * ndtri((np.arange(N) + 0.5) / N)  # deterministic Gaussian sample, mean 0
    G = 1.0 + 2.0 * np.log(B_A)  # geometry factor g = 1 + 2 ln(b/a)
    Z0 = 376.730313412  # free-space impedance mu0*c (ohm)


    def analytic(zrel, b):
        # Longitudinal space charge of a uniform beam in a pipe, per turn:
        # dE = -C g r0 m / gamma^2 * dlambda/dz, with lambda(z) = N_phys/(sqrt(2 pi) sigma) exp(-z^2/(2 sigma^2))
        # particles per metre (the formula of upstream examples/SpaceCharge/sc1d/test_lsc_gauss.py).
        sp = b.getSyncParticle()
        fac = C * G * b.classicalRadius() * b.mass() / sp.gamma() ** 2
        lam = NPHYS / (np.sqrt(2 * np.pi) * SIG) * np.exp(-0.5 * (zrel / SIG) ** 2)
        return -fac * (-zrel / SIG**2) * lam


    def run(shift, use_impedance):
        b = Bunch()
        b.mass(mass_proton)
        b.macroSize(NPHYS / N)
        b.getSyncParticle().kinEnergy(1.0)
        for zi in U:
            b.addParticle(0.0, 0.0, 0.0, 0.0, zi + shift, 0.0)
        if use_impedance:
            # Space-charge impedance per harmonic Z/n = i Z0 g / (2 beta gamma^2), as LSpaceChargeCalc adds it
            sp = b.getSyncParticle()
            calc = LImpedance(C, 0, NBINS)
            calc.assignImpedance(
                [complex(0.0, Z0 * G / (2 * sp.beta() * sp.gamma() ** 2))] * (NBINS // 2 - 1)
            )
        else:
            calc = LSpaceChargeCalc(B_A, C, 0, 1, NBINS)
        calc.trackBunch(b)
        z = np.array([b.z(i) for i in range(N)])
        de = np.array([b.dE(i) for i in range(N)])
        return z, de, b


    worst = 0.0
    control_ok = True
    for use_imp in (False, True):
        for shift in (0.0, 1.0, 5.0, 20.0):
            z, de, b = run(shift, use_imp)
            zc = 0.5 * (z.max() + z.min())  # grid centre used by the code
            exact = analytic(z - shift, b)  # kick at the particle's own position
            moved = analytic(z - shift + zc + DZ / 2, b)  # kick at z + z_c + dz/2
            peak = np.max(np.abs(exact))
            core = np.abs(z - shift) < 3 * SIG
            err = np.max(np.abs(de - exact)[core]) / peak
            err_moved = np.max(np.abs(de - moved)[core]) / peak
            worst = max(worst, err)
            control_ok &= err_moved < 0.01
            name = "LImpedance(Z/n = SC)" if use_imp else "LSpaceChargeCalc"
            print(
                f"{name}, bunch centre {shift:4.1f} m: max|dE - analytic(z)|/peak observed {err:.4f}, expected < 0.01 "
                f"(analytic kick at the particle's z); control max|dE - analytic(z + z_c + dz/2)|/peak {err_moved:.4f}"
            )

    print("DEFECT REPRODUCED" if worst > 0.05 and control_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

