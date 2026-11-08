---
title: "SCT-22: touschek_scatter Monte Carlo rate does not converge when the momentum aperture lies inside the sampled energy spread"
hide:
  - toc
---

# SCT-22 — touschek_scatter Monte Carlo rate does not converge when the momentum aperture lies inside the sampled energy spread

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P4 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Tags | <span class="bb-tag">extreme-regime</span> <span class="bb-tag">contract-unsettled</span> <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [touschekScatter.c:1232–1238](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L1232-L1238)<br>[range, touschekScatter.c:1047–1049](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L1047-L1049)<br>[touschekScatter.nl:17](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.nl#L17)<br>[touschekScatter.c:541–560](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L541-L560)<br>[`moeller`, touschekScatter.c:1405–1415](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L1405-L1415)<br>[touschekScatter.c:519](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L519)<br>[touschekScatter.c:626–627](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L626-L627)<br>[touschekScatter.c:690–693](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L690-L693)<br>[sdds_support.c:2591](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L2591)<br>[touschekScatter.c:179–180](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L179-L180)<br>[touschekScatter.c:618–625](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L618-L625)<br>[touschek_scatter, elegant.tex:7693–7697](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7693-L7697)<br>[elegant.tex:7767](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7767)<br>[elegant.tex:7773–7777](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7773-L7777)<br>[elegant.tex:7798–7799](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7798-L7799) |

touschek_scatter samples the incoming beam out to distribution_cutoff (3 by default) times the rms energy spread σδ and counts every outgoing particle beyond the momentum aperture δm, including particles that were already beyond it. When δm lies inside the sampled spread, nearly forward collisions of those particles, whose Møller cross section is limited only by the sampled angle range, dominate: the Monte Carlo rate SLocalRate grows with n_simulated instead of converging, and a few particles carry the loss weights. With δm = 0.5σδ, SLocalRate is 9.372×10¹², 2.687×10¹³ and 1.395×10¹⁴ s⁻¹ for 2×10⁴, 2×10⁵ and 2×10⁶ samples, against 1.030×10¹⁰ s⁻¹ from Piwinski's formula. With δm = 31.25σδ and 2×10⁶ samples, the rate is 1.748% below Piwinski's.

Run `python SCT-22.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: SCT-22.py"

    [Download SCT-22.py](SCT-22.py)

    ```python
    """SCT-22: touschek_scatter Monte Carlo rate does not converge when the momentum aperture lies inside the sampled energy spread

    touschek_scatter samples the incoming beam out to distribution_cutoff (3 by default) times the
    rms energy spread σδ and counts every outgoing particle beyond the momentum aperture δm,
    including particles that were already beyond it. When δm lies inside the sampled spread, nearly
    forward collisions of those particles, whose Møller cross section is limited only by the sampled
    angle range, dominate: the Monte Carlo rate SLocalRate grows with n_simulated instead of
    converging, and a few particles carry the loss weights. With δm = 0.5σδ, SLocalRate is
    9.372×10¹², 2.687×10¹³ and 1.395×10¹⁴ s⁻¹ for 2×10⁴, 2×10⁵ and 2×10⁶ samples, against 1.030×10¹⁰
    s⁻¹ from Piwinski's formula. With δm = 31.25σδ and 2×10⁶ samples, the rate is 1.748% below
    Piwinski's.

    The script runs touschek_scatter at 2 GeV/c with 0.5 nC on a ring made of a linear one-turn map (βx
    = 12 m, βy = 6 m, no dispersion) with 2×10⁴, 2×10⁵ and 2×10⁶ samples, for a momentum aperture of
    31.25σδ (control), of 0.5σδ, and of 0.5σδ with the sampled longitudinal spread cut at 0.45σδ
    (control), and tracks the scattered particles through a horizontal collimator at |x| = σx. It
    compares SLocalRate with an independent quadrature of Piwinski's formula, reports the effective
    sample size of the weights, and compares the lost fraction with erfc(1), which the factorized
    collision weight gives for any aperture.
    """

    import numpy as np
    from _elegant import run, verdict
    from scipy.integrate import quad
    from scipy.special import erfc, i0e

    # SDDS constants.h
    C, E_CHARGE, ME_MEV = 2.99792458e8, 1.60217733e-19, 0.51099906  # m/s, C, MeV
    RE = 2.81794092e-15  # m, electron classical radius re_mks
    P_MEV, CHARGE = 2000.0, 0.5e-9  # MeV/c, C
    EX, EY, SIGMA_S, SIGMA_DP = 1.5e-9, 1.5e-11, 4e-3, 8e-4  # m, m, m, 1
    BX, BY, NUX, NUY, LENGTH = 12.0, 6.0, 0.23, 0.17, 120.0  # m, m, 1, 1, m (one-turn map)
    SEED = 1357911
    SIGMA_X = float(np.sqrt(BX * EX))  # m, horizontal rms size; no dispersion
    CASES = {  # label: (momentum aperture delta_m, longitudinal distribution_cutoff in sigmas)
        "control, aperture 31.25 sigma_dp": (0.025, 3.0),
        "aperture 0.5 sigma_dp": (4e-4, 3.0),
        "control, aperture 0.5 sigma_dp, sampled spread 0.45 sigma_dp": (4e-4, 0.45),
    }
    SAMPLES = (20000, 200000, 2000000)


    def ring(x_max):
        """TSCATTER, an RCOL collimator, a linear uncoupled one-turn EMATRIX and the closing TSCATTER.

        The map has beta functions BX, BY, alpha = 0 and no dispersion, so the optics are the same at
        both TSCATTER elements. Particles scattered at the second TSCATTER start at the beginning of the
        line and meet the collimator before the map."""
        terms = []
        for k, beta, tune in ((1, BX, NUX), (3, BY, NUY)):
            c, s = float(np.cos(2 * np.pi * tune)), float(np.sin(2 * np.pi * tune))
            terms += [f"R{k}{k}={c!r}", f"R{k}{k + 1}={beta * s!r}", f"R{k + 1}{k}={-s / beta!r}",
                      f"R{k + 1}{k + 1}={c!r}"]  # fmt: skip
        return (
            "TS: TSCATTER\n"
            f"AP: RCOL,X_MAX={x_max!r},Y_MAX=1\n"
            f"M: EMATRIX,L={LENGTH!r},{','.join(terms)},R55=1,R66=1\n"
            "RING: LINE=(TS,AP,M,TS)\n"
        )


    def aperture_file(delta_m):
        return (
            "SDDS1\n"
            "&column name=s, units=m, type=double &end\n"
            "&column name=ElementName, type=string &end\n"
            "&column name=ElementType, type=string &end\n"
            "&column name=ElementOccurence, type=long &end\n"
            "&column name=deltaPositive, type=double &end\n"
            "&column name=deltaNegative, type=double &end\n"
            "&data mode=ascii, no_row_counts=1 &end\n"
            f"0 TS TSCATTER 1 {delta_m!r} {-delta_m!r}\n"
            f"{LENGTH!r} TS TSCATTER 2 {delta_m!r} {-delta_m!r}\n"
        )


    def touschek(delta_m, cutoff, n):
        """Scattered bunch of the second TSCATTER and the losses of tracking it through the ring."""
        commands = (
            f'&run_setup lattice="ring.lte", use_beamline="RING", p_central_mev={P_MEV!r}, '
            f"default_order=1, random_number_seed={SEED} &end\n"
            "&twiss_output matched=1 &end\n"
            "&run_control n_passes=2 &end\n"
            f"&touschek_scatter charge={CHARGE!r}, frequency={C / LENGTH!r}, emit_x={EX!r}, "
            f"emit_y={EY!r}, sigma_dp={SIGMA_DP!r}, sigma_s={SIGMA_S!r}, "
            f"distribution_cutoff[0]=3,3,{cutoff!r}, "
            'Momentum_Aperture="aperture.sdds", Momentum_Aperture_scale=1, '
            f"n_simulated={n}, ignored_portion=0, i_start=2, i_end=2, do_track=1, "
            'bunch="%s.bun", loss="%s.los" &end\n'
        )
        files = {
            "ring.lte": ring(SIGMA_X),
            "aperture.sdds": aperture_file(delta_m),
            "run.ele": commands,
        }
        out = run(files, outputs=("run.bun", "run.los"))
        bun, los = out["run.bun"], out["run.los"]
        par = {p.name: p.data[0] for p in bun.parameters}
        par["TotalLoss"] = {p.name: p.data[0] for p in los.parameters}["TotalLoss"]
        warned = "warning" in out["log"].lower()
        return par, np.asarray(bun[0, "x"]), np.asarray(bun[0, "LRate"]), warned


    def piwinski_rate(delta_m):
        """Particles scattered beyond the +/- aperture per second, N/tau, by Piwinski's formula.

        A. Piwinski, DESY 98-179 (1998), without dispersion (sigma_h = sigma_p):
          1/tau = r^2 c N / (8 sqrt(pi) beta^2 gamma^4 ex ey sigma_s) * F,
          F = int_{tau_m}^inf [(2+1/t)^2 (t/tau_m/(1+t) - 1) + 1 - sqrt(tau_m (1+t)/t)
                               - (4+1/t)/(2t) ln(t/tau_m/(1+t))] sqrt(t/(1+t)) exp(-B1 t) I0(B2 t) dt,
          tau_m = beta^2 delta_m^2, B1 = (bx/ex + by/ey)/(2 beta^2 gamma^2),
          B2 = |bx/ex - by/ey|/(2 beta^2 gamma^2).
        The formula neglects the incoming momentum spread. Each event sends one particle beyond each
        side of a symmetric aperture, so N/tau is the sum of both sides' scattering rates.
        """
        bg = P_MEV / ME_MEV  # elegant's p_central from p_central_mev
        gamma = np.hypot(bg, 1.0)
        beta = bg / gamma
        n = CHARGE / E_CHARGE
        b1 = (BX / EX + BY / EY) / (2 * beta**2 * gamma**2)
        b2 = abs(BX / EX - BY / EY) / (2 * beta**2 * gamma**2)
        tm = (beta * delta_m) ** 2

        def integrand(u):  # t = tau_m e^u, dt = t du; exp(-B1 t) I0(B2 t) = i0e(B2 t) exp(-(B1-B2) t)
            t = tm * np.exp(u)
            log_term = (4 + 1 / t) / (2 * t) * np.log(t / tm / (1 + t))
            bracket = (
                (2 + 1 / t) ** 2 * (t / tm / (1 + t) - 1) + 1 - np.sqrt(tm * (1 + t) / t) - log_term
            )
            return t * bracket * np.sqrt(t / (1 + t)) * i0e(b2 * t) * np.exp(-(b1 - b2) * t)

        f, _ = quad(integrand, 0, 80, limit=500, epsabs=0, epsrel=1e-11)
        return n * RE**2 * C * n * f / (8 * np.sqrt(np.pi) * beta**2 * gamma**4 * EX * EY * SIGMA_S)


    # Lost fraction: with alpha = 0 and no dispersion both colliding particles sit at the same x, the
    # collision weight carries the product of their Gaussian densities, exp(-x^2/sigma_x^2), and the
    # cross section and kinematics depend only on momenta. The rate-weighted x distribution of the
    # scattered particles is therefore Gaussian with rms sigma_x/sqrt(2) for any momentum aperture, and
    # the collimator at |x| = sigma_x removes the fraction erfc(1) of the scattered rate.
    LOSS_EXPECTED = float(erfc(1.0))

    summary = {}
    for label, (delta_m, cutoff) in CASES.items():
        expected = piwinski_rate(delta_m)
        print(f"{label}: delta_m={delta_m:g}, sigma_dp={SIGMA_DP:g}, sampled |delta| <= "
              f"{cutoff:g} sigma_dp; independent Piwinski rate {expected:.6e} /s")  # fmt: skip
        rows = []
        for n in SAMPLES:
            par, x, w, warned = touschek(delta_m, cutoff, n)
            ess = w.sum() ** 2 / (w @ w)  # rate-weighted effective sample size
            lost = np.abs(x) > SIGMA_X
            fraction = par["TotalLoss"] / par["NScatter"]
            stderr = np.sqrt(np.sum(w**2 * (lost - fraction) ** 2)) / w.sum()
            rows.append(dict(rate=par["SLocalRate"], ess=ess, wmax=w.max() / w.sum(),
                             fraction=fraction, stderr=stderr))  # fmt: skip
            print(f"  n_simulated={n}: SLocalRate observed={par['SLocalRate']:.4e} /s "
                  f"(Piwinski's PLocalRate {par['PLocalRate']:.6e} /s); effective sample size "
                  f"{ess:.1f}, largest weight {rows[-1]['wmax']:.2%}; lost fraction observed="
                  f"{fraction:.4f} +/- {stderr:.4f} expected={LOSS_EXPECTED:.4f}; "
                  f"Elegant warning: {'yes' if warned else 'none'}")  # fmt: skip
        summary[label] = dict(expected=expected, rows=rows)

    # Controls: with the aperture well outside the sampled spread, the Monte Carlo rate approaches
    # Piwinski's (which neglects the incoming spread) and the lost fraction erfc(1); with the sampled
    # spread inside the aperture, the rate settles (it is lower, since the sampled beam is truncated).
    far = summary["control, aperture 31.25 sigma_dp"]
    last = far["rows"][-1]
    if not (abs(last["rate"] / far["expected"] - 1) < 0.05
            and abs(last["fraction"] - LOSS_EXPECTED) < 3 * last["stderr"] and last["ess"] > 1000):  # fmt: skip
        raise SystemExit(
            "control: the well-separated aperture does not converge to the expected values"
        )
    inner = summary["control, aperture 0.5 sigma_dp, sampled spread 0.45 sigma_dp"]["rows"]
    if not 0.5 < inner[-1]["rate"] / inner[-2]["rate"] < 2:
        raise SystemExit(
            "control: the rate with the sampled spread inside the aperture does not settle"
        )
    bad = summary["aperture 0.5 sigma_dp"]
    rows = bad["rows"]
    growth = rows[-1]["rate"] / rows[-2]["rate"]
    print(f"aperture 0.5 sigma_dp: SLocalRate at {SAMPLES[-1]} observed={rows[-1]['rate']:.4e} /s "
          f"expected about {bad['expected']:.4e} /s (Piwinski); growth from {SAMPLES[-2]} to "
          f"{SAMPLES[-1]} samples observed={growth:.3f} expected=1 within sampling error; effective "
          f"sample size {rows[-1]['ess']:.1f} against {last['ess']:.1f} for the control")  # fmt: skip
    defect = bool(growth > 3 and rows[-1]["rate"] > 100 * bad["expected"] and rows[-1]["ess"] < 50)
    print("comparison: controls need SLocalRate within 5% of Piwinski's rate, the lost fraction within 3 "
          "standard errors of erfc(1) and an effective sample size above 1000, or a rate that changes "
          "by less than a factor 2 from 2e5 to 2e6 samples; the defect needs SLocalRate growing more "
          "than threefold from 2e5 to 2e6 samples, above 100 times Piwinski's rate, with an effective "
          "sample size below 50")  # fmt: skip
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-04</span> | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed<br><small class="bb-scope">rates and weights with aperture inside sampled spread</small> | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-04</span> | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

