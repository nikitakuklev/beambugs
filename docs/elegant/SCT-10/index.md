---
title: "SCT-10: CBSCAT computes a negative Klein–Nishina total cross section just above its Thomson cutoff"
hide:
  - toc
---

# SCT-10 — CBSCAT computes a negative Klein–Nishina total cross section just above its Thomson cutoff

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [Klein–Nishina total, CBScat.c:447–456](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/CBScat.c#L447-L456)<br>[per-electron total, CBScat.c:217](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/CBScat.c#L217)<br>[scatter probability, CBScat.c:289–300](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/CBScat.c#L289-L300)<br>[cross-section choices, cbscat.tex:52–62](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/cbscat.tex#L52-L62)<br>[positive wavelength, CBScat.c:113–114](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/CBScat.c#L113-L114) |

When Elegant's CBSCAT element, which models laser Compton scattering, uses its default Klein–Nishina cross section, the total cross section it evaluates just above ε = 10⁻⁶ can be negative, where ε is the laser photon energy in the electron rest frame divided by mₑc². Cancellation between large terms destroys the result, and then no electron scatters. For 25 mₑc electrons, a head-on laser and a Thomson probability of 0.025, the code's total is −28.95 times the Thomson value at ε = 1.200×10⁻⁶ instead of 1.000, and none of 24000 electrons scatters instead of about 600; at ε = 5.000×10⁻⁶ the ratio is 0.6069 and 345 scatter.

Run `python SCT-10.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: SCT-10.py"

    [Download SCT-10.py](SCT-10.py)

    ```python
    """SCT-10: CBSCAT computes a negative Klein–Nishina total cross section just above its Thomson cutoff

    When Elegant's CBSCAT element, which models laser Compton scattering, uses its default Klein–Nishina
    cross section, the total cross section it evaluates just above ε = 10⁻⁶ can be negative, where ε is
    the laser photon energy in the electron rest frame divided by mₑc². Cancellation between large terms
    destroys the result, and then no electron scatters. For 25 mₑc electrons, a head-on laser and a
    Thomson probability of 0.025, the code's total is −28.95 times the Thomson value at ε = 1.200×10⁻⁶
    instead of 1.000, and none of 24000 electrons scatters instead of about 600; at ε = 5.000×10⁻⁶ the
    ratio is 0.6069 and 345 scatter.

    The script tracks 24000 electrons of momentum 25 mₑc through a CBSCAT element with a head-on laser
    whose wavelength sets ε to 1.2×10⁻⁶, 5×10⁻⁶ or 1×10⁻⁴, with the number of laser photons chosen so
    that the Thomson scattering probability is 0.025, and counts the scattered electrons. The expected
    count uses the Klein–Nishina total obtained by integrating the differential cross section and
    checked against its closed form in 70-digit arithmetic. The same beam with the Thomson cross
    section, whose total is exact, and the case ε = 1×10⁻⁴, where cancellation does not appreciably
    change the expected event count, are the controls.
    """

    import math
    import sys
    from decimal import Decimal, localcontext

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
    P = 25.0  # electron momentum in units of m_e c; reference and beam momentum
    COUNT, SPOT, LENGTH, PROBABILITY = 24000, 1.5e-3, 8e-6, 0.025
    SIGMA_T = 8 * math.pi / 3 * RE**2
    SIGMAS = 4  # binomial standard deviations allowed for an event count


    def gauss_legendre(function, lower, upper, order=200):
        """Integral of a smooth function by Gauss–Legendre quadrature, error far below 1e-14."""
        nodes, weights = np.polynomial.legendre.leggauss(order)
        x = (upper - lower) / 2 * nodes + (upper + lower) / 2
        return (upper - lower) / 2 * np.sum(weights * function(x))


    def klein_nishina_ratio(eps):
        """sigma_KN / sigma_T: the differential cross section integrated over cos(theta) in [-1, 1].

        d sigma / d Omega = r_e^2/2 q^2 (q + 1/q - sin^2 theta) with q = 1/(1 + eps (1 - cos theta)),
        and sigma_T = 8 pi r_e^2 / 3, so the ratio is (3/8) integral q^2 (q + 1/q - 1 + u^2) du.
        The closed form below, evaluated with 70 digits, must agree with the quadrature.
        """

        def differential(u):
            q = 1 / (1 + eps * (1 - u))
            return 3 / 8 * q * q * (q + 1 / q - 1 + u * u)

        integral = gauss_legendre(differential, -1.0, 1.0)
        with localcontext() as context:
            context.prec = 70
            e = Decimal(eps)
            log = (1 + 2 * e).ln()
            closed = (
                Decimal(3)
                / 4
                * (
                    (1 + e) / e**2 * (2 * (1 + e) / (1 + 2 * e) - log / e)
                    + log / (2 * e)
                    - (1 + 3 * e) / (1 + 2 * e) ** 2
                )
            )
        if abs(integral - float(closed)) > 5e-15:
            sys.exit("the quadrature and the closed form of the Klein-Nishina total disagree")
        return float(closed)


    def source_ratio(eps):
        """Elegant's kleinNishinaTotalXsec(eps) / thomsonTotalXsec(), the same double-precision expression."""
        if eps < 1e-6:
            return 1.0
        pre = 2 * math.pi
        term1 = (1 + eps) / (eps * eps) * (2 * (1 + eps) / (1 + 2 * eps) - math.log(1 + 2 * eps) / eps)
        term2 = math.log(1 + 2 * eps) / (2 * eps)
        term3 = (1 + 3 * eps) / ((1 + 2 * eps) * (1 + 2 * eps))
        return pre * (term1 + term2 - term3) / (8 * math.pi / 3)


    def overlap_integral(wavelength, beta):
        """Scattering probability of a point electron in units of its on-focus Thomson value.

        The electron at z meets the pulse at the pulse coordinate z (1 + 1/beta), so the pulse length
        LENGTH becomes sigma_eff = LENGTH/(1 + 1/beta) along z, and the photon density seen by the
        electron is a Gaussian in z/sigma_eff. The transverse area grows as 1 + (z/z_R)^2 with the
        Rayleigh range z_R = 4 pi SPOT^2/wavelength (the hourglass). Integrating the Gaussian weight
        over the +/-5 sigma window that CBSCAT uses gives the factor; on focus without hourglass it
        would be 1, and the on-focus Thomson probability is PROBABILITY by the choice of photon number.
        """
        sigma_eff = LENGTH / (1 + 1 / beta)
        rayleigh = 4 * math.pi * SPOT**2 / wavelength
        return gauss_legendre(
            lambda u: np.exp(-u * u / 2) / np.sqrt(2 * np.pi) / (1 + (u * sigma_eff / rayleigh) ** 2),
            -5.0,
            5.0,
        )


    def scattered(mode, wavelength, cross_section):
        """Number of electrons that CBSCAT scatters; the beam starts with zero slopes."""
        photons = PROBABILITY * 2 * math.pi * SPOT**2 / SIGMA_T
        lattice = (
            f"K: CBSCAT, LASER_WAVELENGTH={wavelength!r}, N_PHOTONS={photons!r}, SIGMA_X={SPOT!r}, "
            f'SIGMA_Y={SPOT!r}, LASER_PULSE_LENGTH={LENGTH!r}, N_STEPS=81, CROSS_SECTION="{cross_section}", '
            "EXACT_RECOIL=1\nBL: LINE=(K)\n"
        )
        commands = (
            "&global_settings inhibit_seed_permutation=1 &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P!r}, output=out.sdds, "
            "random_number_seed=1357911 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        columns = {name: np.zeros(COUNT) for name in ("x", "xp", "y", "yp", "t")} | {
            "p": np.full(COUNT, P)
        }
        out = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})[
            "out.sdds"
        ]
        # Nonfinite slopes would count as zero scattered electrons and mimic the suppression.
        if out.n_pages != 1 or len(out[0, "p"]) != COUNT:
            sys.exit(f"{mode}: the output is not one page of {COUNT} particles")
        if not all(np.all(np.isfinite(out[0, name])) for name in ("p", "xp", "yp")):
            sys.exit(f"{mode}: the output holds nonfinite p, xp or yp")
        return int(np.count_nonzero(np.hypot(out[0, "xp"], out[0, "yp"]) > 0))


    gamma = math.sqrt(1 + P * P)
    beta = P / gamma
    defect = False
    for mode in modes():
        for target in (1.2e-6, 5e-6, 1e-4):
            # Rest-frame photon energy E' = gamma E_h (1 - beta cos(theta_0)) with cos(theta_0) = -1 for a
            # head-on laser, eps = E'/(m_e c^2): choose the wavelength that gives the target eps.
            wavelength = H * C / E_CHARGE / 1e6 * gamma * (1 + beta) / (target * ELECTRON_MEV)
            eps = gamma * (H * C / wavelength / E_CHARGE / 1e6) * (1 + beta) / ELECTRON_MEV
            ratio = klein_nishina_ratio(eps)
            thomson_events = COUNT * PROBABILITY * overlap_integral(wavelength, beta)
            print(
                f"{mode} eps={eps:.4g} source expression sigma_KN/sigma_T: "
                f"observed=[{source_ratio(eps):.10g}] expected=[{ratio:.10g}]"
            )
            for cross_section, expected in (
                ("thomson", thomson_events),
                ("klein-nishina", thomson_events * ratio),
            ):
                events = scattered(mode, wavelength, cross_section)
                # Binomial count of COUNT electrons, each with probability expected/COUNT.
                tolerance = SIGMAS * math.sqrt(expected * (1 - expected / COUNT))
                bad = abs(events - expected) > tolerance
                print(
                    f"{mode} eps={eps:.4g} {cross_section} scattered electrons: "
                    f"observed=[{events}] expected=[{expected:.1f}] (+-{tolerance:.1f})"
                )
                # Controls: every Thomson case and the Klein-Nishina case at eps = 1e-4. Only the two
                # smaller-eps Klein-Nishina cases decide the verdict.
                control = cross_section == "thomson" or target == 1e-4
                if bad and control:
                    sys.exit(f"{mode}: a control disagrees: the comparison is not valid")
                if not control:
                    defect |= bad
    print(f"comparison: {SIGMAS} binomial standard deviations")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |

