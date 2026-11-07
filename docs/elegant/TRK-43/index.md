---
title: "TRK-43: Matrix-method RFCA tracking uses a different R12 from its own transfer matrix"
hide:
  - toc
---

# TRK-43 — Matrix-method RFCA tracking uses a different R12 from its own transfer matrix

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`, head build) |
| Source | [`trackRfCavityWithWakes`, simple_rfca.c:1047–1066](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1047-L1066)<br>[`rf_cavity_matrix`, compute_matrices.c:2552–2554](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L2552-L2554)<br>[simple_rfca.c:525–529](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L525-L529)<br>[parameter table, track_data.c:612](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L612)<br>[rfca.tex:5–6](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/rfca.tex#L5-L6) |

An RFCA tracked with the default matrix method (N_KICKS = 0) and no body focusing computes R12 for momentum growing linearly along the cavity, while its own transfer matrix takes a uniform gradient, under which energy grows linearly. Conserved transverse momentum then gives R12 = L p₀(arcosh γ₁ − arcosh γ₀)/(γ₁ − γ₀), with γ₀, γ₁ the entrance and exit Lorentz factors and p₀ the entrance βγ; 1024-kick tracking has a relative difference of 1.568×10⁻⁶. For electrons entering a 0.7 m cavity that adds 12 MeV at 1.5 MeV kinetic energy, tracking gives R12 = 0.2229 m instead of 0.2221 m, 0.3685% too large; the error is 0.03247% at 5 MeV and 7.003×10⁻⁶ at 20 MeV.

Run `python TRK-43.py` with [_elegant.py](../helpers/_elegant.py) and [_rfca_lens.py](../helpers/_rfca_lens.py) in the same directory.

??? example "Reproducer: TRK-43.py"

    [Download TRK-43.py](TRK-43.py)

    ```python
    """TRK-43: Matrix-method RFCA tracking uses a different R12 from its own transfer matrix

    An RFCA tracked with the default matrix method (N_KICKS = 0) and no body focusing computes R12
    for momentum growing linearly along the cavity, while its own transfer matrix takes a uniform
    gradient, under which energy grows linearly. Conserved transverse momentum then gives R12 = L
    p₀(arcosh γ₁ − arcosh γ₀)/(γ₁ − γ₀), with γ₀, γ₁ the entrance and exit Lorentz factors and p₀
    the entrance βγ; 1024-kick tracking has a relative difference of 1.568×10⁻⁶. For electrons
    entering a 0.7 m cavity that adds 12 MeV at 1.5 MeV kinetic energy, tracking gives R12 = 0.2229
    m instead of 0.2221 m, 0.3685% too large; the error is 0.03247% at 5 MeV and 7.003×10⁻⁶ at 20
    MeV.

    The script sends electrons of 1.5, 5 and 20 MeV kinetic energy through a 0.7 m RFCA that adds 12 MeV
    in a static field (FREQ=0) without edge or body focusing, and measures R12 from slope pairs in
    matrix-method tracking (N_KICKS = 0), in kick tracking with 1024 kicks and in Elegant's transfer
    matrix. The expected R12 integrates the conserved transverse momentum over the uniform gradient.
    The 1024-kick tracking and the transfer matrix are the controls.
    """

    import numpy as np
    from _elegant import modes, verdict
    from _rfca_lens import LENGTH, ME_MEV, VOLT, momentum, transfer

    KINETIC = (1.5, 5.0, 20.0)  # MeV
    MATRIX_RTOL = 1e-10  # the analytic matrix and the integral are the same closed form
    KICK_RTOL = 1e-5  # 1024 midpoint kicks: discretization error O(N^-2), 1.6e-6 at 1.5 MeV


    defect = False
    for mode in modes():
        for kinetic in KINETIC:
            p_i = momentum(kinetic)
            gamma_i = 1 + kinetic / ME_MEV
            dgamma = VOLT / (1e6 * ME_MEV)
            gamma_f = gamma_i + dgamma
            p_f = float(np.sqrt(gamma_f**2 - 1))
            # A uniform gradient makes gamma linear in z: gamma(z) = gamma_i + dgamma z/L. With no
            # transverse force p_x = p_i x'_i is conserved, so dx/dz = p_i x'_i/p(z) to first order and
            # R12 = p_i * int_0^L dz/sqrt(gamma(z)^2 - 1) = L p_i/dgamma * (acosh gamma_f - acosh gamma_i).
            expected = LENGTH * p_i / dgamma * (np.arccosh(gamma_f) - np.arccosh(gamma_i))
            # Momentum linear in z instead gives L p_i/(p_f - p_i) ln(p_f/p_i): the tracking formula.
            p_linear = LENGTH * p_i / (p_f - p_i) * np.log(p_f / p_i)
            matrix_run = transfer(mode, kinetic, 0, 0, 0)
            tracked = float(matrix_run["tracked"][0, 1])
            analytic = float(matrix_run["matrix"][0, 1])
            kicks = float(transfer(mode, kinetic, 1024, 0, 0)["tracked"][0, 1])
            name = f"{mode} {kinetic:g} MeV"
            print(
                f"{name} tracking N_KICKS=0: R12 observed={tracked:.15g} expected={expected:.15g} m "
                f"(rel {tracked / expected - 1:+.6e}); momentum-linear formula {p_linear:.15g}"
            )
            print(
                f"{name} matrix_output (control): R12={analytic:.15g} (rel {analytic / expected - 1:+.3e})"
            )
            print(
                f"{name} tracking N_KICKS=1024 (control): R12={kicks:.15g} (rel {kicks / expected - 1:+.3e})"
            )
            if abs(analytic / expected - 1) > MATRIX_RTOL or abs(kicks / expected - 1) > KICK_RTOL:
                raise SystemExit(f"{name}: control fails")
            defect |= abs(tracked / expected - 1) > KICK_RTOL
    print(
        f"comparison: matrix tolerance {MATRIX_RTOL:g}, kick-tracking tolerance {KICK_RTOL:g} relative"
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

