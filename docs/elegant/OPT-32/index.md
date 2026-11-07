---
title: "OPT-32: Default analytic RFCA matrices omit finite-speed longitudinal factors"
hide:
  - toc
---

# OPT-32 — Default analytic RFCA matrices omit finite-speed longitudinal factors

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/compute_matrices.c:2504–2556](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L2504-L2556)<br>[src/compute_matrices.c:2642–2654](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L2642-L2654)<br>[src/compute_matrices.c:2492–2503](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L2492-L2503)<br>[doc/elegant.tex:364–380](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L364-L380)<br>[src/simple_rfca.c:998–1001](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L998-L1001)<br>[src/simple_rfca.c:1077–1078](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1077-L1078)<br>[src/compute_matrices.c:1916–1934](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1916-L1934) |

The default analytic RFCA matrix omits finite-speed factors required by the longitudinal coordinate sₑ = βct, where β is particle speed divided by the speed of light. Across a thin accelerating gap, R₅₅ = ∂sₑ,out/∂sₑ,in must equal βout/βin, and R₆₅ = ∂δout/∂sₑ,in requires the entrance factor 1/βin, where δ is relative momentum deviation. At electron momentum 3.000 mc, an 80.00 kV cavity at 48.00° and 650.0 MHz reports R₆₅ = 0.4995 m⁻¹ instead of 0.5265 m⁻¹. Direct particle derivatives and TW1-derived matrices agree with the relativistic impulse. Forward and reverse analytic matrices both fail.

Run `python OPT-32.py` with [_elegant.py](../helpers/_elegant.py) and [_slicing.py](../helpers/_slicing.py) in the same directory.

??? example "Reproducer: OPT-32.py"

    [Download OPT-32.py](OPT-32.py)

    ```python
    """OPT-32: Default analytic RFCA matrices omit finite-speed longitudinal factors

    The default analytic RFCA matrix omits finite-speed factors required by the longitudinal coordinate sₑ = βct, where β is particle speed divided by the speed of light. Across a thin accelerating gap, R₅₅ = ∂sₑ,out/∂sₑ,in must equal βout/βin, and R₆₅ = ∂δout/∂sₑ,in requires the entrance factor 1/βin, where δ is relative momentum deviation. At electron momentum 3.000 mc, an 80.00 kV cavity at 48.00° and 650.0 MHz reports R₆₅ = 0.4995 m⁻¹ instead of 0.5265 m⁻¹. Direct particle derivatives and TW1-derived matrices agree with the relativistic impulse. Forward and reverse analytic matrices both fail.

    The script differentiates a relativistic thin-gap time/energy impulse by complex steps and compares native matrices with four-point particle derivatives. It tests both directions, reference choices and kick counts, with TW1 and zero-voltage controls.
    """

    import numpy as np
    from _elegant import verdict
    from _slicing import MASS_MEV, C, beta, close, discrepancy, track

    VOLTAGE = 80000.0
    FREQUENCY = 650e6
    PHASE = 48.0


    def impulse(q, p0, signed_voltage, phase, reference_out):
        # Time is continuous across a thin cavity. Differentiate the physical
        # time/energy map in complex arithmetic, independently of elegant's matrix.
        s, delta = q
        pin = p0 * (1 + delta)
        g = np.sqrt(1 + pin**2)
        bi = pin / g
        angle = np.deg2rad(phase) + 2 * np.pi * FREQUENCY * s / (C * bi)
        gf = g + signed_voltage / (1e6 * MASS_MEV) * np.sin(angle)
        pf = np.sqrt(gf**2 - 1)
        return np.array([s * (pf / gf) / bi, pf / reference_out - 1])


    def oracle(p0, voltage, phase, change):
        pf = np.sqrt(
            (np.hypot(p0, 1) + voltage / (1e6 * MASS_MEV) * np.sin(np.deg2rad(phase))) ** 2 - 1
        )
        pref = pf if change else p0
        jac = np.column_stack(
            [
                impulse(np.eye(2, dtype=complex)[j] * 1e-30j, p0, voltage, phase, pref).imag / 1e-30
                for j in range(2)
            ]
        )
        return pf, pref, jac


    def main():
        failures = []
        pforward = oracle(3.0, VOLTAGE, PHASE, False)[0]
        for reverse, p0 in ((False, 3.0), (True, pforward)):
            for change in (False, True):
                voltage = -VOLTAGE if reverse else VOLTAGE
                pf, pref, expected = oracle(p0, voltage, PHASE, change)
                for model, nkicks in (("default", 0), ("default", 10), ("TW1", 10)):
                    for h in (1e-4, 5e-5):
                        rays = [np.zeros(6)]
                        for j in (4, 5):
                            for sign in (-2, -1, 1, 2):
                                q = np.zeros(6)
                                q[j] = sign * h
                                rays.append(q)
                        extra = "" if model == "default" else ",BODY_FOCUS_MODEL=TW1"
                        lat = (
                            f"Q: RFCA,L=0,VOLT={VOLTAGE},FREQ={FREQUENCY},PHASE={PHASE},"
                            f"END1_FOCUS=0,END2_FOCUS=0,CHANGE_T=0,N_KICKS={nkicks},"
                            f"CHANGE_P0={int(change)}{extra}\nBL: LINE=(Q)\n"
                        )
                        run = track(
                            f"C3_{reverse}_{change}_{model}_{nkicks}_{h}",
                            lat,
                            rays,
                            p0=p0,
                            reverse=reverse,
                            matrix=True,
                        )
                        out = np.column_stack([run["q"][:, 4], run["raw"][:, 5] / pref - 1])
                        fd = np.column_stack(
                            [
                                (
                                    out[1 + 4 * j]
                                    - 8 * out[2 + 4 * j]
                                    + 8 * out[3 + 4 * j]
                                    - out[4 + 4 * j]
                                )
                                / (12 * h)
                                for j in range(2)
                            ]
                        )
                        fd_err = close(fd, expected, 8e-11, "physical impulse derivative")
                        endpoint_err = close(
                            run["raw"][0, 5], pf, 3e-14, "relativistic energy endpoint"
                        )
                        observed = run["R"][4:6, 4:6]
                        if model == "TW1":
                            close(observed, expected, 8e-10, "TW1 matrix control")
                        else:
                            faulty = expected.copy()
                            faulty[0, 0] = 1
                            faulty[1, 0] *= beta(p0)
                            failures.append(discrepancy(observed, expected, faulty, 4e-14))
                            close(observed[1, 1], expected[1, 1], 4e-14, "correct R66")
                        print(
                            f"reverse={int(reverse)} change_p0={int(change)} model={model} "
                            f"kicks={nkicks} h={h:g}: R55={observed[0, 0]:.16g} expected={expected[0, 0]:.16g}; "
                            f"R65={observed[1, 0]:.16g} expected={expected[1, 0]:.16g} m^-1 "
                            f"(complex-step impulse); particle error={fd_err:.4g}; endpoint error={endpoint_err:.4g}"
                        )
        for phase, voltage in ((90, VOLTAGE), (PHASE, 0)):
            pf, pref, expected = oracle(3.0, voltage, phase, False)
            lat = (
                f"Q: RFCA,L=0,VOLT={voltage},FREQ={FREQUENCY},PHASE={phase},"
                "END1_FOCUS=0,END2_FOCUS=0,CHANGE_T=0\nBL: LINE=(Q)\n"
            )
            run = track(f"C3_control_{phase}_{voltage}", lat, np.zeros(6), p0=3, matrix=True)
            observed = run["R"][4:6, 4:6]
            if voltage == 0:
                close(observed, expected, 1e-14, "zero voltage")
            print(
                f"control phase={phase} voltage={voltage:g}: R55={observed[0, 0]:.16g} "
                f"expected={expected[0, 0]:.16g} (beta_out/beta_in)"
            )
        verdict(any(failures))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

