---
title: "OPT-35: MALIGN energy-offset matrices omit the finite-speed momentum conversion"
hide:
  - toc
---

# OPT-35 — MALIGN energy-offset matrices omit the finite-speed momentum conversion

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant master (`0355dd6`, committed 2026-11-03) |
| Tags | <span class="bb-tag">known-limitation</span> <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/malign_mat.c:170–181](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/malign_mat.c#L170-L181)<br>[src/track_data.c:960–961](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L960-L961)<br>[src/compute_matrices.c:1615–1620](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1615-L1620)<br>[doc/malign.tex:1–7](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/malign.tex#L1-L7)<br>[src/do_tracking.c:1996–1999](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L1996-L1999)<br>[src/do_tracking.c:3644–3650](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L3644-L3650) |

With MALIGN.DE active in matrix calculations, the affine momentum offset treats fractional total-energy change as fractional momentum change. The documented DE = Δγ/γ₀ requires Δp/p₀ = DE/β₀² to first order, where β₀ is reference speed divided by c. At p₀ = 2.000 mc and DE = 0.001000, matrix_output gives the fractional momentum offset C₆ = 0.001000 instead of 0.001250, corresponding to 914.2 eV instead of 1.143 keV of added energy. Native tracking and the relativistic mass shell agree, and an equivalent DP offset gives the correct nominal momentum. The source explicitly acknowledges the missing factor.

Run `python OPT-35.py` with [_elegant.py](../helpers/_elegant.py), [_momentum_matrix.py](../helpers/_momentum_matrix.py) and [_diagnostics.py](../helpers/_diagnostics.py) in the same directory.

??? example "Reproducer: OPT-35.py"

    [Download OPT-35.py](OPT-35.py)

    ```python
    """OPT-35: MALIGN energy-offset matrices omit the finite-speed momentum conversion

    With MALIGN.DE active in matrix calculations, the affine momentum offset treats fractional total-energy change as fractional momentum change. The documented DE = Δγ/γ₀ requires Δp/p₀ = DE/β₀² to first order, where β₀ is reference speed divided by c. At p₀ = 2.000 mc and DE = 0.001000, matrix_output gives the fractional momentum offset C₆ = 0.001000 instead of 0.001250, corresponding to 914.2 eV instead of 1.143 keV of added energy. Native tracking and the relativistic mass shell agree, and an equivalent DP offset gives the correct nominal momentum. The source explicitly acknowledges the missing factor.

    The script compares MALIGN.DE matrix offsets with an 80-digit mass-shell calculation and native nominal-particle tracking. It varies momentum, offset sign, order and pass inclusion, with equivalent DP and zero-offset controls, and checks the leading coefficient as DE tends to zero.
    """

    import mpmath as mp
    from _elegant import verdict
    from _momentum_matrix import close, discrepancy, track, value

    mp.mp.dps = 80
    MASS_EV = mp.mpf("510999.06")  # Electron rest energy in the executable's SDDS constants.


    def case(p0, de, *, dp=False, on_pass=-1, force=0, order=2):
        pe, ee = mp.mpf(str(p0)), mp.mpf(str(de))
        # gamma_f=gamma_0*(1+DE); rationalize sqrt(1+a)-1 to keep the zero exact.
        a = (1 + 1 / pe**2) * (2 * ee + ee**2)
        exact = a / (mp.sqrt(1 + a) + 1)
        option = f"DP={float(exact):.17g}" if dp else f"DE={de:.17g}"
        result = track(
            f"A: MALIGN,{option},ON_PASS={on_pass},FORCE_MODIFY_MATRIX={force}\nBL: LINE=(A)\n",
            p0,
            [0],
            order=order,
        )
        active_matrix, active_track = on_pass == -1 or bool(force), on_pass in (-1, 0)
        expected = float(exact) if active_matrix else 0.0
        faulty = (float(exact) if dp else de) if active_matrix else 0.0
        expected_p = p0 * (1 + float(exact)) if active_track else p0
        observed_p = float(result["particles"]["p"][0])
        close(observed_p, expected_p, 4e-14 * max(1, p0), "nominal mass-shell momentum")
        c6 = value(result, "C6")
        close(
            value(result, "C6", output="individual.sdds", element="A"),
            c6,
            1e-15,
            "individual nominal momentum offset",
        )
        failed = discrepancy(c6, expected, faulty, 1e-15, "missing energy-to-momentum factor")
        # Use high precision for both energy increments; no subtraction of rounded doubles.
        energy = (mp.sqrt(1 + (pe * (1 + mp.mpf(str(c6)))) ** 2) - mp.sqrt(1 + pe**2)) * MASS_EV
        expected_energy = mp.sqrt(1 + pe**2) * ee * MASS_EV if active_matrix else mp.mpf(0)
        print(
            f"p0={p0:g} mc {option} order={order} on_pass={on_pass} force={force}: "
            f"C6={c6:.16g} expected={expected:.16g} (80-digit mass shell); "
            f"tracked p={observed_p:.16g} expected={expected_p:.16g} mc; "
            f"matrix energy={float(energy):.16g} expected={float(expected_energy):.16g} eV"
        )
        return failed, c6, exact


    def main():
        failures = []
        for p0 in (0.5, 2.0, 20.0):
            small = []
            for de in (-1e-3, 1e-3, -1e-5, 1e-5):
                failed, observed, exact = case(p0, de)
                failures.append(failed)
                if abs(de) == 1e-5:
                    small.append((observed, exact))
            observed_coefficient = (small[1][0] - small[0][0]) / (2e-5)
            exact_coefficient = float((small[1][1] - small[0][1]) / mp.mpf("0.00002"))
            leading = 1 + 1 / p0**2
            # Central DE differences converge to 1/beta_0^2; finite-step error is O(DE^2).
            close(exact_coefficient, leading, 6e-8, "exact leading coefficient convergence")
            print(
                f"p0={p0:g} mc leading coefficient: matrix={observed_coefficient:.16g}, "
                f"exact +/-1e-5={exact_coefficient:.16g}, limit={leading:.16g} (1/beta_0^2)"
            )
        for kwargs in ({"dp": True}, {"on_pass": 0}, {"on_pass": 1}):
            if case(2.0, 0.001, **kwargs)[0]:
                raise AssertionError("DP or documented matrix-pass exclusion control failed")
        if case(2.0, 0)[0]:
            raise AssertionError("Zero-offset control failed")
        for kwargs in ({"order": 1}, {"order": 3}, {"on_pass": 0, "force": 1}):
            failures.append(case(2.0, 0.001, **kwargs)[0])
        verdict(any(failures))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

