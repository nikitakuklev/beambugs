---
title: "TRK-53: RFCA and RFCW lose transverse drift through weak-gain logarithmic cancellation"
hide:
  - toc
---

# TRK-53 — RFCA and RFCW lose transverse drift through weak-gain logarithmic cancellation

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/simple_rfca.c:1026–1055](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1026-L1055)<br>[src/compute_matrices.c:2535–2554](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L2535-L2554)<br>[src/compute_matrices.c:2591–2598](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L2591-L2598)<br>[src/track_data.c:608–612](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L608-L612)<br>[src/simple_rfca.c:525–530](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L525-L530)<br>[RFCA contract](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/rfca.tex#L1-L17)<br>[src/simple_rfca.c:550–551](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L550-L551)<br>[src/simple_rfca.c:595–608](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L595-L608)<br>[src/simple_rfca.c:998–1009](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L998-L1009)<br>[src/simple_rfca.c:503–515](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L503-L515)<br>[src/compute_matrices.c:2465–2466](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L2465-L2466) |

RFCA's default matrix tracker can corrupt transverse drift near zero net acceleration, including at ordinary RF voltage. In a 0.4 m, 1 MV, 500 MHz cavity at 180°, a 3 GeV/c electron with 10 µrad initial slope moves 3.996 µm instead of 4.000 µm, a 4.480 nm error. Independent momentum-profile quadrature identifies cancellation in the logarithmic transport coefficient, and kick tracking preserves the near-drift result. RFCW shares the failure; exactly zero voltage bypasses it.

Run `python TRK-53.py` with [_precision.py](../helpers/_precision.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-53.py"

    [Download TRK-53.py](TRK-53.py)

    ```python
    """TRK-53: RFCA and RFCW lose transverse drift through weak-gain logarithmic cancellation

    RFCA's default matrix tracker can corrupt transverse drift near zero net acceleration, including at ordinary RF voltage. In a 0.4 m, 1 MV, 500 MHz cavity at 180°, a 3 GeV/c electron with 10 µrad initial slope moves 3.996 µm instead of 4.000 µm, a 4.480 nm error. Independent momentum-profile quadrature identifies cancellation in the logarithmic transport coefficient, and kick tracking preserves the near-drift result. RFCW shares the failure; exactly zero voltage bypasses it.

    Track fixed rays through ordinary-voltage zero crossings and compare with 70-digit momentum-profile quadrature. Check a zero-voltage drift, aligned kick tracking and finite-gain controls, then exercise the analytic and SRS coefficients at ultralow voltage using their own energy profiles. The native executable is selected by PATH and RPN_DEFNS.
    """

    import math

    from _precision import (
        LENGTH,
        MASS,
        P0,
        C,
        mp,
        native,
        position_line,
        require_close,
        rf_analytic,
        rf_srs,
        rf_transport,
        verdict,
    )


    def cavity(kind, p0, voltage, phase, frequency, kicks, model="none"):
        beta = p0 / math.sqrt(1 + p0 * p0)
        # Kick tracking references its first kick midpoint, matrix tracking the body midpoint.
        reference = LENGTH / (2 * max(kicks, 1) * C * beta)
        extra = f",CELL_LENGTH={LENGTH},LSC=0" if kind == "RFCW" else ""
        return (
            f"{kind},L={LENGTH},VOLT={voltage:.17g},FREQ={frequency:.17g},PHASE={phase},"
            f"T_REFERENCE={reference:.17g},END1_FOCUS=0,END2_FOCUS=0,CHANGE_P0=0,"
            f'N_KICKS={kicks},BODY_FOCUS_MODEL="{model}",STANDING_WAVE={int(model == "srs")}' + extra
        )


    def main():
        defect = False
        for kind, voltage, phase, slope, kicks in (
            ("RFCA", 0.0, 180, 1e-5, 0),
            ("RFCA", 1e6, 180, 1e-5, 0),
            ("RFCW", 1e6, 180, 1e-5, 0),
            ("RFCA", 1e6, 0, 3e-5, 0),
            ("RFCA", 1e6, 180, 1e-5, 16),
            ("RFCA", 1e6, 90, 1e-5, 0),
        ):
            out, _ = native(
                cavity(kind, P0, voltage, phase, 5e8, kicks),
                [{"xp": 0.0}, {"xp": slope}, {"xp": -slope}],
            )
            expected = mp.mpf(slope) * rf_transport(P0, voltage, phase, 5e8, slope)
            error = position_line(
                f"{kind}/V={voltage:.0e}/phase={phase}/kicks={kicks}",
                out["x"][1],
                expected,
                "70-digit momentum-profile integral",
            )
            if voltage == 0 or kicks or phase == 90:
                require_close(out["x"][1], expected, 5e-14)
            else:
                defect |= error > 1e-12
            require_close(out["x"][2], -mp.mpf(float(out["x"][1])), 2e-17)

        # A separate ultralow-voltage test pins the analytic and SRS limits;
        # the ordinary-voltage cases above determine the practical priority.
        p0 = math.sqrt((3e9 / (MASS * 1e6)) ** 2 - 1)
        slope = 1e-4
        for model in ("none", "srs"):
            for voltage in (1e6, 1e-8):
                out, r12 = native(
                    cavity("RFCA", p0, voltage, 90, 1e6, 0, model),
                    [{"xp": 0.0}, {"xp": slope}, {"xp": -slope}],
                    p0,
                    matrix=True,
                )
                expected = (
                    rf_srs(p0, voltage) if model == "srs" else rf_transport(p0, voltage, 90, 1e6, slope)
                )
                error = position_line(
                    f"RFCA/{model}/V={voltage:.0e}",
                    out["x"][1],
                    slope * expected,
                    "70-digit model-matched crest solution",
                )
                analytic = rf_srs(p0, voltage) if model == "srs" else rf_analytic(p0, voltage)
                if r12 is None:
                    raise RuntimeError("Missing analytic matrix")
                print(
                    f"RFCA/{model}/V={voltage:.0e}/analytic: R12={r12:.17g} "
                    f"expected={float(analytic):.17g} m (70-digit energy-profile/SRS solution)"
                )
                if voltage == 1e6:
                    require_close(out["x"][1], slope * expected, 5e-14)
                    require_close(r12, analytic, 5e-11)
                else:
                    defect |= error > 1e-12 or abs(mp.mpf(r12) - analytic) > 1e-10
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial |

