---
title: "OPT-12: Knob construction applies absolute limits to additive corrections"
hide:
  - toc
---

# OPT-12 — Knob construction applies absolute limits to additive corrections

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Source | [KnobOptimizer.__init__, match.py:1334–1351](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/match.py#L1334-L1351)<br>[Vary.limits contract](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/match.py#L454-L456) |

`match_knob()` applies a variable's absolute `Vary.limits` to the additive correction it solves for, without subtracting the variable's starting value. The correction's allowed interval must be the limits shifted by that starting value, and ordinary matching respects the absolute bounds. For a dimensionless variable starting at 8 with limits [−10, 10] and target 12, the generated knob reaches 12.00 instead of stopping at the allowed maximum 10.00. Conversely, starting at 8 with valid limits [7, 10] and target 9 fails during knob setup. The error appears at the matched endpoint itself, without extrapolation or later combination of knobs.

Run `python OPT-12.py` with [_matching.py](../helpers/_matching.py), [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: OPT-12.py"

    [Download OPT-12.py](OPT-12.py)

    ```python
    """OPT-12: Knob construction applies absolute limits to additive corrections

    `match_knob()` applies a variable's absolute `Vary.limits` to the additive correction it solves for,
    without subtracting the variable's starting value. The correction's allowed interval must be the
    limits shifted by that starting value, and ordinary matching respects the absolute bounds. For a
    dimensionless variable starting at 8 with limits [−10, 10] and target 12, the generated knob reaches
    12.00 instead of stopping at the allowed maximum 10.00. Conversely, starting at 8 with valid limits
    [7, 10] and target 9 fails during knob setup. The error appears at the matched endpoint itself,
    without extrapolation or later combination of knobs.

    The script uses one variable k on an empty line, starting at 8 and read back as the target by a
    tracking-free action. With limits [−10, 10] and target 12, ordinary `Line.match()` stops at 10,
    while `match_knob()` keeps the unshifted limits on the correction a in k = 8 + a instead of [−18,
    2]; the generated knob is checked at its start, midpoint and endpoint. With limits [7, 10] and
    target 9, ordinary matching succeeds and knob setup raises an error.
    """

    from _matching import ReadKnobs
    from _prebuilt import identity, np, xt

    identity()
    print("EXECUTION Python only; no tracking kernels invoked")


    def construct(limits, goal, method):
        line = xt.Line(elements=[])
        line.build_tracker(compile=False)  # This check evaluates variables without tracking.
        line["k"] = 8.0
        action = ReadKnobs(line, ("k",))
        params = {
            "vary": [xt.Vary("k", limits=limits, step=1e-4)],
            "targets": [xt.Target("k", goal, tol=1e-11, action=action)],
            "verbose": False,
            "show_call_counter": False,
        }
        if method == "knob":
            opt = line.match_knob(
                knob_name="scan",
                knob_value_start=0.0,
                knob_value_end=1.0,
                run=False,
                **params,
            )
        else:
            opt = line.match(solve=False, **params)
        return line, opt


    for mode in ("direct", "knob"):
        line, opt = construct((-10.0, 10.0), 12.0, mode)
        opt.step(2)
        print(
            mode,
            "after_two_steps k",
            line["k"],
            "varied",
            [(v.name, v.get_value(), v.limits) for v in opt.vary],
        )
        if mode == "direct":
            np.testing.assert_allclose(line["k"], 10.0, rtol=0, atol=1e-12)
        else:
            opt.solve()
            np.testing.assert_allclose(line["k"], 12.0, rtol=0, atol=1e-11)
            opt.generate_knob()
            np.testing.assert_allclose(line["k"], 8.0, rtol=0, atol=1e-12)
            line["scan"] = 0.5
            np.testing.assert_allclose(line["k"], 10.0, rtol=0, atol=1e-11)
            line["scan"] = 1.0
            print("generated endpoint k", line["k"], "specified interval", (-10.0, 10.0))
            np.testing.assert_allclose(line["k"], 12.0, rtol=0, atol=1e-11)

    # Complementary falsification: an initial value and target both within limits.
    line, opt = construct((7.0, 10.0), 9.0, "direct")
    opt.solve()
    np.testing.assert_allclose(line["k"], 9.0, rtol=0, atol=1e-11)
    try:
        construct((7.0, 10.0), 9.0, "knob")
    except ValueError as exc:
        print("legal k=8, target=9, limits=[7,10] rejected during knob setup:", str(exc))
        assert "below lower limit" in str(exc)
    else:
        raise AssertionError("Expected untranslated auxiliary bound to reject zero correction")
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, Python only (no tracking kernels), CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, Python only (no tracking kernels), CPU, serial |

