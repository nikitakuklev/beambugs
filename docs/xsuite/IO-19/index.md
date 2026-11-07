---
title: "IO-19: Spin configuration rejects documented Boolean arguments"
hide:
  - toc
---

# IO-19 — Spin configuration rejects documented Boolean arguments

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on xtrack release [v0.118.2](https://github.com/xsuite/xtrack/tree/v0.118.2) (`ad5e808`) |
| Pull request | [[gh:https://github.com/xsuite/xtrack/pull/923]], fix |
| Discussed in | [[gh:https://github.com/xsuite/xtrack/pull/922]] |
| Source | [Line.configure_spin signature and docstring, line.py:5108–5116](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L5108-L5116)<br>[validation and state update, lines 5117–5127](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L5117-L5127) |

Calling configure_spin with the documented Boolean arguments True or False raises AssertionError before updating the line's spin configuration. True should select automatic spin tracking, and False should disable it. The public signature, docstring and normalization branches establish the intended behavior, and independent state-transition checks reproduce both failures before and after tracker construction. The equivalent 'auto' and None inputs succeed.

Run `python IO-19.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-19.py"

    [Download IO-19.py](IO-19.py)

    ```python
    """IO-19: Spin configuration rejects documented Boolean arguments

    Calling configure_spin with the documented Boolean arguments True or False raises AssertionError before updating the line's spin configuration. True should select automatic spin tracking, and False should disable it. The public signature, docstring and normalization branches establish the intended behavior, and independent state-transition checks reproduce both failures before and after tracker construction. The equivalent 'auto' and None inputs succeed.

    The script starts fresh drift-only lines with spin disabled or automatic and calls each documented input before and after tracker construction. It compares acceptance, final spin state and the radiation compilation switch against the public Boolean contract. The 'auto' and None spellings supply controls, and all comparisons are exact.
    """

    from _prebuilt import CTX, identity, kernels, report, xt


    def main():
        identity()
        cases = {}
        failures = controls = 0
        for built in (False, True):
            for initial in (None, "auto"):
                for value in (True, False, None, "auto"):
                    line = xt.Line(elements=[xt.Drift(length=1.0)])
                    if built:
                        line.build_tracker(_context=CTX)
                    line.configure_spin(initial)
                    error = None
                    try:
                        line.configure_spin(value)
                    except AssertionError:
                        error = "AssertionError"

                    # The public contract maps True to automatic and False to disabled;
                    # None/'auto' already name those states. False compile flags are
                    # represented by absent keys in TrackerConfig.
                    expected = "auto" if value is True else None if value is False else value
                    observed = (
                        error is None,
                        line._spin_model == expected,
                        line.config.get("XTRACK_MULTIPOLE_NO_SYNRAD", False)
                        == (expected is None),
                    )
                    if type(value) is bool:
                        failures += error is not None
                    else:
                        if not all(observed):
                            raise RuntimeError("Documented None/'auto' control failed")
                        controls += 1
                    label = (
                        f"tracker={built} initial={initial!r} input={value!r} "
                        f"error={error!r} after={line._spin_model!r}; "
                        "accepted/state/compile_flag; derivation=public Boolean contract"
                    )
                    cases[label] = (observed, (True, True, True))
        kernels()
        print(f"DOCUMENTED_INPUTS cases={len(cases)} Boolean_exceptions={failures} alternative_pass={controls}")
        report(cases, rtol=0, atol=0)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, Python configuration API (no tracking kernel), CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

