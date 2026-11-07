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
