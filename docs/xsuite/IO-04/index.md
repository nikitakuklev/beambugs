---
title: "IO-04: Renaming a MAD-X deferred variable freezes its contribution to imported dependants"
hide:
  - toc
---

# IO-04 — Renaming a MAD-X deferred variable freezes its contribution to imported dependants

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack release [v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4) (`3cd8bfc`) |
| Source | [`MadLoader.init_line_expressions`, mad_loader.py:437–461](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_loader.py#L437-L461) |

When a MAD-X lattice is imported with deferred expressions and a variable-renaming map (`replace_in_expr`), `MadLoader` renames variables inside expressions but registers each deferred expression under its original name, so the renamed variable keeps only its initial value. A later knob change must propagate, as an independent MAD-X session and the analytic relation confirm. Importing `base_a = 2e-4; derived_a := 2*base_a` and a horizontal kicker with `kick := derived_a`, renaming both variables to `base_b` and `derived_b` and then setting `base_b = 3e-4` leaves `derived_b` and the tracked horizontal momentum `px` at 4.000×10⁻⁴ instead of 6.000×10⁻⁴. Initial values are correct and renaming only numeric variables works; renamed deferred variables stay frozen while other terms of downstream expressions keep updating.

Run `python IO-04.py` with [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-04.py"

    [Download IO-04.py](IO-04.py)

    ```python
    """IO-04: Renaming a MAD-X deferred variable freezes its contribution to imported dependants

    When a MAD-X lattice is imported with deferred expressions and a variable-renaming map
    (`replace_in_expr`), `MadLoader` renames variables inside expressions but registers each deferred
    expression under its original name, so the renamed variable keeps only its initial value. A later
    knob change must propagate, as an independent MAD-X session and the analytic relation confirm.
    Importing `base_a = 2e-4; derived_a := 2*base_a` and a horizontal kicker with `kick := derived_a`,
    renaming both variables to `base_b` and `derived_b` and then setting `base_b = 3e-4` leaves
    `derived_b` and the tracked horizontal momentum `px` at 4.000×10⁻⁴ instead of 6.000×10⁻⁴. Initial
    values are correct and renaming only numeric variables works; renamed deferred variables stay frozen
    while other terms of downstream expressions keep updating.

    The script defines `base_a = 2e-4`, `derived_a := 2*base_a` and a horizontal kicker with `kick :=
    derived_a` in MAD-X, imports the sequence with deferred expressions while renaming both variables to
    their `_b` counterparts, sets `base_b = 3e-4` and tracks an on-axis proton. The expected kick comes
    from MAD-X evaluating the original expression after the same change and is checked against 2 ×
    3×10⁻⁴. The initial imported kick and an import that renames only the numeric base are the controls.
    """

    import importlib.metadata

    import numpy as np
    import xtrack as xt
    from cpymad.madx import Madx

    RTOL, ATOL = 2e-14, 2e-16


    def evaluate():
        cases = {}
        for rename_deferred in (False, True):
            with Madx(stdout=False) as mad:
                mad.input("""
                    base_a=2e-4; derived_a:=2*base_a;
                    hk:hkicker,kick:=derived_a;
                    seq:sequence,l=1; hk,at=0.5; endsequence;
                    beam,particle=proton,pc=1; use,sequence=seq;
                """)
                replacement = {"base_a": "base_b"}
                if rename_deferred:
                    replacement["derived_a"] = "derived_b"
                line = xt.Line.from_madx_sequence(
                    mad.sequence.seq,
                    deferred_expressions=True,
                    replace_in_expr=replacement,
                )
                # Initial values agree, which makes the lost dependency easy to miss.
                initial = xt.Particles(p0c=1e9)
                line.track(initial)
                np.testing.assert_allclose(initial.px[0], 4e-4, rtol=RTOL, atol=ATOL)
                line.vars["base_b"] = 3e-4
                mad.globals["base_a"] = 3e-4
                expected = float(mad.elements.hk.kick)
                np.testing.assert_allclose(expected, 2 * 3e-4, rtol=RTOL, atol=ATOL)
                particle = xt.Particles(p0c=1e9)
                line.track(particle)
                assert particle.state[0] == 1
                if not rename_deferred:
                    np.testing.assert_allclose(particle.px[0], expected, rtol=RTOL, atol=ATOL)
                else:
                    cases["renamed derived variable after knob change"] = (
                        line.vars["derived_b"]._value,
                        expected,
                    )
                    cases["tracked horizontal kick after knob change"] = particle.px[0], expected
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print(f"cpymad {importlib.metadata.version('cpymad')}")
        print(
            "Expected: renaming preserves derived=2*base; after base=3e-4 the horizontal kick must be 6e-4."
        )
        print(
            "MAD-X evaluates the original expression independently; renaming only a numeric base variable is a control."
        )
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

