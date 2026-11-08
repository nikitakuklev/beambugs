---
title: "API-22: MAD-X sequence import places a nested sub-sequence as a zero-length element"
hide:
  - toc
---

# API-22 — MAD-X sequence import places a nested sub-sequence as a zero-length element

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on ImpactX commit [`f9c2a82`](https://github.com/BLAST-ImpactX/impactx/commit/f9c2a82e24b7), committed 2026-11-05 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`MADXParser._get_element_length()`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/impactx/MADXParser.py#L1812-L1821)<br>[`_parse_sequence_definition()`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/impactx/MADXParser.py#L1823-L1857)<br>[REFER conversion and gap loop](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/impactx/MADXParser.py#L1944-L1986)<br>[`_flatten_line()`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/impactx/MADXParser.py#L2460-L2495)<br>[load_file contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/python.rst#L1139-L1160)<br>[permits sub-sequences](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/7998e738eca575c82a424138517c2a17115f3cdb/doc/latexuguide/sequence.tex#L85-L98)<br>[the total sequence length](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/7998e738eca575c82a424138517c2a17115f3cdb/doc/latexuguide/sequence.tex#L32-L37) |

Importing a MAD-X sequence that contains a positive-length sub-sequence treats the child as zero-length while generating the parent's gap drifts. With the default `REFER=centre`, a 4.000 m child centred at 8.000 m starts at 8.000 m instead of 6.000 m; elements placed after the child move by 4.000 m. A 20.00 m drift-only parent therefore imports as 24.00 m and gives R₁₂ = ∂x/∂pₓ = 24.00 m instead of 20.00 m, established by analytic transport and native MAD-X. `REFER=exit` also misplaces the child; `REFER=entry` preserves its entry but misplaces following elements.

Run `python API-22.py` with [_slicing.py](../helpers/_slicing.py) in the same directory.

??? example "Reproducer: API-22.py"

    [Download API-22.py](API-22.py)

    ```python
    """API-22: MAD-X sequence import places a nested sub-sequence as a zero-length element

    Importing a MAD-X sequence that contains a positive-length sub-sequence treats the child as zero-length while generating the parent's gap drifts. With the default `REFER=centre`, a 4.000 m child centred at 8.000 m starts at 8.000 m instead of 6.000 m; elements placed after the child move by 4.000 m. A 20.00 m drift-only parent therefore imports as 24.00 m and gives R₁₂ = ∂x/∂pₓ = 24.00 m instead of 20.00 m, established by analytic transport and native MAD-X. `REFER=exit` also misplaces the child; `REFER=entry` preserves its entry but misplaces following elements.

    Setup: run pinned to one CPU (`taskset -c <n>`); the shared `_slicing` helper refuses to run on more, and child processes inherit that CPU.

    The script imports a 20 m parent sequence holding a 4 m child sequence (a drift between two markers) with every outer REFER convention, both non-default child conventions and all three model floors, and compares marker positions, length, R₁₂/R₃₄, a tracked ray and reference flight with the MAD-X placement rule and native MAD-X TWISS. A flat sequence with the same element positions, an explicit nested LINE and the child sequence alone are passing controls, and a terminal child placed with REFER=entry shows the case without a following element. Temporary decks and MAD-X scratch files live beside the shared helper and are removed on exit.
    """

    from _slicing import environment, run

    TOL = 2e-14
    CHILD = {
        # Child-internal AT values that put the markers at 0 and 4 m and the 4 m drift between them.
        "centre": "ins:sequence,l=4,refer=centre;bi:marker,at=0;b,at=2;bo:marker,at=4;endsequence;",
        "entry": "ins:sequence,l=4,refer=entry;bi:marker,at=0;b,at=0;bo:marker,at=4;endsequence;",
        "exit": "ins:sequence,l=4,refer=exit;bi:marker,at=0;b,at=4;bo:marker,at=4;endsequence;",
    }


    def deck(case, outer, child, length):
        body = "b:drift,l=4;"
        refer = "" if outer == "default" else f",refer={outer}"
        if case == "line":
            body += (
                "d2:drift,l=2;d4:drift,l=4;d6:drift,l=6;"
                "pre:marker;bi:marker;bo:marker;post:marker;last:marker;"
                "ins:line=(bi,b,bo);outer:line=(d2,pre,d4,ins,d4,post,d6,last);"
            )
            period = "outer"
        elif case == "alone":
            body += CHILD[child]
            period = "ins"
        else:
            # MAD-X places the child's entry, centre or exit (outer REFER) at AT.
            at = {"centre": 8, "default": 8, "entry": 6, "exit": 10}[outer]
            if case == "flat":
                # Same elements placed directly; AT refers to each element's REFER point.
                shift = {"centre": 0, "default": 0, "entry": -2, "exit": 2}[outer]
                inner = f"bi:marker,at=6;b,at={8 + shift};bo:marker,at=10;"
            else:
                body += CHILD[child]
                inner = f"ins,at={at};"
            tail = "post:marker,at=14;last:marker,at=20;" if length == 20 else ""
            body += f"outer:sequence,l={length}{refer};pre:marker,at=2;{inner}{tail}endsequence;"
            period = "outer"
        return f"beam,particle=electron,gamma=3;{body}use,period={period};"


    def positions(records):
        # Cumulative ds before each named element of the imported lattice.
        s, out = 0.0, {}
        for record in records:
            out.setdefault(record["name"], s)
            s += record["ds"]
        return out


    def main():
        environment()
        failures = 0
        cases = (
            [
                ("nested", outer, "centre", "linear", 20)
                for outer in ("centre", "default", "entry", "exit")
            ]
            + [("nested", "centre", "centre", model, 20) for model in ("paraxial", "exact")]
            + [("nested", "centre", child, "linear", 20) for child in ("entry", "exit")]
            + [("nested", "entry", "centre", "linear", 10)]
            + [("flat", outer, "-", "linear", 20) for outer in ("centre", "exit")]
            + [("line", "-", "-", "linear", 20), ("alone", "-", "centre", "linear", 4)]
        )
        for case, outer, child, model, total in cases:
            result = run(dict(kind="sequence", model=model, deck=deck(case, outer, child, total)))
            # MAD-X sequence rules: child occupies [6, 10] m, post at 14 m, last at 20 m.
            expected = {"bi": 6.0, "bo": 10.0, "post": 14.0, "last": 20.0}
            if case == "alone":
                expected = {"bi": 0.0, "bo": 4.0}
            elif total == 10:
                expected = {"bi": 6.0, "bo": 10.0}
            native = dict(zip(result["native"]["names"], result["native"]["positions"]))
            assert all(abs(native[k] - v) < TOL for k, v in expected.items()), native
            assert abs(result["native"]["s"] - total) < TOL
            assert abs(result["native"]["matrix"][0][1] - total) < TOL
            observed = positions(result["records"])
            # Free transport: R12 = R34 = L, reference ct = L/beta; the exact drift's
            # finite-angle ray at pt = py = 0 has dx/ds = px/sqrt(1 - px^2).
            expected_x = 0.0002 + total * 0.0007 / ((1 - 0.0007**2) ** 0.5 if model == "exact" else 1)
            residual = max(
                max(abs(observed[k] - v) for k, v in expected.items()),
                abs(result["s"] - total),
                abs(result["matrix"][0][1] - total),
                abs(result["matrix"][2][3] - total),
                abs(result["ct"] - total / result["beta"]),
                abs(result["rays"][1][0] - expected_x),
            )
            if case == "nested" and total == 20:
                failures += residual > TOL
            else:
                assert residual < TOL, (case, outer, residual)
            extra = [w for w in result["warnings"] if "provided as a preview" not in w]
            label = "CONTROL" if case != "nested" else "NESTED" if total == 20 else "TERMINAL"
            print(
                label,
                case,
                "outer",
                outer,
                "child",
                child,
                model,
                "positions",
                {k: observed[k] for k in expected},
                "expected",
                expected,
                "L_R12_R34",
                result["s"],
                result["matrix"][0][1],
                result["matrix"][2][3],
                "expected_L",
                float(total),
                "ray_x",
                result["rays"][1][0],
                "expected_x",
                expected_x,
                "ct",
                result["ct"],
                "expected_ct",
                total / result["beta"],
                "other_warnings",
                len(extra),
                "oracle=MAD-X_placement/free_drift/native_MAD-X",
            )
        print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-06</span> | 26.10 (`615921c6867c895cf3a8fcf1d12991079d0f7d75`), source reading, not executed for this claim |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | 26.10 (`615921c6867c895cf3a8fcf1d12991079d0f7d75`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD |
| Confirmed<br><small class="bb-scope">placement against MAD-X for every outer REFER and model floor</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-06</span> | 26.10 (`615921c6867c895cf3a8fcf1d12991079d0f7d75`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD |

