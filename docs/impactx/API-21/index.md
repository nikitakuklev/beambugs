---
title: "API-21: MAD-X nominal thin dipoles import as straight-reference steering kicks"
hide:
  - toc
---

# API-21 — MAD-X nominal thin dipoles import as straight-reference steering kicks

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on ImpactX commit [`f9c2a82`](https://github.com/BLAST-ImpactX/impactx/commit/f9c2a82e24b7), committed 2026-11-05 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [MULTIPOLE translation branch](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/impactx/madx_to_impactx.py#L1375-L1440)<br>[particle push](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/Multipole.H#L160-L193)<br>[zero-length no-op](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/mixin/thin.H#L25-L33)<br>[ThinDipole](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/ThinDipole.H#L131-L185)<br>[thin-element model-floor contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/python.rst#L1211-L1216)<br>[states](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/7998e738eca575c82a424138517c2a17115f3cdb/doc/latexuguide/elements.tex#L552-L565)<br>[synmadx conversion](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/impactx/synmadx/syn2_to_impactx.py#L360-L390) |

Importing a MAD-X MULTIPOLE with a nonzero dipole coefficient KNL[0], which MAD-X defines as a zero-length reference bend, turns the bend into a steering kick. For a 17.00 mrad bend, the design particle receives normalized horizontal momentum pₓ = −0.01700 instead of zero, and the reference heading does not turn. At reference speed β₀ = 0.9428, the dispersive coefficient R₂₆ = ∂pₓ/∂pt and the path-length coefficient R₅₁ = ∂t/∂x (t the arrival-time distance, pt = −ΔE/(P₀c)) are zero instead of −0.01803 and +0.01803. The MAD-X manual and native MAD-X TRACK, TWISS and SURVEY establish the expected result. An imported HKICKER remains a correct straight-reference kick, and native ThinDipole bends the reference correctly.

Run `python API-21.py` with [_slicing.py](../helpers/_slicing.py) in the same directory.

??? example "Reproducer: API-21.py"

    [Download API-21.py](API-21.py)

    ```python
    """API-21: MAD-X nominal thin dipoles import as straight-reference steering kicks

    Importing a MAD-X MULTIPOLE with a nonzero dipole coefficient KNL[0], which MAD-X defines as a zero-length reference bend, turns the bend into a steering kick. For a 17.00 mrad bend, the design particle receives normalized horizontal momentum pₓ = −0.01700 instead of zero, and the reference heading does not turn. At reference speed β₀ = 0.9428, the dispersive coefficient R₂₆ = ∂pₓ/∂pt and the path-length coefficient R₅₁ = ∂t/∂x (t the arrival-time distance, pt = −ΔE/(P₀c)) are zero instead of −0.01803 and +0.01803. The MAD-X manual and native MAD-X TRACK, TWISS and SURVEY establish the expected result. An imported HKICKER remains a correct straight-reference kick, and native ThinDipole bends the reference correctly.

    Setup: run pinned to one CPU (`taskset -c <n>`); the shared `_slicing` helper refuses to run on more, and child processes inherit that CPU.

    The script compares both signs of a nominal normal dipole at all model floors with native MAD-X TRACK, TWISS and SURVEY and with the R₂₆/R₅₁ coefficients of the analytic thin-bend Hamiltonian. Ordinary-kicker, zero-strength and native ThinDipole cases control kick and reference-bend semantics. Temporary decks and MAD-X scratch files live beside the shared helper and are removed on exit.
    """

    from _slicing import environment, run


    def main():
        environment()
        failures = 0
        cases = [(a, "multipole", m) for a in (0.017, -0.017) for m in ("linear", "paraxial", "exact")]
        cases += [
            (0.017, "kicker", "linear"),
            (0.0, "multipole", "linear"),
            (0.017, "native", "linear"),
        ]
        for angle, kind, model in cases:
            definition = (
                f"b:hkicker,kick={-angle};"
                if kind == "kicker"
                else f"b:multipole,knl={{{angle}}},lrad=0;"
            )
            result = run(
                dict(
                    kind="thin",
                    angle=angle,
                    model=model,
                    native_bend=kind == "native",
                    deck="beam,particle=electron,gamma=3;"
                    + definition
                    + "cell:line=(b);use,period=cell;",
                )
            )
            # pt=-DeltaE/(P0*c), so delta=sqrt(1-2*pt/beta+pt^2)-1.
            # G=-theta*x*delta gives R26=-theta/beta and R51=+theta/beta.
            target = angle / result["beta"] if kind != "kicker" else 0
            expected_design = -angle if kind == "kicker" else 0
            expected_heading = -angle if kind != "kicker" else 0
            matrix = result["matrix"]
            native = result["native"]
            assert abs(native["matrix"][1][5] + target) < 2e-15
            assert abs(native["matrix"][4][0] - target) < 2e-15
            assert abs(native["rays"][0][1] - expected_design) < 2e-15
            assert abs(native["survey_theta"] - expected_heading) < 2e-15
            residual = max(
                abs(matrix[1][5] + target),
                abs(matrix[4][0] - target),
                abs(result["rays"][0][1] - expected_design),
                abs(result["heading"] - expected_heading),
            )
            if kind in ("kicker", "native") or angle == 0:
                assert residual < 2e-15
            else:
                failures += residual > 2e-15
            print(
                "THIN",
                kind,
                angle,
                model,
                "design_px",
                result["rays"][0][1],
                "expected_design_px",
                expected_design,
                "R26_R51",
                matrix[1][5],
                matrix[4][0],
                "expected",
                -target,
                target,
                "heading",
                result["heading"],
                "expected_heading",
                expected_heading,
                "oracle=canonical_generator/native_MAD-X_TRACK_TWISS_SURVEY",
            )
        print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | 26.10 (`615921c6867c895cf3a8fcf1d12991079d0f7d75`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | 26.10 (`615921c6867c895cf3a8fcf1d12991079d0f7d75`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD |
| Confirmed | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | development (`f9c2a82e24b700be2fbe9196a8c7404eaab09dfc`), source, CPU NOACC, double, serial, FFT, no openPMD or SIMD; also the impactx-noacc 26.10 wheel |

