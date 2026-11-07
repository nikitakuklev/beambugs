---
title: "API-20: MAD-X sequence import drops the final implicit drift"
hide:
  - toc
---

# API-20 — MAD-X sequence import drops the final implicit drift

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on ImpactX commit [`f9c2a82`](https://github.com/BLAST-ImpactX/impactx/commit/f9c2a82e24b7), committed 2026-11-05 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/python/impactx/MADXParser.py:1823–1862](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/impactx/MADXParser.py#L1823-L1862)<br>[gap expansion](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/impactx/MADXParser.py#L1959-L1986)<br>[load_file contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/python.rst#L1139-L1160)<br>[defines `L`](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/7998e738eca575c82a424138517c2a17115f3cdb/doc/latexuguide/sequence.tex#L32-L33)<br>[`test_sequence_refer_and_drifts`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/tests/python/test_madx_call.py#L380-L402) |

Importing a MAD-X SEQUENCE whose final element ends before its declared length drops the remaining drift, so particle transport, transfer matrices and reference flight time describe a shorter beamline. A 12.50 m sequence containing a 1.750 m drift centred at 3.125 m imports as 4.000 m: its transfer-matrix element R₁₂ = ∂x/∂pₓ is 4.000 m instead of 12.50 m. Native MAD-X and analytic free transport give the full 12.50 m. Adding a marker at the endpoint or using an equivalent explicit LINE restores the missing transport.

Run `python API-20.py` with [_slicing.py](../helpers/_slicing.py) in the same directory.

??? example "Reproducer: API-20.py"

    [Download API-20.py](API-20.py)

    ```python
    """API-20: MAD-X sequence import drops the final implicit drift

    Importing a MAD-X SEQUENCE whose final element ends before its declared length drops the remaining drift, so particle transport, transfer matrices and reference flight time describe a shorter beamline. A 12.50 m sequence containing a 1.750 m drift centred at 3.125 m imports as 4.000 m: its transfer-matrix element R₁₂ = ∂x/∂pₓ is 4.000 m instead of 12.50 m. Native MAD-X and analytic free transport give the full 12.50 m. Adding a marker at the endpoint or using an equivalent explicit LINE restores the missing transport.

    Setup: run pinned to one CPU (`taskset -c <n>`); the shared `_slicing` helper refuses to run on more, and child processes inherit that CPU.

    The script imports inline drift sequences with all three placement conventions and model floors, then compares particle, matrix and reference transport with analytic free flight and native MAD-X. Endpoint-marker, explicit-LINE and no-tail cases control the physical extent. Temporary decks and MAD-X scratch files live beside the shared helper and are removed on exit.
    """

    from _slicing import environment, run


    def main():
        environment()
        failures = 0
        cases = (
            [("linear", r, False, False, 12.5) for r in ("centre", "entry", "exit")]
            + [(m, "centre", False, False, 12.5) for m in ("paraxial", "exact")]
            + [
                ("linear", "centre", True, False, 12.5),
                ("linear", "centre", False, True, 12.5),
                ("linear", "centre", False, False, 4.0),
            ]
        )
        for model, refer, endpoint, line, total in cases:
            if line:
                layout = "a:drift,l=2.25;d:drift,l=1.75;z:drift,l=8.5;cell:line=(a,d,z);"
            else:
                at = {"centre": 3.125, "entry": 2.25, "exit": 4}[refer]
                layout = f"d:drift,l=1.75;cell:sequence,l={total},refer={refer};d,at={at};"
                if endpoint:
                    layout += f"endmark:marker,at={total};"
                layout += "endsequence;"
            result = run(
                dict(
                    kind="sequence",
                    model=model,
                    deck="beam,particle=electron,gamma=3;" + layout + "use,period=cell;",
                )
            )
            # Free linear transport has R12=R34=L and reference ct=L/beta.
            # The exact drift's finite-angle ray uses pz=sqrt(1-px^2), at pt=py=0.
            expected_x = 0.0002 + total * 0.0007 / ((1 - 0.0007**2) ** 0.5 if model == "exact" else 1)
            assert abs(result["native"]["s"] - total) < 2e-14
            assert abs(result["native"]["matrix"][0][1] - total) < 2e-14
            residual = max(
                abs(result["s"] - total),
                abs(result["matrix"][0][1] - total),
                abs(result["matrix"][2][3] - total),
                abs(result["ct"] - total / result["beta"]),
                abs(result["rays"][1][0] - expected_x),
            )
            if endpoint or line or total == 4:
                assert residual < 2e-14
            else:
                failures += residual > 2e-14
            print(
                "SEQUENCE",
                model,
                refer,
                "endpoint",
                endpoint,
                "line",
                line,
                "L_R12_R34",
                result["s"],
                result["matrix"][0][1],
                result["matrix"][2][3],
                "expected_L",
                total,
                "ray_x",
                result["rays"][1][0],
                "expected_x",
                expected_x,
                "ct",
                result["ct"],
                "expected_ct",
                total / result["beta"],
                "oracle=free_drift/native_MAD-X",
            )
        print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 26.10 (`615921c6867c895cf3a8fcf1d12991079d0f7d75`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 26.10 (`615921c6867c895cf3a8fcf1d12991079d0f7d75`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD |
| Confirmed | `claude-opus-5-5` | high | 2026-11-07 | development (`f9c2a82e24b700be2fbe9196a8c7404eaab09dfc`), source, CPU NOACC, double, serial, FFT, no openPMD or SIMD; also the impactx-noacc 26.10 wheel |

