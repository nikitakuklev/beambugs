---
title: "COL-09: Zero-bunch-charge Gaussian space charge returns nonfinite coordinates"
hide:
  - toc
---

# COL-09 — Zero-bunch-charge Gaussian space charge returns nonfinite coordinates

| | |
|---|---|
| Code | [ImpactX](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on ImpactX commit [`f9c2a82`](https://github.com/BLAST-ImpactX/impactx/commit/f9c2a82e24b7), committed 2026-11-05 |
| Source | [InitDistribution.cpp:303–320](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/initialization/InitDistribution.cpp#L303-L320)<br>[zero-weight diagnostic contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/dataanalysis/dataanalysis.rst#L138-L142)<br>[Gaussian-model contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L2460-L2466)<br>[ReducedBeamCharacteristics.cpp:297–298](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/diagnostics/ReducedBeamCharacteristics.cpp#L297-L298)<br>[ReducedBeamCharacteristics.cpp:545–567](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/diagnostics/ReducedBeamCharacteristics.cpp#L545-L567)<br>[Gauss2p5dPush.cpp:161–166](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/particles/spacecharge/Gauss2p5dPush.cpp#L161-L166)<br>[Gauss2p5dPush.cpp:240–247](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/particles/spacecharge/Gauss2p5dPush.cpp#L240-L247)<br>[Gauss2p5dPush.cpp:280–299](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/particles/spacecharge/Gauss2p5dPush.cpp#L280-L299)<br>[Gauss3dPush.cpp:122–149](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/particles/spacecharge/Gauss3dPush.cpp#L122-L149)<br>[Gauss3dPush.cpp:182–190](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/particles/spacecharge/Gauss3dPush.cpp#L182-L190)<br>[HandleSpacecharge.cpp:36–46](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/particles/spacecharge/HandleSpacecharge.cpp#L36-L46) |

Tracking a zero-bunch-charge test beam with `Gauss2p5D` or `Gauss3D` returns nonfinite particle coordinates instead of the field-free map. Zero bunch charge is accepted with an advisory warning, but the Gaussian pushers consume undefined weighted beam sizes before multiplying by zero source charge. An independent linear-drift calculation predicts finite coordinates for every particle. For 128 initially finite particles through a 3.7 cm drift, `Gauss2p5D` produces 768 nonfinite coordinate entries and `Gauss3D` produces 640. The three PIC models and disabled space charge remain finite and agree with the analytic drift to within 4.941e-16 in reference-normalized momentum.

Run `python COL-09.py` with [_precision.py](../helpers/_precision.py) in the same directory.

??? example "Reproducer: COL-09.py"

    [Download COL-09.py](COL-09.py)

    ```python
    """COL-09: Zero-bunch-charge Gaussian space charge returns nonfinite coordinates

    Tracking a zero-bunch-charge test beam with `Gauss2p5D` or `Gauss3D` returns nonfinite particle coordinates instead of the field-free map. Zero bunch charge is accepted with an advisory warning, but the Gaussian pushers consume undefined weighted beam sizes before multiplying by zero source charge. An independent linear-drift calculation predicts finite coordinates for every particle. For 128 initially finite particles through a 3.7 cm drift, `Gauss2p5D` produces 768 nonfinite coordinate entries and `Gauss3D` produces 640. The three PIC models and disabled space charge remain finite and agree with the analytic drift to within 4.941e-16 in reference-normalized momentum.

    Setup: run pinned to one CPU (`taskset -c <n>`); the shared `_precision` helper refuses to run on more, and child processes inherit that CPU.

    The script initializes Gaussian zero-charge test beams with both species signs and compares their trajectories with an independent linear Drift calculation. Fresh serial children cover Gaussian and PIC models, disabled space charge, Strang splitting, and disabled longitudinal kicks. Positive-charge and single-particle controls isolate the exactly zero-source boundary.
    """

    from _precision import environment, run


    def cases():
        for mode in ("Gauss2p5D", "Gauss3D", "3D", "2D", "2p5D", "off"):
            for sign in (1, -1):
                yield dict(kind="zero", mode=mode, charge=0.0, sign=sign)
        yield dict(kind="zero", mode="Gauss2p5D", charge=0.0, longitudinal=False)
        for mode in ("Gauss2p5D", "Gauss3D"):
            yield dict(kind="zero", mode=mode, charge=0.0, strang=True)
            yield dict(kind="zero", mode=mode, charge=1e-12, weights_after=True)
            yield dict(kind="zero", mode=mode, charge=0.0, n=1)
            yield dict(kind="zero", mode=mode, charge=1e-12)
            yield dict(kind="zero", mode=mode, charge=1e-30)


    def main():
        environment()
        failures = []
        for spec in cases():
            r = run(spec)
            assert r["before"] == r["retained"] == r["requested"]
            is_zero = spec["charge"] == 0.0 or spec.get("weights_after", False)
            gaussian_zero = is_zero and spec["mode"].startswith("Gauss") and r["requested"] > 1
            label = (
                f"{spec['mode']} sign={spec.get('sign', 1):+d} charge={spec['charge']:g} C "
                f"N={r['requested']} Strang={spec.get('strang', False)} "
                f"longitudinal={spec.get('longitudinal', True)} zero_weights_after={spec.get('weights_after', False)}"
            )
            if is_zero:
                assert r["expected_nonfinite"] == 0
                assert r["q_after"] == 0.0
                detail = (
                    f" by_coordinate={r['nonfinite_by_coordinate']}"
                    if r["nonfinite"]
                    else f" drift_max_abs_error={r['drift_max_abs_error']:.17g}"
                )
                print(
                    f"{label}: nonfinite={r['nonfinite']}/{6 * r['requested']} expected_nonfinite=0"
                    f"{detail}; oracle=zero self-field and linear Drift, t'=pt/3"
                )
            else:
                print(
                    f"{label}: nonfinite={r['nonfinite']} expected_nonfinite=0; oracle=finite positive-source control"
                )
            if gaussian_zero:
                failures.append(r["nonfinite"] > 0)
            else:
                assert r["nonfinite"] == 0
        print("DEFECT REPRODUCED" if all(failures) else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 26.10 (`615921c6867c895cf3a8fcf1d12991079d0f7d75`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 26.10 (`615921c6867c895cf3a8fcf1d12991079d0f7d75`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD |

