---
title: "COL-08: Automatic mesh bounds discard their own upper-extremum particles"
hide:
  - toc
---

# COL-08 — Automatic mesh bounds discard their own upper-extremum particles

| | |
|---|---|
| Code | [ImpactX](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on ImpactX commit [`f9c2a82`](https://github.com/BLAST-ImpactX/impactx/commit/f9c2a82e24b7), committed 2026-11-05 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [InitMeshRefinement.cpp:51–63](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/initialization/InitMeshRefinement.cpp#L51-L63)<br>[InitMeshRefinement.cpp:99–133](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/initialization/InitMeshRefinement.cpp#L99-L133)<br>[dynamic-domain contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L2568-L2590)<br>[InitDistribution.cpp:424–433](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/initialization/InitDistribution.cpp#L424-L433)<br>[HandleSpacecharge.cpp:65–82](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/particles/spacecharge/HandleSpacecharge.cpp#L65-L82)<br>[deposit_charge](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/ImpactX.cpp#L706-L725)<br>[AMReX_Geometry.cpp:566–569](https://github.com/AMReX-Codes/amrex/blob/015c14b65715b277fbc54706ebe7455b196d2254/Src/Base/AMReX_Geometry.cpp#L566-L569)<br>[AMReX_Geometry.H:507–522](https://github.com/AMReX-Codes/amrex/blob/015c14b65715b277fbc54706ebe7455b196d2254/Src/Base/AMReX_Geometry.H#L507-L522)<br>[AMReX_Geometry.cpp:708–716](https://github.com/AMReX-Codes/amrex/blob/015c14b65715b277fbc54706ebe7455b196d2254/Src/Base/AMReX_Geometry.cpp#L708-L716)<br>[AMReX_ParticleContainerI.H:451–459](https://github.com/AMReX-Codes/amrex/blob/015c14b65715b277fbc54706ebe7455b196d2254/Src/Particle/AMReX_ParticleContainerI.H#L451-L459) |

Automatic mesh sizing with `geometry.prob_relative=1` can silently delete particles although this setting is documented to cover the beam exactly. Upper-extremum particles fall outside the accepted particle cells and redistribution removes them before deposition. Initialization with any enabled space-charge model is also affected. A 1.000 nC beam of 8192 particles loses 261 particles across 100 slices of a 1 cm drift, leaving 0.9681 nC despite having no aperture or loss process. Charge conservation and an independent boundary calculation establish the error. Near-unit positive padding can also fail through rounding; factors 1.2 and the default 3 preserve this beam.

Run `python COL-08.py` with [_precision.py](../helpers/_precision.py) in the same directory.

??? example "Reproducer: COL-08.py"

    [Download COL-08.py](COL-08.py)

    ```python
    """COL-08: Automatic mesh bounds discard their own upper-extremum particles

    Automatic mesh sizing with `geometry.prob_relative=1` can silently delete particles although this setting is documented to cover the beam exactly. Upper-extremum particles fall outside the accepted particle cells and redistribution removes them before deposition. Initialization with any enabled space-charge model is also affected. A 1.000 nC beam of 8192 particles loses 261 particles across 100 slices of a 1 cm drift, leaving 0.9681 nC despite having no aperture or loss process. Charge conservation and an independent boundary calculation establish the error. Near-unit positive padding can also fail through rounding; factors 1.2 and the default 3 preserve this beam.

    Setup: run pinned to one CPU (`taskset -c <n>`); the shared `_precision` helper refuses to run on more, and child processes inherit that CPU.

    The script uses public deposition, isolated resize and redistribution, high-level Gaussian initialization, and a 100-slice drift. Exact input populations and charges supply conservation oracles, and a 90-digit bound calculation isolates padding lost to rounding. Fresh serial children compare factors 1 and nextafter(1,+∞) with padded and space-charge-disabled controls.
    """

    import math

    from _precision import environment, run


    def main():
        environment()
        failures = []
        factors = (1.0, math.nextafter(1.0, math.inf), 1.2, 3.0)
        for factor in factors:
            r = run(dict(kind="grid", factor=factor))
            expected = r["q_before"]
            print(
                f"deposit factor={factor:.17g}: retained={r['retained']}/27 "
                f"live_charge={r['q_after']:.17g} C deposited_charge={r['q_deposited']:.17g} C "
                f"expected_charge={expected:.17g} C; oracle=input population and species charge times summed weights"
            )
            if factor >= 1.2:
                assert r["retained"] == 27
                assert abs(r["q_deposited"] - expected) <= 8 * math.ulp(expected)
            else:
                failures.append(r["retained"] < 27)
        for factor in factors:
            r = run(dict(kind="boundary", mode="2D", factor=factor))
            assert r["retained"] == r["predicted_accepted"]
            print(
                f"boundary factor={factor:.17g}: retained={r['retained']}/27 "
                f"AMReX_accepted={r['predicted_accepted']} rounded_upper_x={r['rounded_hi_x']:.17g} m "
                f"exact_padding_x={r['exact_padding_x']} m; oracle=27 input particles, 90-digit bound arithmetic"
            )
            if factor < 1.2:
                failures.append(r["retained"] < 27)
            else:
                assert r["retained"] == 27
        for mode, factor in [
            ("3D", 1.0),
            ("2D", 1.0),
            ("2p5D", 1.0),
            ("Gauss2p5D", 1.0),
            ("Gauss3D", 1.0),
            ("Gauss2p5D", 1.2),
            ("Gauss3D", 1.2),
            ("off", 1.0),
        ]:
            r = run(dict(kind="initialize", mode=mode, factor=factor, n=4096))
            print(
                f"initialize {mode} factor={factor:g}: retained={r['retained']}/4096 "
                f"charge={r['q_after']:.17g} C expected_charge=1e-09 C; oracle=requested population and charge"
            )
            if factor == 1.0 and mode != "off":
                failures.append(r["retained"] < 4096)
            else:
                assert r["retained"] == 4096
        for mode, factor in [("3D", 1.0), ("2D", 1.0), ("2p5D", 1.0), ("3D", 1.2), ("3D", 3.0)]:
            r = run(dict(kind="track", mode=mode, factor=factor))
            print(
                f"track {mode} factor={factor:g} L=0.01 m slices=100: retained={r['retained']}/8192 "
                f"charge={r['q_after']:.17g} C expected_charge={r['q_before']:.17g} C; "
                "oracle=particle and charge conservation without apertures"
            )
            if factor == 1.0:
                failures.append(r["retained"] < 8192)
            else:
                assert r["retained"] == 8192
                assert abs(r["q_after"] - r["q_before"]) <= math.ulp(r["q_before"])
        print("DEFECT REPRODUCED" if all(failures) else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 26.10 (`615921c6867c895cf3a8fcf1d12991079d0f7d75`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 26.10 (`615921c6867c895cf3a8fcf1d12991079d0f7d75`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD |
| Confirmed | `claude-opus-5-5` | high | 2026-11-07 | development (`f9c2a82e24b700be2fbe9196a8c7404eaab09dfc`), source, CPU NOACC, double, serial, FFT, no openPMD or SIMD; also the impactx-noacc 26.10 wheel |

