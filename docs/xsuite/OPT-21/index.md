---
title: "OPT-21: Propagated longitudinal W uses the initialization reference speed"
hide:
  - toc
---

# OPT-21 — Propagated longitudinal W uses the initialization reference speed

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Source | [W construction, optics_propagation.py:188–201](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/optics_propagation.py#L188-L201)<br>[recorded pζ:172–176](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/optics_propagation.py#L172-L176)<br>[published pζ:211–222](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/optics_propagation.py#L211-L222)<br>[transfer_matrices.py:96–108](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/transfer_matrices.py#L96-L108)<br>[114–122](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/transfer_matrices.py#L114-L122)<br>[local longitudinal coordinates](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L335-L381)<br>[canonical optics vector](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L4287-L4294)<br>[W propagation](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L4442-L4446)<br>[LocalParticle_update_p0c:398–425](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/headers/local_particle_common.h#L398-L425)<br>[LineSegmentMap:360–389](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/linesegmentmap.h#L360-L389) |

When reference momentum changes along an open Twiss calculation, the sixth row of its normalizing matrix divides energy deviations by the initialization reference speed instead of the local speed. Rebasing protons from 650.0 MeV/c to 1.170 GeV/c gives an amplitude-restored longitudinal momentum derivative of 0.5556 instead of 0.4055, a 36.99% excess. The exact reference-coordinate transformation and direct tracked matrices establish the correct derivative. A ratio cancelling modal amplitudes independently exposes the same error. Internal reference-energy changes during segment acceleration also fail; fixed-reference controls pass.

Run `python OPT-21.py` with [_prebuilt.py](../helpers/_prebuilt.py), [_reference_momentum.py](../helpers/_reference_momentum.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: OPT-21.py"

    [Download OPT-21.py](OPT-21.py)

    ```python
    """OPT-21: Propagated longitudinal W uses the initialization reference speed

    When reference momentum changes along an open Twiss calculation, the sixth row of its normalizing matrix divides energy deviations by the initialization reference speed instead of the local speed. Rebasing protons from 650.0 MeV/c to 1.170 GeV/c gives an amplitude-restored longitudinal momentum derivative of 0.5556 instead of 0.4055, a 36.99% excess. The exact reference-coordinate transformation and direct tracked matrices establish the correct derivative. A ratio cancelling modal amplitudes independently exposes the same error. Internal reference-energy changes during segment acceleration also fail; fixed-reference controls pass.

    The script restores the recorded modal amplitudes and compares the longitudinal basis with the exact reference-coordinate transformation and a directly tracked matrix. A second comparison takes a longitudinal row ratio that cancels modal amplitudes. Both open Twiss modes and standalone and internal reference changes are exercised, with simultaneous segment acceleration, unchanged reference and fixed-reference cavity controls.
    """

    from _prebuilt import identity, kernels, np
    from _reference_momentum import W0, cases, twiss, twiss_modes

    identity()
    failed = 0
    for kind, ratio, t, line, p, expected in cases():
        direct = line.get_R_matrix(particle_on_co=p, symmetrize=False)["R_matrix"]
        np.testing.assert_allclose(direct, expected, atol=3e-10, rtol=3e-10)
        scalar = p.copy()
        line.track(scalar)
        for mode in twiss_modes(kind):
            tw = twiss(line, p, mode)
            restored = tw.W_matrix[-1] * np.repeat([tw.nux[-1], tw.nuy[-1], tw.nuzeta[-1]], 2)
            basis = restored @ np.linalg.inv(W0)
            np.testing.assert_allclose(basis[:5], expected[:5], atol=3e-10, rtol=3e-10)
            np.testing.assert_allclose(tw.pzeta[-1], scalar.pzeta[0], atol=2e-14, rtol=2e-14)
            # Divide out the entrance scales; any common mode amplitude cancels.
            row_ratio = tw.W_matrix[-1, 5, 5] * W0[4, 4] / (tw.W_matrix[-1, 4, 4] * W0[5, 5])
            expected_ratio = expected[5, 5] / expected[4, 4]
            ok = np.allclose(basis[5], expected[5], atol=3e-10, rtol=3e-10)
            ratio_ok = np.isclose(row_ratio, expected_ratio, atol=3e-10, rtol=3e-10)
            assert ok == ratio_ok
            if ratio == 1:
                np.testing.assert_allclose(basis, expected, atol=3e-10, rtol=3e-10)
            elif not ok:
                np.testing.assert_allclose(basis[5] / t, expected[5], atol=3e-10, rtol=3e-10)
            failed += not ok
            print(
                f"{kind} a={ratio:g} mode={mode}: observed_restored_W66={basis[5, 5]:.15g} expected_restored_W66={expected[5, 5]:.15g} observed_row_ratio={row_ratio:.15g} expected_row_ratio={expected_ratio:.15g} beta_ratio={t:.15g} derivation=W66=1/(a*t),row_ratio=1/(a*t*t) physical_match={ok}"
            )
    kernels()
    print("DEFECT REPRODUCED" if failed else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

