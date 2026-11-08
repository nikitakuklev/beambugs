---
title: "OPT-18: Twiss transfer matrices omit reference-change amplitudes"
hide:
  - toc
---

# OPT-18 — Twiss transfer matrices omit reference-change amplitudes

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Source | [TwissTable.get_R_matrix, twiss_table.py:775–797](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_table.py#L775-L797)<br>[get_R_matrix_table:833–848](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_table.py#L833-L848)<br>[809–829](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_table.py#L809-L829)<br>[850–874](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_table.py#L850-L874)<br>[_renormalize_eigenvectors:195–234](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/lattice_functions_from_W.py#L195-L234)<br>[manual transfer definition](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L4287-L4294)<br>[LocalParticle_update_p0c:398–425](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/headers/local_particle_common.h#L398-L425)<br>[LineSegmentMap:360–389](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/linesegmentmap.h#L360-L389) |

Reconstructing a transfer matrix from a Twiss table across a reference-momentum change discards the mode-amplitude change. The public matrix APIs promise transfer between the selected table locations. Rebasing protons from 650.0 MeV/c to 1.170 GeV/c gives a horizontal matrix with diagonal (1.342, 0.7454), instead of (1, 0.5556), and determinant 1 instead of 0.5556. Physical momentum conservation and tracked finite differences establish the expected map. Both reconstruction APIs fail; direct tracked matrices and the unchanged-reference control pass.

Run `python OPT-18.py` with [_prebuilt.py](../helpers/_prebuilt.py), [_reference_momentum.py](../helpers/_reference_momentum.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: OPT-18.py"

    [Download OPT-18.py](OPT-18.py)

    ```python
    """OPT-18: Twiss transfer matrices omit reference-change amplitudes

    Reconstructing a transfer matrix from a Twiss table across a reference-momentum change discards the mode-amplitude change. The public matrix APIs promise transfer between the selected table locations. Rebasing protons from 650.0 MeV/c to 1.170 GeV/c gives a horizontal matrix with diagonal (1.342, 0.7454), instead of (1, 0.5556), and determinant 1 instead of 0.5556. Physical momentum conservation and tracked finite differences establish the expected map. Both reconstruction APIs fail; direct tracked matrices and the unchanged-reference control pass.

    The script compares both Twiss-table matrix APIs with the analytic reference-coordinate Jacobian and a directly tracked finite-difference matrix. It exercises both open Twiss modes, two standalone reference-change elements and a segment with internal reference changes, including simultaneous acceleration. Same-row maps, unchanged reference and a fixed-reference cavity gain are controls.
    """

    from _prebuilt import identity, kernels, np
    from _reference_momentum import cases, twiss, twiss_modes

    identity()
    failed = 0
    for kind, ratio, _t, line, p, expected in cases():
        direct = line.get_R_matrix(particle_on_co=p, symmetrize=False)["R_matrix"]
        np.testing.assert_allclose(direct, expected, atol=3e-10, rtol=3e-10)
        for mode in twiss_modes(kind):
            tw = twiss(line, p, mode)
            matrix = tw.get_R_matrix(0, len(tw.name) - 1)
            table = tw.get_R_matrix_table().R_matrix[-1]
            np.testing.assert_allclose(matrix, table, atol=2e-14, rtol=2e-14)
            np.testing.assert_allclose(tw.get_R_matrix(0, 0), np.eye(6), atol=2e-14, rtol=2e-14)
            ok = np.allclose(matrix[:4, :4], expected[:4, :4], atol=3e-10, rtol=3e-10)
            if ratio == 1:
                np.testing.assert_allclose(matrix, expected, atol=3e-10, rtol=3e-10)
            elif not ok:
                # Each uncoupled transverse mode has its own removed amplitude.
                amplitudes = np.repeat([tw.nux[-1] / tw.nux[0], tw.nuy[-1] / tw.nuy[0]], 2)
                np.testing.assert_allclose(
                    matrix[:4, :4] * amplitudes, expected[:4, :4], atol=3e-10, rtol=3e-10
                )
            failed += not ok
            print(
                f"{kind} a={ratio:g} mode={mode}: observed_diag=({matrix[0, 0]:.15g},{matrix[1, 1]:.15g}) expected_diag=({expected[0, 0]:.15g},{expected[1, 1]:.15g}) observed_det={np.linalg.det(matrix[:2, :2]):.15g} expected_det={1 / ratio:.15g} derivation=det(Rx)=Pold/Pnew physical_match={ok}"
            )
    kernels()
    print("DEFECT REPRODUCED" if failed else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

