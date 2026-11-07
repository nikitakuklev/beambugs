---
title: "TRK-42: RFCA END1_FOCUS and END2_FOCUS edge lenses omit the 1/β² factor"
hide:
  - toc
---

# TRK-42 — RFCA END1_FOCUS and END2_FOCUS edge lenses omit the 1/β² factor

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`trackRfCavityWithWakes`, simple_rfca.c:745–768](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L745-L768)<br>[simple_rfca.c:804–807](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L804-L807)<br>[simple_rfca.c:1019–1024](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1019-L1024)<br>[1072–1076](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1072-L1076)<br>[`rf_cavity_matrix`, compute_matrices.c:2604–2610](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L2604-L2610)<br>[parameter table, track_data.c:608–609](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L608-L609)<br>[RFCA manual, rfca.tex](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/rfca.tex) |

RFCA's END1_FOCUS and END2_FOCUS edge lenses, in tracking and in the transfer matrix, have inverse focal length γ′/(2γ), where γ′ is the energy gain per length in units of mc². Integrating the radial field −(r/2)∂E_z/∂z of a hard field edge over the transit time dz/(βc) gives γ′/(2β²γ) at the local β and γ, so both lenses are too weak by the factor β². For electrons entering a 0.7 m cavity that adds 12 MeV at 1.5 MeV kinetic energy, the entrance lens is 4.262 m⁻¹ instead of 4.556 m⁻¹, 6.457% weak; at 5 MeV it is 0.8598% weak and at 20 MeV 0.06207% weak.

Run `python TRK-42.py` with [_elegant.py](../helpers/_elegant.py) and [_rfca_lens.py](../helpers/_rfca_lens.py) in the same directory.

??? example "Reproducer: TRK-42.py"

    [Download TRK-42.py](TRK-42.py)

    ```python
    """TRK-42: RFCA END1_FOCUS and END2_FOCUS edge lenses omit the 1/β² factor

    RFCA's END1_FOCUS and END2_FOCUS edge lenses, in tracking and in the transfer matrix, have inverse
    focal length γ′/(2γ), where γ′ is the energy gain per length in units of mc². Integrating the radial
    field −(r/2)∂E_z/∂z of a hard field edge over the transit time dz/(βc) gives γ′/(2β²γ) at the local
    β and γ, so both lenses are too weak by the factor β². For electrons entering a 0.7 m cavity that
    adds 12 MeV at 1.5 MeV kinetic energy, the entrance lens is 4.262 m⁻¹ instead of 4.556 m⁻¹, 6.457%
    weak; at 5 MeV it is 0.8598% weak and at 20 MeV 0.06207% weak.

    The script sends electrons of 1.5, 5 and 20 MeV kinetic energy through a 0.7 m RFCA that adds 12 MeV
    in a static field (FREQ=0), with each edge lens alone and with none, in matrix-method tracking
    (N_KICKS = 0), one-kick tracking and Elegant's own transfer matrix. The lens strengths follow from
    the tracked and matrix maps and are compared with the finite-β hard-edge impulse. The control is the
    cavity without edge lenses, whose map must conserve transverse momentum.
    """

    import numpy as np
    from _elegant import modes, verdict
    from _rfca_lens import LENGTH, ME_MEV, VOLT, momentum, transfer

    KINETIC = (1.5, 5.0, 20.0)  # MeV
    KICKS = (0, 1)
    RTOL = 1e-9  # relative; the maps are exact up to rounding


    def lenses(no_edge, edge1, edge2):
        """Entrance and exit inverse focal lengths (positive = focusing) from three 2x2 maps.

        With the body map B, END1 alone gives B @ [[1, 0], [-k1, 1]], so R21 = B21 - k1 B22;
        END2 alone gives [[1, 0], [-k2, 1]] @ B, so R21 = B21 - k2 B11.
        """
        k1 = (no_edge[1, 0] - edge1[1, 0]) / no_edge[1, 1]
        k2 = (no_edge[1, 0] - edge2[1, 0]) / no_edge[0, 0]
        return float(k1), float(k2)


    defect = False
    for mode in modes():
        for kinetic in KINETIC:
            maps = {
                (e1, e2, n): transfer(mode, kinetic, n, e1, e2)
                for n in KICKS
                for e1, e2 in ((0, 0), (1, 0), (0, 1))
            }
            ref = maps[(0, 0, 0)]
            gamma_i, gamma_f = ref["gamma_in"], ref["gamma_out"]
            beta2_i, beta2_f = 1 - 1 / gamma_i**2, 1 - 1 / gamma_f**2
            gradient = VOLT / (1e6 * ME_MEV) / LENGTH  # gamma' = dgamma/dz, 1/m
            # Hard edge of an axisymmetric field: E_r = -(r/2) dE_z/dz, no magnetic field for FREQ=0.
            # dp_x/dt = q E_r with dt = dz/(beta c) integrates over the edge to
            # dp_x = -q x E_z/(2 beta c), and dx' = dp_x/p_z with p_z = beta gamma m c gives
            # dx' = -x gamma'/(2 beta^2 gamma) at the entrance (E_z switches on) and
            # dx' = +x gamma'/(2 beta^2 gamma) at the exit, each with the local beta and gamma.
            k1_phys = gradient / (2 * beta2_i * gamma_i)
            k2_phys = -gradient / (2 * beta2_f * gamma_f)
            # Control: without edges p_x is conserved, so R21 = 0 and R22 = p_in/p_out.
            p_ratio = momentum(kinetic) / np.sqrt(gamma_f**2 - 1)
            for n in KICKS:
                for source in ("tracked", "matrix"):
                    if source == "matrix" and n:
                        continue  # the matrix does not depend on N_KICKS
                    body = maps[(0, 0, n)][source]
                    label = f"{mode} {kinetic:g} MeV {'matrix_output' if source == 'matrix' else f'tracking N_KICKS={n}'}"
                    if abs(body[1, 0]) > RTOL * gradient or abs(body[1, 1] - p_ratio) > RTOL:
                        raise SystemExit(
                            f"{label}: control fails: the edge-free map does not conserve p_x"
                        )
                    k1, k2 = lenses(body, maps[(1, 0, n)][source], maps[(0, 1, n)][source])
                    e1, e2 = k1 / k1_phys - 1, k2 / k2_phys - 1
                    print(
                        f"{label}: entrance 1/f observed={k1:.12g} expected={k1_phys:.12g} /m "
                        f"(rel {e1:+.6e}, -1/gamma_in^2 = {-1 / gamma_i**2:+.6e}); exit 1/f "
                        f"observed={k2:.12g} expected={k2_phys:.12g} /m (rel {e2:+.6e})"
                    )
                    defect |= abs(e1) > 1e-6 and kinetic < 10
    print(f"control: edge-free maps give R21 = 0 and R22 = p_in/p_out within {RTOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

