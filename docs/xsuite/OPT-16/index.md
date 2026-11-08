---
title: "OPT-16: Radiation integrals omit the dipole-face term of I₄"
hide:
  - toc
---

# OPT-16 — Radiation integrals omit the dipole-face term of I₄

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Source | [`_compute_radiation_integrals`, radiation.py:399–401](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/radiation.py#L399-L401)<br>[radiation.py:415–417](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/radiation.py#L415-L417)<br>[lines 428–429](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/radiation.py#L428-L429)<br>[Lines 435–453](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/radiation.py#L435-L453)<br>[`twiss` docstring, twiss.py:186–187 and 363–380](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss.py#L363-L380)<br>[physics guide, xtrack.tex:6100–6105](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L6100-L6105)<br>[line 6167](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L6167)<br>[`Magnet` edge parameters, magnet.py:110–126](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/magnet.py#L110-L126)<br>[track_dipole_edge_linear.h:17–36](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_dipole_edge_linear.h#L17-L36) |

With `radiation_integrals=True`, Twiss computes the fourth radiation integral I₄ of a bend with angled hard-edge faces, including a rectangular `RBend`, from its body alone, omitting the face term −h²(D₁ tan e₁ + D₂ tan e₂); h is the curvature, D₁, D₂ the horizontal dispersion and e₁, e₂ the angles at the two faces. The exact path length ℓ between the inclined field boundaries of a ray with momentum offset δ gives I₄ = h² dℓ/dδ, which contains this term. In a 132 m electron ring of rectangular bends, I₄ is 0.1157 m⁻¹ instead of −8.403 × 10⁻⁴ m⁻¹, so the horizontal damping partition number is 0.8030 instead of 1.001 and the integral-based equilibrium emittance is 24.71% too large. Perpendicular faces are unaffected.

Run `python OPT-16.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: OPT-16.py"

    [Download OPT-16.py](OPT-16.py)

    ```python
    """OPT-16: Radiation integrals omit the dipole-face term of I₄

    With `radiation_integrals=True`, Twiss computes the fourth radiation integral I₄ of a bend with
    angled hard-edge faces, including a rectangular `RBend`, from its body alone, omitting the face term
    −h²(D₁ tan e₁ + D₂ tan e₂); h is the curvature, D₁, D₂ the horizontal dispersion and e₁, e₂ the
    angles at the two faces. The exact path length ℓ between the inclined field boundaries of a ray with
    momentum offset δ gives I₄ = h² dℓ/dδ, which contains this term. In a 132 m electron ring of
    rectangular bends, I₄ is 0.1157 m⁻¹ instead of −8.403 × 10⁻⁴ m⁻¹, so the horizontal damping
    partition number is 0.8030 instead of 1.001 and the integral-based equilibrium emittance is 24.71%
    too large. Perpendicular faces are unaffected.

    The script builds 3 GeV/c electron lines of uniform-field `Bend` and `RBend` magnets with linear
    hard edges, slices each bend into 256 thin Teapot kicks and runs 4D Twiss with
    `radiation_integrals=True`. The expected I₄ is h² dℓ/dδ, where ℓ(δ) is the exact path length of
    an off-momentum circular ray between the two inclined field boundaries, differentiated by complex
    step; perpendicular faces are the control. The ring's periodic dispersion comes from independent
    affine 3×3 transfer matrices, and the reported I₄ is also compared with the analytic body integral
    alone.
    """

    import json

    from _prebuilt import CTX, identity, kernels, np, xt

    identity()

    P0C = 3e9
    STEP = 1e-20  # complex-step size; the derivative carries no cancellation error


    def ray_length(delta, length, h, e1, e2, d, dp):
        """Path length in the field of a ray with momentum offset delta.

        Laboratory frame: the reference ray enters at the origin along +y and bends towards -x on a
        circle of radius 1/h centred at (-1/h, 0). The field fills the region between the entrance
        plane through the origin and the exit plane through the reference exit point, each rotated by
        its face angle from the radial direction (positive angles close the outer side, as in a
        rectangular magnet). The off-momentum ray starts on the entrance line at x = d*delta with
        slope px/(1+delta) = dp*delta/(1+delta) and radius (1+delta)/h at fixed field.
        """
        theta = h * length
        alpha = np.arcsin(dp * delta / (1 + delta))
        tangent = np.array([np.sin(alpha), np.cos(alpha)])
        point = np.array([d * delta, 0 * delta])
        entry_normal = np.array([-np.sin(e1), np.cos(e1)])
        entry = point - (entry_normal @ point) / (entry_normal @ tangent) * tangent
        radius = (1 + delta) / h
        center = entry + radius * np.array([-tangent[1], tangent[0]])
        ref_exit = np.array([(np.cos(theta) - 1) / h, np.sin(theta) / h])
        exit_normal = np.array([-np.sin(theta - e2), np.cos(theta - e2)])
        # Turning angle from the entry point to the exit plane, times the radius.
        phi = alpha + theta - e2 + np.arcsin(exit_normal @ (ref_exit - center) / radius)
        return radius * phi


    def expected_i4(length, h, e1, e2, d, dp):
        # At fixed field the emitted energy scales as (1+delta)^2 h^2 ell(delta) in the
        # ultrarelativistic limit; with dU/dE ∝ 2 I2 + I4 and I2 = h^2 L this gives I4 = h^2 ell'(0).
        return h * h * np.imag(ray_length(STEP * 1j, length, h, e1, e2, d, dp)) / STEP


    def body_i4(length, h, e1, d, dp):
        # Body term h^3 ∫ D ds inside a sector body, after the entrance lens D' -> D' + h tan(e1) D:
        # D(s) = D cos(hs) + D'_in sin(hs)/h + (1 - cos(hs))/h.
        theta = h * length
        slope = dp + h * np.tan(e1) * d
        return (
            h * h * d * np.sin(theta)
            + h * slope * (1 - np.cos(theta))
            + h * h * length
            - h * np.sin(theta)
        )


    def edge_kwargs(e1, e2):
        return dict(
            edge_entry_angle=e1,
            edge_exit_angle=e2,
            edge_entry_model="linear",
            edge_exit_model="linear",
        )


    def twiss(elements, slices, **init):
        line = xt.Line(elements=elements)
        line.particle_ref = xt.Particles(_context=CTX, p0c=P0C, mass0=xt.ELECTRON_MASS_EV, q0=-1)
        line.slice_thick_elements(
            [
                xt.Strategy(slicing=None),
                xt.Strategy(slicing=xt.Teapot(slices), element_type=xt.Bend),
                xt.Strategy(slicing=xt.Teapot(slices), element_type=xt.RBend),
            ]
        )
        line.build_tracker(_context=CTX)
        kwargs = dict(init=xt.TwissInit(betx=4.0, bety=3.0, **init)) if init else {}
        return line.twiss(method="4d", radiation_integrals=True, chrom=False, **kwargs)


    # Open line: one 0.9 m, 0.09 rad bend entered with D = 0.65 m, D' = -0.015.
    L, THETA, D0, DP0 = 0.9, 0.09, 0.65, -0.015
    H = THETA / L
    defects = []
    for kind, e1, e2 in (
        ("Bend", 0.0, 0.0),
        ("Bend", THETA / 2, THETA / 2),
        ("Bend", 0.025, 0.065),
        ("RBend", THETA / 2, THETA / 2),
    ):
        if kind == "Bend":
            element = xt.Bend(length=L, angle=THETA, k0=H, **edge_kwargs(e1, e2))
        else:
            # An RBend with zero edge angles has faces at THETA/2 from the radial direction.
            element = xt.RBend(
                length_straight=2 * np.sin(THETA / 2) / H,
                angle=THETA,
                k0_from_h=True,
                edge_entry_model="linear",
                edge_exit_model="linear",
            )
        tw = twiss([element], 256, dx=D0, dpx=DP0)
        expected = expected_i4(L, H, e1, e2, D0, DP0)
        body = body_i4(L, H, e1, D0, DP0)
        # The face part of the exact result equals -h^2 (D1 tan e1 + D2 tan e2).
        d_exit = D0 * np.cos(THETA) + (DP0 + H * np.tan(e1) * D0) * np.sin(THETA) / H
        d_exit += (1 - np.cos(THETA)) / H
        face = -H * H * (D0 * np.tan(e1) + d_exit * np.tan(e2))
        assert abs(expected - body - face) < 5e-18
        # Controls: the faces act on the dispersion, and the reported value is the body integral.
        assert abs(tw.dx[-1] - d_exit) < 1e-9
        assert abs(tw.rad_int_i4 - body) < 1e-12
        if e1 == e2 == 0:
            assert abs(tw.rad_int_i4 - expected) < 1e-12
        else:
            defects.append(abs(tw.rad_int_i4 - expected) > 0.99 * abs(face))
        print(
            f"BEND {kind} e1={e1:.4g} e2={e2:.4g}: I4 observed={tw.rad_int_i4:.16g} "
            f"expected={expected:.16g} (h^2 dl/ddelta, exact ray) body={body:.16g} "
            f"face={face:.16g} (-h^2(D1 tan e1+D2 tan e2))"
        )

    # 132 m ring of 24 cells with rectangular bends (faces at half the bend angle).
    NCELL, BL, QL, K1 = 24, 1.4, 0.25, 1.2
    HR = np.pi / (NCELL * BL)
    ER = HR * BL / 2
    specs = [
        ("q", K1, QL),
        ("d", 0.0, 0.7),
        ("b", HR, BL),
        ("d", 0.0, 0.4),
        ("q", -K1, QL),
        ("d", 0.0, 0.4),
        ("b", HR, BL),
        ("d", 0.0, 0.7),
    ]


    def affine(kind, k, length):
        # Affine maps on (D, D', 1) for the dispersion equation D'' + K D = h.
        if kind == "d":
            return np.array([[1, length, 0], [0, 1, 0], [0, 0, 1.0]])
        if kind == "q":
            r = np.sqrt(abs(k))
            c, s = (np.cos, np.sin) if k > 0 else (np.cosh, np.sinh)
            sign = -1 if k > 0 else 1
            return np.array(
                [
                    [c(r * length), s(r * length) / r, 0],
                    [sign * r * s(r * length), c(r * length), 0],
                    [0, 0, 1.0],
                ]
            )
        t = k * length
        body = np.array(
            [
                [np.cos(t), np.sin(t) / k, (1 - np.cos(t)) / k],
                [-k * np.sin(t), np.cos(t), np.sin(t)],
                [0, 0, 1.0],
            ]
        )
        lens = np.array([[1, 0, 0], [k * np.tan(ER), 1, 0], [0, 0, 1.0]])  # hard-edge face lens
        return lens @ body @ lens


    maps = [affine(*spec) for spec in specs]
    cell_map = np.eye(3)
    for m in maps:
        cell_map = m @ cell_map
    # Periodic dispersion: (I - M) D0 = b for the 2x2 block M and inhomogeneous column b.
    d_periodic = np.linalg.solve(np.eye(2) - cell_map[:2, :2], cell_map[:2, 2])
    state = np.r_[d_periodic, 1.0]
    i4_cell = body_cell = 0.0
    cell = []
    for (kind, k, length), m in zip(specs, maps):
        if kind == "q":
            cell.append(xt.Quadrupole(length=length, k1=k))
        elif kind == "d":
            cell.append(xt.Drift(length=length))
        else:
            i4_cell += expected_i4(length, k, ER, ER, *state[:2])
            body_cell += body_i4(length, k, ER, *state[:2])
            cell.append(xt.Bend(length=length, angle=k * length, k0=k, **edge_kwargs(ER, ER)))
        state = m @ state
    tw = twiss([el.copy() for _ in range(NCELL) for el in cell], 256)
    i2 = NCELL * 2 * BL * HR**2
    i4 = NCELL * i4_cell
    jx = 1 - i4 / i2
    assert abs(tw.line_length - 132.0) < 1e-9
    # Thin Teapot slicing changes the periodic dispersion by 2.6e-8 m and the body sum by 1.3e-9 m^-1
    # at 256 slices per bend; both fall as slices^-2 (64 slices give 4.1e-7 m and 2.2e-8 m^-1).
    assert abs(tw.dx[0] - d_periodic[0]) < 5e-8
    assert abs(tw.rad_int_i2 - i2) < 1e-12
    assert abs(tw.rad_int_i4 - NCELL * body_cell) < 3e-9
    # The equilibrium emittance scales as I5/(I2 - I4x); at Xtrack's I5 the correct value is:
    emit = tw.rad_int_eq_gemitt_x * (i2 - tw.rad_int_i4x) / (i2 - i4)
    defects.append(abs(tw.rad_int_partition_number_x - jx) > 0.1)
    ring = dict(
        circumference=float(tw.line_length),
        D0_observed=float(tw.dx[0]),
        D0_expected=float(d_periodic[0]),
        I2_observed=float(tw.rad_int_i2),
        I2_expected=i2,
        I4_observed=float(tw.rad_int_i4),
        I4_expected=float(i4),
        I4_body=float(NCELL * body_cell),
    )
    print("RING", json.dumps(ring))
    print(
        f"RING Jx observed={tw.rad_int_partition_number_x:.10g} expected={jx:.10g} "
        "(1 - I4/I2, exact-ray I4)"
    )
    print(
        f"RING eq_gemitt_x observed={tw.rad_int_eq_gemitt_x:.6g} expected={emit:.6g} m "
        "(same I5, I2 - I4 with exact-ray I4)"
    )
    kernels()
    print("DEFECT REPRODUCED" if all(defects) else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, prebuilt kernels (compilation forbidden), CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, prebuilt kernels (compilation forbidden), CPU, serial |

