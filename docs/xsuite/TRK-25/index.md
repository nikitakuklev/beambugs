---
title: "TRK-25: Loss refinement constructs false walls for apertures excluding the origin"
hide:
  - toc
---

# TRK-25 — Loss refinement constructs false walls for apertures excluding the origin

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Source | [characterize_aperture, loss_location_refinement.py:594–659](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/loss_location_refinement/loss_location_refinement.py#L594-L659)<br>[The path selection, lines 159–185](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/loss_location_refinement/loss_location_refinement.py#L159-L185)<br>[Refinement and writeback, lines 354–383](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/loss_location_refinement/loss_location_refinement.py#L354-L383)<br>[LossLocationRefinement parameters, lines 31–58](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/loss_location_refinement/loss_location_refinement.py#L31-L58)<br>[the interpolated-impact guide, collimation.rst:248–255](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/collimation.rst#L248-L255)<br>[example, 001_loss_location_refinement.py:20–45](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/examples/collimation/001_loss_location_refinement.py#L20-L45) |

Loss-location refinement can return impacts well downstream of the wall when interpolating shifted apertures whose opening excludes the laboratory origin. The boundary search starts every ray at that origin and turns immediate loss into a false wall at the last coarse scan radius. Between apertures at 0.1000 and 1.100 m, a rectangular opening with 10.00 mm horizontal half-width shifted by 100.0 mm returns 1.100 m instead of 0.6100 m on a 10.00 mm sampling grid, a 0.4900 m error. An independent ray–wall intersection and boundary-vertex checks establish the discrepancy. Identical embedded apertures use replication and refine correctly.

Run `python TRK-25.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-25.py"

    [Download TRK-25.py](TRK-25.py)

    ```python
    """TRK-25: Loss refinement constructs false walls for apertures excluding the origin

    Loss-location refinement can return impacts well downstream of the wall when interpolating shifted apertures whose opening excludes the laboratory origin. The boundary search starts every ray at that origin and turns immediate loss into a false wall at the last coarse scan radius. Between apertures at 0.1000 and 1.100 m, a rectangular opening with 10.00 mm horizontal half-width shifted by 100.0 mm returns 1.100 m instead of 0.6100 m on a 10.00 mm sampling grid, a 0.4900 m error. An independent ray–wall intersection and boundary-vertex checks establish the discrepancy. Identical embedded apertures use replication and refine correctly.

    Build shifted rectangles and ellipses with native placement or surrounding transforms, including straight and tapered walls. Compare refined impacts with independent wall inequalities on the requested longitudinal grid and inspect generated boundary vertices. Identical native apertures and origin-containing openings provide controls.
    """

    from _prebuilt import CTX, identity, kernels, np, xt

    records = []


    def run_case(
        shape,
        representation,
        shift,
        roll=0.0,
        rmax=0.1,
        dr=1e-5,
        ds=0.02,
        ntheta=128,
        s0=0.2,
        span=0.8,
        width=0.009,
        slope=0.02411,
    ):
        taper = 0.002 if representation == "native-taper" else 0.0
        c, s = np.cos(roll), np.sin(roll)

        def opening(end):
            half = width + taper * end
            aperture = (
                xt.LimitRect(min_x=-half, max_x=half, min_y=-0.006, max_y=0.006)
                if shape == "rect"
                else xt.LimitEllipse(a=half, b=0.006)
            )
            if representation.startswith("native"):
                aperture.shift_x = shift
                aperture.rot_s_rad = roll
                return [aperture]
            return [
                xt.Translation(shift_x=shift),
                xt.Rotation(rot_s_rad=roll),
                aperture,
                xt.Rotation(rot_s_rad=-roll),
                xt.Translation(shift_x=-shift),
            ]

        line = xt.Line(
            elements=[
                xt.Drift(length=s0),
                *opening(0),
                xt.Drift(length=span),
                *opening(1),
                xt.Drift(length=0.15),
            ]
        )
        line.build_tracker(_context=CTX)
        line.reset_s_at_end_turn = False
        p = particle(
            x=[shift, shift - s0 * slope * c],
            y=[0, -s0 * slope * s],
            px=[0, slope * c],
            py=[0, slope * s],
        )
        line.track(p)
        order = np.argsort(p.particle_id)
        assert np.array_equal(p.state[order], [1, 0])
        assert abs(p.s[order][1] - (s0 + span)) < 3e-15

        # Expanded drift and delta=0 give u_x(t)=slope*t, u_y(t)=0.
        # Both shapes intersect the local horizontal axis at +/- half_width(t).
        crossing = s0 + width / (slope - taper / span)
        nseg = int(np.ceil(span / ds))
        tau = np.linspace(0, span, nseg + 1)
        first_out = np.flatnonzero(slope * tau > width + taper * tau / span)[0]
        expected_s = float(s0 + tau[first_out])
        expected_xy = np.array([shift + slope * tau[first_out] * c, slope * tau[first_out] * s])

        refine = xt.LossLocationRefinement(
            line, n_theta=ntheta, r_max=rmax, dr=dr, ds=ds, save_refine_lines=True
        )
        refine.refine_loss_location(p, with_progress=False)
        order = np.argsort(p.particle_id)
        observed_s = float(p.s[order][1])
        observed_xy = np.array([p.x[order][1], p.y[order][1]])
        assert np.array_equal(p.state[order], [1, 0])
        assert np.max(np.abs([p.x[order][0] - shift, p.y[order][0]])) < 3e-15
        defect = abs(observed_s - expected_s) > 3e-14
        local_origin = np.array([-shift * c, shift * s])
        origin_inside = (
            abs(local_origin[0]) < width and abs(local_origin[1]) < 0.006
            if shape == "rect"
            else (local_origin[0] / width) ** 2 + (local_origin[1] / 0.006) ** 2 < 1
        )
        if origin_inside or representation == "native-identical":
            assert not defect
        if not defect:
            assert np.max(np.abs(observed_xy - expected_xy)) < 3e-14
        label = f"{shape} {representation} dx={shift} roll={roll} rmax={rmax} dr={dr} ds={ds} ntheta={ntheta}"
        print(
            "LOSS",
            label,
            "continuous_crossing",
            crossing,
            "sample_expected",
            expected_s,
            "observed",
            observed_s,
            "expected_xy",
            expected_xy,
            "observed_xy",
            observed_xy,
            "error_s",
            observed_s - expected_s,
            "origin_inside",
            origin_inside,
            "derivation=first sampled plane outside independent wall inequality",
            "defect",
            defect,
        )
        record = dict(
            case=label,
            crossing=crossing,
            expected_s=expected_s,
            observed_s=observed_s,
            expected_xy=expected_xy.tolist(),
            observed_xy=observed_xy.tolist(),
            error_s=observed_s - expected_s,
            defect=defect,
        )
        interp = refine.refine_lines[refine.i_apertures[-1]]
        polygon = interp.elements[0]
        if hasattr(polygon, "x_vertices"):
            xy = np.column_stack([polygon.x_vertices, polygon.y_vertices])
            radius = np.linalg.norm(xy, axis=1)
            body = (xy - [shift, 0]) @ np.array([[c, -s], [s, c]])
            boundary_residual = (
                np.maximum(abs(body[:, 0]) / width, abs(body[:, 1]) / 0.006) - 1
                if shape == "rect"
                else (body[:, 0] / width) ** 2 + (body[:, 1] / 0.006) ** 2 - 1
            )
            print(
                "BOUNDARY",
                label,
                "x_bounds",
                [xy[:, 0].min(), xy[:, 0].max()],
                "r_bounds",
                [radius.min(), radius.max()],
                "max_normalized_residual",
                abs(boundary_residual).max(),
            )
            record["polygon_x_bounds"] = [float(xy[:, 0].min()), float(xy[:, 0].max())]
            record["polygon_r_bounds"] = [float(radius.min()), float(radius.max())]
            if defect:
                # r=0 is lost on every ray, so argmin selects index 0 and the
                # preceding coarse sample is r[-1]. Use the actual array: floating
                # arange can include its nominal endpoint (e.g. r_max=.065).
                coarse = np.arange(0.0, rmax, rmax / 100.0)
                false_radius = coarse[-1]
                assert np.max(abs(radius - false_radius)) < 2e-15
                radius_at_grid = np.hypot(shift + slope * tau * c, slope * tau * s)
                false_indices = np.flatnonzero(radius_at_grid > false_radius)
                false_grid_s = s0 + tau[false_indices[0]] if len(false_indices) else s0 + span
                assert abs(observed_s - false_grid_s) < 3e-14
                record["false_circle_sample_s"] = float(false_grid_s)
                print("FALSE_CIRCLE_ORACLE", "radius", false_radius, "sample_s", false_grid_s)
        records.append(record)


    def particle(**kwargs):
        return xt.Particles(_context=CTX, p0c=2.1e9, mass0=xt.PROTON_MASS_EV, q0=1, **kwargs)


    def finish(records):
        kernels()
        failures = sum(record["defect"] for record in records)
        print(f"COMPARISONS total={len(records)} pass={len(records) - failures} fail={failures}")
        print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


    identity()
    for shape in ("rect", "ellipse"):
        for representation in ("native-identical", "native-taper", "external"):
            for shift in (0.0, 0.002, 0.05, -0.05):
                run_case(shape, representation, shift)
    # A rotation, finer radial/angular/longitudinal grids, and a second valid r_max.
    for representation in ("native-taper", "external"):
        run_case("rect", representation, 0.05, roll=0.31)
        run_case("rect", representation, 0.05, dr=5e-6, ds=0.01, ntheta=256)
        run_case("rect", representation, 0.05, rmax=0.065)
    # Separate reproduction of the discovery's magnitude, with the same independent
    # straight-wall predicate and newly built line rather than its discovery script.
    run_case("rect", "external", 0.1, rmax=0.2, ds=0.01, s0=0.1, span=1.0, width=0.01, slope=0.01979)
    finish(records)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

