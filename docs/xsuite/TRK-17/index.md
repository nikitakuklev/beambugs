---
title: "TRK-17: Off-origin polygon apertures retain particles outside their boundary"
hide:
  - toc
---

# TRK-17 — Off-origin polygon apertures retain particles outside their boundary

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`LimitPolygon_track_local_particle`, limitpolygon.h:24–89](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/limitpolygon.h#L24-L89)<br>[vertex contract and constructor](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/limit_polygon.py#L12-L128)<br>[centroid tests](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/tests/test_apertures.py#L397-L416) |

On serial CPU, `LimitPolygon` also accepts a disk around its local origin when the polygon lies away from that origin, so particles outside the aperture survive. Independent convex half-plane tests give the correct acceptance, and writing the same geometry as centered vertices plus an element translation restores it. For a rectangle spanning 4–8 mm horizontally and ±1.5 mm vertically before rotation, 358 of 2114 sampled particles survive instead of 56, and all 302 false survivors lie in the spurious 4 mm-radius disk. The error occurs for both vertex orderings and in line tracking. Source inspection also finds that on GPU a particle exactly at the local origin bypasses the test.

Run `python TRK-17.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-17.py"

    [Download TRK-17.py](TRK-17.py)

    ```python
    """TRK-17: Off-origin polygon apertures retain particles outside their boundary

    On serial CPU, `LimitPolygon` also accepts a disk around its local origin when the polygon lies away
    from that origin, so particles outside the aperture survive. Independent convex half-plane tests
    give the correct acceptance, and writing the same geometry as centered vertices plus an element
    translation restores it. For a rectangle spanning 4–8 mm horizontally and ±1.5 mm vertically before
    rotation, 358 of 2114 sampled particles survive instead of 56, and all 302 false survivors lie in
    the spurious 4 mm-radius disk. The error occurs for both vertex orderings and in line tracking.
    Source inspection also finds that on GPU a particle exactly at the local origin bypasses the test.

    The script tracks 2114 deterministic positions per shape through off-origin convex rectangle,
    triangle and pentagon apertures with rotated vertices, in 36 configurations covering both vertex
    orderings, element and line tracking, three length scales, translations, an element roll and two
    momenta, and compares the survivors with the half-plane acceptance; positions near the boundary are
    excluded. Controls must pass: the same apertures recentered through the element translation,
    recentered particles and vertices, an equivalent `LimitRect` for the rectangle, and a line with
    local aperture checks disabled, where every particle survives. It runs serially on the installed
    prebuilt CPU kernels with kernel compilation forbidden.
    """

    from _prebuilt import CTX, identity, kernels, np, xt


    def rotation(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s], [s, c]])


    def geometry(vertices, points):
        """Convex intersection of inward half-planes; finite-segment clearances."""
        edge = np.roll(vertices, -1, axis=0) - vertices
        offset = points[:, None, :] - vertices[None, :, :]
        signed = edge[:, 0] * offset[:, :, 1] - edge[:, 1] * offset[:, :, 0]
        inside = np.all(signed > 0, axis=1) | np.all(signed < 0, axis=1)
        projection = np.sum(offset * edge, axis=2) / np.sum(edge * edge, axis=1)
        closest = vertices + np.clip(projection, 0, 1)[:, :, None] * edge
        clearance = np.min(np.linalg.norm(points[:, None, :] - closest, axis=2), axis=1)
        return inside, clearance


    def track(vertices, points, *, mode, shift, angle, p0c):
        aperture = xt.LimitPolygon(
            _context=CTX,
            x_vertices=vertices[:, 0],
            y_vertices=vertices[:, 1],
            shift_x=shift[0],
            shift_y=shift[1],
            rot_s_rad=angle,
        )
        return track_element(aperture, points, mode=mode, p0c=p0c)


    def track_element(aperture, points, *, mode, p0c):
        particles = xt.Particles(
            _context=CTX, p0c=p0c, mass0=xt.PROTON_MASS_EV, q0=1, x=points[:, 0], y=points[:, 1]
        )
        if mode == "element":
            aperture.track(particles)
        else:
            line = xt.Line(elements=[aperture])
            line.build_tracker(_context=CTX)
            assert not line.tracker.track_flags.XS_FLAG_IGNORE_LOCAL_APERTURE
            line.track(particles)
        order = np.argsort(np.asarray(particles.particle_id))
        states = np.asarray(particles.state)[order]
        assert np.all(np.isin(states, [0, 1]))
        return states > 0


    def main():
        identity(random_seed=91843)
        rng = np.random.default_rng(91843)
        angles = np.arange(5) * 2 * np.pi / 5 + 0.17
        shapes = {
            "rectangle": np.array([[4.0, -1.5], [8.0, -1.5], [8.0, 1.5], [4.0, 1.5]]),
            "triangle": np.array([[3.0, 1.0], [8.0, 2.0], [5.0, 6.0]]),
            "pentagon": np.c_[6 + 2 * np.cos(angles), -2 + 2 * np.sin(angles)],
        }
        results = []
        for name, original in shapes.items():
            # Rotate the geometry in its own frame independently of element roll.
            vertex_rotation = rotation(-0.29)
            vertices = original @ vertex_rotation.T
            _, radius = geometry(vertices, np.zeros((1, 2)))
            radius = float(radius[0])
            theta = np.arange(32) * 2 * np.pi / 32
            circle = np.c_[np.cos(theta), np.sin(theta)]
            points = np.vstack(
                [
                    rng.uniform(-10, 10, (2048, 2)),
                    np.zeros((1, 2)),
                    0.8 * radius * circle,
                    1.2 * radius * circle,
                    np.mean(vertices, axis=0),
                ]
            )
            expected, clearances = geometry(vertices, points)
            radii = np.linalg.norm(points, axis=1)
            safe = (clearances > 1e-5) & (np.abs(radii - radius) > 1e-5)
            points, expected, clearances = points[safe], expected[safe], clearances[safe]
            # The kernel accepts every point within the minimum origin-to-edge distance without
            # checking that the origin is inside: the polygon plus an origin-centred disk.
            predicted = expected | (np.linalg.norm(points, axis=1) < radius)
            assert expected.sum() >= 8
            assert np.sum(predicted & ~expected) >= 50
            center = np.mean(vertices, axis=0)
            for scale in [0.0001, 0.001, 0.007]:
                for reverse in [False, True]:
                    ordered = vertices[::-1] if reverse else vertices
                    for mode in ["element", "line"]:
                        angle = 0.61 if mode == "line" else 0.0
                        turn = rotation(angle)
                        shift = scale * np.array([1.2, -0.9])
                        physical = scale * points @ turn.T + shift
                        momentum = 0.7e9 if scale == 0.0001 else 3e9
                        actual = track(
                            scale * ordered, physical, mode=mode, shift=shift, angle=angle, p0c=momentum
                        )
                        np.testing.assert_array_equal(actual, predicted)
                        recentered = track(
                            scale * (ordered - center),
                            physical,
                            mode=mode,
                            shift=shift + scale * center @ turn.T,
                            angle=angle,
                            p0c=momentum,
                        )
                        np.testing.assert_array_equal(recentered, expected)
                        # An aperture with the origin inside it validates the shortcut's premise.
                        centered_points = scale * (points - center) @ turn.T + shift
                        centered = track(
                            scale * (ordered - center),
                            centered_points,
                            mode=mode,
                            shift=shift,
                            angle=angle,
                            p0c=momentum,
                        )
                        np.testing.assert_array_equal(centered, expected)
                        if name == "rectangle":
                            # Cartesian bounds supply the discovery's separate aperture control.
                            # Vertex rotation and element roll compose in the same transverse plane.
                            rectangle = xt.LimitRect(
                                _context=CTX,
                                min_x=scale * 4,
                                max_x=scale * 8,
                                min_y=-scale * 1.5,
                                max_y=scale * 1.5,
                                shift_x=shift[0],
                                shift_y=shift[1],
                                rot_s_rad=angle - 0.29,
                            )
                            rectangular = track_element(rectangle, physical, mode=mode, p0c=momentum)
                            np.testing.assert_array_equal(rectangular, expected)
                        record = dict(
                            shape=name,
                            scale=scale,
                            reverse=reverse,
                            mode=mode,
                            particles=len(points),
                            expected_alive=int(expected.sum()),
                            actual_alive=int(actual.sum()),
                            false_positive=int(np.sum(actual & ~expected)),
                            false_negative=int(np.sum(~actual & expected)),
                            shortcut_radius_m=scale * radius,
                            min_edge_clearance_m=scale * float(clearances.min()),
                        )
                        results.append(record)
                        print("POLYGON", record)
        # Explicit aperture disabling is a supported alternative branch.
        aperture = xt.LimitPolygon(
            _context=CTX,
            x_vertices=[0.004, 0.008, 0.008, 0.004],
            y_vertices=[-0.0015, -0.0015, 0.0015, 0.0015],
        )
        line = xt.Line(elements=[aperture])
        line.build_tracker(_context=CTX)
        line.tracker.track_flags.XS_FLAG_IGNORE_LOCAL_APERTURE = True
        particles = xt.Particles(_context=CTX, p0c=3e9, x=np.linspace(-0.02, 0.02, 32))
        line.track(particles)
        assert np.all(particles.state == 1)
        print("IGNORE_LOCAL_APERTURE control: 32/32 survive")
        print(
            "SWEEP",
            len(results),
            "minimum_edge_clearance_m",
            min(row["min_edge_clearance_m"] for row in results),
        )
        print("CONTROLS centered/recentered polygons and Cartesian rectangle acceptance pass")
        kernels()
        print(
            "POLYGON CONFIRMED: every error is an outside point in the shortcut disk; "
            "all centered and recentered controls pass"
        )
        print("DEFECT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, serial |

