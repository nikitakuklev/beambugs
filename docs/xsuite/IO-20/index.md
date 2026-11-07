---
title: "IO-20: Reverse MAD-X import keeps forward horizontal aperture offsets"
hide:
  - toc
---

# IO-20 — Reverse MAD-X import keeps forward horizontal aperture offsets

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack main (`f09e0fb`, head build) |
| Source | [load signature](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/load.py#L49-L55)<br>[MAD-X-only reverse_lines restriction](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/load.py#L80-L81)<br>[MAD-X dispatch](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/load.py#L104-L105)<br>[reads `aper_offset`, loader.py:618–620](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_parser/loader.py#L618-L620)<br>[stores it as `shift_x`/`shift_y` with `aper_tilt` as `rot_s_rad`, lines 647–658](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_parser/loader.py#L647-L658)<br>[reverse loader, lines 799–816](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_parser/loader.py#L799-L816)<br>[`_reverse_element`, environment.py:3152–3233](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/environment.py#L3152-L3233)<br>[negates `rot_s_rad` with the strengths, lines 3195–3201](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/environment.py#L3195-L3201)<br>[physics manual, xtrack.tex:5725–5752](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L5725-L5752)<br>[Twiss reverse frame, twiss.rst:131–138](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/twiss.rst#L131-L138)<br>[REFLECT](https://github.com/MethodicalAcceleratorDesign/MAD-X-User-Manual/blob/46a8a655e56ef9920532f1f4249392caf3c9bbe8/seqedit.tex#L97-L110)<br>[line reflection](https://github.com/MethodicalAcceleratorDesign/MAD-X-User-Manual/blob/46a8a655e56ef9920532f1f4249392caf3c9bbe8/line.tex#L100-L103)<br>[APER_OFFSET](https://github.com/MethodicalAcceleratorDesign/MAD-X-User-Manual/blob/46a8a655e56ef9920532f1f4249392caf3c9bbe8/aperture.tex#L249-L263)<br>[reverse-import test](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/tests/test_native_madloader.py#L584-L610) |

Loading a MAD-X sequence with reverse_lines leaves each aperture's horizontal offset unchanged instead of reflecting it into the reversed frame, where x changes sign. Rectangular, circular, elliptical, rectangle–ellipse and polygon apertures can therefore lose valid particles and keep particles outside the physical opening; asymmetric polygon vertices are not reflected either. For a rectangle with 1.000 mm horizontal half-width and a 3.000 mm offset, the imported centre is 6.000 mm from its required position. Independent Cartesian membership tests establish the required reflection, and reflecting only the horizontal geometry restores tracking; forward imports and zero-horizontal-offset controls of symmetric shapes pass.

Run `python IO-20.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-20.py"

    [Download IO-20.py](IO-20.py)

    ```python
    """IO-20: Reverse MAD-X import keeps forward horizontal aperture offsets

    Loading a MAD-X sequence with reverse_lines leaves each aperture's horizontal offset unchanged instead of reflecting it into the reversed frame, where x changes sign. Rectangular, circular, elliptical, rectangle–ellipse and polygon apertures can therefore lose valid particles and keep particles outside the physical opening; asymmetric polygon vertices are not reflected either. For a rectangle with 1.000 mm horizontal half-width and a 3.000 mm offset, the imported centre is 6.000 mm from its required position. Independent Cartesian membership tests establish the required reflection, and reflecting only the horizontal geometry restores tracking; forward imports and zero-horizontal-offset controls of symmetric shapes pass.

    The script loads a 3 m drift sequence with one aperture marker forward and with reverse_lines, for five aperture shapes and three placements (positive and negative horizontal offset with roll, and zero horizontal offset with vertical offset and roll). A fixed grid of zero-momentum rays is tracked through each line, entered as x̃ = −x, ỹ = y in the reversed frame, and the surviving particle IDs are compared with Cartesian membership predicates written independently of Xtrack. Forward imports and a reversed line whose horizontal aperture geometry alone is reflected by hand are controls; a final two-ray case isolates the rectangular centre error.
    """

    from _prebuilt import CTX, identity, kernels, np, particles, xt

    LENGTH, AT = 3.0, 0.8
    # Grid nodes avoid every aperture boundary of the cases below by at least 1e-5 m.
    GX, GY = (a.ravel() for a in np.meshgrid(np.linspace(-6.05e-3, 6.05e-3, 56), np.linspace(-4.05e-3, 4.05e-3, 38)))
    # Asymmetric triangle (m), containing the local origin.
    TRI_X, TRI_Y = np.array([2.6e-3, -1.2e-3, -1.2e-3]), np.array([0.0, 1.9e-3, -1.9e-3])
    SHAPES = {
        "rectangle": ("apertype=rectangle, aperture={1.2e-3,2.1e-3}", lambda u, v: (abs(u) <= 1.2e-3) & (abs(v) <= 2.1e-3)),
        "ellipse": ("apertype=ellipse, aperture={1.2e-3,2.1e-3}", lambda u, v: (u / 1.2e-3) ** 2 + (v / 2.1e-3) ** 2 <= 1),
        "circle": ("apertype=circle, aperture={1.2e-3}", lambda u, v: u**2 + v**2 <= 1.2e-3**2),
        "rectellipse": (
            "apertype=rectellipse, aperture={1.2e-3,2.1e-3,1.7e-3,2.7e-3}",
            lambda u, v: (abs(u) <= 1.2e-3) & (abs(v) <= 2.1e-3) & ((u / 1.7e-3) ** 2 + (v / 2.7e-3) ** 2 <= 1),
        ),
        "polygon": (
            f"apertype=polygon, aper_vx={{{','.join(map(str, TRI_X))}}}, aper_vy={{{','.join(map(str, TRI_Y))}}}",
            # Inside the triangle: right of the vertical edge and under both slanted edges.
            lambda u, v: (u >= -1.2e-3) & (abs(v) <= 1.9e-3 * (2.6e-3 - u) / 3.8e-3),
        ),
    }
    PLACEMENTS = {"+dx": (2.5e-3, -0.7e-3, 0.43), "-dx": (-2.5e-3, -0.7e-3, -0.43), "dx=0": (0.0, -0.7e-3, 0.43)}


    def inside(predicate, x, y, dx, dy, tilt):
        # Membership of a physical ray (forward-frame x, y): shift to the aperture centre, then
        # undo its roll about s. The reversed frame (x~ = -x, y~ = y, D = diag(-1, 1)) must see
        # the same physical opening, so the same rays survive: centre D d, roll -tilt because
        # D R(tilt) D = R(-tilt), and polygon vertices reflected through x = 0.
        c, s = np.cos(tilt), np.sin(tilt)
        u, v = c * (x - dx) + s * (y - dy), -s * (x - dx) + c * (y - dy)
        return predicate(u, v)


    def load(deck, reverse):
        env = xt.load(string=deck, format="madx", reverse_lines=["s"] if reverse else None)
        line = env.lines["s"]
        table = line.get_table()
        (name,) = [n for n, t in zip(table.name, table.element_type) if t.startswith("Limit")]
        return line, name


    def survivors(line, x, y):
        line.build_tracker(_context=CTX)
        p = particles(x=x, y=y)
        line.track(p)
        return p.state[np.argsort(p.particle_id)] > 0


    def reflect_by_hand(aperture, dx):
        # Required reversed-frame geometry set from the deck values, independently of the import:
        # x-offset -dx and polygon vertices reflected through x = 0 (order reversed to keep the
        # polygon's orientation).
        aperture.shift_x = -dx
        if hasattr(aperture, "x_vertices"):
            aperture.x_vertices, aperture.y_vertices = -TRI_X[::-1], TRI_Y[::-1]


    def main():
        identity()
        print(f"RAYS {GX.size} per case, zero momenta; marker aperture at s={AT} m of a {LENGTH} m sequence")
        failures = 0
        for shape, (spec, predicate) in SHAPES.items():
            for label, (dx, dy, tilt) in PLACEMENTS.items():
                deck = (
                    f"m: marker, {spec}, aper_offset={{{dx},{dy}}}, aper_tilt={tilt};"
                    f"s: sequence, l={LENGTH}; m, at={AT}; endsequence;"
                )
                expected = inside(predicate, GX, GY, dx, dy, tilt)
                fwd_line, _ = load(deck, False)
                fwd = survivors(fwd_line, GX, GY)
                rev_line, name = load(deck, True)
                ap = rev_line[name]
                centre, roll = (float(ap.shift_x), float(ap.shift_y)), float(ap.rot_s_rad)
                s_rev = float(rev_line.get_table()["s", name])
                # Same physical rays in the reversed frame: x~ = -x, y~ = y.
                rev = survivors(rev_line, -GX, GY)
                rep_line, name = load(deck, True)
                reflect_by_hand(rep_line[name], dx)
                rep = survivors(rep_line, -GX, GY)
                assert np.array_equal(fwd, expected), f"forward control failed: {shape} {label}"
                assert np.array_equal(rep, expected), f"reflected control failed: {shape} {label}"
                assert abs(roll + tilt) == 0 and abs(s_rev - (LENGTH - AT)) < 1e-15
                bad = not np.array_equal(rev, expected)
                failures += bad
                print(
                    f"{shape} {label}: reverse survivors={rev.sum()} expected={expected.sum()} "
                    f"false_loss={(expected & ~rev).sum()} false_survival={(rev & ~expected).sum()} "
                    f"centre_mm=({1e3 * centre[0]:.4g}, {1e3 * centre[1]:.4g}) "
                    f"expected=({-1e3 * dx + 0.0:.4g}, {1e3 * dy:.4g}) roll={roll} s={s_rev:.15g}; "
                    f"forward and hand-reflected controls exact; "
                    f"{'FAIL' if bad else 'pass'}; derivation=x~=-x reflection of the Cartesian predicate"
                )
        # Two rays at the required reversed centre x~=-3 mm and at the forward centre x~=+3 mm.
        deck = (
            "m: marker, apertype=rectangle, aperture={1e-3,2e-3}, aper_offset={3e-3,0};"
            f"s: sequence, l={LENGTH}; m, at={AT}; endsequence;"
        )
        line, name = load(deck, True)
        centre = float(line[name].shift_x)
        state = survivors(line, np.array([-3e-3, 3e-3]), np.zeros(2)).astype(int)
        failures += not np.array_equal(state, [1, 0])
        print(
            f"rectangle half-width 1 mm, offset +3 mm, reversed: survival at x~=(-3, +3) mm "
            f"observed={state} expected=[1 0] centre_mm={1e3 * centre:.4g} expected=-3 "
            f"centre_error_mm={1e3 * abs(centre + 3e-3):.4g}; derivation=x~=-x"
        )
        kernels()
        print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed<br><small>polygon offsets and asymmetric polygon vertices</small> | `claude-opus-5-5` | high | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

