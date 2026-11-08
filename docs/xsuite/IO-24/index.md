---
title: "IO-24: Exported bend body slices each repeat the parent's edge faces"
hide:
  - toc
---

# IO-24 — Exported bend body slices each repeat the parent's edge faces

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Source | [`bend_to_mad_str`, mad_writer.py:448–483](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L448-L483)<br>[writes the parent's `e1`, `e2`, `fint`, `fintx` and `hgap`, lines 467–476](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L467-L476)<br>[`rbend_to_mad_str`, lines 485–487](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L485-L487)<br>[slice dispatch, lines 670–680](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L670-L680)<br>[Bend body kernel, thick_slice_bend.h:64–76](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/thick_slice_bend.h#L64-L76)<br>[RBend body kernel, thick_slice_rbend.h:64–76](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/thick_slice_rbend.h#L64-L76)<br>[entry map, thin_slice_bend_entry.h:64–76](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/thin_slice_bend_entry.h#L64-L76)<br>[exit map, thin_slice_bend_exit.h:64–76](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/thin_slice_bend_exit.h#L64-L76)<br>[E1, E2, FINT, FINTX and HGAP, elements.tex:115–137](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/doc/latexuguide/elements.tex#L115-L137)<br>[fringe integral and gap g = 2·hgap, xtrack.tex:1592–1604](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L1592-L1604)<br>[effective-angle edge model, lines 2151–2193](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L2151-L2193)<br>[linear edge coefficient, track_dipole_edge_linear.h:12–25](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_dipole_edge_linear.h#L12-L25) |

Exporting native bend body slices to MAD-X or MAD-NG copies the parent's entrance and exit face settings onto every body. These numerical slice boundaries therefore acquire extra edge focusing. For a 1.200 m sector bend of 0.1200 rad with 0.06000/0.04000 rad faces, four thick slices send a small test ray's normalized vertical momentum pᵧ to −1.872×10⁻⁹ instead of +1.029×10⁻⁹. PTC and whole-line MAD-NG tracking match independent matrices that repeat both faces four times; native tracking matches one physical pair. Rectangular bends and unequal slices are affected. MAD-NG with both faces inactive avoids the repeated focusing.

Run `python IO-24.py` with [_madx_io.py](../helpers/_madx_io.py), [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-24.py"

    [Download IO-24.py](IO-24.py)

    ```python
    """IO-24: Exported bend body slices each repeat the parent's edge faces

    Exporting native bend body slices to MAD-X or MAD-NG copies the parent's entrance and exit face settings onto every body. These numerical slice boundaries therefore acquire extra edge focusing. For a 1.200 m sector bend of 0.1200 rad with 0.06000/0.04000 rad faces, four thick slices send a small test ray's normalized vertical momentum pᵧ to −1.872×10⁻⁹ instead of +1.029×10⁻⁹. PTC and whole-line MAD-NG tracking match independent matrices that repeat both faces four times; native tracking matches one physical pair. Rectangular bends and unequal slices are affected. MAD-NG with both faces inactive avoids the repeated focusing.

    Slice a sector bend and a rectangular bend with finite face angles, fringe integrals and half-gaps into thick bodies (Uniform 1, 2 and 4; Teapot 4), load the whole line into MAD-NG through Line.to_madng and track a small vertical ray. The expected value is an independent product of 2x2 vertical maps with one physical face pair; each failing case is also compared with the same product repeating the pair on every body. Exported body-only selections run through PTC and MAD-X TWISS, whose whole sliced lines MAD-X refuses on an unrelated keyword. Controls are native tracking, the single-body export and MAD-NG with both faces inactive; the script uses cpymad, pymadng and installed serial prebuilt Xtrack kernels with compilation forbidden, with scratch files in automatically removed directories beside the script.
    """

    import sys
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from _madx_io import madng_track, ptc_track
    from _prebuilt import CTX, identity, kernels, np, xt

    # Import after the helpers cap numerical threads and forbid compilation.
    # isort: split
    from importlib.metadata import version

    from cpymad.madx import Madx
    from xtrack import mad_writer as writer

    PC = 3e9
    LENGTH = 1.2
    ANGLE = 0.12
    E_ENTRY, E_EXIT = 0.06, 0.04
    FINT_ENTRY, FINT_EXIT = 0.5, 0.4
    HGAP = 0.02
    # Extra RBend face rotations beyond its parallel (rectangular) faces.
    RBEND_E_ENTRY, RBEND_E_EXIT = 0.03, 0.02
    # A small vertical ray probes the linear map; nonlinear terms stay far below TOL.
    RAY = [0.0, 0.0, 1e-7, 2e-9]
    TOL = 3e-19
    failures = []


    def faces(**extra):
        return dict(edge_entry_fint=FINT_ENTRY, edge_exit_fint=FINT_EXIT,
                    edge_entry_hgap=HGAP, edge_exit_hgap=HGAP, model="bend-kick-bend", **extra)


    def bend(**extra):
        return xt.Bend(length=LENGTH, angle=ANGLE, edge_entry_angle=E_ENTRY, edge_exit_angle=E_EXIT,
                       **faces(**extra))


    def rbend():
        return xt.RBend(length_straight=LENGTH, angle=ANGLE, edge_entry_angle=RBEND_E_ENTRY,
                        edge_exit_angle=RBEND_E_EXIT, **faces())


    def sliced(element, scheme):
        line = xt.Line(elements={"m": element})
        line.particle_ref = xt.Particles(p0c=PC)
        line.slice_thick_elements([xt.Strategy(slicing=scheme)], with_progress=False)
        return line


    def body_weights(line):
        return [float(line[nn].weight) for nn in line.element_names if nn.startswith("m..") and nn[3:].isdigit()]


    # Independent vertical linear optics. A face of angle e (sector frame) with fringe
    # integral F and half-gap g on a bend of curvature h kicks py by -h tan(e - psi) y,
    # psi = 2 h g F (1 + sin^2 e) / cos e (MAD-X and the Xsuite physics manual use the
    # same finite-gap correction); the sector body is a vertical drift of its arc length.
    def face(h, e, fint):
        psi = 2 * h * HGAP * fint * (1 + np.sin(e) ** 2) / np.cos(e)
        return np.array([[1.0, 0.0], [-h * np.tan(e - psi), 1.0]])


    def drift(length):
        return np.array([[1.0, length], [0.0, 1.0]])


    def chain(h, bodies):
        # bodies: (arc length, entrance face, exit face) in beam order.
        m = np.eye(2)
        for length, e_in, e_out in bodies:
            m = face(h, e_out, FINT_EXIT) @ drift(length) @ face(h, e_in, FINT_ENTRY) @ m
        return m


    def sector_maps(weights):
        h = ANGLE / LENGTH
        physical = chain(h, [(LENGTH, E_ENTRY, E_EXIT)])
        repeated = chain(h, [(w * LENGTH, E_ENTRY, E_EXIT) for w in weights])
        return physical, repeated


    def rbend_maps(weights):
        # Arc of the chord LENGTH: radius R = LENGTH / (2 sin(ANGLE/2)). In the sector
        # frame each RBend face is half the bend angle plus its extra rotation; an
        # exported body of angle w*ANGLE gets half of its own angle.
        radius = LENGTH / (2 * np.sin(ANGLE / 2))
        arc, h = radius * ANGLE, 1 / radius
        physical = chain(h, [(arc, ANGLE / 2 + RBEND_E_ENTRY, ANGLE / 2 + RBEND_E_EXIT)])
        repeated = chain(h, [(w * arc, w * ANGLE / 2 + RBEND_E_ENTRY, w * ANGLE / 2 + RBEND_E_EXIT)
                             for w in weights])
        return physical, repeated


    def vertical(m):
        return m @ np.array(RAY[2:])


    def compare(label, observed, expected, derivation, defect, repeated=None, n_pairs=None):
        observed = np.asarray(observed, dtype=float)
        ok = bool(np.allclose(observed, expected, atol=TOL, rtol=0))
        if defect:
            failures.append(not ok)
        else:
            assert ok, (label, observed, expected)
        print(f"{label} (y,py): observed={observed.tolist()} expected={expected.tolist()} "
              f"derivation={derivation} tol={TOL:g} {'pass' if ok else 'FAIL'}")
        if defect:
            assert repeated is not None
            match = bool(np.allclose(observed, repeated, atol=TOL, rtol=0))
            assert match, (label, observed, repeated)
            print(f"{label} vs {n_pairs} repeated face pairs (y,py): observed={observed.tolist()} "
                  f"expected={repeated.tolist()} tol={TOL:g} pass")


    def native_y(line):
        p = xt.Particles(_context=CTX, p0c=PC, x=RAY[0], px=RAY[1], y=RAY[2], py=RAY[3])
        line.build_tracker(_context=CTX)
        line.track(p)
        return [p.y[0], p.py[0]]


    def ng_y(line):
        sys.stdout.flush()
        end, _ = madng_track(line, RAY)
        return end[2:]


    def text_checks():
        # Native body slices disable both faces (thick_slice_bend.h); the entry and
        # exit maps own them. Count exported bodies that carry the entrance face.
        line = sliced(bend(), xt.Uniform(4, mode="thick"))
        bodies = [nn for nn in line.element_names if nn.startswith("m..") and nn[3:].isdigit()]
        for dialect in (writer.MadType.MADX, writer.MadType.MADNG):
            texts = [writer.element_to_mad_str(nn, nn, line, mad_type=dialect, substituted_vars=[])
                     for nn in bodies]
            count = sum(f"e1 = {E_ENTRY}" in t and f"e2 = {E_EXIT}" in t and f"fint = {FINT_ENTRY}" in t
                        for t in texts)
            failures.append(count != 0)
            print(f"Bend Uniform(4) {dialect.name}: bodies carrying e1/e2/fint={count} of {len(bodies)} "
                  f"expected=0 derivation=native body slices disable both faces "
                  f"{'pass' if count == 0 else 'FAIL'}; first body: {texts[0]}")


    def whole_line_checks():
        physical, _ = sector_maps([1.0])
        for n in (1, 2, 4):
            line = sliced(bend(), xt.Uniform(n, mode="thick"))
            _, repeated = sector_maps(body_weights(line))
            if n == 4:
                compare("native Bend Uniform(4)", native_y(line), vertical(physical),
                        "one physical face pair", False)
            compare(f"MAD-NG Bend Uniform({n})", ng_y(line), vertical(physical),
                    "one physical face pair", n > 1, vertical(repeated), n)

        line = sliced(bend(), xt.Teapot(4, mode="thick"))
        weights = body_weights(line)
        _, repeated = sector_maps(weights)
        compare(f"MAD-NG Bend Teapot(4) weights={weights}", ng_y(line), vertical(physical),
                "one physical face pair", True, vertical(repeated), 4)

        for n in (1, 4):
            line = sliced(rbend(), xt.Uniform(n, mode="thick"))
            rb_physical, rb_repeated = rbend_maps(body_weights(line))
            compare(f"MAD-NG RBend Uniform({n})", ng_y(line), vertical(rb_physical),
                    "one physical RBend face pair", n > 1, vertical(rb_repeated), n)

        # Both faces inactive: the magnet is a vertical drift natively and in MAD-NG,
        # which receives kill_ent_fringe/kill_exi_fringe.
        line = sliced(bend(edge_entry_active=False, edge_exit_active=False), xt.Uniform(4, mode="thick"))
        field_free = vertical(drift(LENGTH))
        compare("native Bend Uniform(4) faces inactive", native_y(line), field_free,
                "vertical drift over the arc", False)
        compare("MAD-NG Bend Uniform(4) faces inactive", ng_y(line), field_free,
                "vertical drift over the arc", False)


    def body_only_checks():
        # MAD-X refuses the whole sliced line on the generated entry/exit maps, so the
        # four bodies are selected natively; without the maps they carry no face.
        line = sliced(bend(), xt.Uniform(4, mode="thick"))
        body = line.select(start="m..0", end="m..3")
        physical, repeated = sector_maps(body_weights(line))
        field_free = drift(LENGTH)
        compare("native Bend Uniform(4) body-only selection", native_y(body), vertical(field_free),
                "vertical drift over the arc", False)
        source = body.to_madx_sequence("testseq")
        sys.stdout.flush()
        _, end = ptc_track(source, RAY, PC / 1e9)
        compare("PTC Bend Uniform(4) body-only selection", end[2:], vertical(field_free),
                "vertical drift over the arc", True, vertical(repeated), 4)

        with (
            TemporaryDirectory(prefix="io24-madx-", dir=Path(__file__).resolve().parent) as scratch,
            Madx(stdout=False, cwd=scratch) as mad,
        ):
            mad.input(source)
            mad.beam(particle="proton", pc=PC / 1e9)
            mad.use(sequence="testseq")
            tw = mad.twiss(betx=1, bety=1, rmatrix=True)
            observed = np.array([[tw.re33[-1], tw.re34[-1]], [tw.re43[-1], tw.re44[-1]]])
        ok = bool(np.allclose(observed, field_free, atol=1e-15, rtol=0))
        failures.append(not ok)
        assert np.allclose(observed, repeated, atol=1e-15, rtol=0), (observed, repeated)
        print(f"MAD-X TWISS Bend Uniform(4) body-only vertical R: observed={observed.tolist()} "
              f"expected={field_free.tolist()} derivation=vertical drift over the arc tol=1e-15 "
              f"{'pass' if ok else 'FAIL'}; 4 repeated face pairs={repeated.tolist()} pass; "
              f"whole-magnet R43 with one pair={float(physical[1, 0])!r} 1/m, "
              f"excess {100 * (observed[1, 0] / physical[1, 0] - 1):.4f}%")


    def main():
        identity()
        print(f"PACKAGE cpymad {version('cpymad')}; pymadng {version('pymadng')}")
        text_checks()
        whole_line_checks()
        body_only_checks()
        kernels()
        print(f"COMPARISONS expected-correct cases fail={sum(failures)} of {len(failures)}")
        print("DEFECT REPRODUCED" if any(failures) else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, exported MAD-X/MAD-NG text |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03 PTC; pymadng 0.9.6, MAD-NG 1.1.14 |
| Confirmed<br><small class="bb-scope">MAD-X TWISS of exported bodies</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03 TWISS and PTC; pymadng 0.9.6, MAD-NG 1.1.14 |

