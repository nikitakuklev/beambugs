---
title: "IO-27: Bend export replaces the exit half-gap by the entrance half-gap"
hide:
  - toc
---

# IO-27 — Bend export replaces the exit half-gap by the entrance half-gap

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`bend_to_mad_str`, mad_writer.py:467–471](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L467-L471)<br>[`rbend_to_mad_str`, lines 485–487](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L485-L487)<br>[_common.py:137–148](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/_common.py#L137-L148)<br>[Bend kernel, bend.h:70–73](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/bend.h#L70-L73)<br>[RBend kernel, rbend.h:70–73](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/rbend.h#L70-L73)<br>[linear edge coefficient, track_dipole_edge_linear.h:12–25](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_dipole_edge_linear.h#L12-L25)<br>[fringe integral through the full gap g = 2·hgap, xtrack.tex:1592–1604](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L1592-L1604)<br>[separate FINT and FINTX but one HGAP, elements.tex:129–137](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/doc/latexuguide/elements.tex#L129-L137)<br>[fringe-field definition, lines 197–228](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/doc/latexuguide/elements.tex#L197-L228) |

Exporting a bend with different entrance and exit half-gaps to MAD-X or MAD-NG silently applies the entrance gap to both faces while retaining the original fringe integrals. This changes exit focusing even for an unsliced magnet. For a 1.200 m, 0.1200 rad sector bend with 0.02000/0.04000 m half-gaps, the vertical matrix coefficient R₄₃ is −0.009620 m⁻¹ instead of −0.009461 m⁻¹, a 1.686% excess in magnitude. Independent edge matrices and native tracking agree on the intended result; preserving the exit gap–fringe-integral product restores it. Rectangular bends also fail; equal gaps avoid this error.

Run `python IO-27.py` with [_madx_io.py](../helpers/_madx_io.py), [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-27.py"

    [Download IO-27.py](IO-27.py)

    ```python
    """IO-27: Bend export replaces the exit half-gap by the entrance half-gap

    Exporting a bend with different entrance and exit half-gaps to MAD-X or MAD-NG silently applies the entrance gap to both faces while retaining the original fringe integrals. This changes exit focusing even for an unsliced magnet. For a 1.200 m, 0.1200 rad sector bend with 0.02000/0.04000 m half-gaps, the vertical matrix coefficient R₄₃ is −0.009620 m⁻¹ instead of −0.009461 m⁻¹, a 1.686% excess in magnitude. Independent edge matrices and native tracking agree on the intended result; preserving the exit gap–fringe-integral product restores it. Rectangular bends also fail; equal gaps avoid this error.

    Export an unsliced sector bend and an unsliced rectangular bend with finite face angles, fringe integrals and unequal half-gaps, then compare the vertical transfer matrix from MAD-X TWISS and small-ray tracking in PTC and in MAD-NG (whole line through Line.to_madng) with an independent product of 2x2 linear edge and drift matrices that uses each face's own gap; each failing case is also compared with the same product using the entrance gap at the exit. Controls are native tracking, equal gaps, and the exported MAD-X text with only FINTX changed to keep the exit gap-fringe product. The script uses cpymad, pymadng and installed serial prebuilt Xtrack kernels with compilation forbidden; scratch files stay in automatically removed directories beside the script.
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
    HGAP_ENTRY, HGAP_EXIT = 0.02, 0.04
    # A small vertical ray probes the linear map; nonlinear terms stay far below TOL.
    RAY = [0.0, 0.0, 1e-7, 2e-9]
    TOL = 3e-19
    MATRIX_TOL = 1e-15
    failures = []


    def faces(hgap_exit=HGAP_EXIT):
        return dict(edge_entry_angle=E_ENTRY, edge_exit_angle=E_EXIT,
                    edge_entry_fint=FINT_ENTRY, edge_exit_fint=FINT_EXIT,
                    edge_entry_hgap=HGAP_ENTRY, edge_exit_hgap=hgap_exit, model="bend-kick-bend")


    def line_of(element):
        line = xt.Line(elements={"m": element})
        line.particle_ref = xt.Particles(p0c=PC)
        return line


    def sbend_line(hgap_exit=HGAP_EXIT):
        return line_of(xt.Bend(length=LENGTH, angle=ANGLE, **faces(hgap_exit)))


    def rbend_line():
        # E_ENTRY/E_EXIT are rotations beyond the RBend's parallel faces.
        return line_of(xt.RBend(length_straight=LENGTH, angle=ANGLE, **faces()))


    # Independent vertical linear optics (MAD-X user guide and Xsuite physics manual):
    # a face of sector-frame angle e, fringe integral F and half-gap g on a bend of
    # curvature h kicks py by -h tan(e - psi) y with psi = 2 h g F (1 + sin^2 e) / cos e;
    # a sector body is a vertical drift of its arc length.
    def face(h, e, fint, hgap):
        psi = 2 * h * hgap * fint * (1 + np.sin(e) ** 2) / np.cos(e)
        return np.array([[1.0, 0.0], [-h * np.tan(e - psi), 1.0]])


    def magnet(h, arc, e_in, e_out, exit_gap):
        return (face(h, e_out, FINT_EXIT, exit_gap) @ np.array([[1.0, arc], [0.0, 1.0]])
                @ face(h, e_in, FINT_ENTRY, HGAP_ENTRY))


    def sector_maps(hgap_exit=HGAP_EXIT):
        h = ANGLE / LENGTH
        return (magnet(h, LENGTH, E_ENTRY, E_EXIT, hgap_exit),
                magnet(h, LENGTH, E_ENTRY, E_EXIT, HGAP_ENTRY))


    def rbend_maps():
        # Arc of the chord LENGTH: R = LENGTH / (2 sin(ANGLE/2)); in the sector frame each
        # RBend face is half the bend angle plus its extra rotation.
        radius = LENGTH / (2 * np.sin(ANGLE / 2))
        args = (1 / radius, radius * ANGLE, ANGLE / 2 + E_ENTRY, ANGLE / 2 + E_EXIT)
        return magnet(*args, HGAP_EXIT), magnet(*args, HGAP_ENTRY)


    def record(label, ok, defect):
        if defect:
            failures.append(not ok)
        else:
            assert ok, label
        return "pass" if ok else "FAIL"


    def ray_compare(label, observed, intended, substituted, defect):
        observed = np.asarray(observed, dtype=float)
        expected = intended @ np.array(RAY[2:])
        ok = bool(np.allclose(observed, expected, atol=TOL, rtol=0))
        print(f"{label} (y,py): observed={observed.tolist()} expected={expected.tolist()} "
              f"derivation=each face with its own half-gap tol={TOL:g} {record(label, ok, defect)}")
        if defect:
            alt = substituted @ np.array(RAY[2:])
            assert np.allclose(observed, alt, atol=TOL, rtol=0), (label, observed, alt)
            print(f"{label} vs entrance gap at the exit (y,py): observed={observed.tolist()} "
                  f"expected={alt.tolist()} tol={TOL:g} pass")


    def twiss_vertical(source):
        with (
            TemporaryDirectory(prefix="io27-madx-", dir=Path(__file__).resolve().parent) as scratch,
            Madx(stdout=False, cwd=scratch) as mad,
        ):
            mad.input(source)
            mad.beam(particle="proton", pc=PC / 1e9)
            mad.use(sequence="testseq")
            tw = mad.twiss(betx=1, bety=1, rmatrix=True)
            print(f"CONSUMER cpymad={version('cpymad')} MAD-X={mad.version} TWISS")
            return np.array([[tw.re33[-1], tw.re34[-1]], [tw.re43[-1], tw.re44[-1]]])


    def twiss_compare(label, source, intended, substituted, defect):
        observed = twiss_vertical(source)
        ok = bool(np.allclose(observed, intended, atol=MATRIX_TOL, rtol=0))
        excess = 100 * (abs(observed[1, 0]) / abs(intended[1, 0]) - 1)
        print(f"MAD-X TWISS {label} vertical R: observed={observed.tolist()} expected={intended.tolist()} "
              f"derivation=each face with its own half-gap tol={MATRIX_TOL:g} {record(label, ok, defect)}")
        if defect:
            print(f"MAD-X TWISS {label} |R43| excess {excess:.6f}%")
            assert np.allclose(observed, substituted, atol=MATRIX_TOL, rtol=0), (label, observed, substituted)
            print(f"MAD-X TWISS {label} vs entrance gap at the exit: expected={substituted.tolist()} "
                  f"tol={MATRIX_TOL:g} pass")


    def native_y(line):
        p = xt.Particles(_context=CTX, p0c=PC, x=RAY[0], px=RAY[1], y=RAY[2], py=RAY[3])
        line.build_tracker(_context=CTX)
        line.track(p)
        return [p.y[0], p.py[0]]


    def ng_y(line):
        sys.stdout.flush()
        end, _ = madng_track(line, RAY)
        return end[2:]


    def ptc_y(source):
        sys.stdout.flush()
        _, end = ptc_track(source, RAY, PC / 1e9)
        return end[2:]


    def text_checks():
        # The written exit product FINTX * HGAP must equal the native F_exit * g_exit.
        line = sbend_line()
        for dialect in (writer.MadType.MADX, writer.MadType.MADNG):
            text = writer.element_to_mad_str("m", "m", line, mad_type=dialect, substituted_vars=[])
            tokens = dict(t.split(" = ") for t in text.split(", ") if t.count(" = ") == 1)
            hgap, fint, fintx = (float(tokens[k].rstrip(" }")) for k in ("hgap", "fint", "fintx"))
            entry_ok = abs(fint * hgap - FINT_ENTRY * HGAP_ENTRY) < 1e-15
            exit_ok = abs(fintx * hgap - FINT_EXIT * HGAP_EXIT) < 1e-15
            assert entry_ok, text
            print(f"TEXT {dialect.name}: hgap={hgap} fint={fint} fintx={fintx}; entrance product "
                  f"{fint * hgap:g} expected {FINT_ENTRY * HGAP_ENTRY:g} pass; exit product {fintx * hgap:g} "
                  f"expected {FINT_EXIT * HGAP_EXIT:g} derivation=native edge_exit_fint * edge_exit_hgap "
                  f"{record('text', exit_ok, True)}")


    def main():
        identity()
        print(f"PACKAGE cpymad {version('cpymad')}; pymadng {version('pymadng')}")
        text_checks()

        intended, substituted = sector_maps()
        print(f"INTENDED sector R43={float(intended[1, 0])!r} 1/m; entrance gap at the exit "
              f"R43={float(substituted[1, 0])!r} 1/m")
        line = sbend_line()
        source = line.to_madx_sequence("testseq")
        ray_compare("native Bend", native_y(line), intended, substituted, False)
        twiss_compare("Bend", source, intended, substituted, True)
        ray_compare("PTC Bend", ptc_y(source), intended, substituted, True)
        ray_compare("MAD-NG Bend", ng_y(line), intended, substituted, True)

        # Equivalent exit fringe integral: FINTX * HGAP_ENTRY = FINT_EXIT * HGAP_EXIT.
        fintx = FINT_EXIT * HGAP_EXIT / HGAP_ENTRY
        patched = source.replace(f"fintx = {FINT_EXIT}", f"fintx = {fintx}")
        assert patched.count(f"fintx = {fintx}") == 1, source
        twiss_compare(f"Bend with FINTX={fintx} only changed", patched, intended, substituted, False)

        equal, _ = sector_maps(HGAP_ENTRY)
        line = sbend_line(HGAP_ENTRY)
        twiss_compare("Bend equal gaps", line.to_madx_sequence("testseq"), equal, equal, False)
        ray_compare("MAD-NG Bend equal gaps", ng_y(line), equal, equal, False)

        rb_intended, rb_substituted = rbend_maps()
        print(f"INTENDED RBend R43={float(rb_intended[1, 0])!r} 1/m; entrance gap at the exit "
              f"R43={float(rb_substituted[1, 0])!r} 1/m")
        line = rbend_line()
        ray_compare("native RBend", native_y(line), rb_intended, rb_substituted, False)
        twiss_compare("RBend", line.to_madx_sequence("testseq"), rb_intended, rb_substituted, True)
        ray_compare("MAD-NG RBend", ng_y(line), rb_intended, rb_substituted, True)

        kernels()
        print(f"COMPARISONS expected-correct cases fail={sum(failures)} of {len(failures)}")
        print("DEFECT REPRODUCED" if any(failures) else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, exported MAD-X/MAD-NG text |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-07 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03; pymadng 0.9.6, MAD-NG 1.1.14 |
| Confirmed<br><small>PTC tracking of the unsliced export</small> | `claude-opus-5-5` | high | 2026-11-07 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03 TWISS and PTC; pymadng 0.9.6, MAD-NG 1.1.14 |

