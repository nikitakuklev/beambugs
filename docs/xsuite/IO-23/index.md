---
title: "IO-23: Exported thin-mode slices occupy the parent's length a second time"
hide:
  - toc
---

# IO-23 — Exported thin-mode slices occupy the parent's length a second time

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack main (`f09e0fb`, head build) |
| Source | [`_ThinSliceElementBase`, slice_elements_thin.py:18–23](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/slice_elements_thin.py#L18-L23)<br>[thin slicing, slicing.py:402–441](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/slicing.py#L402-L441)<br>[default of `Uniform` and `Teapot`, lines 22–26](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/slicing.py#L22-L26)<br>[slice dispatch, mad_writer.py:670–680](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L670-L680)<br>[cavity, lines 183–194](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L183-L194)<br>[Multipole, lines 312–358](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L312-L358)<br>[bend, lines 448–464](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L448-L464)<br>[sextupole, lines 501–509](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L501-L509)<br>[octupole, lines 530–538](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L530-L538)<br>[quadrupole, lines 559–567](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L559-L567)<br>[MAD-X sequence writer, lines 720–745](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L720-L745)<br>[MAD-NG writer, lines 786–826](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L786-L826)<br>[`Line.to_madx_sequence`, line.py:877–894](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L877-L894)<br>[`Line.to_madng`, lines 897–918](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L897-L918)<br>[`build_madng_model`, madng_interface.py:98–124](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/madng_interface.py#L98-L124)<br>[lines 251–253](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/madng_interface.py#L251-L253)<br>[sequence.tex:32–37](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/doc/latexuguide/sequence.tex#L32-L37)<br>[sequence.tex:125–141](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/doc/latexuguide/sequence.tex#L125-L141) |

Exporting native thin-mode slices to MAD-X or MAD-NG writes their kicks as finite-length parent elements while retaining the drift slices that already span the magnet. Four thin slices of a 1.200 m quadrupole therefore occupy 2.400 m. MAD-X rejects the overlapping sequence; MAD-NG accepts the consecutive elements and silently tracks the extra length. With zero fields, a proton starting at x = 3.000 mm and normalized horizontal momentum pₓ = 0.001 exits at 5.400 mm instead of the independently calculated 4.200 mm. Thick slicing, unsliced export and equivalent zero-length Multipole kicks preserve the length.

Run `python IO-23.py` with [_madx_io.py](../helpers/_madx_io.py), [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-23.py"

    [Download IO-23.py](IO-23.py)

    ```python
    """IO-23: Exported thin-mode slices occupy the parent's length a second time

    Exporting native thin-mode slices to MAD-X or MAD-NG writes their kicks as finite-length parent elements while retaining the drift slices that already span the magnet. Four thin slices of a 1.200 m quadrupole therefore occupy 2.400 m. MAD-X rejects the overlapping sequence; MAD-NG accepts the consecutive elements and silently tracks the extra length. With zero fields, a proton starting at x = 3.000 mm and normalized horizontal momentum pₓ = 0.001 exits at 5.400 mm instead of the independently calculated 4.200 mm. Thick slicing, unsliced export and equivalent zero-length Multipole kicks preserve the length.

    Slice seven zero-field 1.2 m magnet families (and a dipole-only kicker Multipole) in Uniform and Teapot thin mode, export every element in both writer dialects and sum the occupied lengths `l`, against the parent length; thick slicing is the control. MAD-X then parses the exported cavity and a quadrupole whose identity entry/exit maps are removed natively, and MAD-NG loads the whole line through Line.to_madng and tracks one ray, against exact field-free flight over the parent length. Controls are unsliced and thick-sliced exports, a natively built line of drifts and zero-length Multipole kicks with `lrad`, and native tracking. The script uses cpymad, pymadng and installed serial prebuilt Xtrack kernels with compilation forbidden; scratch files stay in automatically removed directories beside the script.
    """

    import re
    import sys

    from _madx_io import consume, madng_track
    from _prebuilt import CTX, identity, kernels, np, xt

    # Import after the helpers cap numerical threads and forbid compilation.
    # isort: split
    from importlib.metadata import version

    from xtrack import mad_writer as writer

    LENGTH = 1.2
    PC = 3e9
    INITIAL = [0.003, 1e-3, 0.0, 0.0]
    # Every field is zero: a slice kick then has no effect, and only geometry matters.
    FAMILIES = {
        "Quadrupole": lambda: xt.Quadrupole(length=LENGTH),
        "Sextupole": lambda: xt.Sextupole(length=LENGTH),
        "Octupole": lambda: xt.Octupole(length=LENGTH),
        "Bend": lambda: xt.Bend(length=LENGTH, angle=0.0),
        # Zero angle: arc and chord coincide, so no RBend length convention enters.
        "RBend": lambda: xt.RBend(length_straight=LENGTH, angle=0.0),
        "Multipole": lambda: xt.Multipole(length=LENGTH, isthick=True, knl=[0, 0], ksl=[0, 0]),
        "Multipole kicker": lambda: xt.Multipole(length=LENGTH, isthick=True, knl=[0], ksl=[0]),
        "Cavity": lambda: xt.Cavity(length=LENGTH, frequency=400e6),
    }
    SCHEMES = {
        "Uniform(4) thin": lambda: xt.Uniform(4, mode="thin"),
        "Teapot(4) thin": lambda: xt.Teapot(4, mode="thin"),
        "Uniform(4) thick": lambda: xt.Uniform(4, mode="thick"),
        "Teapot(4) thick": lambda: xt.Teapot(4, mode="thick"),
    }
    # Generated entry/exit maps are written with their weight floored to 1e-10, so two
    # of them occupy 2.4e-10 m; the tolerance admits that and nothing of slice size.
    GEOMETRY_TOL = 1e-9
    # Occupied length `l`; `lrad` (a thin element's radiation length) is not matched.
    OCCUPIED = re.compile(r"(?<![\w.])l = ([^,}]+)")
    failures = []


    def line_of(element, scheme=None):
        line = xt.Line(elements={"m": element})
        line.particle_ref = xt.Particles(p0c=PC)
        if scheme is not None:
            line.slice_thick_elements([xt.Strategy(slicing=scheme)], with_progress=False)
        return line


    def occupied_length(line, dialect):
        total = 0.0
        for nn in line.element_names:
            text = writer.element_to_mad_str(nn, nn, line, mad_type=dialect, substituted_vars=[])
            total += sum(float(v) for v in OCCUPIED.findall(text))
        return total


    def geometry_checks():
        for family, make in FAMILIES.items():
            for scheme_name, scheme in SCHEMES.items():
                line = line_of(make(), scheme())
                for dialect in (writer.MadType.MADX, writer.MadType.MADNG):
                    observed = occupied_length(line, dialect)
                    # A thin kick has zero extent; the drift slices (thin mode) or the
                    # thick bodies already span the parent, so the total is its length.
                    ok = abs(observed - LENGTH) < GEOMETRY_TOL
                    if "thick" in scheme_name:
                        assert ok, (family, scheme_name, dialect, observed)
                    else:
                        failures.append(not ok)
                    print(f"{family} {scheme_name} {dialect.name}: exported occupied length={observed!r} m "
                          f"expected={LENGTH} m derivation=parent length, thin kicks have zero extent "
                          f"tol={GEOMETRY_TOL:g} {'pass' if ok else 'FAIL'}")


    def without_maps(line):
        # The quadrupole's generated entry/exit maps are identity maps without fringes;
        # removing them natively keeps the line's physics and isolates the kick lengths.
        for nn in ("m..entry_map", "m..exit_map"):
            line.remove(nn)
        return line


    def lowered_line():
        # Independent native construction of the thin quadrupole: five 0.24 m drifts
        # around four zero-length kicks carrying the slice length as `lrad`.
        elements, names = {}, []
        for i in range(5):
            elements[f"d{i}"] = xt.Drift(length=LENGTH / 5)
            names.append(f"d{i}")
            if i < 4:
                elements[f"k{i}"] = xt.Multipole(knl=[0, 0], ksl=[0, 0], length=LENGTH / 4)
                names.append(f"k{i}")
        line = xt.Line(elements=elements, element_names=names)
        line.particle_ref = xt.Particles(p0c=PC)
        return line


    def madx_checks():
        cases = (
            ("Cavity Uniform(4) thin", line_of(FAMILIES["Cavity"](), xt.Uniform(4, mode="thin")), True),
            ("Quadrupole Uniform(4) thin, maps removed",
             without_maps(line_of(FAMILIES["Quadrupole"](), xt.Uniform(4, mode="thin"))), True),
            ("Cavity Uniform(4) thick", line_of(FAMILIES["Cavity"](), xt.Uniform(4, mode="thick")), False),
            ("Quadrupole Uniform(4) thick, maps removed",
             without_maps(line_of(FAMILIES["Quadrupole"](), xt.Uniform(4, mode="thick"))), False),
            ("Quadrupole unsliced", line_of(FAMILIES["Quadrupole"]()), False),
            ("drifts and zero-length Multipole kicks", lowered_line(), False),
        )
        for label, line, defect in cases:
            # MAD-X writes its diagnostics directly; flushing keeps them in order.
            sys.stdout.flush()
            result = consume(line.to_madx_sequence("testseq"))
            ok = result["status"] == "accepted" and abs(result["length"] - LENGTH) < 1e-12
            if defect:
                failures.append(not ok)
            else:
                assert ok, (label, result)
            print(f"MAD-X {label}: status={result['status']} length={result.get('length')} "
                  f"error={result.get('error')!r} expected=accepted, {LENGTH} m "
                  f"derivation=MAD-X refuses overlaps beyond 1 um {'pass' if ok else 'FAIL'}")


    def tracking_checks():
        x0, px0 = INITIAL[0], INITIAL[1]
        # Exact field-free flight over the parent length S: x' = px / ps with
        # ps = sqrt(1 - px^2 - py^2), as MAD-NG's exact drift integrates it.
        exact = np.array([x0 + LENGTH * px0 / np.sqrt(1 - px0**2), px0, 0.0, 0.0])
        doubled = np.array([x0 + 2 * LENGTH * px0 / np.sqrt(1 - px0**2), px0, 0.0, 0.0])
        # Rounding over up to 15 elements of 20 integration steps stays below 1e-16 m.
        tol = 1e-16

        thin = line_of(FAMILIES["Quadrupole"](), xt.Uniform(4, mode="thin"))
        p = xt.Particles(_context=CTX, p0c=PC, x=x0, px=px0)
        thin.build_tracker(_context=CTX)
        thin.track(p)
        # Native drifts are expanded: x' = px, so x = x0 + L px.
        native = np.array([x0 + LENGTH * px0, px0, 0.0, 0.0])
        observed = np.array([p.x[0], p.px[0], p.y[0], p.py[0]])
        ok = bool(np.allclose(observed, native, atol=tol, rtol=0))
        assert ok, observed
        print(f"native Uniform(4) thin (x,px,y,py): observed={observed.tolist()} expected={native.tolist()} "
              f"derivation=expanded drift over {LENGTH} m tol={tol:g} pass")

        for label, line, defect in (
            ("MAD-NG unsliced", line_of(FAMILIES["Quadrupole"]()), False),
            ("MAD-NG Uniform(4) thick", line_of(FAMILIES["Quadrupole"](), xt.Uniform(4, mode="thick")), False),
            ("MAD-NG drifts and zero-length Multipole kicks", lowered_line(), False),
            ("MAD-NG Uniform(4) thin", line_of(FAMILIES["Quadrupole"](), xt.Uniform(4, mode="thin")), True),
        ):
            sys.stdout.flush()
            end, seq_length = madng_track(line, INITIAL)
            end = np.array(end)
            ok = bool(np.allclose(end, exact, atol=tol, rtol=0))
            if defect:
                failures.append(not ok)
            else:
                assert ok, (label, end)
            print(f"{label} (x,px,y,py): observed={end.tolist()} expected={exact.tolist()} "
                  f"MAD-NG sequence length={seq_length!r} derivation=exact drift over {LENGTH} m "
                  f"tol={tol:g} {'pass' if ok else 'FAIL'}")
            if defect:
                match = bool(np.allclose(end, doubled, atol=tol, rtol=0))
                assert match, (label, end, doubled)
                print(f"{label} vs doubled length (x,px,y,py): observed={end.tolist()} "
                      f"expected={doubled.tolist()} derivation=exact drift over {2 * LENGTH} m "
                      f"tol={tol:g} pass; excess x={float(end[0] - exact[0])!r} m")


    def main():
        identity()
        print(f"PACKAGE cpymad {version('cpymad')}; pymadng {version('pymadng')}")
        geometry_checks()
        madx_checks()
        tracking_checks()
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
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03; pymadng 0.9.6, MAD-NG 1.1.14 |
| Confirmed<br><small>exported lengths of eight converter branches, MAD-X refusal and MAD-NG tracking</small> | `claude-opus-5-5` | high | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03; pymadng 0.9.6, MAD-NG 1.1.14 |

