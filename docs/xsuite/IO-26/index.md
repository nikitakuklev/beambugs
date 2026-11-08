---
title: "IO-26: Exported Cavity slices each repeat the parent's full voltage"
hide:
  - toc
---

# IO-26 — Exported Cavity slices each repeat the parent's full voltage

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Source | [`cavity_to_mad_str`, mad_writer.py:183–194](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L183-L194)<br>[slice dispatch, lines 659–680](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L659-L680)<br>[Cavity contract, cavity.py:26–29](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/cavity.py#L26-L29)<br>[longitudinal field integral, with the kick pτ ← pτ + χV_n sin(φ − kτ), xtrack.tex:2936–2954](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L2936-L2954)<br>[RF body, track_rf.h:354–367](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_rf.h#L354-L367)<br>[`ThinSliceCavity.get_equivalent_element`, slice_elements_thin.py:139–151](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/slice_elements_thin.py#L139-L151)<br>[RFCAVITY VOLT as the peak voltage with ΔE = VOLT·sin(2π(LAG − HARMON·f₀t)), elements.tex:892–903](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/doc/latexuguide/elements.tex#L892-L903) |

Exporting native Cavity slices to MAD-X or MAD-NG writes the parent's full RF voltage on every piece instead of multiplying it by the slice weight. Four equal slices of a 1 MV cavity therefore carry 4 MV in total rather than 1 MV. Thick slices preserve the cavity's length, so their export loads successfully and gives a crest energy gain of 4.000 MeV instead of 1.000 MeV for a unit-charge particle. The integrated RF-kick law, native tracking and independently weighted external elements establish the intended gain. Unsliced and weight-one exports preserve the voltage.

Run `python IO-26.py` with [_madx_io.py](../helpers/_madx_io.py), [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-26.py"

    [Download IO-26.py](IO-26.py)

    ```python
    """IO-26: Exported Cavity slices each repeat the parent's full voltage

    Exporting native Cavity slices to MAD-X or MAD-NG writes the parent's full RF voltage on every piece instead of multiplying it by the slice weight. Four equal slices of a 1 MV cavity therefore carry 4 MV in total rather than 1 MV. Thick slices preserve the cavity's length, so their export loads successfully and gives a crest energy gain of 4.000 MeV instead of 1.000 MeV for a unit-charge particle. The integrated RF-kick law, native tracking and independently weighted external elements establish the intended gain. Unsliced and weight-one exports preserve the voltage.

    Slice a 1.2 m, 1 MV Cavity in Uniform and Teapot thin and thick mode, export every slice in both writer dialects and compare each written voltage with the slice weight times the parent voltage. MAD-X then parses the whole thick-sliced export and its voltages are read back, and MAD-NG loads whole thick-sliced lines through Line.to_madng and tracks a reference proton at crest, against the integrated RF-kick law Delta E = q V sin(phi); each failing case is also compared with the sum of repeated full voltages. Controls are unsliced and weight-one exports, native tracking of the slices and a natively built line of four independently weighted 0.3 m, 0.25 MV cavities exported to MAD-NG. The script uses cpymad, pymadng and installed serial prebuilt Xtrack kernels with compilation forbidden; scratch files stay in automatically removed directories beside the script.
    """

    import re
    import sys
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from _madx_io import madng_track_6d
    from _prebuilt import CTX, identity, kernels, np, xt

    # Import after the helpers cap numerical threads and forbid compilation.
    # isort: split
    from importlib.metadata import version

    from cpymad.madx import Madx
    from xtrack import mad_writer as writer

    PC = 3e9
    LENGTH = 1.2
    VOLTAGE = 1e6
    # MAD-NG refuses an accelerating voltage at zero frequency. At 1 Hz the RF phase of a
    # particle that stays within nanoseconds of the reference changes by < 1e-8 rad, so
    # the crest gain stays V sin(pi/2) = V to far better than the tolerance below.
    FREQUENCY = 1.0
    PHASE = np.pi / 2
    # Gains are ~1e6 eV; binary64 rounding of p0c * pt is ~1e-9 eV.
    GAIN_TOL = 2e-8
    VOLT_TOL = 1e-12  # MV, on written decimal voltages
    VOLT = re.compile(r"(?<![\w.])volt = ([^,}]+)")
    failures = []


    def cavity(length=LENGTH, voltage=VOLTAGE):
        return xt.Cavity(length=length, voltage=voltage, frequency=FREQUENCY, phase=PHASE)


    def line_of(elements, scheme=None):
        line = xt.Line(elements=elements, element_names=list(elements))
        line.particle_ref = xt.Particles(p0c=PC)
        if scheme is not None:
            line.slice_thick_elements([xt.Strategy(slicing=scheme)], with_progress=False)
        return line


    def cavity_names(line):
        return [nn for nn in line.element_names
                if isinstance(line[nn], (xt.Cavity, xt.ThinSliceCavity, xt.ThickSliceCavity))]


    def weight(line, name):
        el = line[name]
        return float(el.weight) if hasattr(el, "_parent") else 1.0


    def text_checks():
        schemes = {
            "unsliced": None,
            "Uniform(1) thick": xt.Uniform(1, mode="thick"),
            "Uniform(4) thin": xt.Uniform(4, mode="thin"),
            "Uniform(4) thick": xt.Uniform(4, mode="thick"),
            "Teapot(4) thin": xt.Teapot(4, mode="thin"),
            "Teapot(4) thick": xt.Teapot(4, mode="thick"),
        }
        for label, scheme in schemes.items():
            line = line_of({"m": cavity()}, scheme)
            names = cavity_names(line)
            # A slice of weight w carries the integrated voltage w V (native kernels and
            # ThinSliceCavity.get_equivalent_element both apply voltage * weight).
            expected = [weight(line, nn) * VOLTAGE * 1e-6 for nn in names]
            for dialect in (writer.MadType.MADX, writer.MadType.MADNG):
                written = []
                for nn in names:
                    text = writer.element_to_mad_str(nn, nn, line, mad_type=dialect, substituted_vars=[])
                    (value,) = VOLT.findall(text)
                    written.append(float(value))
                ok = bool(np.allclose(written, expected, atol=VOLT_TOL, rtol=0))
                if all(w == 1.0 for w in (weight(line, nn) for nn in names)):
                    assert ok, (label, dialect, written, expected)
                else:
                    failures.append(not ok)
                print(f"TEXT {label} {dialect.name}: written volt={written} MV sum={sum(written):.10g} "
                      f"expected={expected} MV sum={sum(expected):.10g} derivation=slice weight x parent "
                      f"voltage tol={VOLT_TOL:g} {'pass' if ok else 'FAIL'}")


    def madx_readback(line, label, defect):
        source = line.to_madx_sequence("testseq")
        with (
            TemporaryDirectory(prefix="io26-madx-", dir=Path(__file__).resolve().parent) as scratch,
            Madx(stdout=False, cwd=scratch) as mad,
        ):
            mad.input(source)
            mad.beam(particle="proton", pc=PC / 1e9)
            mad.use(sequence="testseq")
            seq = mad.sequence["testseq"]
            volts = [float(e.volt) for e in seq.elements if e.base_type.name == "rfcavity"]
            length = float(seq.length)
            print(f"CONSUMER cpymad={version('cpymad')} MAD-X={mad.version}")
        ok = abs(sum(volts) - VOLTAGE * 1e-6) < VOLT_TOL and abs(length - LENGTH) < 1e-12
        if defect:
            failures.append(not ok)
        else:
            assert ok, (label, volts, length)
        print(f"MAD-X {label}: parsed volt={volts} MV total={sum(volts):.10g} MV length={length!r} m "
              f"expected total={VOLTAGE * 1e-6:g} MV length={LENGTH} m derivation=integrated parent "
              f"voltage and geometry {'pass' if ok else 'FAIL'}")


    def gain_compare(label, observed, defect, repeated=None):
        # Integrated RF-kick law at crest for charge q = +e: Delta E = q V sin(pi/2) = V.
        expected = float(VOLTAGE * np.sin(PHASE))
        ok = abs(observed - expected) < GAIN_TOL
        if defect:
            failures.append(not ok)
        else:
            assert ok, (label, observed, expected)
        print(f"{label} crest gain: observed={observed!r} eV expected={expected!r} eV "
              f"derivation=Delta E = q V sin(phi), q = +e tol={GAIN_TOL:g} {'pass' if ok else 'FAIL'}")
        if defect:
            assert repeated is not None
            match = abs(observed - repeated) < GAIN_TOL
            assert match, (label, observed, repeated)
            print(f"{label} vs every piece at full voltage: observed={observed!r} eV "
                  f"expected={repeated!r} eV tol={GAIN_TOL:g} pass")


    def native_gain(line):
        p = xt.Particles(_context=CTX, p0c=PC)
        line.build_tracker(_context=CTX)
        line.track(p)
        return float(p.ptau[0]) * PC


    def ng_gain(line):
        sys.stdout.flush()
        end, _ = madng_track_6d(line, [0.0] * 6)
        return end[5] * PC


    def tracking_checks():
        for mode in ("thin", "thick"):
            line = line_of({"m": cavity()}, xt.Uniform(4, mode=mode))
            gain_compare(f"native Uniform(4) {mode}", native_gain(line), False)

        gain_compare("MAD-NG unsliced", ng_gain(line_of({"m": cavity()})), False)
        for n in (1, 2, 4):
            line = line_of({"m": cavity()}, xt.Uniform(n, mode="thick"))
            gain_compare(f"MAD-NG Uniform({n}) thick", ng_gain(line), n > 1, n * VOLTAGE)
        line = line_of({"m": cavity()}, xt.Teapot(4, mode="thick"))
        gain_compare("MAD-NG Teapot(4) thick", ng_gain(line), True, 4 * VOLTAGE)

        # Independently weighted pieces: four ordinary 0.3 m cavities of 0.25 MV each.
        pieces = {f"c{i}": cavity(LENGTH / 4, VOLTAGE / 4) for i in range(4)}
        gain_compare("MAD-NG four 0.3 m, 0.25 MV cavities", ng_gain(line_of(pieces)), False)

        madx_readback(line_of({"m": cavity()}), "unsliced", False)
        madx_readback(line_of({"m": cavity()}, xt.Uniform(4, mode="thick")), "Uniform(4) thick", True)


    def main():
        identity()
        print(f"PACKAGE cpymad {version('cpymad')}; pymadng {version('pymadng')}")
        text_checks()
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
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, exported MAD-X/MAD-NG text |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-07</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03; pymadng 0.9.6, MAD-NG 1.1.14 |
| Confirmed<br><small class="bb-scope">written and MAD-X-parsed slice voltages, MAD-NG crest gains</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03; pymadng 0.9.6, MAD-NG 1.1.14 |

