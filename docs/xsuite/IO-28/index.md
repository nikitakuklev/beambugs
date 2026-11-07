---
title: "IO-28: Sliced CrabCavity export raises AttributeError"
hide:
  - toc
---

# IO-28 — Sliced CrabCavity export raises AttributeError

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Source | [`crabcavity_to_mad_str`, mad_writer.py:198–219](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L198-L219)<br>[`cavity_to_mad_str`, lines 183–186](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L183-L186)<br>[slice dispatch, lines 659–680](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L659-L680)<br>[crab_cavity.py:122–133](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/crab_cavity.py#L122-L133)<br>[`ThinSliceCrabCavity`, slice_elements_thin.py:154–173](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/slice_elements_thin.py#L154-L173)<br>[`ThickSliceCrabCavity`, slice_elements_thick.py:173–179](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/slice_elements_thick.py#L173-L179)<br>[RF-multipole model of integrated transverse impulse, xtrack.tex:2958–2961](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L2958-L2961) |

Exporting a natively sliced crab cavity to MAD-X or MAD-NG raises AttributeError instead of producing a sequence. The converter reads RF fields from the slice object even though they belong to its parent. Thin and thick slices both fail, including a single slice and zero voltage. Four equal pieces of a 1.200 m, 1 MV cavity should retain the parent's RF settings and carry integrated deflecting voltages of 0.2500 MV each; native tracking and explicit equivalent elements establish those values. Unsliced crab cavities serialize successfully.

Run `python IO-28.py` with [_madx_io.py](../helpers/_madx_io.py), [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-28.py"

    [Download IO-28.py](IO-28.py)

    ```python
    """IO-28: Sliced CrabCavity export raises AttributeError

    Exporting a natively sliced crab cavity to MAD-X or MAD-NG raises AttributeError instead of producing a sequence. The converter reads RF fields from the slice object even though they belong to its parent. Thin and thick slices both fail, including a single slice and zero voltage. Four equal pieces of a 1.200 m, 1 MV cavity should retain the parent's RF settings and carry integrated deflecting voltages of 0.2500 MV each; native tracking and explicit equivalent elements establish those values. Unsliced crab cavities serialize successfully.

    Slice a 1.2 m, 1 MV CrabCavity in Uniform and Teapot thin and thick mode and export each slice in both writer dialects, then call the public Line.to_madx_sequence and Line.to_madng on a sliced line; a zero-voltage single-slice cavity tests independence from strength and slice count. The expected export carries the parent's frequency and phase with integrated deflecting voltage w V per slice; native tracking at zero frequency and crest phase checks the integrated kick V/(p0 c) and each thin slice's equivalent element its weighted voltage. Controls are the unsliced export, which MAD-X parses, and native tracking of the slices. The script uses cpymad and pymadng (Line.to_madng raises before starting MAD-NG) and installed serial prebuilt Xtrack kernels with compilation forbidden; scratch files stay in automatically removed directories beside the script.
    """

    from pathlib import Path
    from tempfile import TemporaryDirectory

    from _madx_io import consume
    from _prebuilt import CTX, identity, kernels, np, xt

    # Import after the helpers cap numerical threads and forbid compilation.
    # isort: split
    from importlib.metadata import version

    from xtrack import mad_writer as writer

    PC = 3e9
    LENGTH = 1.2
    VOLTAGE = 1e6
    # Zero frequency and crest phase make the integrated kick exactly V sin(pi/2) / (p0 c).
    PHASE = np.pi / 2
    KICK_TOL = 1e-18
    failures = []


    def crab(voltage=VOLTAGE, frequency=0.0):
        return xt.CrabCavity(length=LENGTH, crab_voltage=voltage, frequency=frequency, phase=PHASE)


    def line_of(element, scheme=None):
        line = xt.Line(elements={"m": element})
        line.particle_ref = xt.Particles(p0c=PC)
        if scheme is not None:
            line.slice_thick_elements([xt.Strategy(slicing=scheme)], with_progress=False)
        return line


    def crab_slices(line):
        return [nn for nn in line.element_names
                if isinstance(line[nn], (xt.ThinSliceCrabCavity, xt.ThickSliceCrabCavity))]


    def attempt(func, *args, **kwargs):
        try:
            return func(*args, **kwargs), None
        except AttributeError as exc:
            return None, exc


    def slice_exports():
        cases = {f"{name}({n}) {mode}": (scheme(n, mode=mode), VOLTAGE, 0.0)
                 for name, scheme in (("Uniform", xt.Uniform), ("Teapot", xt.Teapot))
                 for n in (4,) for mode in ("thin", "thick")}
        cases["Uniform(1) thick, zero voltage, 400 MHz"] = (xt.Uniform(1, mode="thick"), 0.0, 400e6)
        for label, (scheme, voltage, frequency) in cases.items():
            line = line_of(crab(voltage, frequency), scheme)
            names = crab_slices(line)
            # Each slice keeps the parent's RF settings and integrated voltage w V.
            expected = [round(float(line[nn].weight) * voltage * 1e-6, 12) for nn in names]
            for dialect in (writer.MadType.MADX, writer.MadType.MADNG):
                errors = [attempt(writer.element_to_mad_str, nn, nn, line, mad_type=dialect,
                                  substituted_vars=[])[1] for nn in names]
                failed = sum(e is not None for e in errors)
                failures.append(failed > 0)
                print(f"EXPORT {label} {dialect.name}: observed={failed} of {len(names)} slices raise "
                      f"{type(errors[0]).__name__}: {errors[0]} expected=crabcavity with freq="
                      f"{frequency * 1e-6:g} MHz and volt={expected} MV derivation=parent RF settings, "
                      f"slice weight x parent voltage {'FAIL' if failed else 'pass'}")


    def public_exports():
        line = line_of(crab(), xt.Uniform(4, mode="thin"))
        _, exc = attempt(line.to_madx_sequence, "testseq")
        failures.append(exc is not None)
        print(f"Line.to_madx_sequence Uniform(4) thin: observed={type(exc).__name__}: {exc} "
              f"expected=MAD-X sequence derivation=documented corresponding-sequence contract "
              f"{'FAIL' if exc else 'pass'}")
        line = line_of(crab(), xt.Uniform(4, mode="thick"))
        with TemporaryDirectory(prefix="io28-madng-", dir=Path(__file__).resolve().parent) as scratch:
            mng, exc = attempt(line.to_madng, temp_fname=str(Path(scratch) / "line"),
                               stdout=str(Path(scratch) / "stdout.txt"))
            if mng is not None:
                mng.close()
        failures.append(exc is not None)
        print(f"Line.to_madng Uniform(4) thick: observed={type(exc).__name__}: {exc} "
              f"expected=MAD-NG instance derivation=documented build-from-line contract "
              f"{'FAIL' if exc else 'pass'}")


    def controls():
        source = line_of(crab()).to_madx_sequence("testseq")
        result = consume(source)
        assert result["status"] == "accepted" and abs(result["length"] - LENGTH) < 1e-12, result
        print(f"CONTROL unsliced export: observed=MAD-X {result['status']} length={result['length']!r} "
              f"expected=accepted length={LENGTH} derivation=parent geometry pass")

        # Integrated transverse impulse of a crab cavity at zero frequency and crest
        # phase for q = +e: Delta px = q V sin(pi/2) / (p0 c).
        expected_kick = VOLTAGE / PC
        for mode in ("thin", "thick"):
            line = line_of(crab(), xt.Uniform(4, mode=mode))
            p = xt.Particles(_context=CTX, p0c=PC)
            line.build_tracker(_context=CTX)
            line.track(p)
            observed = float(p.px[0])
            assert abs(observed - expected_kick) < KICK_TOL, (mode, observed, expected_kick)
            print(f"CONTROL native Uniform(4) {mode}: Delta px={observed!r} expected={expected_kick!r} "
                  f"derivation=V/(p0 c) at crest tol={KICK_TOL:g} pass")

        line = line_of(crab(), xt.Uniform(4, mode="thin"))
        equivalents = [line[nn].get_equivalent_element() for nn in crab_slices(line)]
        volts = [float(e.crab_voltage) for e in equivalents]
        frequencies = [float(e.frequency) for e in equivalents]
        assert volts == [VOLTAGE / 4] * 4 and frequencies == [0.0] * 4, (volts, frequencies)
        print(f"CONTROL thin equivalent elements: crab_voltage={volts} V frequency={frequencies} Hz "
              f"expected={[VOLTAGE / 4] * 4} V, parent frequency derivation=slice weight x parent pass")


    def main():
        identity()
        print(f"PACKAGE cpymad {version('cpymad')}; pymadng {version('pymadng')}")
        slice_exports()
        public_exports()
        controls()
        kernels()
        print(f"COMPARISONS expected-correct exports fail={sum(failures)} of {len(failures)}")
        print("DEFECT REPRODUCED" if any(failures) else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, exported MAD-X/MAD-NG text |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-07 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03; pymadng 0.9.6, MAD-NG 1.1.14 |
| Confirmed<br><small>both public export methods and both slice modes</small> | `claude-opus-5-5` | high | 2026-11-07 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03; pymadng 0.9.6 |

