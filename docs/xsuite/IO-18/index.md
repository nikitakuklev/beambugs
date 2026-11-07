---
title: "IO-18: MAD-X LINE export calls converters with obsolete arguments"
hide:
  - toc
---

# IO-18 — MAD-X LINE export calls converters with obsolete arguments

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack main (`f09e0fb`, head build) |
| Source | [MAD-X LINE branch, mad_writer.py:702–709](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L702-L709)<br>[quadrupole signature](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L547-L565)<br>[Bend signature](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L435-L449)<br>[_get_expr](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/_common.py#L837-L857)<br>[Marker retains the old signature](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L221-L242)<br>[registration and aperture tables](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L613-L640)<br>[generic dispatcher](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L653-L691)<br>[Line.to_madx_sequence, line.py:877–894](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L877-L894)<br>[API generation](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/conf.py#L73-L79)<br>[API inclusion](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/apireference.rst#L833-L833) |

Exporting a line with `Line.to_madx_sequence(..., mode="line")` raises exceptions for ordinary elements instead of returning the documented MAD-X LINE. The branch passes element names and the Line object to converters that now expect an element reference, causing duplicate-argument TypeErrors for quadrupoles, drifts and other types; bends raise a ValueError. Thirteen of fourteen registered element types that serialize in sequence mode fail in line mode. Marker-only lines succeed. A separately handwritten MAD-X LINE preserves a 2.625 m test lattice and its quadrupole parameters exactly, as does default sequence export.

Run `python IO-18.py` with [_madx_io.py](../helpers/_madx_io.py), [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-18.py"

    [Download IO-18.py](IO-18.py)

    ```python
    """IO-18: MAD-X LINE export calls converters with obsolete arguments

    Exporting a line with `Line.to_madx_sequence(..., mode="line")` raises exceptions for ordinary elements instead of returning the documented MAD-X LINE. The branch passes element names and the Line object to converters that now expect an element reference, causing duplicate-argument TypeErrors for quadrupoles, drifts and other types; bends raise a ValueError. Thirteen of fourteen registered element types that serialize in sequence mode fail in line mode. Marker-only lines succeed. A separately handwritten MAD-X LINE preserves a 2.625 m test lattice and its quadrupole parameters exactly, as does default sequence export.

    Export every registered element type in both modes and independently inspect converter argument binding. Validate a handwritten 2.625 m MAD-X LINE against sequence export, and check marker-only, replica, aperture and sliced controls.
    """

    import inspect

    from _madx_io import consume
    from _prebuilt import identity, xt
    from xtrack import mad_writer as writer

    drift_parent = xt.Drift(length=1.5)
    fixtures = {
        xt.Cavity: xt.Cavity(voltage=1e6, frequency=400e6),
        xt.Marker: xt.Marker(),
        xt.Drift: xt.Drift(length=0.75),
        xt.Device: xt.Device(length=0.75),
        xt.Multipole: xt.Multipole(knl=[0, 0.125]),
        xt.DipoleEdge: xt.DipoleEdge(k=0.1, e1=0.02),
        xt.Bend: xt.Bend(length=1.375, angle=0.04),
        xt.RBend: xt.RBend(length_straight=1.375, angle=0.04),
        xt.Sextupole: xt.Sextupole(length=1.375, k2=0.2),
        xt.Octupole: xt.Octupole(length=1.375, k3=0.2),
        xt.Quadrupole: xt.Quadrupole(length=1.375, k1=0.3125),
        xt.Solenoid: xt.Solenoid(length=1.375, ks=0.2),
        xt.UniformSolenoid: xt.UniformSolenoid(length=1.375, ks=0.2),
        xt.SRotation: xt.SRotation(angle=0.5),
        xt.RFMultipole: xt.RFMultipole(knl=[0, 0.125]),
        xt.CrabCavity: xt.CrabCavity(crab_voltage=1e6, frequency=400e6),
        xt.DriftSlice: xt.DriftSlice(_parent=drift_parent, _buffer=drift_parent._buffer, weight=0.5),
        xt.ACDipole: xt.ACDipole(volt=100, freq=0.2, plane="h"),
    }


    def main():
        identity()
        print("EXECUTION Python serialization and MAD-X parsing; no tracker or tracking kernel")
        assert set(fixtures) == set(writer.xsuite_to_mad_converters)
        supported = failures = 0
        unsupported = {xt.DipoleEdge, xt.SRotation, xt.RFMultipole, xt.ACDipole}
        for cls, element in fixtures.items():
            if cls is xt.DriftSlice:
                assert isinstance(element, xt.DriftSlice)
                # Slice parent metadata is installed dynamically by Xobjects.
                element.parent_name = "parent"  # ty: ignore[unresolved-attribute]
                line = xt.Line(elements={"parent": drift_parent, "e": element}, element_names=["e"])
            else:
                line = xt.Line(elements={"e": element})
            try:
                sequence = line.to_madx_sequence("testseq")
            except NotImplementedError:
                assert cls in unsupported
                continue
            assert cls not in unsupported
            assert line.to_madx_sequence("testseq", mode="sequence") == sequence
            supported += 1
            converter = writer.xsuite_to_mad_converters[cls]
            try:
                inspect.signature(converter).bind("e", line, mad_type=writer.MadType.MADX)
                binding = "accepted"
            except TypeError as exc:
                binding = str(exc)
            try:
                line.to_madx_sequence("testseq", mode="line")
                observed = "accepted"
            except (TypeError, ValueError) as exc:
                failures += 1
                observed = f"{type(exc).__name__}: {exc}"
            print(
                f"REGISTERED {cls.__name__}: observed={observed} expected=MAD-X LINE "
                f"binding={binding} derivation=public mode contract and independent argument binding"
            )
        print(f"SUPPORTED_REGISTERED_TYPES {supported} LINE_FAILURES {failures}")
        # Sequence mode resolves these supported classes through the generic dispatcher.
        extras = {
            "aperture": xt.Line(elements={"e": xt.LimitRect()}),
            "replica": xt.Line(
                elements={"p": xt.Marker(), "e": xt.Replica(parent_name="p")}, element_names=["e"]
            ),
        }
        for mode in ("thin", "thick"):
            line = xt.Line(elements={"q": xt.Quadrupole(length=1.375, k1=0.3125)})
            line.slice_thick_elements([xt.Strategy(xt.Teapot(2, mode=mode))])
            extras["sliced_" + mode] = line
        for label, line in extras.items():
            line.to_madx_sequence("testseq")
            try:
                line.to_madx_sequence("testseq", mode="line")
                observed = "accepted"
            except KeyError as exc:
                failures += 1
                observed = f"KeyError: {exc}"
            print(
                f"DISPATCH {label}: observed={observed} expected=MAD-X LINE derivation=generic dispatcher"
            )
        mixed = xt.Line(
            elements={
                "a": xt.Marker(),
                "d1": xt.Drift(length=0.75),
                "q": xt.Quadrupole(length=1.375, k1=0.3125),
                "d2": xt.Drift(length=0.5),
                "b": xt.Marker(),
            }
        )
        handwritten = (
            "a:marker; d1:drift,l=0.75; q:quadrupole,l=1.375,k1=0.3125; "
            "d2:drift,l=0.5; b:marker; testseq:line=(a,d1,q,d2,b);"
        )
        for label, source in (
            ("handwritten", handwritten),
            ("sequence", mixed.to_madx_sequence("testseq")),
        ):
            result = consume(source)
            assert result["status"] == "accepted"
            assert result["length"] == 2.625
            (q,) = [e for e in result["elements"] if e[0] == "q"]
            assert q[2:] == (1.375, 0.3125)
            assert result["expanded_names"] == [
                "testseq$start",
                "a",
                "d1",
                "q",
                "d2",
                "b",
                "testseq$end",
            ]
            print(
                f"TARGET_CONTROL {label}: observed_length={result['length']} q_L_k1={q[2:]} "
                "expected=(2.625,1.375,0.3125) derivation=independently specified binary-exact lattice"
            )
        markers = xt.Line(elements={"a": xt.Marker(), "b": xt.Marker()})
        result = consume(markers.to_madx_sequence("testseq", mode="line"))
        assert result["status"] == "accepted" and result["length"] == 0
        print("MARKER_CONTROL: observed=accepted length=0 expected=accepted length=0")
        print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, Python MAD-X export (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, Python MAD-X export (compilation forbidden), CPU, double, serial |

