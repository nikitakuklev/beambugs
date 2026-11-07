---
title: "IO-25: MAD-X rejects sliced-magnet exports containing kill_body"
hide:
  - toc
---

# IO-25 — MAD-X rejects sliced-magnet exports containing kill_body

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack main (`f09e0fb`, head build) |
| Source | [`element_to_mad_str`, mad_writer.py:670–680](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/mad_writer.py#L670-L680)<br>[edge-slice base, slice_elements_edge.py:34–39](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/slice_elements_edge.py#L34-L39)<br>[`Line.to_madx_sequence`, line.py:877–894](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L877-L894)<br>[quadrupole attribute dictionary, mad_dict.c:1740–1793](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/3865cb633e66fb4cf53d2f37b00c919dbd6ab495/src/mad_dict.c#L1740-L1793)<br>[thick elements, madl_element.mad:174–177](https://github.com/MethodicalAcceleratorDesign/MAD-NG/blob/6e9f93fa222d4f3a3eb4b2685b2c45f2d07feb62/src/madl_element.mad#L174-L177) |

Exporting native sliced magnets as a MAD-X sequence writes `kill_body = True` on their generated entry and exit maps. MAD-X has no `kill_body` attribute and terminates while parsing the export, even when the native magnet's edge effects are disabled. The failure affects sliced quadrupoles, sextupoles, octupoles, bends, rectangular bends and uniform solenoids. An independently handwritten quadrupole is accepted until that single keyword is added; unsliced exports and selections containing only thick body slices parse successfully. The keyword belongs to MAD-NG, whose consumer accepts these maps.

Run `python IO-25.py` with [_madx_io.py](../helpers/_madx_io.py), [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-25.py"

    [Download IO-25.py](IO-25.py)

    ```python
    """IO-25: MAD-X rejects sliced-magnet exports containing kill_body

    Exporting native sliced magnets as a MAD-X sequence writes `kill_body = True` on their generated entry and exit maps. MAD-X has no `kill_body` attribute and terminates while parsing the export, even when the native magnet's edge effects are disabled. The failure affects sliced quadrupoles, sextupoles, octupoles, bends, rectangular bends and uniform solenoids. An independently handwritten quadrupole is accepted until that single keyword is added; unsliced exports and selections containing only thick body slices parse successfully. The keyword belongs to MAD-NG, whose consumer accepts these maps.

    Check MAD-X's keyword contract with a handwritten quadrupole, then export native slices of six magnet families in every applicable slicing mode. Compare parsing with unsliced, body-only and lowered thin-quadrupole controls, including a sliced bend with disabled faces. Load the whole thick-sliced quadrupole through the public MAD-NG interface as a dialect control, using the shared helpers with compilation forbidden.
    """

    from _madx_io import consume, madng_track
    from _prebuilt import identity, np, xt

    LENGTH = 1.2
    FAMILIES = ("Quadrupole", "Sextupole", "Octupole", "Bend", "RBend", "UniformSolenoid")


    def sliced(element, mode="thick"):
        line = xt.Line(elements={"m": element})
        line.particle_ref = xt.Particles(p0c=3e9)
        line.slice_thick_elements([xt.Strategy(xt.Uniform(4, mode=mode))], with_progress=False)
        return line


    def accepted_control(label, source):
        result = consume(source)
        assert result["status"] == "accepted", result
        # Serialization/sequence expansion must preserve the parent's geometry.
        np.testing.assert_allclose(result["length"], LENGTH, rtol=0, atol=5e-15)
        print(
            f"CONTROL {label}: observed={result['status']} length={result['length']!r} "
            f"expected=accepted length={LENGTH} derivation=specified parent geometry"
        )


    def main():
        identity()
        print("EXECUTION Python serialization, MAD-X parsing and MAD-NG tracking; no native tracker")
        base = "testseq:sequence,l=1.2; q:quadrupole,l=.6,k1=.3,at=.6%s; endsequence;"
        accepted_control("handwritten quadrupole", base % "")
        # The sole change is an attribute absent from MAD-X's quadrupole grammar.
        # A numeric value isolates attribute recognition from Boolean spelling.
        for token in (",kill_body=True", ",kill_body=1"):
            result = consume(base % token)
            assert result["status"] == "RuntimeError", result
            print(
                f"GRAMMAR {token}: observed={result['status']} error={result['error']!r} "
                "expected=keyword rejected derivation=single-attribute mutation of accepted input"
            )

        failures = total = 0

        def check(label, line):
            nonlocal failures, total
            source = line.to_madx_sequence("testseq")
            assert source == line.to_madx_sequence("testseq", mode="sequence")
            result = consume(source)
            if result["status"] != "accepted":
                assert result["status"] == "RuntimeError", result
                assert source.count("kill_body = True") == 2
                failures += 1
            total += 1
            print(
                f"EXPORT {label}: observed={result['status']} "
                f"kill_body_tokens={source.count('kill_body = True')} expected=accepted "
                "derivation=public corresponding-sequence contract"
            )

        for mode in ("thin", "thick"):
            for family in FAMILIES:
                # The public slicer supports UniformSolenoid only in thick mode.
                if family == "UniformSolenoid" and mode == "thin":
                    continue
                length = {"length_straight": LENGTH} if family == "RBend" else {"length": LENGTH}
                check(f"{family} Uniform(4,{mode})", sliced(getattr(xt, family)(**length), mode))
        check(
            "Bend disabled faces",
            sliced(xt.Bend(length=LENGTH, angle=.12, edge_entry_active=False, edge_exit_active=False)),
        )
        quadrupole = xt.Quadrupole(length=LENGTH, k1=.3)
        accepted_control("unsliced quadrupole", xt.Line(elements={"m": quadrupole}).to_madx_sequence("testseq"))
        thick = sliced(quadrupole.copy())
        bodies = thick.copy()
        bodies.element_names = [name for name in bodies.element_names if isinstance(bodies[name], xt.ThickSliceQuadrupole)]
        assert len(bodies.element_names) == 4
        accepted_control("four thick bodies only", bodies.to_madx_sequence("testseq"))
        lowered = sliced(quadrupole.copy(), "thin")
        lowered._replace_with_equivalent_elements()
        accepted_control("lowered thin quadrupole", lowered.to_madx_sequence("testseq"))

        # MAD-NG defines kill_body and accepts the same generated maps. A zero ray
        # checks successful consumption without relying on a fringe/field oracle.
        end, length = madng_track(thick, [0, 0, 0, 0])
        np.testing.assert_array_equal(end, [0, 0, 0, 0])
        np.testing.assert_allclose(length, LENGTH, rtol=0, atol=3e-10)
        print(
            f"CONTROL MAD-NG whole sliced quadrupole: observed=accepted length={length!r} "
            "zero_ray=[0,0,0,0] expected=accepted zero ray derivation=MAD-NG keyword contract"
        )
        print(f"COMPARISONS expected-correct exports fail={failures} of {total}")
        print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, exported MAD-X/MAD-NG text; cpymad 1.19.0, MAD-X 5.09.03 |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial; cpymad 1.19.0, MAD-X 5.09.03; pymadng 0.9.6, MAD-NG 1.1.14 |

