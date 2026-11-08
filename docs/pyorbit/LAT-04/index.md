---
title: "LAT-04: RF_AxisFieldsStore returns the first table read for a file name, from any directory and after the file changes"
hide:
  - toc
---

# LAT-04 — `RF_AxisFieldsStore` returns the first table read for a file name, from any directory and after the file changes

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice API |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">contract-unsettled</span> [?](../../methodology.md#tags) |
| Source | [LinacRfGapNodes.py#L341-L407](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L341-L407)<br>[LinacRfGapNodes.py#L493-L505](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L493-L505)<br>[rf_models_modifications_lib.py#L254-L259](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice_modifications/rf_models_modifications_lib.py#L254-L259) |

`RF_AxisFieldsStore.addAxisField(file_name, dir_location)`, which `AxisFieldRF_Gap.readAxisFieldFile` and the axis-field gap replacement function call, caches each field table in a process-wide dictionary keyed by the file name alone. A later request for a file of the same name in another directory, or for the same file after it was rewritten, silently returns the first table read. Two lattices built in one process from field-map sets that share file names, such as two versions of a cavity's maps, both get the first set's fields. A file holding the constant field 2.0 in a second directory reads back as 1.0, and the first file rewritten with 7.0 still reads 1.0.

Run `python LAT-04.py`.

??? example "Reproducer: LAT-04.py"

    [Download LAT-04.py](LAT-04.py)

    ```python
    """LAT-04: `RF_AxisFieldsStore` returns the first table read for a file name, from any directory and after the file changes

    `RF_AxisFieldsStore.addAxisField(file_name, dir_location)`, which `AxisFieldRF_Gap.readAxisFieldFile` and the axis-field gap replacement function call, caches each field table in a process-wide dictionary keyed by the file name alone. A later request for a file of the same name in another directory, or for the same file after it was rewritten, silently returns the first table read. Two lattices built in one process from field-map sets that share file names, such as two versions of a cavity's maps, both get the first set's fields. A file holding the constant field 2.0 in a second directory reads back as 1.0, and the first file rewritten with 7.0 still reads 1.0.

    The script writes two axis-field files of the same name with constant values 1.0 and 2.0 into two fresh temporary directories and reads them in turn through `AxisFieldRF_Gap.readAxisFieldFile`, then rewrites the first file with 7.0 and reads it again. The expected field at z = 0 is the value in the file at the requested path. The first read, which fills the empty store, is the control.
    """

    import os
    import tempfile

    from orbit.py_linac.lattice import AxisFieldRF_Gap, BaseRF_Gap

    NAME = "cavity_field.txt"


    def write_field(directory, value):
        # Constant field on -10 mm .. 10 mm in 2 mm steps; two columns z [m], E [arbitrary].
        with open(os.path.join(directory, NAME), "w") as fl:
            fl.writelines(f"{-0.01 + 0.002 * i:.4f} {value}\n" for i in range(11))


    def read_field(directory):
        gap = AxisFieldRF_Gap(BaseRF_Gap("gap"))
        gap.readAxisFieldFile(directory + os.sep, NAME, 0.002)
        return gap.getAxisFieldFunction().getY(0.0)


    reproduced = False
    with tempfile.TemporaryDirectory() as root:
        first, second = os.path.join(root, "set_a"), os.path.join(root, "set_b")
        os.mkdir(first)
        os.mkdir(second)
        write_field(first, 1.0)
        write_field(second, 2.0)
        cases = [
            ("control, first file (value 1.0)", first, 1.0, True),
            ("same name, second directory (value 2.0)", second, 2.0, False),
        ]
        for label, directory, expected, control in cases:
            got = read_field(directory)
            print(f"{label}: field read {got:.4g}, expected {expected:.4g} (file content)")
            if control and got != expected:
                raise RuntimeError(
                    "control case failed: the first read does not return the file content"
                )
            reproduced |= not control and got != expected
        write_field(first, 7.0)
        got = read_field(first)
        print(f"first file rewritten with 7.0: field read {got:.4g}, expected 7 (file content)")
        reproduced |= got != 7.0
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

