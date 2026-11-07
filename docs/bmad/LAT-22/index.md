---
title: "LAT-22: openPMD readers reject a numeric or absent openPMDextension"
hide:
  - toc
---

# LAT-22 — openPMD readers reject a numeric or absent openPMDextension

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [hdf5_read_beam header, bmad/hdf5/hdf5_read_beam.f90:70–78](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/hdf5/hdf5_read_beam.f90#L70-L78)<br>[hdf5_read_grid_field header, bmad/hdf5/hdf5_read_grid_field.f90:49–55](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/hdf5/hdf5_read_grid_field.f90#L49-L55)<br>[openPMD 1.1.0 root-header contract](https://github.com/openPMD/openPMD-standard/blob/1.1.0/STANDARD.md#hierarchy-of-the-data-file)<br>[upcoming 2.0 draft](https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/STANDARD.md#hierarchy-of-the-data-file) |

Reading an openPMD beam fails at the header when openPMDextension is an integer or is absent. The 1.1 standard defines a uint32 extension bit mask, while the 2.0 draft permits omission when no extension is used. Bmad requires a string before decoding particle records, so numeric zero is rejected as not a string and an absent attribute is rejected as missing. The same base-standard particle records decode when the extension attribute is a string.

Run `python LAT-22.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: LAT-22.py"

    [Download LAT-22.py](LAT-22.py)

    ```python
    """LAT-22: openPMD readers reject a numeric or absent openPMDextension

    Reading an openPMD beam fails at the header when openPMDextension is an integer or is absent. The 1.1 standard defines a uint32 extension bit mask, while the 2.0 draft permits omission when no extension is used. Bmad requires a string before decoding particle records, so numeric zero is rejected as not a string and an absent attribute is rejected as missing. The same base-standard particle records decode when the extension attribute is a string.

    The script builds a one-electron file from base-standard position and momentum records, supplying the species and reference momentum through an inline lattice. For each version it first decodes a string-header control and checks the particle, then changes only the extension attribute to the standard-defined form. Only the matching extension diagnostic supplies a defect verdict.
    """

    import io

    import h5py
    import numpy as np
    from _bmad import compare, fortran, verdict

    PROGRAM = """
    program read_external
    use bmad
    implicit none
    type(lat_struct) lat
    type(beam_struct) beam
    logical err
    call bmad_parser('lat.bmad', lat)
    call hdf5_read_beam('beam.h5',beam,err,ele=lat%ele(0))
    if (err) then
      print *, 'READ_REJECTED'
    else
      if (size(beam%bunch) /= 1) error stop 'wrong bunch count'
      if (size(beam%bunch(1)%particle) /= 1) error stop 'wrong particle count'
      write(*,'(a,6es26.17e3)') 'PARTICLE ', beam%bunch(1)%particle(1)%vec
      print *, 'READ_ACCEPTED'
    endif
    end program
    """
    LATTICE = """parameter[particle]=electron
    parameter[p0c]=1e6
    d: drift,l=0.1
    l: line=(d)
    use,l
    """


    def fixture(version, extension):
        buffer = io.BytesIO()
        with h5py.File(buffer, "w") as file:
            file.attrs["openPMD"] = np.bytes_(version)
            if extension is not None:
                file.attrs["openPMDextension"] = extension
            file.attrs["basePath"] = np.bytes_("/data/%T/")
            file.attrs["particlesPath"] = np.bytes_("particles/")
            file.attrs["iterationEncoding"] = np.bytes_("groupBased")
            file.attrs["iterationFormat"] = np.bytes_("/data/%T/")
            iteration = file.create_group("data/00001")
            iteration.attrs.update(time=0.0, dt=0.0, timeUnitSI=1.0)
            species = iteration.create_group("particles/electrons")
            # Bmad uses this additional count attribute; it changes no base record semantics.
            species.attrs["numParticles"] = np.uint64(1)
            for name, values, dimension, unit_si in (
                ("position", [0.002, -0.003, 0], [1, 0, 0, 0, 0, 0, 0], 1.0),
                ("positionOffset", [0, 0, 0], [1, 0, 0, 0, 0, 0, 0], 1.0),
                # eV/c units match the reader's native units and isolate the header gate.
                ("momentum", [0, 0, 1.1e6], [1, 1, -1, 0, 0, 0, 0], 1.602176634e-19 / 299792458),
            ):
                record = species.create_group(name)
                record.attrs["unitDimension"] = np.array(dimension, dtype=np.float64)
                record.attrs["timeOffset"] = 0.0
                for axis, value in zip("xyz", values, strict=True):
                    component = record.create_dataset(axis, data=np.array([value], dtype=np.float64))
                    component.attrs["unitSI"] = unit_si
        return buffer.getvalue()


    def read(data):
        log = fortran(PROGRAM, files={"lat.bmad": LATTICE, "beam.h5": data}).log
        if "READ_ACCEPTED" in log:
            if "READ_REJECTED" in log:
                raise RuntimeError(log)
            records = [line.split()[1:] for line in log.splitlines() if line.startswith("PARTICLE ")]
            if len(records) != 1:
                raise RuntimeError(log)
            observed = np.array(records[0], dtype=float)
            np.testing.assert_allclose(observed, [0.002, 0, -0.003, 0, 0, 0.1], atol=3e-16, rtol=3e-14)
            return True, log
        if "READ_REJECTED" not in log:
            raise RuntimeError(log)
        return False, log


    bad = False
    for name, version, extension, signature in (
        ("numeric", "1.1.0", np.uint32(0), "IS NOT A STRING"),
        ("missing", "2.0.0", None, "ATTRIBUTE IS NOT PRESENT"),
    ):
        accepted, log = read(fixture(version, np.bytes_("")))
        if not accepted:
            raise RuntimeError(log)
        print(f"{version} string-header control: one electron decoded with expected coordinates")
        accepted, log = read(fixture(version, extension))
        if not accepted:
            if "openPMDextension" not in log or signature not in log:
                raise RuntimeError(log)
            print(f"{name} extension diagnostic: {signature}")
        bad |= compare(
            f"{name} extension header accepted (standard contract)", int(accepted), 1, atol=0
        )
    verdict(bad)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-21 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |

