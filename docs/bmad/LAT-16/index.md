---
title: "LAT-16: The beam-writer header documents the wrong default format"
hide:
  - toc
---

# LAT-16 — The beam-writer header documents the wrong default format

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [write_beam_file header, declaration and dispatch, bmad/multiparticle/beam_file_io.f90:20–59](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_file_io.f90#L20-L59) |

Calling write_beam_file without a format argument writes an extensionless output as ASCII, although the API header labels HDF5 as the default. The header also calls the format argument logical, while the implementation accepts an integer enumeration. The extensionless output is byte-identical to explicitly selected ASCII, and both default.h5 and default.hdf5 produce HDF5 files. The header should describe filename-dependent selection and the integer argument type.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _s3_12_particle_contracts.py](../helpers/_s3_12_particle_contracts.py) · [View source](../helpers/index.md#_s3_12_particle_contracts)

??? example "Reproducer: LAT-16.py"

    [Download LAT-16.py](LAT-16.py)

    ```python
    """LAT-16: The beam-writer header documents the wrong default format

    Calling write_beam_file without a format argument writes an extensionless output as ASCII, although the API header labels HDF5 as the default. The header also calls the format argument logical, while the implementation accepts an integer enumeration. The extensionless output is byte-identical to explicitly selected ASCII, and both default.h5 and default.hdf5 produce HDF5 files. The header should describe filename-dependent selection and the integer argument type.

    The script exercises omitted and explicit format arguments on the same deterministic beam and checks the file signatures. BMAD_SOURCE supplies the API header whose type and default claims are compared with those results.
    """

    from _bmad import verdict
    from _s3_12_particle_contracts import beam_files, document_claim

    files = beam_files()
    assert files["default_beam"] == files["beam.txt"]
    for name in ("default.h5", "default.hdf5"):
        assert files[name].startswith(b"\x89HDF\r\n\x1a\n")
    print(
        "omitted format: extensionless=ASCII (byte-identical to explicit ASCII); .h5=HDF5; .hdf5=HDF5"
    )
    verdict(
        document_claim(
            "bmad/multiparticle/beam_file_io.f90",
            "file_format -- logical, optional: ascii$, or hdf5$ (default).",
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

