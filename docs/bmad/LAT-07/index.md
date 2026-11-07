---
title: "LAT-07: Modern particle-file readers ignore the requested bunch count"
hide:
  - toc
---

# LAT-07 — Modern particle-file readers ignore the requested bunch count

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`read_beam_file` HDF5 branch, bmad/multiparticle/beam_file_io.f90:418–444](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_file_io.f90#L418-L444)<br>[`read_beam_ascii` bunch loop, bmad/multiparticle/beam_file_io.f90:876–879](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_file_io.f90#L876-L879)<br>[`n_bunch` contract, bmad/doc/beam-init.tex:211–216](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/beam-init.tex#L211-L216)<br>[reader API contract, bmad/multiparticle/beam_file_io.f90:358–365](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_file_io.f90#L358-L365) |

Reading a modern ASCII or HDF5 particle file ignores a nonzero requested bunch count. Requesting one bunch from a two-bunch file returns both with no error. The documented contract requires exactly the requested number, with an error when the file contains too few. A zero request correctly returns all bunches.

Run `python LAT-07.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_11_particle_io.py](../helpers/_s3_11_particle_io.py) in the same directory.

??? example "Reproducer: LAT-07.py"

    [Download LAT-07.py](LAT-07.py)

    ```python
    """LAT-07: Modern particle-file readers ignore the requested bunch count

    Reading a modern ASCII or HDF5 particle file ignores a nonzero requested bunch count. Requesting one bunch from a two-bunch file returns both with no error. The documented contract requires exactly the requested number, with an error when the file contains too few. A zero request correctly returns all bunches.

    The script reads a two-bunch ASCII file and a two-bunch HDF5 file with requests for zero and one bunch. Zero selects all bunches as a control; the nonzero request is checked against the documented exact count.
    """

    from _bmad import compare, verdict
    from _s3_11_particle_io import HEADER_FILE, execute

    h5_setup = """
    call reallocate_beam(beam, 2, 2)
    do i = 1, 2
      beam%bunch(i)%particle%state = alive$
      beam%bunch(i)%particle%species = electron$
      beam%bunch(i)%particle%p0c = 1e9_rp
      beam%bunch(i)%particle%beta = 1
      beam%bunch(i)%particle%charge = 1
    enddo
    call hdf5_write_beam('beam.h5', beam%bunch, .false., err)
    if (err) error stop 'HDF5 write failed'
    """
    defect = False
    for kind in ("ASCII", "HDF5"):
        for request in (0, 1):
            setup = h5_setup if kind == "HDF5" else ""
            filename = "beam.h5" if kind == "HDF5" else "beam.txt"
            result = execute(
                setup
                + f"""
    bi%n_bunch = {request}
    call read_beam_file('{filename}', back, bi, err)
    if (err) error stop 'beam read failed'
    call emit('count', [real(size(back%bunch),rp)])
    """,
                files={
                    "beam.txt": HEADER_FILE.replace(
                        "# species = proton", "# next bunch\n# species = proton"
                    )
                },
            )
            expected = 2 if request == 0 else request
            mismatch = compare(
                f"{kind} n_bunch={request}; documented selection", result["count"][0], expected, atol=0
            )
            if request == 0 and mismatch:
                raise RuntimeError("All-bunches control failed")
            defect |= mismatch
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

