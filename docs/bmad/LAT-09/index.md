---
title: "LAT-09: ASCII output omits persistent particle identifiers"
hide:
  - toc
---

# LAT-09 — ASCII output omits persistent particle identifiers

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`write_ascii_beam_file` column list, bmad/multiparticle/beam_file_io.f90:155–157](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_file_io.f90#L155-L157)<br>[identifier header formatting, bmad/multiparticle/beam_file_io.f90:224–235](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_file_io.f90#L224-L235)<br>[identifier value formatting, bmad/multiparticle/beam_file_io.f90:257–277](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_file_io.f90#L257-L277)<br>[ASCII reader identifier/index handling, bmad/multiparticle/beam_file_io.f90:1001–1016](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_file_io.f90#L1001-L1016)<br>[column and row-index contract, bmad/doc/beam-init.tex:363–382](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/beam-init.tex#L363-L382) |

Writing a beam as ASCII omits the persistent particle identifier ix_user. Identifiers 101, 102, 103 read back as −1, −1, −1 while the particles’ horizontal positions survive. The emitted row index is ignored by the reader and cannot preserve identity through filtering or reordering. The same identifiers survive an HDF5 round trip.

Run `python LAT-09.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_11_particle_io.py](../helpers/_s3_11_particle_io.py) in the same directory.

??? example "Reproducer: LAT-09.py"

    [Download LAT-09.py](LAT-09.py)

    ```python
    """LAT-09: ASCII output omits persistent particle identifiers

    Writing a beam as ASCII omits the persistent particle identifier ix_user. Identifiers 101, 102, 103 read back as −1, −1, −1 while the particles’ horizontal positions survive. The emitted row index is ignored by the reader and cannot preserve identity through filtering or reordering. The same identifiers survive an HDF5 round trip.

    The script writes three particles with identifiers 101–103 as ASCII and HDF5 and reads each file in a fresh native process. It checks horizontal positions as a control and compares identifiers directly with their input values.
    """

    from _bmad import compare, verdict
    from _s3_11_particle_io import execute

    setup = """
    call reallocate_beam(beam, 1, 3)
    do i = 1, 3
      p => beam%bunch(1)%particle(i)
      call init_coord(p, [0.001_rp*i, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp], ele, upstream_end$, electron$)
      p%ix_user = 100 + i
      p%charge = 1e-12_rp
    enddo
    """
    defect = False
    for kind in ("HDF5", "ASCII"):
        writer = (
            """
    call hdf5_write_beam('beam.h5', beam%bunch, .false., err)
    if (err) error stop 'HDF5 write failed'
    """
            if kind == "HDF5"
            else "call write_ascii_beam_file('beam.txt', beam, .true.)"
        )
        filename = "beam.h5" if kind == "HDF5" else "beam.txt"
        result = execute(
            setup
            + writer
            + f"""
    call read_beam_file('{filename}', back, bi, err)
    if (err) error stop 'beam read failed'
    if (size(back%bunch) /= 1) error stop 'wrong bunch count'
    if (size(back%bunch(1)%particle) /= 3) error stop 'wrong particle count'
    call emit('ids', real(back%bunch(1)%particle%ix_user,rp))
    call emit('x', back%bunch(1)%particle%vec(1))
    """
        )
        if compare(
            f"{kind} coordinate control (m)", result["x"], [0.001, 0.002, 0.003], atol=2e-18, rtol=2e-15
        ):
            raise RuntimeError("Coordinate control failed")
        mismatch = compare(
            f"{kind} persistent identifiers; input identity", result["ids"], [101, 102, 103], atol=0
        )
        if kind == "HDF5" and mismatch:
            raise RuntimeError("HDF5 identifier control failed")
        defect |= mismatch
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

