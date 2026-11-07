---
title: "LAT-01: Live-only ASCII output corrupts retained particles"
hide:
  - toc
---

# LAT-01 — Live-only ASCII output corrupts retained particles

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [`write_ascii_beam_file`, bmad/multiparticle/beam_file_io.f90:155–214](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_file_io.f90#L155-L214)<br>[routine contract, bmad/multiparticle/beam_file_io.f90:115–125](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_file_io.f90#L115-L125)<br>[particle serialization, bmad/multiparticle/beam_file_io.f90:249–277](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_file_io.f90#L249-L277) |

When live-only ASCII output filters a bunch whose first particle is lost, it can replace the retained particles’ charge, direction and field with values inferred from the discarded particle. A two-particle example retains one particle but reads back charge 1 instead of 2, direction +1 instead of −1, and zero field instead of (0.6, 0.8). Filtering should preserve the supported data of every retained particle. Writing the same bunch without filtering preserves these values.

Run `python LAT-01.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_11_particle_io.py](../helpers/_s3_11_particle_io.py) in the same directory.

??? example "Reproducer: LAT-01.py"

    [Download LAT-01.py](LAT-01.py)

    ```python
    """LAT-01: Live-only ASCII output corrupts retained particles

    When live-only ASCII output filters a bunch whose first particle is lost, it can replace the retained particles’ charge, direction and field with values inferred from the discarded particle. A two-particle example retains one particle but reads back charge 1 instead of 2, direction +1 instead of −1, and zero field instead of (0.6, 0.8). Filtering should preserve the supported data of every retained particle. Writing the same bunch without filtering preserves these values.

    The script writes the same two particles with and without alive_only, then reads both files. It compares the retained particle’s charge, direction and field with the input values and checks the unfiltered control.
    """

    from _bmad import compare, verdict
    from _s3_11_particle_io import execute

    setup = """
    call reallocate_beam(beam, 1, 2)
    beam%bunch(1)%particle = coord_struct()
    beam%bunch(1)%particle%species = electron$
    beam%bunch(1)%particle%p0c = 1e9_rp
    beam%bunch(1)%particle%beta = 1
    beam%bunch(1)%particle(1)%state = lost$
    beam%bunch(1)%particle(1)%charge = 1
    beam%bunch(1)%particle(2)%state = alive$
    beam%bunch(1)%particle(2)%charge = 2
    beam%bunch(1)%particle(2)%direction = -1
    beam%bunch(1)%particle(2)%field = [0.6_rp, 0.8_rp]
    """


    def case(filtered):
        return execute(
            setup
            + f"""
    call write_ascii_beam_file('beam.txt', beam, .true., {".true." if filtered else ".false."})
    call read_beam_file('beam.txt', back, bi, err)
    if (err) error stop 'ASCII read failed'
    if (size(back%bunch) /= 1) error stop 'bunch count changed'
    if (size(back%bunch(1)%particle) /= {1 if filtered else 2}) error stop 'wrong filter count'
    p => back%bunch(1)%particle({1 if filtered else 2})
    if (p%state /= alive$) error stop 'retained state changed'
    call emit('retained', [p%charge, real(p%direction,rp), p%field])
    """
        )["retained"]


    expected = [2, -1, 0.6, 0.8]
    if compare(
        "unfiltered control: charge, direction, field; input identity",
        case(False),
        expected,
        atol=2e-15,
    ):
        raise RuntimeError("Unfiltered control failed")
    verdict(
        compare(
            "alive_only: charge, direction, field; retained input identity",
            case(True),
            expected,
            atol=2e-15,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

