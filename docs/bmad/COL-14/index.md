---
title: "COL-14: A one-element high-energy SC branch receives only half its length"
hide:
  - toc
---

# COL-14 — A one-element high-energy SC branch receives only half its length

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [setup_high_energy_space_charge_calc, bmad/space_charge/high_energy_space_charge_mod.f90:132–143](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/high_energy_space_charge_mod.f90#L132-L143) |

A branch containing one tracking element receives only half of its prescribed high-energy space-charge interaction length. The last-element rule overwrites the first-element rule, leaving L/2 instead of the full branch length L. For a 0.6 m drift, the kick constant is 5.095e−10 instead of 1.019e−9, a factor-of-two underestimate.

Run `python COL-14.py` with [_bmad.py](../helpers/_bmad.py) and [_collective_fields.py](../helpers/_collective_fields.py) in the same directory.

??? example "Reproducer: COL-14.py"

    [Download COL-14.py](COL-14.py)

    ```python
    """COL-14: A one-element high-energy SC branch receives only half its length

    A branch containing one tracking element receives only half of its prescribed high-energy
    space-charge interaction length. The last-element rule overwrites the first-element rule,
    leaving L/2 instead of the full branch length L. For a 0.6 m drift, the kick constant is
    5.095e−10 instead of 1.019e−9, a factor-of-two underestimate.

    The script prepares a 0.6 m branch with exactly one tracking element, retaining allocated padding to
    isolate the length rule. It checks the count and beam widths, then compares the stored kick constant
    with the full-length analytic expression.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _collective_fields import rows

    PROGRAM = r"""

    program high_energy_checks
      use bmad
      use high_energy_space_charge_mod
      implicit none
      type(ele_struct) ele
      type(lat_struct) lat
      type(normal_modes_struct) mode
      type(coord_struct) p
      type(lat_param_struct) param
      real(rp) response(2,2), phi
      integer j
      call bmad_parser('lat.bmad',lat)
      ! The parser appends an end marker. Exercise the supported branch structure
      ! with exactly one tracking element while retaining allocated padding.
      lat%branch(0)%n_ele_track = 1
      write(*,'(a,i4)') 'RESULT track_count ',lat%branch(0)%n_ele_track
      lat%ele(1)%a%beta = 3; lat%ele(1)%b%beta = 5
      mode%a%emittance = 2.0e-6_rp; mode%b%emittance = 1.0e-6_rp
      mode%sig_z = 0.003_rp; mode%sigE_E = 0
      call setup_high_energy_space_charge_calc(.true.,lat%branch(0),1.0e9_rp,mode)
      write(*,'(a,6es25.16e3)') 'RESULT setup ',lat%ele(1)%high_energy_space_charge%kick_const, &
        lat%ele(1)%high_energy_space_charge%sig_x,lat%ele(1)%high_energy_space_charge%sig_y, &
        lat%ele(1)%value(l$),m_electron,classical_radius_factor
    end program
    """


    data = rows(
        fortran(
            PROGRAM,
            files={
                "lat.bmad": "parameter[particle]=electron\nparameter[p0c]=1e8\nd: drift, l=0.6\nl: line=(d)\nuse, l\n"
            },
        ).log
    )
    if not np.array_equal(data["track_count"], [1]):
        raise RuntimeError("One-element branch control failed")
    kick, sx, sy, length, mass, radius_factor = data["setup"]
    if length != 0.6 or not np.allclose(
        sorted([sx, sy]), sorted([np.sqrt(6e-6), np.sqrt(5e-6)]), rtol=2e-15, atol=0
    ):
        raise RuntimeError("Length or independent beam-width control failed")
    # Integrating a constant prescribed kick over a one-element branch gives the full L.
    expected = (
        length
        * radius_factor
        * 1e9
        / ((2 * np.pi) ** 1.5 * (1e8 / mass) ** 3 * mass * (sx + sy) * 0.003)
    )
    defect = compare(
        "one-element kick constant, full interaction length", kick, expected, atol=0, rtol=3e-15
    )
    if defect and not np.isclose(kick, expected / 2, rtol=3e-15, atol=0):
        raise RuntimeError("Kick mismatch does not match the half-length signature")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

