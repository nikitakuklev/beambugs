---
title: "COL-37: Basic Touschek lifetime averaging reads beyond the tracking elements"
hide:
  - toc
---

# COL-37 — Basic Touschek lifetime averaging reads beyond the tracking elements

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [touschek_lifetime, bmad/multiparticle/touschek_mod.f90:49–108](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/touschek_mod.f90#L49-L108) |

The basic Touschek lifetime routine uses the length after the final tracking element as a physical integration weight. A uniform local loss rate must retain its lifetime when averaged around a ring. For four equal-length elements, zero length in the following storage slot makes the lifetime 14.29% too long; setting that nontracking length to twice an element length makes it 11.11% too short. The same indexing reads outside the element array when storage ends at the last tracking element.

Run `python COL-37.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-37.py"

    [Download COL-37.py](COL-37.py)

    ```python
    """COL-37: Basic Touschek lifetime averaging reads beyond the tracking elements

    The basic Touschek lifetime routine uses the length after the final tracking element as a physical integration weight. A uniform local loss rate must retain its lifetime when averaged around a ring. For four equal-length elements, zero length in the following storage slot makes the lifetime 14.29% too long; setting that nontracking length to twice an element length makes it 11.11% too short. The same indexing reads outside the element array when storage ends at the last tracking element.

    The script compiles a four-element uniform-rate lattice driver against BMAD_DIST. It varies only the nontracking storage length and compares the average lifetime with particle count divided by the independently called local loss rate.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict

    PROGRAM = """program probe_touschek
    use bmad
    use touschek_mod, only: touschek_rate1, touschek_lifetime
    implicit none
    type(lat_struct) lat
    type(normal_modes_struct) mode
    real(rp) rate, lifetime, expected
    integer i, k
    call init_lat(lat,5)
    lat%n_ele_track = 4
    lat%param%total_length=4.0_rp
    lat%param%particle = electron$
    lat%param%geometry = closed$
    lat%param%n_part = 1.0e10_rp
    do i=0,5
      lat%ele(i)%value(l$)=1.0_rp
      lat%ele(i)%value(e_tot$) = 2.0e9_rp
      lat%ele(i)%a%beta = 10.0_rp
      lat%ele(i)%b%beta = 20.0_rp
    enddo
    mode%a%emittance = 1.0e-9_rp
    mode%b%emittance = 1.0e-11_rp
    mode%sigE_E = 1.0e-3_rp
    mode%sig_z = 0.01_rp
    mode%pz_aperture = 0.03_rp
    call touschek_rate1(mode,rate,lat,ix=1)
    expected=lat%param%n_part/rate
    do k=1,3
      select case(k)
      case(1)
        lat%ele(5)%value(l$)=1.0_rp
      case(2)
        lat%ele(5)%value(l$)=0.0_rp
      case(3)
        lat%ele(5)%value(l$)=2.0_rp
      end select
      call touschek_lifetime(mode,lifetime,lat)
      write(*,'(3es25.16)') lat%ele(5)%value(l$), lifetime, expected
    enddo
    end program
    """
    rows = np.array([[float(x) for x in line.split()] for line in fortran(PROGRAM).log.splitlines()])
    if rows.shape != (3, 3) or not np.isfinite(rows).all() or not (rows[:, 1:] > 0).all():
        raise RuntimeError("Expected three finite positive lifetime cases")
    if compare(
        "matching padding control, lifetime/local lifetime", rows[0, 1] / rows[0, 2], 1, atol=2e-14
    ):
        raise RuntimeError("Uniform-rate control failed")
    defect = False
    for row in rows[1:]:
        defect |= compare(
            f"nontracking length {row[0]:g} m, lifetime/local lifetime", row[1] / row[2], 1, atol=2e-14
        )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small class="bb-scope">source check</small> | <span class="bb-nowrap">`gemini-4-argon-high`</span> |  | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small class="bb-scope">source check</small> | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small class="bb-scope">Touschek lifetime averaging</small> | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

