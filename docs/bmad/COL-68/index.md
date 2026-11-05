---
title: "COL-68: Saved strong-beam centers are overwritten with the weak particle position"
hide:
  - toc
---

# COL-68 — Saved strong-beam centers are overwritten with the weak particle position

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [save_a_step, bmad/code/save_a_step.f90:108–116](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/save_a_step.f90#L108-L116)<br>[frame contract, lines 7–8](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/save_a_step.f90#L7-L8)<br>[strong_beam_struct, bmad/modules/bmad_struct.f90:2080–2085](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bmad_struct.f90#L2080-L2085)<br>[track_a_beambeam, lines 229–231](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_beambeam.f90#L229-L231) |

When beam–beam slice data are saved from element body coordinates, the saved strong-beam center is overwritten with the weak particle position. A supplied slice center at (3, −5) mm and element offsets (1, −2) mm require the laboratory center (4, −7) mm by coordinate translation. With the weak particle at (4, 6) mm in body coordinates, the routine instead records its laboratory position (5, 4) mm as the strong-beam center. Saving data already in laboratory coordinates preserves the supplied center.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)

??? example "Reproducer: COL-68.py"

    [Download COL-68.py](COL-68.py)

    ```python
    """COL-68: Saved strong-beam centers are overwritten with the weak particle position

    When beam–beam slice data are saved from element body coordinates, the saved strong-beam center is overwritten with the weak particle position. A supplied slice center at (3, −5) mm and element offsets (1, −2) mm require the laboratory center (4, −7) mm by coordinate translation. With the weak particle at (4, 6) mm in body coordinates, the routine instead records its laboratory position (5, 4) mm as the strong-beam center. Saving data already in laboratory coordinates preserves the supplied center.

    The script calls save_a_step from BMAD_DIST with a parsed, translated beam–beam element and distinct particle and strong-beam positions. It first checks the laboratory-frame copy control, then compares the body-frame saved center with the analytic coordinate translation.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict

    LATTICE = """parameter[geometry]=open
    parameter[e_tot]=1e9
    parameter[particle]=electron
    beginning[beta_a]=10
    beginning[beta_b]=20
    bb: beambeam, sig_x=.001, sig_y=.0005, beta_a_strong=10, beta_b_strong=20, x_offset=.001, y_offset=-.002
    line: line=(bb)
    use, line
    """
    PROGRAM = """program probe
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(coord_struct) particle
    type(track_struct) track
    type(strong_beam_struct) strong
    logical err, local
    integer k
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) stop 1
    call init_coord(particle,[.004_rp,0.0_rp,.006_rp,0.0_rp,0.0_rp,0.0_rp],lat%ele(1),upstream_end$)
    strong%x_center=.003_rp
    strong%y_center=-.005_rp
    do k=1,2
      local=(k==2)
      track%n_pt=-1
      call save_a_step(track,lat%ele(1),lat%param,local,particle,0.0_rp,strong_beam=strong)
      if(track%n_pt/=0) stop 2
      write(*,'(a,i1,4es25.16)') 'CENTER ',k,track%pt(0)%strong_beam%x_center, &
          track%pt(0)%strong_beam%y_center,track%pt(0)%orb%vec(1),track%pt(0)%orb%vec(3)
    enddo
    end program
    """
    output = fortran(PROGRAM, files={"lat.bmad": LATTICE}).log
    rows = np.array(
        [
            [float(v) for v in line.split()[2:]]
            for line in output.splitlines()
            if line.startswith("CENTER ")
        ]
    )
    if rows.shape != (2, 4) or not np.isfinite(rows).all():
        raise RuntimeError("Expected two complete finite center records")
    center_body = np.array([0.003, -0.005])
    offset = np.array([0.001, -0.002])
    particle_body = np.array([0.004, 0.006])
    if compare("laboratory-frame supplied-center control (m)", rows[0, :2], center_body, atol=1e-16):
        raise RuntimeError("Laboratory-frame control failed")
    # Pure translations give r_lab = r_body + offset for both beam and particle.
    if compare("particle translation control (m)", rows[1, 2:], particle_body + offset, atol=1e-16):
        raise RuntimeError("Body-to-laboratory translation control failed")
    verdict(
        compare(
            "saved strong-beam center (m), supplied center plus element offset",
            rows[1, :2],
            center_body + offset,
            atol=1e-16,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

