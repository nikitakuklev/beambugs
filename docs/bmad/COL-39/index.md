---
title: "COL-39: Beam–beam crab coefficients define an angle instead of the documented displacement"
hide:
  - toc
---

# COL-39 — Beam–beam crab coefficients define an angle instead of the documented displacement

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [strong_beam_center, bmad/low_level/track_a_beambeam.f90:312–324](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_beambeam.f90#L312-L324)<br>[Bmad manual, BeamBeam crabbing polynomial, bmad/doc/elements.tex:375–389](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L375-L389) |

Beam–beam crabbing interprets its coefficients as an angle polynomial, while the manual defines a transverse displacement polynomial. For a linear coefficient a and strong-beam longitudinal coordinate u, tracking uses x = u sin(a) instead of the documented x = ua, and also changes the longitudinal coordinate by cos(a). At a = 0.3 rad the transverse slice offset is 1.493% smaller than the documented displacement. The formulas agree to first order in small angle.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)

??? example "Reproducer: COL-39.py"

    [Download COL-39.py](COL-39.py)

    ```python
    """COL-39: Beam–beam crab coefficients define an angle instead of the documented displacement

    Beam–beam crabbing interprets its coefficients as an angle polynomial, while the manual defines a transverse displacement polynomial. For a linear coefficient a and strong-beam longitudinal coordinate u, tracking uses x = u sin(a) instead of the documented x = ua, and also changes the longitudinal coordinate by cos(a). At a = 0.3 rad the transverse slice offset is 1.493% smaller than the documented displacement. The formulas agree to first order in small angle.

    The script tracks a parsed two-slice beam–beam element using BMAD_DIST and reconstructs the slice centers from recorded particle positions and particle-to-slice offsets. It compares finite crabbing with the manual displacement formula and checks a zero-crab control.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict

    LATTICE = """parameter[geometry] = open
    parameter[particle] = electron
    parameter[e_tot] = 1e9
    beginning[beta_a] = 10
    beginning[beta_b] = 20
    bb: beambeam, sig_x=1e-3, sig_y=5e-4, sig_z=.02, n_slice=2, n_particle=1e7, charge=-1, beta_a_strong=10, beta_b_strong=20, crab_x1=.3
    line: line=(bb)
    use, line
    """
    PROGRAM = """program probe
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(coord_struct) orb
    type(track_struct) tr
    real(rp) z(2)
    integer i,k
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) stop 1
    do k=1,2
      lat%ele(1)%value(crab_x1$)=0.3_rp*(k-1)
      call init_coord(orb,lat%ele(1),upstream_end$)
      tr%ds_save=-1
      tr%n_pt=-1
      call track_a_beambeam(orb,lat%ele(1),lat%param,track=tr)
      if(orb%state /= alive$) stop 2
      call bbi_slice_calc(lat%ele(1),2,z)
      do i=1,2
        write(*,'(a,4es25.16)') 'RESULT ', lat%ele(1)%value(crab_x1$), z(i), &
                 tr%pt(2*i-1)%orb%vec(1)-tr%pt(2*i-1)%strong_beam%dx, &
                 tr%pt(2*i-1)%orb%vec(3)-tr%pt(2*i-1)%strong_beam%dy
        write(*,'(a,2es25.16)') 'SAVED_CENTER ', tr%pt(2*i-1)%strong_beam%x_center, tr%pt(2*i-1)%strong_beam%y_center
      enddo
    enddo
    end program
    """
    output = fortran(PROGRAM, files={"lat.bmad": LATTICE}).log
    print(
        "\n".join(line for line in output.splitlines() if line.startswith(("RESULT ", "SAVED_CENTER ")))
    )
    rows = np.array(
        [
            [float(x) for x in line.split()[1:]]
            for line in output.splitlines()
            if line.startswith("RESULT ")
        ]
    )
    if rows.shape != (4, 4):
        raise RuntimeError("Expected four tracked slice centers")
    if compare("zero-crab slice-center control (m)", rows[:2, 2:], np.zeros((2, 2)), atol=1e-15):
        raise RuntimeError("Zero-crab control failed")
    # With no element offsets, x_center = particle x - recorded dx in the same frame.
    # bbi_slice_calc uses positive z toward the tail; the manual's u points toward the head.
    if compare(
        "sine geometry control (m)",
        rows[2:, 2],
        -rows[2:, 1] * np.sin(rows[2:, 0]),
        atol=1e-14,
        rtol=2e-12,
    ):
        raise RuntimeError("Recorded centers do not match the source geometry")
    expected = -rows[2:, 1] * rows[2:, 0]
    verdict(
        compare("crabbed x centers (m), manual x=u*a", rows[2:, 2], expected, atol=1e-14, rtol=2e-12)
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>three-slice quadratic/cubic crab geometry and tilt</small> | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

