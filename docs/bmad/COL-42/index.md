---
title: "COL-42: Beam–beam longitudinal kick hardcodes equal beam speeds"
hide:
  - toc
---

# COL-42 — Beam–beam longitudinal kick hardcodes equal beam speeds

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [strong-beam speed and encounter geometry, bmad/low_level/track_a_beambeam.f90:161–171](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_beambeam.f90#L161-L171)<br>[kinematic energy update, bmad/low_level/track_a_beambeam.f90:241–250](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_beambeam.f90#L241-L250)<br>[Bmad manual, unequal beam velocities, bmad/doc/elements.tex:407–412](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L407-L412)<br>[BeamBeam Tracking, equation dpz12, bmad/doc/charged-tracking.tex:377–396](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/charged-tracking.tex#L377-L396) |

For unequal beam speeds, the beam–beam tracker uses the equal-speed coefficient in its kinematic longitudinal kick. The documented coefficient is 1/[2(1 + βw/βs)(1 + δ)], where βw and βs are the weak- and strong-beam speeds in units of c and δ is relative momentum; tracking uses 1/[4(1 + δ)]. At βw = √3/2 and βs = 1/2, the longitudinal kick is 36.60% too large. Equal-speed beams recover the documented coefficient.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)

??? example "Reproducer: COL-42.py"

    [Download COL-42.py](COL-42.py)

    ```python
    """COL-42: Beam–beam longitudinal kick hardcodes equal beam speeds

    For unequal beam speeds, the beam–beam tracker uses the equal-speed coefficient in its kinematic longitudinal kick. The documented coefficient is 1/[2(1 + βw/βs)(1 + δ)], where βw and βs are the weak- and strong-beam speeds in units of c and δ is relative momentum; tracking uses 1/[4(1 + δ)]. At βw = √3/2 and βs = 1/2, the longitudinal kick is 36.60% too large. Equal-speed beams recover the documented coefficient.

    The script tracks a parsed single-slice collision at the strong-beam waist with BMAD_DIST. It verifies zero hourglass derivatives and compares the longitudinal kick with the manual equation using the measured transverse kicks, with equal speeds as a control.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict

    LATTICE = """parameter[geometry] = open
    parameter[particle] = electron
    parameter[e_tot] = 2*m_electron
    beginning[beta_a] = 10
    beginning[beta_b] = 20
    bb: beambeam, sig_x=1e-3, sig_y=5e-4, sig_z=.02, n_slice=1, n_particle=1e7, charge=-1, beta_a_strong=10, beta_b_strong=20, crab_x1=0
    line: line=(bb)
    use, line
    """
    PROGRAM = """program probe
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(coord_struct) orb
    real(rp) bw,bs,old(6),sigma(2),deriv(2),constant
    integer k
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) stop 1
    do k=1,2
      call init_coord(orb,[1e-4_rp,0.0_rp,2e-4_rp,0.0_rp,0.0_rp,0.0_rp],lat%ele(1),upstream_end$)
      bw=orb%beta
      bs=bw
      if(k==2) bs=0.5_rp
      lat%ele(1)%value(species_strong$)=electron$
      lat%ele(1)%value(e_tot_strong$)=m_electron/sqrt(1-bs**2)
      call strong_beam_sigma_calc(lat%ele(1),0.0_rp,sigma,constant,deriv)
      old=orb%vec
      call track_a_beambeam(orb,lat%ele(1),lat%param)
      if(orb%state /= alive$) stop 2
      write(*,'(a,7es25.16)') 'RESULT ',bw,bs,orb%vec(6),orb%vec(2)-old(2),orb%vec(4)-old(4),deriv
    enddo
    end program
    """
    output = fortran(PROGRAM, files={"lat.bmad": LATTICE}).log
    rows = np.array(
        [
            [float(x) for x in line.split()[1:]]
            for line in output.splitlines()
            if line.startswith("RESULT ")
        ]
    )
    if rows.shape != (2, 7) or not np.isfinite(rows).all():
        raise RuntimeError("Expected finite equal- and unequal-speed records")
    if compare("waist hourglass derivatives control", rows[:, 5:], np.zeros((2, 2)), atol=0):
        raise RuntimeError("Hourglass contribution must vanish")
    # With zero initial transverse momenta and delta=0, dpz12 depends only on
    # the measured transverse kicks and the beam-speed ratio. Waist derivatives vanish.
    expected = (rows[:, 3] ** 2 + rows[:, 4] ** 2) / (2 * (1 + rows[:, 0] / rows[:, 1]))
    if compare("equal-speed longitudinal-kick control", rows[0, 2], expected[0], atol=0, rtol=2e-12):
        raise RuntimeError("Equal-speed control failed")
    verdict(
        compare(
            "unequal-speed longitudinal kick, manual dpz12", rows[1, 2], expected[1], atol=0, rtol=2e-12
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>source check</small> | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>unequal-speed kinematic kick at the waist</small> | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

