---
title: "TRK-30: Solenoid strength counts tracking charge twice"
hide:
  - toc
---

# TRK-30 — Solenoid strength counts tracking charge twice

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`solenoid_track_and_mat`, bmad/low_level/solenoid_track_and_mat.f90:38–57](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/solenoid_track_and_mat.f90#L38-L57)<br>[Bmad manual, “Solenoid Tracking” (`s:solenoid.std`), bmad/doc/charged-tracking.tex:1623–1653](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/charged-tracking.tex#L1623-L1653)<br>[`rel_tracking_charge_to_mass`, bmad/code/rel_tracking_charge_to_mass.f90:27–32](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/rel_tracking_charge_to_mass.f90#L27-L32) |

A solenoid with an electron reference rotates equal-mass electrons and positrons with the same handedness in native tracking. The relative charge factor already contains the tracked charge sign, and multiplying by the tracked charge again cancels that reversal. For a 1 m solenoid of strength 1 m⁻¹ and entrance x = 1 mm, both species leave at y = −0.4207 mm; the positron should leave at +0.4207 mm. The analytic solenoid map and Runge–Kutta tracking agree on the required sign.

Run `python TRK-30.py` with [_bmad.py](../helpers/_bmad.py) and [_orbital_matrices_radiation.py](../helpers/_orbital_matrices_radiation.py) in the same directory.

??? example "Reproducer: TRK-30.py"

    [Download TRK-30.py](TRK-30.py)

    ```python
    """TRK-30: Solenoid strength counts tracking charge twice

    A solenoid with an electron reference rotates equal-mass electrons and positrons with the same handedness in native tracking. The relative charge factor already contains the tracked charge sign, and multiplying by the tracked charge again cancels that reversal. For a 1 m solenoid of strength 1 m⁻¹ and entrance x = 1 mm, both species leave at y = −0.4207 mm; the positron should leave at +0.4207 mm. The analytic solenoid map and Runge–Kutta tracking agree on the required sign.

    The script tracks electrons and positrons through a fringe-free solenoid using both native and Runge–Kutta methods. It derives the transverse positions from the hard-edge solenoid body solution at fixed canonical entrance momentum.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _orbital_matrices_radiation import records

    PROGRAM = """
    program probe
    use bmad

    implicit none
    type(lat_struct), target :: lat
    type(coord_struct) a,b
    real(rp) v(6)
    logical err
    integer species,method
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) error stop 'parse failed'
    bmad_com%rel_tol_adaptive_tracking=1e-13_rp
    bmad_com%abs_tol_adaptive_tracking=1e-16_rp
    ! Preserve the historical default-real literal, then convert it to rp.
    v=0; v(1)=0.001
    do species=1,2
     if (species==1) call init_coord(a,v,lat%ele(1),upstream_end$,particle=electron$)
     if (species==2) call init_coord(a,v,lat%ele(1),upstream_end$,particle=positron$)
     do method=1,2
      if (method==1) lat%ele(1)%tracking_method=bmad_standard$
      if (method==2) lat%ele(1)%tracking_method=runge_kutta$
      call track1(a,lat%ele(1),lat%param,b,err_flag=err)
      if (err.or.b%state/=alive$) error stop 'tracking failed'
      write(*,'(a,i0,i0,2es26.17)') 'RESULT case',species,method,b%vec(1),b%vec(3)
     enddo
    enddo
    end program
    """
    LATTICE = """
    no_digested
    parameter[geometry]=open
    parameter[particle]=electron
    beginning[p0c]=1e6
    e: solenoid,l=1,ks=1,fringe_type=none
    lat: line=(e)
    use,lat
    """
    rows = records(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
    x0 = float(np.float32(0.001))  # The original Fortran input was a default-real literal.
    theta = 0.5 / np.sqrt(1 - (x0 / 2) ** 2)
    # ks changes sign with charge, while pz and cos(theta)^2 do not.
    electron = np.array([np.cos(theta) ** 2 * x0, -np.sin(theta) * np.cos(theta) * x0])
    positron = electron * [1, -1]
    for key, expected in (("case11", electron), ("case12", electron), ("case22", positron)):
        if compare(f"{key} control x,y (m), analytic solenoid", rows[key], expected, atol=2e-15):
            raise RuntimeError("Solenoid control failed")
    verdict(compare("native positron x,y (m), analytic solenoid", rows["case21"], positron, atol=2e-15))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

