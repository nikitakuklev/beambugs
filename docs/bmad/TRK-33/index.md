---
title: "TRK-33: Reverse sol_quad and bend maps omit backward reference-clock motion"
hide:
  - toc
---

# TRK-33 — Reverse sol_quad and bend maps omit backward reference-clock motion

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`sol_quad_mat6_calc`, bmad/low_level/sol_quad_mat6_calc.f90:323–326](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/sol_quad_mat6_calc.f90#L323-L326)<br>[`track_a_sol_quad`, bmad/low_level/track_a_sol_quad.f90:75–103](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_sol_quad.f90#L75-L103)<br>[general bend body, bmad/low_level/track_a_bend.f90:202–209](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_bend.f90#L202-L209)<br>[Bmad manual, “Tracking Backwards” (`s:back.track`), bmad/doc/tracking.tex:728–748](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/tracking.tex#L728-L748)<br>[`solenoid_track_and_mat`, bmad/low_level/solenoid_track_and_mat.f90:55–57](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/solenoid_track_and_mat.f90#L55-L57) |

Native reverse-direction tracking through sol_quad and general sector-bend maps omits the backward motion of the reference clock. Physical particle time should increase while the reference time decreases. A force-free on-axis 1 m sol_quad at 1 MeV/c returns z = 0 and elapsed time −3.746 ns instead of z = −2 m and +3.746 ns. A displaced reverse bend likewise misses −2 m in z and returns a negative flight time; circular geometry and Runge–Kutta tracking give a positive flight time.

Run `python TRK-33.py` with [_bmad.py](../helpers/_bmad.py) and [_orbital_matrices_radiation.py](../helpers/_orbital_matrices_radiation.py) in the same directory.

??? example "Reproducer: TRK-33.py"

    [Download TRK-33.py](TRK-33.py)

    ```python
    """TRK-33: Reverse sol_quad and bend maps omit backward reference-clock motion

    Native reverse-direction tracking through sol_quad and general sector-bend maps omits the backward motion of the reference clock. Physical particle time should increase while the reference time decreases. A force-free on-axis 1 m sol_quad at 1 MeV/c returns z = 0 and elapsed time −3.746 ns instead of z = −2 m and +3.746 ns. A displaced reverse bend likewise misses −2 m in z and returns a negative flight time; circular geometry and Runge–Kutta tracking give a positive flight time.

    The script tracks a reverse-moving electron through sol_quad and a sector bend. Straight-flight and circle/exit-plane geometry supply the expected path lengths and times, with Runge–Kutta as a control. The bend begins 0.1 μm off axis to select its general map.
    """

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
    integer j,method
    real(rp) path,theta,phi
    type(ele_struct),pointer::e
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) error stop 'parse failed'
    do j=1,2
     e=>lat%ele(j)
     v=0;path=1
     if(j==2) then
      v(1)=1e-7_rp
      theta=.1_rp
      phi=asin((2+.1_rp*v(1))*sin(theta))-theta
      path=phi/.1_rp
     endif
     call init_coord(a,v,e,downstream_end$,direction=-1)
     write(*,'(a,i0,2es26.17)') 'RESULT expected',j,-1-path,path/(a%beta*c_light)
     do method=1,2
      if(method==1) e%tracking_method=bmad_standard$
      if(method==2) e%tracking_method=runge_kutta$
      call track1(a,e,lat%param,b,err_flag=err)
      if (err.or.b%state/=alive$) error stop 'tracking failed'
      write(*,'(a,i0,i0,2es26.17)') 'RESULT case',j,method,b%vec(5),b%t-a%t
     enddo
    enddo
    end program
    """
    LATTICE = """
    no_digested
    parameter[geometry]=open
    parameter[particle]=electron
    beginning[p0c]=1e6
    e: sol_quad,l=1,ks=1,k1=.3,fringe_type=none
    b: sbend,l=1,g=.1,fringe_type=none
    lat: line=(e,b)
    use,lat
    """
    rows = records(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
    defect = False
    for j, label in ((1, "sol_quad"), (2, "bend")):
        expected = rows[f"expected{j}"]
        if compare(f"{label} RK z (m)", rows[f"case{j}2"][:1], expected[:1], atol=2e-10):
            raise RuntimeError("RK path control failed")
        if compare(f"{label} RK flight (s)", rows[f"case{j}2"][1:], expected[1:], atol=2e-18):
            raise RuntimeError("RK time control failed")
        defect |= compare(
            f"{label} native z (m), reference minus path",
            rows[f"case{j}1"][:1],
            expected[:1],
            atol=2e-14,
        )
        defect |= compare(
            f"{label} native flight (s), path/(beta*c)", rows[f"case{j}1"][1:], expected[1:], atol=2e-22
        )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

