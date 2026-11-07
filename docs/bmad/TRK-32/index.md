---
title: "TRK-32: Reverse quadrupole timing correction repeats the full length at every step"
hide:
  - toc
---

# TRK-32 — Reverse quadrupole timing correction repeats the full length at every step

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [`track_a_quadrupole`, bmad/low_level/track_a_quadrupole.f90:75–120](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_quadrupole.f90#L75-L120)<br>[Bmad manual, “Tracking Backwards” (`s:back.track`), bmad/doc/tracking.tex:728–748](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/tracking.tex#L728-L748) |

A particle travelling backward through a stepped quadrupole receives the full element reference-clock correction at every integration step. On axis the force vanishes, so the flight time and longitudinal coordinate must be independent of the number of steps. For a 1 m quadrupole at 1 MeV/c, one step gives z = −2 m and elapsed time 3.746 ns, while eight steps give z = −16 m and 56.19 ns. The correction uses the element reference time without the signed step fraction.

Run `python TRK-32.py` with [_bmad.py](../helpers/_bmad.py) and [_orbital_matrices_radiation.py](../helpers/_orbital_matrices_radiation.py) in the same directory.

??? example "Reproducer: TRK-32.py"

    [Download TRK-32.py](TRK-32.py)

    ```python
    """TRK-32: Reverse quadrupole timing correction repeats the full length at every step

    A particle travelling backward through a stepped quadrupole receives the full element reference-clock correction at every integration step. On axis the force vanishes, so the flight time and longitudinal coordinate must be independent of the number of steps. For a 1 m quadrupole at 1 MeV/c, one step gives z = −2 m and elapsed time 3.746 ns, while eight steps give z = −16 m and 56.19 ns. The correction uses the element reference time without the signed step fraction.

    The script tracks the same on-axis reverse-moving electron with one and eight integration steps. A tiny sextupole coefficient activates stepping without producing an on-axis force; the expected time is the straight flight length divided by βc.
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
    integer j
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) error stop 'parse failed'
    v=0
    call init_coord(a,v,lat%ele(1),downstream_end$,direction=-1)
    write(*,'(a,es26.17)') 'RESULT flight ',1/(a%beta*c_light)
    do j=1,2
     lat%ele(1)%value(ds_step$)=1.0_rp/j**3
     call track1(a,lat%ele(1),lat%param,b,err_flag=err)
     if (err.or.b%state/=alive$) error stop 'tracking failed'
     write(*,'(a,i0,2es26.17)') 'RESULT case',j,b%vec(5),b%t-a%t
    enddo
    end program
    """
    LATTICE = """
    no_digested
    parameter[geometry]=open
    parameter[particle]=electron
    beginning[p0c]=1e6
    e: quadrupole,l=1,k1=.3,fringe_type=none,b2=1e-8
    lat: line=(e)
    use,lat
    """
    rows = records(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
    if compare("one-step z (m), -2L", rows["case1"][:1], [-2], atol=2e-15) or compare(
        "one-step flight (s), L/(beta*c)", rows["case1"][1:], rows["flight"], atol=2e-22
    ):
        raise RuntimeError("One-step control failed")
    defect = compare("eight-step z (m), -2L", rows["case2"][:1], [-2], atol=2e-14)
    defect |= compare(
        "eight-step flight time (s), L/(beta*c)", rows["case2"][1:], rows["flight"], atol=2e-22
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

