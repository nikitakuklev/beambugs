---
title: "TRK-31: Negative-time combined-function bends select invalid small-argument series"
hide:
  - toc
---

# TRK-31 — Negative-time combined-function bends select invalid small-argument series

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`sbend_body_with_k1_map`, bmad/low_level/track_a_bend.f90:525–570](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_bend.f90#L525-L570)<br>[Bmad manual, coordinate time direction, bmad/doc/tracking.tex:34–78](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/tracking.tex#L34-L78) |

Backward-time tracking through a combined-function bend uses a small-phase polynomial for every negative phase, even when its magnitude is large. The body map should evaluate the same trigonometric or hyperbolic solution at negative length. For a 2 m bend with curvature 0.1 m⁻¹, quadrupole strength 1 m⁻² and entrance x = y = 1 mm, it returns x = −1.020 mm and y = 3.000 mm instead of −0.4252 mm and 3.762 mm. A forward/backward round trip leaves a −1.803 mm vertical residual.

Run `python TRK-31.py` with [_bmad.py](../helpers/_bmad.py) and [_orbital_matrices_radiation.py](../helpers/_orbital_matrices_radiation.py) in the same directory.

??? example "Reproducer: TRK-31.py"

    [Download TRK-31.py](TRK-31.py)

    ```python
    """TRK-31: Negative-time combined-function bends select invalid small-argument series

    Backward-time tracking through a combined-function bend uses a small-phase polynomial for every negative phase, even when its magnitude is large. The body map should evaluate the same trigonometric or hyperbolic solution at negative length. For a 2 m bend with curvature 0.1 m⁻¹, quadrupole strength 1 m⁻² and entrance x = y = 1 mm, it returns x = −1.020 mm and y = 3.000 mm instead of −0.4252 mm and 3.762 mm. A forward/backward round trip leaves a −1.803 mm vertical residual.

    The script compares positive- and negative-time tracking with the combined-function body solution and checks a forward/backward round trip. The particle keeps forward spatial direction while time_dir changes sign.
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
    type(coord_struct) c
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) error stop 'parse failed'
    v=0;v(1)=.001_rp;v(3)=.001_rp
    call init_coord(a,v,lat%ele(1),downstream_end$)
    a%time_dir=-1
    call track1(a,lat%ele(1),lat%param,b,err_flag=err)
    if (err.or.b%state/=alive$) error stop 'reverse tracking failed'
    write(*,'(a,4es26.17)') 'RESULT backward ',b%vec(1:4)
    call init_coord(a,v,lat%ele(1),upstream_end$)
    call track1(a,lat%ele(1),lat%param,b,err_flag=err)
    if (err.or.b%state/=alive$) error stop 'forward tracking failed'
    write(*,'(a,4es26.17)') 'RESULT forward ',b%vec(1:4)
    b%time_dir=-1
    call track1(b,lat%ele(1),lat%param,c,err_flag=err)
    if (err.or.c%state/=alive$) error stop 'round trip failed'
    write(*,'(a,4es26.17)') 'RESULT residual ',c%vec(1:4)-v(1:4)
    end program
    """
    LATTICE = """
    no_digested
    parameter[geometry]=open
    parameter[particle]=electron
    beginning[p0c]=1e9
    e: sbend,l=2,g=.1,k1=1,fringe_type=none
    lat: line=(e)
    use,lat
    """
    rows = records(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)


    def body(length):
        hx = np.sqrt(1.01)
        return np.array(
            [
                0.001 * np.cos(hx * length),
                -0.001 * hx * np.sin(hx * length),
                0.001 * np.cosh(length),
                0.001 * np.sinh(length),
            ]
        )


    if compare("positive time x,px,y,py, body solution", rows["forward"], body(2), atol=3e-15):
        raise RuntimeError("Positive-time control failed")
    defect = compare("negative time x,px,y,py, body solution", rows["backward"], body(-2), atol=3e-15)
    defect |= compare("round-trip orbit residual", rows["residual"], np.zeros(4), atol=3e-14)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

