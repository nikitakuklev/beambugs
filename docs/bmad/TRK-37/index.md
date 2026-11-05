---
title: "TRK-37: Thin-multipole dispatch ignores and fails to compose the supplied matrix"
hide:
  - toc
---

# TRK-37 — Thin-multipole dispatch ignores and fails to compose the supplied matrix

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`track1_bmad` thin-multipole branch, bmad/low_level/track1_bmad.f90:145–166](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track1_bmad.f90#L145-L166)<br>[argument contract, bmad/low_level/track1_bmad.f90:8–18](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track1_bmad.f90#L8-L18)<br>[`multipole_kick_mat` contract and assignments, bmad/low_level/multipole_kick_mat.f90:17–45](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/multipole_kick_mat.f90#L17-L45) |

Native thin-multipole tracking writes kick terms to the element matrix instead of propagating the matrix supplied by its caller. Even when the supplied matrix is the element matrix, the kick terms overwrite entries rather than composing with the incoming map. A thin quadrupole with integrated strength 0.7 m⁻¹ leaves a separate identity matrix with horizontal kick derivative 0 instead of −0.7 m⁻¹. With an incoming 2 m drift map and an aliased matrix, the horizontal momentum diagonal remains 1 instead of −0.4, although the particle kick is correct.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _orbital_matrices_radiation.py](../helpers/_orbital_matrices_radiation.py) · [View source](../helpers/index.md#_orbital_matrices_radiation)

??? example "Reproducer: TRK-37.py"

    [Download TRK-37.py](TRK-37.py)

    ```python
    """TRK-37: Thin-multipole dispatch ignores and fails to compose the supplied matrix

    Native thin-multipole tracking writes kick terms to the element matrix instead of propagating the matrix supplied by its caller. Even when the supplied matrix is the element matrix, the kick terms overwrite entries rather than composing with the incoming map. A thin quadrupole with integrated strength 0.7 m⁻¹ leaves a separate identity matrix with horizontal kick derivative 0 instead of −0.7 m⁻¹. With an incoming 2 m drift map and an aliased matrix, the horizontal momentum diagonal remains 1 instead of −0.4, although the particle kick is correct.

    The script calls track1_bmad with a separate identity matrix and then with an aliased matrix containing a drift term. The expected matrix is the thin-quadrupole Jacobian multiplied by the incoming matrix. Set BMAD_DIST to the build root.
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
    real(rp) mat(6,6)
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) error stop 'parse failed'
    v=0;v(1)=.01_rp
    call init_coord(a,v,lat%ele(1),upstream_end$)
    call mat_make_unit(mat)
    call track1_bmad(a,lat%ele(1),lat%param,err,mat6=mat,make_matrix=.true.)
    if(err.or.a%state/=alive$) error stop 'separate matrix failed'
    write(*,'(a,2es26.17)') 'RESULT separate ',a%vec(2),mat(2,1)
    call init_coord(a,v,lat%ele(1),upstream_end$)
    call mat_make_unit(lat%ele(1)%mat6);lat%ele(1)%mat6(1,2)=2
    call track1_bmad(a,lat%ele(1),lat%param,err,mat6=lat%ele(1)%mat6,make_matrix=.true.)
    if(err.or.a%state/=alive$) error stop 'aliased matrix failed'
    write(*,'(a,es26.17)') 'RESULT composed ',lat%ele(1)%mat6(2,2)
    end program
    """
    LATTICE = """
    no_digested
    parameter[geometry]=open
    parameter[particle]=electron
    beginning[p0c]=1e6
    e: multipole,k1l=.7
    lat: line=(e)
    use,lat
    """
    rows = records(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
    if compare("particle px/p0, -k1l*x", rows["separate"][0], -0.007, atol=2e-17):
        raise RuntimeError("Particle-kick control failed")
    defect = compare("separate M21 (1/m), -k1l", rows["separate"][1], -0.7, atol=2e-15)
    defect |= compare("composed M22, 1-k1l*M12", rows["composed"], [1 - 0.7 * 2], atol=2e-15)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

