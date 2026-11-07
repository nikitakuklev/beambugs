---
title: "LAT-30: Center-fiducial bend angle changes corrupt symmetric geometry"
hide:
  - toc
---

# LAT-30 — Center-fiducial bend angle changes corrupt symmetric geometry

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`bend_angle_fiducial` and `bend_angle_fiducial2`, bmad/low_level/set_flags_for_changed_attribute.f90:949–998](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/set_flags_for_changed_attribute.f90#L949-L998)<br>[Bmad manual, “Bend: Fiducial Point Calculations” (`s:bend.fiducial`), bmad/doc/charged-tracking.tex:715–733](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/charged-tracking.tex#L715-L733) |

Changing the angle of a symmetric bend about its center fiducial produces asymmetric face angles and the wrong arc length. Starting from length 1 m, angle 0.1 rad and zero face angles, setting the angle to 0.2 rad gives length 1.002 m, entrance face angle zero and exit face angle 0.1503 rad. Holding the center and face planes fixed requires length 1 m and both face angles 0.05000 rad, as the circle-plane intersection shows. The implementation updates the exit angle twice and reuses already changed curvature.

Run `python LAT-30.py` with [_bmad.py](../helpers/_bmad.py) and [_multipass_slicing.py](../helpers/_multipass_slicing.py) in the same directory.

??? example "Reproducer: LAT-30.py"

    [Download LAT-30.py](LAT-30.py)

    ```python
    """LAT-30: Center-fiducial bend angle changes corrupt symmetric geometry

    Changing the angle of a symmetric bend about its center fiducial produces asymmetric face angles and the wrong arc length. Starting from length 1 m, angle 0.1 rad and zero face angles, setting the angle to 0.2 rad gives length 1.002 m, entrance face angle zero and exit face angle 0.1503 rad. Holding the center and face planes fixed requires length 1 m and both face angles 0.05000 rad, as the circle-plane intersection shows. The implementation updates the exit angle twice and reuses already changed curvature.

    The script changes only ANGLE through Bmad's attribute setter and runs bookkeeping. It verifies the initial symmetric geometry and computes the new radius from the intersection of the new circle with the original face plane.
    """

    import math

    from _bmad import compare, fortran, verdict
    from _multipass_slicing import control, rows

    PROGRAM = r"""program probe
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: ele, lord, slave
    type(ele_struct) :: s1,s2
    type(coord_struct) :: start, orb, ending, whole
    type(coord_struct), allocatable :: orbit(:)
    integer i, j, n
    real(rp) v(6)
    logical err
    character(40) label, arg
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) stop 1
    ele=>lat%ele(1)
    call row('initial',[ele%value(l$),ele%value(angle$),ele%value(e1$),ele%value(e2$)])
    call set_ele_real_attribute(ele,'ANGLE',0.2_rp,err)
    if (err) stop 2
    call lattice_bookkeeper(lat,err)
    if (err) stop 3
    call row('changed',[ele%value(l$),ele%value(angle$),ele%value(e1$),ele%value(e2$)])
    call deallocate_lat_pointers(lat)
    contains
    subroutine row(name,values)
    character(*) name
    real(rp) values(:)
    write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
    end subroutine
    end program
    """
    LATTICE = r"""no_digested
    parameter[geometry] = open
    beginning[p0c] = 1e9
    beginning[beta_a] = 10
    beginning[beta_b] = 10
    b: sbend, l=1, angle=0.1, fiducial_pt=center
    lat: line=(b)
    use, lat
    """

    data = rows(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
    control("initial [L (m), angle, e1, e2 (rad)]", data["initial"], [1, 0.1, 0, 0], atol=2e-16)
    # In center-tangent coordinates the old exit plane intersects the new circle at
    # rho1*(sin(t0)+sin(t1-t0))=rho0*sin(t0), with old/new half angles t0,t1.
    t0, t1, rho0 = 0.05, 0.1, 10.0
    rho1 = rho0 * math.sin(t0) / (math.sin(t0) + math.sin(t1 - t0))
    expected = [2 * t1 * rho1, 2 * t1, t1 - t0, t1 - t0]
    defect = compare(
        "changed [L (m), angle, e1, e2 (rad)], fixed-center face-plane geometry",
        data["changed"],
        expected,
        atol=2e-14,
        rtol=2e-14,
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

