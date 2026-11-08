---
title: "LAT-27: Taylor slices each apply the complete map"
hide:
  - toc
---

# LAT-27 — Taylor slices each apply the complete map

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`makeup_super_slave1` Taylor entrance rule, bmad/modules/bookkeeper_mod.f90:1225–1237](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bookkeeper_mod.f90#L1225-L1237)<br>[later lord-state copy, bmad/modules/bookkeeper_mod.f90:1423–1434](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bookkeeper_mod.f90#L1423-L1434)<br>[`create_element_slice` API, bmad/code/create_element_slice.f90:1–29](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/create_element_slice.f90#L1-L29) |

Splitting a finite-length Taylor element through the element-slicing API applies its complete map in each slice. A map that adds 1 mm to horizontal position produces 1 mm as a whole element and 2 mm as two half-length slices. The slicing routine explicitly assigns the zero-length map to the physical entrance, but a later copy of the lord's enabled state reactivates the second slice. The map must act once at the entrance.

Run `python LAT-27.py` with [_bmad.py](../helpers/_bmad.py) and [_multipass_slicing.py](../helpers/_multipass_slicing.py) in the same directory.

??? example "Reproducer: LAT-27.py"

    [Download LAT-27.py](LAT-27.py)

    ```python
    """LAT-27: Taylor slices each apply the complete map

    Splitting a finite-length Taylor element through the element-slicing API applies its complete map in each slice. A map that adds 1 mm to horizontal position produces 1 mm as a whole element and 2 mm as two half-length slices. The slicing routine explicitly assigns the zero-length map to the physical entrance, but a later copy of the lord's enabled state reactivates the second slice. The map must act once at the entrance.

    The script creates two half-length slices of a Taylor element that translates x by 1 mm. It checks the whole element against that translation, then checks slice activation and their combined displacement.
    """

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
    call create_element_slice(s1,lat%ele(1),0.5_rp,0.0_rp,lat%param,.true.,.false.,err)
    if (err) stop 2
    call create_element_slice(s2,lat%ele(1),0.5_rp,0.5_rp,lat%param,.false.,.true.,err)
    if (err) stop 3
    call row('slice_on',real([merge(1,0,s1%is_on),merge(1,0,s2%is_on)],rp))
    call init_coord(start,lat%particle_start,lat%ele(1),upstream_end$)
    call track1(start,lat%ele(1),lat%param,whole,err_flag=err)
    if (err .or. whole%state /= alive$) stop 4
    call track1(start,s1,lat%param,orb,err_flag=err)
    if (err .or. orb%state /= alive$) stop 5
    call track1(orb,s2,lat%param,ending,err_flag=err)
    if (err .or. ending%state /= alive$) stop 6
    call row('whole_x',[whole%vec(1)])
    call row('split_x',[ending%vec(1)])
    call deallocate_ele_pointers(s1)
    call deallocate_ele_pointers(s2)
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
    t: taylor, l=1, {1: 0.001 | }
    lat: line=(t)
    use, lat
    """

    data = rows(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
    control("whole Taylor translation x (m)", data["whole_x"], [0.001], atol=2e-16)
    # The source contract places the entire zero-length map at the physical entrance.
    defect = compare("slice activation [entrance, exit]", data["slice_on"], [1, 0], atol=0)
    defect |= compare("two slices: single translation x (m)", data["split_x"], [0.001], atol=2e-16)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

