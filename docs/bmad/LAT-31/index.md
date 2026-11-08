---
title: "LAT-31: Super-lord spin-method edits do not reach single-lord slaves"
hide:
  - toc
---

# LAT-31 — Super-lord spin-method edits do not reach single-lord slaves

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`makeup_super_slave1` method inheritance, bmad/modules/bookkeeper_mod.f90:1423–1434](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bookkeeper_mod.f90#L1423-L1434)<br>[multipass-only integer-attribute propagation, bmad/low_level/set_flags_for_changed_attribute.f90:87–104](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/set_flags_for_changed_attribute.f90#L87-L104)<br>[`attribute_free1` super-slave restrictions, bmad/modules/attribute_mod.f90:3294–3303](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/attribute_mod.f90#L3294-L3303)<br>[Bmad manual, “Spin Tracking Methods” (`s:spin.methods`), bmad/doc/methods.tex:436–459](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L436-L459) |

After a midpoint marker splits a quadrupole, setting its super lord’s spin tracking method to off leaves the tracked slices using spin tracking. A particle starting at x = 1 mm with spin (0, 0, 1) exits a 1 m quadrupole with its spin rotated by 2.751 mrad, while the whole disabled-spin element preserves (0, 0, 1) exactly. The slices must inherit the lord's method because the spin method is not an independently adjustable super-slave attribute. The missing method copy silently defeats the requested switch.

Run `python LAT-31.py` with [_bmad.py](../helpers/_bmad.py) and [_multipass_slicing.py](../helpers/_multipass_slicing.py) in the same directory.

??? example "Reproducer: LAT-31.py"

    [Download LAT-31.py](LAT-31.py)

    ```python
    """LAT-31: Super-lord spin-method edits do not reach single-lord slaves

    After a midpoint marker splits a quadrupole, setting its super lord’s spin tracking method to off leaves the tracked slices using spin tracking. A particle starting at x = 1 mm with spin (0, 0, 1) exits a 1 m quadrupole with its spin rotated by 2.751 mrad, while the whole disabled-spin element preserves (0, 0, 1) exactly. The slices must inherit the lord's method because the spin method is not an independently adjustable super-slave attribute. The missing method copy silently defeats the requested switch.

    The script inserts a midpoint marker into a quadrupole, disables the lord's spin method and runs bookkeeping. It compares the two quadrupole slices and their tracked spin with the exact off-state invariant and the whole-lord control.
    """

    import numpy as np
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
    bmad_com%spin_tracking_on=.true.
    lord=>lat%ele(lat%n_ele_max)
    if (lord%key /= quadrupole$ .or. lord%n_slave /= 2) stop 2
    lord%spin_tracking_method=off$
    call set_flags_for_changed_attribute(lord,lord%spin_tracking_method)
    call lattice_bookkeeper(lat,err)
    if (err) stop 3
    call row('lord_off',[real(merge(1,0,lord%spin_tracking_method==off$),rp)])
    do j=1,2
      slave=>pointer_to_slave(lord,1+(j-1)*(lord%n_slave-1))
      if (slave%key /= quadrupole$) stop 4
      write(label,'(a,i0)') 'slave_off',j
      call row(label,[real(merge(1,0,slave%spin_tracking_method==off$),rp)])
    enddo
    v=0;v(1)=0.001_rp
    call init_coord(start,v,lat%ele(1),upstream_end$)
    start%spin=[0.0_rp,0.0_rp,1.0_rp]
    orb=start
    do i=1,lat%n_ele_track
      call track1(orb,lat%ele(i),lat%param,ending,err_flag=err)
      if (err .or. ending%state /= alive$) stop 5
      orb=ending
    enddo
    call row('split_spin',orb%spin)
    call track1(start,lord,lat%param,ending,err_flag=err)
    if (err .or. ending%state /= alive$) stop 6
    call row('whole_spin',ending%spin)
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
    q: quadrupole, l=1, k1=1, spin_tracking_method=tracking
    m: marker, superimpose, ref=q
    lat: line=(q)
    use, lat
    """

    data = rows(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
    control("lord spin method is off", data["lord_off"], [1])
    control("whole-lord spin, off-state invariant", data["whole_spin"], [0, 0, 1], atol=0)
    control("split spin norm", np.linalg.norm(data["split_spin"]), 1.0, atol=2e-14)
    defect = compare(
        "quadrupole slices inherit off method", data["slave_off1"] + data["slave_off2"], [1, 1], atol=0
    )
    defect |= compare("split spin, off-state invariant", data["split_spin"], [0, 0, 1], atol=2e-14)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

