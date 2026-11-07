---
title: "LAT-29: A girder's off state is lost at creation and ignored during alignment"
hide:
  - toc
---

# LAT-29 — A girder's off state is lost at creation and ignored during alignment

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [`create_girder` parsed-state copy, bmad/parsing/create_girder.f90:50–62](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/create_girder.f90#L50-L62)<br>[`makeup_control_slave` girder transform, bmad/modules/bookkeeper_mod.f90:1590–1649](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bookkeeper_mod.f90#L1590-L1649)<br>[Bmad manual, “Girder” (`s:girder`), off-state contract, bmad/doc/elements.tex:2874–2875](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L2874-L2875) |

A girder declared off is created enabled, and explicitly turning it off leaves its displacement applied to attached magnets. A 1 cm girder offset remains on a 1 m quadrupole with strength 1 m⁻², moving a centered particle to x = 4.597 mm with normalized horizontal momentum pₓ = 0.008415. An off girder is documented as unshifted relative to its reference frame, so this centered quadrupole must leave x and pₓ zero. The measured orbit matches the analytic map of the displaced magnet.

Run `python LAT-29.py` with [_bmad.py](../helpers/_bmad.py) and [_multipass_slicing.py](../helpers/_multipass_slicing.py) in the same directory.

??? example "Reproducer: LAT-29.py"

    [Download LAT-29.py](LAT-29.py)

    ```python
    """LAT-29: A girder's off state is lost at creation and ignored during alignment

    A girder declared off is created enabled, and explicitly turning it off leaves its displacement applied to attached magnets. A 1 cm girder offset remains on a 1 m quadrupole with strength 1 m⁻², moving a centered particle to x = 4.597 mm with normalized horizontal momentum pₓ = 0.008415. An off girder is documented as unshifted relative to its reference frame, so this centered quadrupole must leave x and pₓ zero. The measured orbit matches the analytic map of the displaced magnet.

    The script parses a disabled girder, then explicitly switches it off and on through the API with bookkeeping. It checks the on-state orbit against the displaced-quadrupole map and the off state against zero displacement and zero orbit.
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
    lord=>lat%ele(lat%n_ele_max)
    if (lord%key /= girder$) stop 2
    call row('parsed_on',[real(merge(1,0,lord%is_on),rp)])
    do j=1,2
      lord%is_on=(j==2)
      call set_flags_for_changed_attribute(lord,lord%is_on)
      call lattice_bookkeeper(lat,err)
      if (err) stop 3
      call init_coord(orb,lat%particle_start,lat%ele(1),upstream_end$)
      do i=1,lat%n_ele_track
        call track1(orb,lat%ele(i),lat%param,ending,err_flag=err)
        if (err .or. ending%state /= alive$) stop 4
        orb=ending
      enddo
      write(label,'(a,i0)') 'state',j
      call row(label,[real(merge(1,0,lord%is_on),rp),lat%ele(1)%value(x_offset_tot$),orb%vec(1:2)])
    enddo
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
    q: quadrupole, l=1, k1=1
    g: girder={q}, x_offset=0.01, is_on=F
    lat: line=(q)
    use, lat
    """

    data = rows(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
    # For L=1 and k1=1, translating the focusing map by dx gives x=dx*(1-cos(1)), px=dx*sin(1).
    shifted = [0.01 * (1 - math.cos(1)), 0.01 * math.sin(1)]
    control(
        "enabled [is_on, x_offset_tot (m), x (m), px], translated quadrupole",
        data["state2"],
        [1, 0.01, *shifted],
        atol=2e-15,
        rtol=2e-13,
    )
    control("API off flag", data["state1"][:1], [0])
    defect = compare("parsed girder is_on", data["parsed_on"], [0], atol=0)
    defect |= compare(
        "disabled [is_on, x_offset_tot (m), x (m), px], unshifted contract",
        data["state1"],
        [0, 0, 0, 0],
        atol=2e-15,
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

