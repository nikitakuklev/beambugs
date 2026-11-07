---
title: "LAT-28: RF slicing invents couplers and doubles both-end kicks"
hide:
  - toc
---

# LAT-28 — RF slicing invents couplers and doubles both-end kicks

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`compute_slave_coupler`, bmad/modules/bookkeeper_mod.f90:1530–1543](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bookkeeper_mod.f90#L1530-L1543)<br>[`rf_coupler_kick`, bmad/low_level/rf_coupler_kick.f90:39–75](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/rf_coupler_kick.f90#L39-L75)<br>[Bmad manual, “RF Couplers” (`s:rf.coupler`), bmad/doc/attributes.tex:2797–2828](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L2797-L2828) |

Inserting a midpoint marker into an rfcavity element changes its transverse coupler kicks. With couplers enabled at both ends or only at the exit, the split cavity produces horizontal normalized momentum pₓ = 0.0001618 instead of 0.00008090; with couplers disabled, it produces the same nonzero kick instead of zero. Slicing chooses couplers from geometric edge alignment without respecting the enabled ends, and loses the half-strength factor for a both-end coupler. Preserving the physical ends and their strengths must preserve the kick.

Run `python LAT-28.py` with [_bmad.py](../helpers/_bmad.py) and [_multipass_slicing.py](../helpers/_multipass_slicing.py) in the same directory.

??? example "Reproducer: LAT-28.py"

    [Download LAT-28.py](LAT-28.py)

    ```python
    """LAT-28: RF slicing invents couplers and doubles both-end kicks

    Inserting a midpoint marker into an rfcavity element changes its transverse coupler kicks. With couplers enabled at both ends or only at the exit, the split cavity produces horizontal normalized momentum pₓ = 0.0001618 instead of 0.00008090; with couplers disabled, it produces the same nonzero kick instead of zero. Slicing chooses couplers from geometric edge alignment without respecting the enabled ends, and loses the half-strength factor for a both-end coupler. Preserving the physical ends and their strengths must preserve the kick.

    The script compares split and whole RF cavities for both-end, exit-only and disabled couplers. It also evaluates individual edge kicks at a fixed phase against the analytic gradient-times-strength formula, separating location and normalization errors from orbit-dependent phase changes.
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
    if (lord%key /= rfcavity$ .or. lord%n_slave /= 2) stop 2
    call init_coord(start,lat%particle_start,lat%ele(1),upstream_end$)
    orb=start
    do i=1,lat%n_ele_track
      call track1(orb,lat%ele(i),lat%param,ending,err_flag=err)
      if (err .or. ending%state /= alive$) stop 3
      orb=ending
    enddo
    call row('split_px',[orb%vec(2)])
    call init_coord(start,lat%particle_start,lord,upstream_end$)
    call track1(start,lord,lat%param,whole,err_flag=err)
    if (err .or. whole%state /= alive$) stop 4
    call row('whole_px',[whole%vec(2)])
    ! Evaluate each edge at identical phase and zero transverse position: no orbit-induced energy kick.
    do j=1,2
      if (j==1) then
        slave=>pointer_to_slave(lord,1)
      else
        slave=>pointer_to_slave(lord,lord%n_slave)
      endif
      if (slave%key /= rfcavity$) stop 5
      v=0
      call init_coord(orb,v,lord,upstream_end$)
      call rf_coupler_kick(lord,lat%param,first_track_edge$+(j-1),pi/2,orb)
      write(label,'(a,i0)') 'whole_edge',j
      call row(label,[orb%vec(2)])
      call init_coord(orb,v,slave,upstream_end$)
      call rf_coupler_kick(slave,lat%param,first_track_edge$+(j-1),pi/2,orb)
      write(label,'(a,i0)') 'slice_edge',j
      call row(label,[orb%vec(2)])
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
    r: rfcavity, l=1, voltage=1e6, rf_frequency=1e6, phi0=0.25, coupler_strength=0.1, coupler_phase=0.1, coupler_at=both_ends
    m: marker, superimpose, ref=r
    lat: line=(r)
    use, lat
    """
    # At phase pi/2 the RFCAVITY phase conversion leaves 2*pi*coupler_phase.
    # Each enabled end gets its share of gradient*strength*cos(phase)/p0c.
    kick = 1e6 * 0.1 * math.cos(2 * math.pi * 0.1) / 1e9
    defect = False
    for mode, weights in (("both_ends", [0.5, 0.5]), ("exit_end", [0, 1]), ("no_end", [0, 0])):
        data = rows(
            fortran(
                PROGRAM,
                files={"lat.bmad": LATTICE.replace("coupler_at=both_ends", f"coupler_at={mode}")},
            ).log
        )
        for j, weight in enumerate(weights, 1):
            expected = [kick * weight]
            control(
                f"{mode} whole edge {j}: analytic px",
                data[f"whole_edge{j}"],
                expected,
                atol=2e-18,
                rtol=2e-14,
            )
            defect |= compare(
                f"{mode} slice edge {j}: analytic px",
                data[f"slice_edge{j}"],
                expected,
                atol=2e-18,
                rtol=2e-14,
            )
        if mode == "no_end":
            control("disabled whole-cavity px", data["whole_px"], [0], atol=0)
        # Whole tracking retains the physical end locations and strengths, including phase evolution.
        defect |= compare(
            f"{mode} tracked split px, whole-cavity expectation",
            data["split_px"],
            data["whole_px"],
            atol=2e-13,
            rtol=0,
        )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

