---
title: "TRK-22: PTC fibre update does not refresh cavity geometry"
hide:
  - toc
---

# TRK-22 — PTC fibre update does not refresh cavity geometry

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [update_fibre_from_ele length update, bmad/ptc/update_fibre_from_ele.f90:68–74](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/update_fibre_from_ele.f90#L68-L74)<br>[cavity update, bmad/ptc/update_fibre_from_ele.f90:151–170](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/update_fibre_from_ele.f90#L151-L170)<br>[cavity creation, bmad/ptc/ele_to_fibre.f90:384–416](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/ele_to_fibre.f90#L384-L416) |

Changing a cavity’s geometry in an existing PTC layout produces a different orbit from rebuilding that same layout. The update changes some length fields but leaves geometry derived during fibre creation stale. At zero voltage, increasing a traveling-wave linear cavity from 0.5 to 0.55 m leaves a 50.00 mm longitudinal discrepancy. Increasing a standing-wave cavity’s frequency from 500 to 515 MHz changes its active length and produces an 8.732 mm discrepancy.

Run `python TRK-22.py` with [_bmad.py](../helpers/_bmad.py) and [_rf_and_spin.py](../helpers/_rf_and_spin.py) in the same directory.

??? example "Reproducer: TRK-22.py"

    [Download TRK-22.py](TRK-22.py)

    ```python
    """TRK-22: PTC fibre update does not refresh cavity geometry

    Changing a cavity’s geometry in an existing PTC layout produces a different orbit from rebuilding that same layout. The update changes some length fields but leaves geometry derived during fibre creation stale. At zero voltage, increasing a traveling-wave linear cavity from 0.5 to 0.55 m leaves a 50.00 mm longitudinal discrepancy. Increasing a standing-wave cavity’s frequency from 500 to 515 MHz changes its active length and produces an 8.732 mm discrepancy.

    The script compares an updated PTC layout with a freshly rebuilt copy after changing cavity length or standing-wave frequency, using BMAD_DIST. The cavities have zero voltage to isolate geometry from RF phase; an unchanged-attribute update is the control.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _rf_and_spin import control, rows

    PROGRAM = """program compare_update
    use bmad
    use ptc_layout_mod, only: lat_to_ptc_layout, ptc_track_all, kill_ptc_layouts
    use s_tracking, only: check_stable
    implicit none
    type(lat_struct), target :: lat
    type(coord_struct), allocatable :: orbit(:)
    type(ele_struct), pointer :: ele
    character(300) :: filename, action
    logical :: err
    real(rp), allocatable :: updated(:,:), rebuilt(:,:), state_update(:,:), state_rebuilt(:,:)
    integer :: i, track_state
    call get_command_argument(1, filename)
    call get_command_argument(2, action)
    call bmad_parser(trim(filename), lat, err_flag=err)
    if (err) error stop 'parse'
    call lat_to_ptc_layout(lat)
    ele => lat%ele(1)
    select case(trim(action))
    case('same')
      call set_flags_for_changed_attribute(ele, ele%value(voltage$))
    case('frequency')
      ele%value(rf_frequency$) = 1.03_rp * ele%value(rf_frequency$)
      call set_flags_for_changed_attribute(ele, ele%value(rf_frequency$))
    case('length')
      ele%value(l$) = 1.1_rp * ele%value(l$)
      call set_flags_for_changed_attribute(ele, ele%value(l$))
    case default
      error stop 'action'
    end select
    call lattice_bookkeeper(lat, err)
    if (err) error stop 'bookkeeping'
    allocate(orbit(0:lat%n_ele_track), updated(6,lat%n_ele_track), rebuilt(6,lat%n_ele_track))
    allocate(state_update(15,lat%n_ele_track), state_rebuilt(15,lat%n_ele_track))
    call measure(updated, state_update)
    call kill_ptc_layouts(lat)
    call lat_to_ptc_layout(lat)
    call measure(rebuilt, state_rebuilt)
    print '(a,es24.15)', 'MAX_ORBIT_DIFF ', maxval(abs(updated-rebuilt))
    print '(a,es24.15)', 'MAX_FIBRE_DIFF ', maxval(abs(state_update-state_rebuilt))
    do i=1,lat%n_ele_track
      block
        write(*,'(a,i0,6es25.16)') 'RESULT update',i, updated(:,i)
        write(*,'(a,i0,6es25.16)') 'RESULT rebuilt',i, rebuilt(:,i)
      end block
      block
        write(*,'(a,i0,15es25.16)') 'RESULT state_update',i, state_update(:,i)
        write(*,'(a,i0,15es25.16)') 'RESULT state_rebuilt',i, state_rebuilt(:,i)
      end block
    enddo
    contains
    subroutine measure(vecs, states)
    real(rp) :: vecs(:,:), states(:,:)
    integer :: j
    call init_coord(orbit(0), ele=lat%ele(0), element_end=downstream_end$)
    orbit(0)%vec = [1e-4_rp, 2e-5_rp, -2e-4_rp, 1e-5_rp, 3e-4_rp, 1e-3_rp]
    ! On this pin ptc_track_all leaves its optional err_flag true on success.
    ! The PTC stability flag, track state and each particle state validate the run.
    call ptc_track_all(lat%branch(0), orbit, track_state)
    if (.not. check_stable .or. track_state /= moving_forward$) error stop 'track'
    do j=1,lat%n_ele_track
      if (orbit(j)%state /= alive$) error stop 'lost'
      vecs(:,j) = orbit(j)%vec
      states(:,j) = 0
      associate(f => lat%ele(j)%ptc_fibre)
      states(1:6,j) = [f%mag%p%p0c, f%beta0, f%gamma0i, f%mag%p%ld, real(f%mag%p%nst,rp), real(f%mag%kind,rp)]
      if (associated(f%mag%volt)) states(7,j)=f%mag%volt
      if (associated(f%mag%lag)) states(8,j)=f%mag%lag
      if (associated(f%mag%freq)) states(9,j)=f%mag%freq
      if (associated(f%mag%h1)) states(10,j)=f%mag%h1
      if (associated(f%mag%h2)) states(11,j)=f%mag%h2
      if (associated(f%mag%bn)) states(12,j)=f%mag%bn(1)
      if (associated(f%mag%thin)) states(13,j)=merge(1.0_rp,0.0_rp,f%mag%thin)
      states(14,j)=f%mag%l
      states(15,j)=f%mag%p%lc
      end associate
    enddo
    end subroutine
    end program
    """
    CASES = [
        (
            "traveling",
            """no_digested
    parameter[particle]=electron
    parameter[p0c]=1e8
    parameter[geometry]=open
    beginning[beta_a]=10
    beginning[beta_b]=10
    c: lcavity, l=0.5, cavity_type=traveling_wave, num_steps=3, ds_step=0.16666666666666666, tracking_method=symp_lie_ptc, voltage=0, phi0=0, rf_frequency=5e8, autoscale_phase=F, autoscale_amplitude=F, phi0_autoscale=0
    q: quadrupole, l=0.2, k1=0.8, num_steps=3
    ln: line=(c,q)
    use, ln
    """,
            "length",
        ),
        (
            "standing",
            """no_digested
    parameter[particle]=electron
    parameter[p0c]=1e8
    parameter[geometry]=open
    beginning[beta_a]=10
    beginning[beta_b]=10
    c: lcavity, l=0.5, cavity_type=standing_wave, num_steps=5, ds_step=0.1, tracking_method=symp_lie_ptc, voltage=0, phi0=0, rf_frequency=5e8, autoscale_phase=F, autoscale_amplitude=F, phi0_autoscale=0
    q: quadrupole, l=0.2, k1=0.8, num_steps=3
    ln: line=(c,q)
    use, ln
    """,
            "frequency",
        ),
    ]
    defect = False
    for name, lattice, action in CASES:
        for change in ("same", action):
            d = rows(fortran(PROGRAM, ["lat.bmad", change], files={"lat.bmad": lattice}).log)
            if change == "same":
                control(f"{name} unchanged-update final orbit", d["update2"], d["rebuilt2"], atol=2e-13)
            else:
                defect |= compare(
                    f"{name} {change} final orbit (m, normalized momenta), fresh layout",
                    d["update2"],
                    d["rebuilt2"],
                    atol=2e-13,
                )
                print(
                    f"{name} {change} max orbit difference={np.max(np.abs(d['update2'] - d['rebuilt2'])):.16g}"
                )
                print(
                    f"{name} {change} fibre mag%l, p%lc: updated={d['state_update1'][13:].tolist()} rebuilt={d['state_rebuilt1'][13:].tolist()}"
                )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-22 | ptc-cavity-phase-update (`b48e02e3d9fa`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

