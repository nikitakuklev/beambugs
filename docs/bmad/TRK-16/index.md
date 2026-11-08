---
title: "TRK-16: DC gun autoscaling restores the old amplitude"
hide:
  - toc
---

# TRK-16 — DC gun autoscaling restores the old amplitude

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [autoscale_phase_and_amp, DC branch, bmad/low_level/autoscale_phase_and_amp.f90:277–305](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/autoscale_phase_and_amp.f90#L277-L305)<br>[Bmad Manual, Automatic Phase and Amplitude Scaling of RF Fields, `s:autoscale`, bmad/doc/attributes.tex:2876–2925](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L2876-L2925) |

A zero-frequency electron gun with amplitude autoscaling enabled reports success but keeps its original field scale. The converged scale is overwritten when saved element attributes are restored. A 0.1 m constant field of −500.0 kV/m gives 50.00 keV instead of the requested 100.0 keV; integrating the electric field requires a scale of two.

Run `python TRK-16.py` with [_bmad.py](../helpers/_bmad.py) and [_rf_and_spin.py](../helpers/_rf_and_spin.py) in the same directory.

??? example "Reproducer: TRK-16.py"

    [Download TRK-16.py](TRK-16.py)

    ```python
    """TRK-16: DC gun autoscaling restores the old amplitude

    A zero-frequency electron gun with amplitude autoscaling enabled reports success but keeps its original field scale. The converged scale is overwritten when saved element attributes are restored. A 0.1 m constant field of −500.0 kV/m gives 50.00 keV instead of the requested 100.0 keV; integrating the electric field requires a scale of two.

    The script compiles and tracks a constant grid field and an analytic-field gun, using BMAD_DIST. It checks successful autoscaling and compares the retained amplitude and energy gain with the electric-field integral. Setting the required scale explicitly supplies a passing acceleration control.
    """

    from _bmad import compare, fortran, verdict
    from _rf_and_spin import control, rows

    PROGRAM = """program gun_probe
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: ele
    type(coord_struct) :: a,b
    logical :: err
    real(rp) :: e0,e1,required_scale
    character(80) :: arg
    call get_command_argument(1,arg)
    read(arg,*) required_scale
    call bmad_parser('gun.bmad',lat,err_flag=err)
    if (err) error stop 'parse'
    ele=>lat%ele(1)
    if (.not. is_true(ele%value(autoscale_amplitude$))) error stop 'autoscale disabled'
    write(*,'(a,es25.16)') 'RESULT initial_scale ',ele%value(field_autoscale$)
    call autoscale_phase_and_amp(ele,lat%param,err,call_bookkeeper=.false.)
    if (err) error stop 'autoscale'
    write(*,'(a,es25.16)') 'RESULT scale ',ele%value(field_autoscale$)
    call init_coord(a,ele=ele,element_end=upstream_end$)
    call track1(a,ele,lat%param,b,err_flag=err)
    if (err .or. b%state /= alive$) error stop 'track'
    e0=hypot(a%p0c*(1+a%vec(6)),mass_of(a%species))
    e1=hypot(b%p0c*(1+b%vec(6)),mass_of(b%species))
    write(*,'(a,2es25.16)') 'RESULT gain ',e1-e0,ele%value(voltage$)
    ! Hold the analytic field integral's required amplitude fixed and track again.
    ! This separates a discarded autoscale result from an invalid field or voltage.
    ele%value(field_autoscale$)=required_scale
    bmad_com%rel_tol_adaptive_tracking=1e-12_rp
    bmad_com%abs_tol_adaptive_tracking=1e-14_rp
    call track1(a,ele,lat%param,b,err_flag=err)
    if (err .or. b%state /= alive$) error stop 'manual-scale control track'
    e1=hypot(b%p0c*(1+b%vec(6)),mass_of(b%species))
    write(*,'(a,es25.16)') 'RESULT control_gain ',e1-e0
    end program
    """
    GRID = """{
    geometry = xyz,
    r0 = (0,0,0),
    dr = (0.01,0.01,0.05),
    harmonic = 0,
    field_scale = 1,
    master_parameter = none,
    ele_anchor_pt = beginning,
    pt(-1,-1,0) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(-1,-1,1) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(-1,-1,2) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(-1,0,0) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(-1,0,1) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(-1,0,2) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(-1,1,0) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(-1,1,1) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(-1,1,2) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(0,-1,0) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(0,-1,1) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(0,-1,2) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(0,0,0) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(0,0,1) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(0,0,2) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(0,1,0) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(0,1,1) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(0,1,2) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(1,-1,0) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(1,-1,1) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(1,-1,2) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(1,0,0) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(1,0,1) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(1,0,2) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(1,1,0) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(1,1,1) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    pt(1,1,2) = ((0,0),(0,0),(-500000,0),(0,0),(0,0),(0,0)),
    }
    """
    CASES = [
        (
            "grid",
            """no_digested
    parameter[particle] = electron
    parameter[p0c] = 0
    parameter[geometry] = open
    gun: e_gun, l = 0.1, voltage = 1e5, rf_frequency = 0, autoscale_phase = F, field_calc = fieldmap, grid_field = call::dc_grid.bmad
    ln: line = (gun)
    use, ln
    """,
            1.0,
            2.0,
        ),
        (
            "analytic",
            """no_digested
    parameter[particle] = electron
    parameter[p0c] = 0
    parameter[geometry] = open
    gun: e_gun, l = 0.1, voltage = 1e5, rf_frequency = 0, field_autoscale = 0.5
    ln: line = (gun)
    use, ln
    """,
            0.5,
            1.0,
        ),
    ]

    defect = False
    for name, lattice, initial, expected_scale in CASES:
        d = rows(
            fortran(
                PROGRAM, [str(expected_scale)], files={"gun.bmad": lattice, "dc_grid.bmad": GRID}
            ).log
        )
        print(
            f"{name} parser scale={d['initial_scale'][0]:.16g}; requested scale={expected_scale:.16g}"
        )
        # For the grid q*Ez*L = 50000 eV; analytic unit scale gives 100000 eV.
        control(f"{name} requested voltage (eV)", d["gain"][1], 100000, atol=1e-10)
        control(
            f"{name} gain at required scale (eV), field integral",
            d["control_gain"][0],
            100000,
            atol=2e-9,
        )
        defect |= compare(
            f"{name} retained scale, field integral", d["scale"][0], expected_scale, atol=1e-10
        )
        defect |= compare(f"{name} gain (eV), requested voltage", d["gain"][0], d["gain"][1], atol=1e-5)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

