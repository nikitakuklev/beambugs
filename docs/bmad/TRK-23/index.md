---
title: "TRK-23: Legacy lcavity tracking skips reference-momentum changes at zero length and in backward time"
hide:
  - toc
---

# TRK-23 — Legacy lcavity tracking skips reference-momentum changes at zero length and in backward time

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [track_a_lcavity_old, bmad/low_level/track_a_lcavity_old.f90:449–490](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity_old.f90#L449-L490)<br>[ele_reference_energy_correction, bmad/low_level/ele_reference_energy_correction.f90:47–54](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/ele_reference_energy_correction.f90#L47-L54)<br>[Bmad Manual, tracking and time direction, bmad/doc/tracking.tex:50–77](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/tracking.tex#L50-L77) |

The legacy linear-cavity tracker selected by n_rf_steps = 0 mishandles reference momentum at zero length and in backward time. A zero-length 2 MV cavity leaves a 5 MeV/c particle unaccelerated and retains the 5 MeV/c reference instead of the required 7.007 MeV/c exit reference. After a powered finite cavity is tracked forward and then backward, the returned particle has momentum 4.178 MeV/c instead of its initial 5 MeV/c and retains the exit reference. The stepped tracker restores the initial orbit and reference in both powered and off finite-cavity round trips.

Run `python TRK-23.py` with [_bmad.py](../helpers/_bmad.py) and [_rf_and_spin.py](../helpers/_rf_and_spin.py) in the same directory.

??? example "Reproducer: TRK-23.py"

    [Download TRK-23.py](TRK-23.py)

    ```python
    """TRK-23: Legacy lcavity tracking skips reference-momentum changes at zero length and in backward time

    The legacy linear-cavity tracker selected by n_rf_steps = 0 mishandles reference momentum at zero length and in backward time. A zero-length 2 MV cavity leaves a 5 MeV/c particle unaccelerated and retains the 5 MeV/c reference instead of the required 7.007 MeV/c exit reference. After a powered finite cavity is tracked forward and then backward, the returned particle has momentum 4.178 MeV/c instead of its initial 5 MeV/c and retains the exit reference. The stepped tracker restores the initial orbit and reference in both powered and off finite-cavity round trips.

    The script tests forward/backward transport and zero-length acceleration with the legacy tracker, using BMAD_DIST. Relativistic energy addition and the RF phase give independent reference and particle momenta for the thin cavity. Stepped tracking supplies powered and off controls for thin acceleration and finite-cavity round trips.
    """

    import math

    from _bmad import compare, fortran, verdict
    from _rf_and_spin import control, rows

    PROGRAM = """program legacy
    use bmad
    implicit none
    type(lat_struct),target::lat
    type(ele_struct),pointer::cav
    type(coord_struct)::a,b,c
    real(rp)::v(6)
    integer k
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) error stop 'parse'
    cav=>lat%ele(1)
    do k=1,2
     cav%is_on=(k==1)
     call set_flags_for_changed_attribute(cav,cav%is_on)
     call lattice_bookkeeper(lat,err)
     if(err) error stop 'bookkeeping'
     v=0; v(1)=1e-3_rp; v(5)=0.003_rp
     call init_coord(a,v,cav,upstream_end$,electron$)
     call track1(a,cav,lat%param,b,err_flag=err)
     if(err .or. b%state/=alive$) error stop 'forward track'
     if(cav%value(l$)==0) then
      write(*,'(a,i0,3es25.16)') 'RESULT thin',k,b%p0c,cav%value(p0c$),b%p0c*(1+b%vec(6))
      cycle
     endif
     b%time_dir=-1
     call track1(b,cav,lat%param,c,err_flag=err)
     if(err .or. c%state/=alive$) error stop 'backward track'
     write(*,'(a,i0,*(es25.16,1x))') 'RESULT case',k,b%p0c,cav%value(p0c$), &
     c%p0c,a%p0c,b%p0c*(1+b%vec(6)),c%p0c*(1+c%vec(6)),c%vec-a%vec
    enddo
    end program
    """
    CASES = [
        (
            "stepped thin",
            """no_digested
    beginning[p0c]=5e6
    parameter[particle]=electron
    parameter[geometry]=open
    c: lcavity, l=0, voltage=2e6, rf_frequency=1.3e9, n_rf_steps=10, cavity_type=traveling_wave
    ln: line=(c)
    use, ln
    """,
        ),
        (
            "stepped",
            """no_digested
    beginning[p0c]=5e6
    parameter[particle]=electron
    parameter[geometry]=open
    c: lcavity, l=1, voltage=2e6, rf_frequency=1.3e9, n_rf_steps=10
    ln: line=(c)
    use, ln
    """,
        ),
        (
            "legacy finite",
            """no_digested
    beginning[p0c]=5e6
    parameter[particle]=electron
    parameter[geometry]=open
    c: lcavity, l=1, voltage=2e6, rf_frequency=1.3e9, n_rf_steps=0
    ln: line=(c)
    use, ln
    """,
        ),
        (
            "legacy thin",
            """no_digested
    beginning[p0c]=5e6
    parameter[particle]=electron
    parameter[geometry]=open
    c: lcavity, l=0, voltage=2e6, rf_frequency=1.3e9, n_rf_steps=0, cavity_type=traveling_wave
    ln: line=(c)
    use, ln
    """,
        ),
    ]
    # Relativistic energy addition fixes the nominal reference independently of
    # lattice bookkeeping. The particle at z=3 mm sees the RF phase -omega*z/(beta*c).
    mass = 510998.95069  # Electron rest energy in eV, the pin's species convention.
    initial_energy = math.hypot(5e6, mass)
    exit_reference = math.sqrt((initial_energy + 2e6) ** 2 - mass**2)
    beta = 5e6 / initial_energy
    phase = 2 * math.pi * 1.3e9 * 0.003 / (beta * 299792458)
    thin_momentum = math.sqrt((initial_energy + 2e6 * math.cos(phase)) ** 2 - mass**2)
    defect = False
    for name, lattice in CASES:
        d = rows(fortran(PROGRAM, files={"lat.bmad": lattice}).log)
        keys = {"thin1", "thin2"} if name.endswith("thin") else {"case1", "case2"}
        if set(d) != keys:
            raise ValueError(f"Incomplete legacy-cavity cases: {sorted(d)}")
        for key, r in d.items():
            state = "on" if key.endswith("1") else "off"
            control(
                f"{name} {state} lattice exit reference, relativistic energy addition (eV/c)",
                r[1],
                exit_reference,
                atol=1e-8,
            )
            if name.endswith("thin"):
                target = thin_momentum if state == "on" else 5e6
                if name == "stepped thin":
                    control(f"{name} {state} exit reference (eV/c)", r[0], exit_reference, atol=1e-8)
                    control(
                        f"{name} {state} physical momentum, analytic RF kick (eV/c)",
                        r[2],
                        target,
                        atol=1e-8,
                    )
                else:
                    defect |= compare(
                        f"{name} {state} exit reference (eV/c)", r[0], exit_reference, atol=1e-8
                    )
                    if state == "on":
                        defect |= compare(
                            f"{name} {state} physical momentum, analytic RF kick (eV/c)",
                            r[2],
                            target,
                            atol=1e-8,
                        )
                    else:
                        control(f"{name} {state} physical momentum (eV/c)", r[2], target, atol=1e-8)
                continue
            if name == "stepped":
                control(f"{name} {state} exit reference (eV/c)", r[0], r[1], atol=1e-8)
                control(f"{name} {state} return reference (eV/c)", r[2], r[3], atol=1e-8)
                control(f"{name} {state} orbit round trip", r[6:], [0] * 6, atol=3e-15)
            else:
                defect |= compare(f"{name} {state} exit reference (eV/c)", r[0], r[1], atol=1e-8)
                defect |= compare(f"{name} {state} return reference (eV/c)", r[2], r[3], atol=1e-8)
                print(
                    f"{name} {state} forward, return physical momentum (eV/c)={r[4:6].tolist()}; initial=5000000"
                )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-22 | lcavity-is-on (`1df84a7745db`), source, production, CPU, double, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |

