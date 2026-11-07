---
title: "TRK-55: Foil energy loss leaves longitudinal position inconsistent with arrival time"
hide:
  - toc
---

# TRK-55 — Foil energy loss leaves longitudinal position inconsistent with arrival time

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [track_a_foil energy update](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_foil.f90#L130-L135)<br>[Bmad Manual, “Charged Particle Phase Space Coordinates,” Eq. zbctt](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/phase-space.tex#L80-L94)<br>[“Foil” (s:foil)](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L2445-L2450) |

A foil changes particle speed through energy loss but leaves its longitudinal coordinate z unchanged. Since z = −βc(t − t₀), with β the speed divided by the speed of light c and t − t₀ the arrival-time offset, unchanged arrival and reference times require z to scale by the final-to-initial speed ratio. A 150 MeV/c proton entering a carbon foil at z = 23 mm exits with z = 23 mm instead of 22.96 mm, an error of 40.58 μm. The stored time and independently reconstructed absolute-time coordinate identify the omitted z update.

Run `python TRK-55.py` with [_bmad.py](../helpers/_bmad.py) and [_tracking_contracts_materials.py](../helpers/_tracking_contracts_materials.py) in the same directory.

??? example "Reproducer: TRK-55.py"

    [Download TRK-55.py](TRK-55.py)

    ```python
    """TRK-55: Foil energy loss leaves longitudinal position inconsistent with arrival time

    A foil changes particle speed through energy loss but leaves its longitudinal coordinate z
    unchanged. Since z = −βc(t − t₀), with β the speed divided by the speed of light c and t − t₀
    the arrival-time offset, unchanged arrival and reference times require z to scale by the
    final-to-initial speed ratio. A 150 MeV/c proton entering a carbon foil at z = 23 mm exits with
    z = 23 mm instead of 22.96 mm, an error of 40.58 μm. The stored time and independently
    reconstructed absolute-time coordinate identify the omitted z update.

    The script tracks a fixed proton through a five-step carbon foil with scattering disabled. It
    verifies the unchanged times and refreshed speed, then compares z with both the speed-ratio
    prediction and the absolute-time identity.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _tracking_contracts_materials import rows

    PROGRAM = """
    program foil_clock
    use bmad
    implicit none
    type(lat_struct) lat
    type(coord_struct) a,b
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) error stop 'parse failed'
    call init_coord(a,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.023_rp,0.0_rp],lat%ele(1),upstream_end$)
    call track1(a,lat%ele(1),lat%param,b,err_flag=err)
    if(err.or.b%state/=alive$) error stop 'track failed'
    write(*,'(a,12es26.17e3)') 'RESULT clock ',a%beta,b%beta,a%t,b%t,a%vec(5),b%vec(5), &
     lat%ele(1)%ref_time,lat%ele(1)%value(delta_ref_time$), &
     b%p0c*(1+b%vec(6)),m_proton,c_light,lat%ele(1)%value(l$)
    end program
    """
    LAT = """no_digested
    parameter[geometry]=open
    beginning[beta_a]=10
    beginning[beta_b]=10
    parameter[particle]=proton
    parameter[p0c]=150e6
    e:foil,material_type="C",area_density=.012,scatter_method=off,num_steps=5,final_charge=1
    bl:line=(e)
    use,bl
    """
    b0, b1, t0, t1, z0, z1, tref, dtref, pc, mass, c, length = rows(
        fortran(PROGRAM, files={"lat.bmad": LAT}).log
    )["clock"]
    if compare("zero-length unchanged-clock control", [t1 - t0, dtref, length], [0, 0, 0], atol=0):
        raise AssertionError("Foil clock control failed")
    if not (0 < b1 < b0):
        raise AssertionError("Expected positive speed reduced by energy loss")
    if compare("final beta control, relativistic pc/E", b1, pc / np.hypot(pc, mass), atol=3e-16):
        raise AssertionError("Final speed control failed")
    # At fixed t and t0 the coordinate identity makes z proportional to particle beta.
    expected = z0 * b1 / b0
    absolute = -b1 * c * (t1 - tref)
    if compare("two independent clock reconstructions (m)", expected, absolute, atol=2e-17):
        raise AssertionError("Clock oracles disagree")
    print(f"beta before={b0:.17g} after={b1:.17g}; stored times before={t0:.17g} after={t1:.17g}")
    verdict(compare("foil z (m), speed-ratio identity", z1, expected, atol=2e-17))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

