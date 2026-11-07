---
title: "TRK-46: Aperture header states an obsolete transverse-momentum limit"
hide:
  - toc
---

# TRK-46 — Aperture header states an obsolete transverse-momentum limit

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [check_aperture_limit header](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/check_aperture_limit.f90#L4-L20)<br>[momentum check before geometric apertures](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/check_aperture_limit.f90#L57-L63)<br>[charged-particle momentum guard](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/orbit_too_large.f90#L85-L118) |

The aperture-check header promises particle loss when either |pₓ| or |pᵧ| exceeds one, where pₓ and pᵧ are transverse momenta divided by the reference momentum. The routine instead requires their squared sum to stay at or below (1 + δ)², where δ is the relative total-momentum offset. A particle with pₓ = 1.1 and δ = 0.2 survives, while pₓ = pᵧ = 0.8 at δ = 0 is lost, in agreement with the requirement for real longitudinal momentum. With default momentum checking, this physical guard remains active when geometric apertures are disabled.

Run `python TRK-46.py` with [_bmad.py](../helpers/_bmad.py) and [_tracking_contracts_materials.py](../helpers/_tracking_contracts_materials.py) in the same directory.

??? example "Reproducer: TRK-46.py"

    [Download TRK-46.py](TRK-46.py)

    ```python
    """TRK-46: Aperture header states an obsolete transverse-momentum limit

    The aperture-check header promises particle loss when either |pₓ| or |pᵧ| exceeds one, where
    pₓ and pᵧ are transverse momenta divided by the reference momentum. The routine instead
    requires their squared sum to stay at or below (1 + δ)², where δ is the relative
    total-momentum offset. A particle with pₓ = 1.1 and δ = 0.2 survives, while pₓ = pᵧ = 0.8 at
    δ = 0 is lost, in agreement with the requirement for real longitudinal momentum. With
    default momentum checking, this physical guard remains active when geometric apertures are
    disabled.

    The script calls check_aperture_limit for a small-momentum control and the two counterexamples,
    with geometric apertures both enabled and disabled. It compares the results with the combined
    momentum criterion and the header text from BMAD_SOURCE."""

    import numpy as np
    from _bmad import compare, fortran, source, verdict
    from _tracking_contracts_materials import rows

    PROGRAM = """
    program aperture_contract
    use bmad
    implicit none
    type(ele_struct) ele
    type(coord_struct) orb
    type(lat_param_struct) param
    real(rp) v(6)
    integer i,j
    character(20) name
    call init_ele(ele,drift$)
    ele%ref_species=electron$
    ele%value(p0c$)=1e6_rp
    ele%value(p0c_start$)=1e6_rp
    do j=0,1
     bmad_com%aperture_limit_on=(j==1)
     do i=1,3
      v=0
      if(i==1) v(2)=.1_rp
      if(i==2) then
       v(2)=1.1_rp; v(6)=.2_rp
      endif
      if(i==3) then
       v(2)=.8_rp; v(4)=.8_rp
      endif
      ! Assign the deliberately inadmissible case after init_coord: only the guard is tested.
      call init_coord(orb, [0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp], ele, upstream_end$, electron$)
      orb%vec=v
      call check_aperture_limit(orb,ele,first_track_edge$,param)
      write(name,'(a,i0,a,i0)') 'case',i,'_',j
      write(*,'(3a,i3)') 'RESULT ',trim(name),' ',merge(1,0,orb%state==alive$)
     enddo
    enddo
    end program
    """
    records = rows(fortran(PROGRAM).log)
    header = (source() / "bmad/low_level/check_aperture_limit.f90").read_text().split("!-", 1)[0]
    defects = []
    # A real longitudinal momentum requires |p_perp| <= p. The header's
    # component bound is a sufficient loss condition, not an exhaustive rule.
    momenta = np.array([[0.1, 0, 1], [1.1, 0, 1.2], [0.8, 0.8, 1]])
    expected_alive = (np.linalg.norm(momenta[:, :2], axis=1) <= momenta[:, 2]).astype(int)
    for enabled in (0, 1):
        observed = np.array([records[f"case{i}_{enabled}"][0] for i in (1, 2, 3)])
        if compare(
            f"apertures={enabled}: real-longitudinal-momentum control, alive flags",
            observed,
            expected_alive,
            atol=0,
        ):
            raise AssertionError("Combined-momentum guard changed")
        defects.append(
            compare(
                f"apertures={enabled}: px=1.1, delta=0.2; header promises loss",
                observed[1],
                0,
                atol=0,
            )
        )
    verdict("|p_x| or |p_y| > 1" in header and any(defects))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

