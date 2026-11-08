---
title: "TRK-54: Foils apply full-thickness scattering at every integration step"
hide:
  - toc
---

# TRK-54 — Foils apply full-thickness scattering at every integration step

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [track_a_foil scattering loop](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_foil.f90#L58-L109)<br>[energy-loss subdivision](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_foil.f90#L114-L135)<br>[Bmad Manual, “Scattering in a Foil” (s:foil.scatter), Eq. sszpb](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/charged-tracking.tex#L1032-L1097)<br>[“ds_step and num_steps Parameters” (s:ds.step)](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L664-L677)<br>[foil defaults](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/set_ele_defaults.f90#L180-L188) |

A foil reapplies its full-thickness scattering kick at every integration subdivision instead of sharing the scattering variance across the foil. For a 150 MeV/c proton in carbon with areal density 0.012 kg/m², deterministic unit draws give a normalized transverse kick of 0.002255 with one step and 0.02259 with the default ten steps. Independent integration of the Highland scattering and mean energy-loss formulas reproduces those repeated full-material kicks. For independent random draws and small energy loss, the loop implies approximately ten times the intended variance at the default setting.

Run `python TRK-54.py` with [_bmad.py](../helpers/_bmad.py) and [_tracking_contracts_materials.py](../helpers/_tracking_contracts_materials.py) in the same directory.

??? example "Reproducer: TRK-54.py"

    [Download TRK-54.py](TRK-54.py)

    ```python
    """TRK-54: Foils apply full-thickness scattering at every integration step

    A foil reapplies its full-thickness scattering kick at every integration subdivision instead of
    sharing the scattering variance across the foil. For a 150 MeV/c proton in carbon with areal
    density 0.012 kg/m², deterministic unit draws give a normalized transverse kick of 0.002255 with
    one step and 0.02259 with the default ten steps. Independent integration of the Highland
    scattering and mean energy-loss formulas reproduces those repeated full-material kicks. For
    independent random draws and small energy loss, the loop implies approximately ten times the
    intended variance at the default setting.

    The script compiles a deterministic foil driver using scatter_test and compares one, three and
    ten steps. A separate integration evaluates the manual formulas. An equal-variance reference
    retains the whole-foil logarithmic correction and energy history while dividing the variance
    of each draw among the steps.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _tracking_contracts_materials import rows

    PROGRAM = """
    program foil_scattering
    use bmad
    implicit none
    type(lat_struct),target::lat
    type(ele_struct),pointer::e
    type(coord_struct)::a,b
    real(rp) area,x0,amat,excit,ne,pc,beta,px,loss,en,pcnew,sigma,xx,whole
    integer i,n,ns,zz
    logical err
    character(20) name
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) error stop 'parse failed'
    e=>lat%ele(1)
    area=e%foil%material(1)%area_density_used
    x0=e%foil%material(1)%radiation_length_used
    zz=atomic_number(e%foil%material(1)%species)
    amat=mass_of(e%foil%material(1)%species)/atomic_mass_unit
    excit=mean_excitation_energy_over_z(zz)*zz
    ne=1e3_rp*N_avogadro*area*zz/amat
    write(*,'(a,5es26.17e3)') 'RESULT material ',area,x0,amat,excit,ne
    do i=1,3
     e=>lat%ele(i)
     call init_coord(a,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.023_rp,0.0_rp],e,upstream_end$)
     call track1(a,e,lat%param,b,err_flag=err)
     if(err.or.b%state/=alive$) error stop 'track failed'
     pc=a%p0c;beta=a%beta;px=0
     n=nint(e%value(num_steps$))
     do ns=1,n
      ! Highland sigma uses total areal density; repeated full-density kicks are the defect hypothesis.
      xx=area/x0
      sigma=13.6e6_rp*sqrt(xx)/(pc*beta)*(1+0.038_rp*log(xx/beta**2))
      if(ns==1) whole=sigma
      px=px+sigma*pc/a%p0c
      ! Bethe mean loss, with energy divided into n steps and direction preserved after each loss.
      loss=-fourpi*mass_of(electron$)*ne*(r_e/beta)**2 * &
           (log(2*mass_of(electron$)*(pc/mass_of(a%species))**2/excit)-beta**2)
      en=sqrt(pc**2+mass_of(a%species)**2)+loss/n
      pcnew=sqrt((en-mass_of(a%species))*(en+mass_of(a%species)))
      px=px*pcnew/pc
      pc=pcnew;beta=pc/en
     enddo
     write(name,'(a,i0)') 'steps',n
     write(*,'(3a,6es26.17e3)') 'RESULT ',trim(name),' ',b%vec(2),px,b%beta,pc,whole, &
        b%p0c*(1+b%vec(6))
    enddo
    end program
    """
    LAT = """no_digested
    parameter[geometry]=open
    beginning[beta_a]=10
    beginning[beta_b]=10
    parameter[particle]=proton
    parameter[p0c]=150e6
    f1: foil,material_type="C",area_density=.012,scatter_test=T,num_steps=1,final_charge=1
    f2: f1,num_steps=3
    f3: foil,material_type="C",area_density=.012,scatter_test=T,final_charge=1
    bl:line=(f1,f2,f3)
    use,bl
    """
    records = rows(fortran(PROGRAM, files={"lat.bmad": LAT}).log)
    if set(records) != {"material", "steps1", "steps3", "steps10"}:
        raise ValueError("Missing foil records or changed default step count")
    print(
        "material (area kg/m2, radiation length kg/m2, atomic mass, excitation eV, electrons/m2):",
        records["material"].tolist(),
    )
    defects = []
    for n in (1, 3, 10):
        actual, repeated, beta, pc, whole, tracked_pc = records[f"steps{n}"]
        if compare(f"steps={n}: energy-loss control pc (eV/c)", tracked_pc, pc, atol=2e-6):
            raise AssertionError("Energy-loss integration control failed")
        bad = compare(
            f"steps={n}: px, repeated full-material analytic prediction", actual, repeated, atol=2e-15
        )
        if n == 1 and bad:
            raise AssertionError("Single-step Highland control failed")
        # Unit draws add coherently. A variance-preserving n-step partition would scale
        # each full-foil width by 1/sqrt(n), not repeat that width n times.
        normalized = actual / whole
        print(
            f"steps={n}: px/entrance whole-foil sigma={normalized:.16g}; repeated-full prediction near {n}; variance-preserving unit-draw scale near {np.sqrt(n):.16g}"
        )
        if n > 1:
            # This equal-variance reference retains the whole-foil logarithmic
            # correction and the independently integrated energy history. Each
            # angular draw receives 1/n of the full-foil variance.
            partitioned = repeated / np.sqrt(n)
            discrepancy = compare(
                f"steps={n}: px, equal-variance partition reference",
                actual,
                partitioned,
                atol=2e-15,
            )
            defects.append(not bad and discrepancy)
    verdict(all(defects))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

