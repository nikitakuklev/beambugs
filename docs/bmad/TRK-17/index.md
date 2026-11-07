---
title: "TRK-17: RF and crab kicks use charge-to-mass instead of charge"
hide:
  - toc
---

# TRK-17 — RF and crab kicks use charge-to-mass instead of charge

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [track_a_rfcavity, bmad/low_level/track_a_rfcavity.f90:56–76](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_rfcavity.f90#L56-L76)<br>[track_a_crab_cavity, bmad/low_level/track_a_crab_cavity.f90:68–94](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_crab_cavity.f90#L68-L94)<br>[rel_tracking_charge_to_mass, bmad/code/rel_tracking_charge_to_mass.f90:27–32](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/rel_tracking_charge_to_mass.f90#L27-L32)<br>[Bmad Manual, RFCavity, bmad/doc/elements.tex:4874–4891](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L4874-L4891)<br>[Bmad Manual, Crab_Cavity, bmad/doc/elements.tex:1201–1222](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L1201-L1222) |

Tracking a different-mass species through an electron-reference RF or crab cavity adds an erroneous inverse-mass factor to the voltage kick. Equal-charge electrons and muons should receive equal energy changes from a crest RF voltage, and equal normalized transverse kicks at the same reference momentum. A 1 MV RF cavity gives a muon 4.836 keV instead of 1.000 MeV, and a crab cavity gives 0.0008503 instead of 0.1758 in normalized horizontal momentum. Electron controls agree with the charge-only formulas.

Run `python TRK-17.py` with [_bmad.py](../helpers/_bmad.py) and [_rf_and_spin.py](../helpers/_rf_and_spin.py) in the same directory.

??? example "Reproducer: TRK-17.py"

    [Download TRK-17.py](TRK-17.py)

    ```python
    """TRK-17: RF and crab kicks use charge-to-mass instead of charge

    Tracking a different-mass species through an electron-reference RF or crab cavity adds an erroneous inverse-mass factor to the voltage kick. Equal-charge electrons and muons should receive equal energy changes from a crest RF voltage, and equal normalized transverse kicks at the same reference momentum. A 1 MV RF cavity gives a muon 4.836 keV instead of 1.000 MeV, and a crab cavity gives 0.0008503 instead of 0.1758 in normalized horizontal momentum. Electron controls agree with the charge-only formulas.

    The script tracks electrons and muons through thin crest RF and crab cavities with an electron reference, using BMAD_DIST. Electron controls and charge-only voltage formulas establish the energy and normalized momentum expectations.
    """

    from _bmad import compare, fortran, verdict
    from _rf_and_spin import control, rows

    PROGRAM = """program charge_probe
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: ele
    type(coord_struct) :: a,b
    integer :: i, isp
    logical :: err
    real(rp) :: v(6),e0,e1,expected,phase
    call bmad_parser('probe.bmad',lat,err_flag=err)
    if (err) error stop 'parse'
    v=0
    v(1)=1e-3_rp
    do i=1,3,2
     ele=>lat%ele(i)
     ele%value(phi0$)=0.25_rp
     do isp=1,2
      if(isp==1) then
       call init_coord(a,v,ele,upstream_end$,particle=electron$)
      else
       call init_coord(a,v,ele,upstream_end$,particle=muon$)
      endif
      call convert_pc_to(a%p0c*(1+a%vec(6)),a%species,beta=a%beta)
      call track1(a,ele,lat%param,b,err_flag=err)
      if (err .or. b%state /= alive$) error stop 'track'
      if (b%species /= a%species) error stop 'species'
      e0=hypot(a%p0c*(1+a%vec(6)),mass_of(a%species))
      e1=hypot(b%p0c*(1+b%vec(6)),mass_of(b%species))
      if(i==1) then
       expected=ele%value(voltage$)
       write(*,'(a,i0,3es25.16)') 'RESULT RF',isp,a%p0c,e1-e0,expected
      else
       expected=ele%value(voltage$)/a%p0c
       write(*,'(a,i0,3es25.16)') 'RESULT CR',isp,a%p0c,b%vec(2)-a%vec(2),expected
      endif
     enddo
    enddo
    end program
    """
    LATTICE = """no_digested
    parameter[particle] = electron
    parameter[p0c] = 5e6
    parameter[geometry] = open
    rf: rfcavity, l = 0, rf_frequency = 1e9, voltage = 1e6, phi0 = 0.13
    lc: lcavity, l = 0, rf_frequency = 1e9, voltage = 1e6, phi0 = 0.13, n_rf_steps = 4, cavity_type = traveling_wave
    cr: crab_cavity, l = 0, rf_frequency = 1e9, voltage = 1e6, phi0 = 0.13
    ln: line = (rf, lc, cr)
    use, ln
    """
    d = rows(fortran(PROGRAM, files={"probe.bmad": LATTICE}).log)
    defect = False
    for kind, unit, atol in [("RF", "eV", 3e-8), ("CR", "p0", 3e-15)]:
        reference, observed, expected = d[kind + "1"]
        control(f"{kind} electron kick ({unit})", observed, expected, atol=atol)
        control(f"{kind} shared reference momentum (eV/c)", d[kind + "2"][0], reference, atol=1e-8)
        # q/q_ref = 1 for both negative species; mass does not enter qV or qV/p0c.
        defect |= compare(
            f"{kind} muon kick ({unit}), charge-only voltage law", d[kind + "2"][1], expected, atol=atol
        )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

