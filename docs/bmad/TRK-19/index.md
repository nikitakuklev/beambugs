---
title: "TRK-19: Native lcavity body kicks omit the tracked/reference charge ratio"
hide:
  - toc
---

# TRK-19 — Native lcavity body kicks omit the tracked/reference charge ratio

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [track_a_lcavity::this_energy_kick, bmad/low_level/track_a_lcavity.f90:354–364](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L354-L364)<br>[Bmad Manual, LCavity, equation `dergl`, bmad/doc/elements.tex:3313–3318](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L3313-L3318)<br>[Bmad Manual, LCavity Tracking, equation `vkrq`, bmad/doc/charged-tracking.tex:1173–1179](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/charged-tracking.tex#L1173-L1179)<br>[fringe_kick, bmad/low_level/track_a_lcavity.f90:269–276](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L269-L276) |

A positron tracked through an electron-reference linear cavity receives the electron energy-kick sign. The native body kick omits the tracked-to-reference charge ratio required by the cavity equation. In a thin 10 kV traveling-wave cavity at crest, a positron 10% above the 1 MeV/c reference gains 10.00 keV instead of losing 10.00 keV. The electron linear-cavity case and both charge signs in an RF cavity agree with the voltage law.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _rf_and_spin.py](../helpers/_rf_and_spin.py) · [View source](../helpers/index.md#_rf_and_spin)

??? example "Reproducer: TRK-19.py"

    [Download TRK-19.py](TRK-19.py)

    ```python
    """TRK-19: Native lcavity body kicks omit the tracked/reference charge ratio

    A positron tracked through an electron-reference linear cavity receives the electron energy-kick sign. The native body kick omits the tracked-to-reference charge ratio required by the cavity equation. In a thin 10 kV traveling-wave cavity at crest, a positron 10% above the 1 MeV/c reference gains 10.00 keV instead of losing 10.00 keV. The electron linear-cavity case and both charge signs in an RF cavity agree with the voltage law.

    The script tracks both charge signs through thin RF and linear cavities at crest, using BMAD_DIST. It checks survival, species, endpoint reference, beta, clock and transverse coordinates before comparing the energy changes with the charge-scaled voltage.
    """

    from _bmad import compare, fortran, verdict
    from _rf_and_spin import control, rows

    PROGRAM = """program charge
    use bmad
    implicit none
    type(lat_struct) lat
    type(coord_struct) a,b
    integer i,j,particle_kind
    real(rp) v(6),gain
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) error stop 'parse'
    v=0; v(6)=0.1_rp
    do i=1,2
     do j=1,2
      particle_kind=electron$
      if(j==2) particle_kind=positron$
      call init_coord(a,v,lat%ele(i),upstream_end$,particle=particle_kind)
      call convert_pc_to(a%p0c*(1+a%vec(6)),particle_kind,beta=a%beta)
      call track1(a,lat%ele(i),lat%param,b,err_flag=err)
      if(err .or. b%state/=alive$ .or. b%species/=particle_kind) error stop 'track'
      gain=hypot(b%p0c*(1+b%vec(6)),mass_of(particle_kind))-hypot(a%p0c*(1+a%vec(6)),mass_of(particle_kind))
      write(*,'(a,i0,i0,*(es25.16,1x))') 'RESULT case',i,j,gain,b%p0c,lat%ele(i)%value(p0c$), &
       b%beta,b%p0c*(1+b%vec(6))/hypot(b%p0c*(1+b%vec(6)),mass_of(particle_kind)),a%t,b%t,b%vec(1:5)
     enddo
    enddo
    end program
    """
    LATTICE = """no_digested
    parameter[particle]=electron
    parameter[p0c]=1e6
    parameter[geometry]=open
    r: rfcavity, l=0, voltage=1e4, rf_frequency=1e8, phi0=0.25
    c: lcavity, l=0, voltage=1e4, rf_frequency=1e8, phi0=0, cavity_type=traveling_wave
    ln: line=(r,c)
    use,ln
    """
    d = rows(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
    for key, row in d.items():
        control(f"{key} endpoint reference (eV/c)", row[1], row[2], atol=1e-8)
        control(f"{key} beta from momentum", row[3], row[4], atol=3e-15)
        control(f"{key} clock (s)", row[5:7], [0, 0], atol=2e-23)
        control(f"{key} first five coordinates", row[7:], [0] * 5, atol=2e-16)
    for key, gain in [("case11", 1e4), ("case12", -1e4), ("case21", 1e4)]:
        control(f"{key} energy gain (eV), relative charge times voltage", d[key][0], gain, atol=8e-10)
    verdict(compare("lcavity positron gain (eV), q/q_ref times V", d["case22"][0], -1e4, atol=8e-10))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

