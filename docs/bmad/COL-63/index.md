---
title: "COL-63: Modal momentum kicks omit the manual’s finite-velocity factor"
hide:
  - toc
---

# COL-63 — Modal momentum kicks omit the manual’s finite-velocity factor

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">contract-unsettled</span> [?](../../methodology.md#tags) |
| Source | [`sr_longitudinal_wake_particle`, bmad/multiparticle/wake_mod.f90:320–380](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L320-L380)<br>[`sr_transverse_wake_particle`, bmad/multiparticle/wake_mod.f90:416–456](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L416-L456)<br>[`Short--Range Wakes (s:sr.wake.eq), delvp1 and pelqxw2`, bmad/doc/wakefields.tex:26–69](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/wakefields.tex#L26-L69) |

The short-range modal wake routines use a momentum-kick normalization that omits the inverse particle-speed factor shown in the manual. With β = v/c = 0.5000, the computed kicks are half the values obtained from the documented equations for the same supplied wake. The longitudinal update also leaves stored β and the longitudinal coordinate unchanged after changing momentum. The implementation and documentation therefore specify different finite-velocity models; their ultrarelativistic limits agree.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _wake_tables_and_bbu.py](../helpers/_wake_tables_and_bbu.py) · [View source](../helpers/index.md#_wake_tables_and_bbu)

??? example "Reproducer: COL-63.py"

    [Download COL-63.py](COL-63.py)

    ```python
    """COL-63: Modal momentum kicks omit the manual’s finite-velocity factor

    The short-range modal wake routines use a momentum-kick normalization that omits the inverse particle-speed factor shown in the manual. With β = v/c = 0.5000, the computed kicks are half the values obtained from the documented equations for the same supplied wake. The longitudinal update also leaves stored β and the longitudinal coordinate unchanged after changing momentum. The implementation and documentation therefore specify different finite-velocity models; their ultrarelativistic limits agree.

    The script initializes electrons at β = 0.5 and near unit β, applies longitudinal and transverse modal wakes, and compares with the manual’s normalization. It also records the longitudinal update’s stored speed and coordinate. BMAD_SOURCE supplies the equations and BMAD_DIST selects the build.
    """

    import re

    import numpy as np
    from _bmad import compare, fortran, source, verdict
    from _wake_tables_and_bbu import require, rows

    PROGRAM = r"""
    program finite_beta
    use bmad
    use wake_mod
    implicit none
    type(ele_struct) ele
    type(coord_struct) orb
    real(rp) pc, strength, before_beta, before_z
    integer pass, kind
    character(20) name
    call init_ele(ele,drift$)
    ele%value(l$)=0.6_rp
    do pass=1,2
     pc=m_electron/sqrt(3.0_rp)
     if (pass==2) pc=1e12_rp
     ele%value(p0c$)=pc
     ele%value(p0c_start$)=pc
     ele%value(e_tot$)=hypot(pc,m_electron)
     ele%value(e_tot_start$)=ele%value(e_tot$)
     strength=pc*1e6_rp
     do kind=1,2
      call init_wake(ele%wake,1,1,0,0)
      ele%wake%sr%long(1)=wake_sr_mode_struct()
      ele%wake%sr%trans(1)=wake_sr_mode_struct()
      ele%wake%sr%long(1)%amp=strength
      ele%wake%sr%long(1)%damp=3
      ele%wake%sr%long(1)%k=4
      ele%wake%sr%long(1)%phi=0.13_rp
      ele%wake%sr%long(1)%position_dependence=none$
      ele%wake%sr%trans(1)%amp=strength
      ele%wake%sr%trans(1)%damp=3
      ele%wake%sr%trans(1)%k=4
      ele%wake%sr%trans(1)%phi=0.13_rp
      ele%wake%sr%trans(1)%polarization=x_polarization$
      ele%wake%sr%trans(1)%position_dependence=none$
      ele%wake%sr%z_ref_long=0.01_rp
      ele%wake%sr%z_ref_trans=0.01_rp
      call init_coord(orb,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.01_rp,0.0_rp],ele,upstream_end$,electron$)
      orb%charge=2e-12_rp
      if (kind==1) call sr_longitudinal_wake_particle(ele,orb)
      if (kind==2) call sr_transverse_wake_particle(ele,orb)
      call init_coord(orb,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,-0.02_rp,0.0_rp],ele,upstream_end$,electron$)
      orb%charge=0
      before_beta=orb%beta
      before_z=orb%vec(5)
      if (kind==1) call sr_longitudinal_wake_particle(ele,orb)
      if (kind==2) call sr_transverse_wake_particle(ele,orb)
      write(name,'(a,i0,a,i0)') 'case',pass,'_',kind
      write(*,'(3a,8es25.16)') 'RESULT ',trim(name),' ',orb%vec(6),orb%vec(2),before_beta,orb%beta, &
       before_z,orb%vec(5),pc,m_electron
     enddo
    enddo
    end program
    """
    manual = (source() / "bmad/doc/wakefields.tex").read_text()
    equations = re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}", manual, re.DOTALL)


    def denominator(label):
        eq = next(s for s in equations if f"\\label{{{label}}}" in s)
        if r"{v \, P_0}" in eq:
            return True
        if r"{c \, P_0}" in eq:
            return False
        raise RuntimeError(f"Unrecognized normalization in {label}")


    finite = [denominator("delvp1"), denominator("pelqxw2")]
    actual = rows(fortran(PROGRAM).log)
    # A/p0c=1e6, q=2e-12 C, L=0.6 m; the manual adds c/v to this impulse.
    base = -1e6 * 2e-12 * 0.6 * np.exp(-0.03 * 3) * np.sin(-0.03 * 4 + 2 * np.pi * 0.13)
    defect = False
    for kind, label in [(1, "longitudinal"), (2, "transverse")]:
        control = actual[f"case2_{kind}"]
        column = 0 if kind == 1 else 1
        require("ultrarelativistic input beta", control[2], 1, atol=2e-13)
        require(
            f"ultrarelativistic {label} delta; manual",
            control[column],
            base / control[2],
            rtol=3e-13,
            atol=1e-22,
        )
        case = actual[f"case1_{kind}"]
        require(f"finite-beta {label} input beta", case[2], 0.5, atol=2e-16)
        expected = base / case[2] if finite[kind - 1] else base
        defect |= compare(
            f"beta=0.5 {label} delta; displayed equation",
            case[column],
            expected,
            rtol=3e-13,
            atol=1e-22,
        )
    case = actual["case1_1"]
    # Bmad stores z=-beta*c*(t-t_ref); a pure longitudinal impulse preserves arrival time.
    p = case[6] * (1 + case[0])
    beta_from_momentum = p / np.hypot(p, case[7])
    print(
        f"longitudinal update: stored beta={case[3]:.17g}, momentum-implied beta={beta_from_momentum:.17g}; z(m)={case[5]:.17g}, fixed-arrival-time z(m)={case[4] * beta_from_momentum / case[2]:.17g}"
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found (source/contract discrepancy) | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed (source/contract discrepancy) | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`) |
| Confirmed (supplied-wake equation/implementation normalization and unchanged longitudinal beta/z) | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

