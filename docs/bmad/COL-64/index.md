---
title: "COL-64: Manual long-range accumulator signs disagree with stored coefficients"
hide:
  - toc
---

# COL-64 — Manual long-range accumulator signs disagree with stored coefficients

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [track1_lr_wake stored coefficients](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L263-L276)<br>[compensating kick weights](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L244-L255)<br>[“Long-Range Wakes”, ac2rq and akz2q](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/wakefields.tex#L167-L209)<br>[long-range state input contract](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L3243-L3249) |

Using the manual’s formulas to reconstruct an unpolarized long-range wake gives the wrong
signs for the stored normal cosine and skew sine coefficients. In the order-two example,
Bmad stores −2326 V m⁻¹ and −4618 V m⁻¹, while the formulas give +2326 V m⁻¹ and +4618 V m⁻¹.
The kick equations compensate for these sign conventions: a delayed witness agrees with an
independent resonator calculation.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)

??? example "Reproducer: COL-64.py"

    [Download COL-64.py](COL-64.py)

    ```python
    """COL-64: Manual long-range accumulator signs disagree with stored coefficients

    Using the manual’s formulas to reconstruct an unpolarized long-range wake gives the wrong
    signs for the stored normal cosine and skew sine coefficients. In the order-two example,
    Bmad stores −2326 V m⁻¹ and −4618 V m⁻¹, while the formulas give +2326 V m⁻¹ and +4618 V m⁻¹.
    The kick equations compensate for these sign conventions: a delayed witness agrees with an
    independent resonator calculation.

    The script executes the guide’s unpolarized order-two source and delayed-witness case. It
    compares all four stored coefficients with the manual definitions and requires the physical
    delayed kick to agree with an independent resonator calculation. This separates the state-
    documentation mismatch from the polarization defect.
    """

    import math

    import numpy as np
    from _bmad import compare, fortran, verdict

    PROGRAM = """
    program long_range_modes
      use bmad
      use wake_mod
      implicit none
      type(ele_struct) ele
      type(bunch_struct) bunch
      integer polarized
      character(16) name
      call init_ele(ele,lcavity$)
      ele%value(p0c$)=1.0e12_rp; ele%value(p0c_start$)=1.0e12_rp
      bmad_com%lr_wakes_on = .true.
      do polarized = 0, 0
        call init_wake(ele%wake,0,0,0,1)
        ele%wake%lr = wake_lr_struct(mode=ele%wake%lr%mode)
        ele%wake%lr%mode(1) = wake_lr_mode_struct()
        ele%wake%lr%self_wake_on = .false.
        ele%wake%lr%mode(1)%freq = 2.0e8_rp
        ele%wake%lr%mode(1)%freq_in = 2.0e8_rp
        ele%wake%lr%mode(1)%r_over_q = 1.0e12_rp
        ele%wake%lr%mode(1)%damp = 1.0e7_rp
        ele%wake%lr%mode(1)%phi = 0.17_rp
        ele%wake%lr%mode(1)%m = 2
        ele%wake%lr%mode(1)%angle = 0.13_rp
        ele%wake%lr%mode(1)%polarized = (polarized == 1)
        call reallocate_bunch(bunch,2)
        call init_coord(bunch%particle(1),[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.1_rp,0.0_rp],ele,upstream_end$,electron$)
        call init_coord(bunch%particle(2),[0.002_rp,0.0_rp,0.003_rp,0.0_rp,0.0_rp,0.0_rp],ele,upstream_end$,electron$)
        bunch%particle(1)%charge=0; bunch%particle(2)%charge=2.0e-12_rp
        bunch%particle(1)%t=0; bunch%particle(2)%t=0.7e-9_rp
        call order_particles_in_z(bunch)
        call track1_lr_wake(bunch,ele)
        write(name,'(a,i0)') 'amplitudes',polarized
        write(*,'(3a,4es25.16e3)') 'RESULT ',trim(name),' ',ele%wake%lr%mode(1)%b_sin, &
          ele%wake%lr%mode(1)%b_cos,ele%wake%lr%mode(1)%a_sin,ele%wake%lr%mode(1)%a_cos
        call reallocate_bunch(bunch,1)
        call init_coord(bunch%particle(1),[0.007_rp,0.0_rp,-0.004_rp,0.0_rp,0.0_rp,0.0_rp],ele,upstream_end$,electron$)
        bunch%particle(1)%charge=0; bunch%particle(1)%t=1.9e-9_rp
        call order_particles_in_z(bunch)
        call track1_lr_wake(bunch,ele)
        write(name,'(a,i0)') 'kick',polarized
        write(*,'(3a,3es25.16e3)') 'RESULT ',trim(name),' ',bunch%particle(1)%vec(2), &
          bunch%particle(1)%vec(4),bunch%particle(1)%vec(6)
      enddo
    end program
    """
    log = fortran(PROGRAM).log
    rows = {
        p[1]: list(map(float, p[2:]))
        for line in log.splitlines()
        if (p := line.split()) and p[0] == "RESULT"
    }
    if set(rows) != {"amplitudes0", "kick0"}:
        raise RuntimeError(log)
    omega, damping, phi = 2 * math.pi * 2e8, 1e7, 2 * math.pi * 0.17
    time, delay = 0.7e-9, 1.2e-9
    z = complex(0.002, 0.003) ** 2
    scale = 2e-12 * 1e12 * 299792458 * math.exp(damping * time)
    # Manual ac2rq: b_sin=Re(z^m)cos, b_cos=-Re(z^m)sin,
    # a_sin=Im(z^m)cos, a_cos=-Im(z^m)sin, times the positive amplitude.
    c, s = math.cos(omega * time), math.sin(omega * time)
    expected = scale * np.array([z.real * c, -z.real * s, z.imag * c, -z.imag * s])
    moment = np.array([-z.real, z.imag])
    coefficient = 2e-12 * 1e12 * 299792458 * math.exp(-damping * delay) / 1e12 * moment
    a, b = coefficient[1], coefficient[0]
    transverse = (
        2
        * math.sin(omega * delay + phi)
        * np.array([-b * 0.007 + a * (-0.004), a * 0.007 + b * (-0.004)])
    )
    w = complex(0.007, -0.004) ** 2
    longitudinal = (
        coefficient
        @ np.array([-w.real, w.imag])
        / 299792458
        * (damping * math.sin(omega * delay + phi) - omega * math.cos(omega * delay + phi))
    )
    if compare(
        "unpolarized delayed kick control; independent resonator response",
        rows["kick0"],
        np.r_[transverse, longitudinal],
        atol=1e-25,
        rtol=5e-14,
    ):
        raise RuntimeError("Physical delayed-kick control failed")
    verdict(
        compare(
            "stored [b_sin,b_cos,a_sin,a_cos]; manual ac2rq",
            rows["amplitudes0"],
            expected,
            atol=1e-12,
            rtol=3e-14,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed (unpolarized order-two accumulator contract and delayed physical kick) | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

