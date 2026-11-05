---
title: "COL-50: Fundamental wakes lose elapsed phase during absolute-time tracking"
hide:
  - toc
---

# COL-50 — Fundamental wakes lose elapsed phase during absolute-time tracking

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [track1_lr_wake elapsed-phase rotation](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L151-L174)<br>[absolute/relative clock use](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L209-L245)<br>[“Long-Range Wakes”, negative-frequency fundamental designation](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L3251-L3256) |

With absolute-time tracking enabled, a fundamental long-range wake mode advances its reference
time without rotating its stored phase. An ordinary mode with the same physical frequency
retains the elapsed phase and gives the correct delayed wake. For a 1 nC source and a witness
arriving 0.125 ns later at 1 GHz, the fundamental gives a relative momentum kick of −6.283 ×
10⁻⁷ instead of −4.443 × 10⁻⁷. The excess is the missing cosine of the π/4 phase advance.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)

??? example "Reproducer: COL-50.py"

    [Download COL-50.py](COL-50.py)

    ```python
    """COL-50: Fundamental wakes lose elapsed phase during absolute-time tracking

    With absolute-time tracking enabled, a fundamental long-range wake mode advances its reference
    time without rotating its stored phase. An ordinary mode with the same physical frequency
    retains the elapsed phase and gives the correct delayed wake. For a 1 nC source and a witness
    arriving 0.125 ns later at 1 GHz, the fundamental gives a relative momentum kick of −6.283 ×
    10⁻⁷ instead of −4.443 × 10⁻⁷. The excess is the missing cosine of the π/4 phase advance.

    The script tracks a source and a delayed zero-charge witness through ordinary and fundamental
    zero-voltage cavities. Both have the same physical mode frequency and R/Q, so the ordinary mode
    is a control for the analytic cosine response. Absolute-time tracking is enabled explicitly.
    """

    import math

    from _bmad import compare, fortran, verdict

    PROGRAM = """program probe
    use bmad
    use beam_mod
    use wake_mod
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct) :: ele
    type(bunch_struct) :: bunch, one
    type(beam_init_struct) :: bi
    real(rp) :: result(3), expected, q, dt
    integer :: i, n
    logical :: done, err
    character(200) :: which, file
    q=1e-9_rp
      call bmad_parser('fundamental.bmad',lat)
      ! The parser also appends an END element; identify the two test cavities by name.
      if(lat%n_ele_track<2) error stop 'missing test cavities'
      if(lat%ele(1)%name/='HOM'.or.lat%ele(2)%name/='FUND') error stop 'test cavity names differ'
      if(.not.bmad_com%absolute_time_tracking) error stop 'absolute-time tracking is disabled'
      do i=1,2
        if(size(lat%ele(i)%wake%lr%mode)/=1) error stop 'expected one mode per cavity'
        if(lat%ele(i)%wake%lr%mode(1)%freq/=1e9_rp) error stop 'physical frequencies differ'
        if(lat%ele(i)%wake%lr%mode(1)%r_over_q/=100.0_rp) error stop 'R/Q differs'
        if(lat%ele(i)%wake%lr%mode(1)%damp/=0) error stop 'nonzero damping'
        if(lat%ele(i)%value(voltage$)/=0) error stop 'nonzero accelerating voltage'
      enddo
      if(lat%ele(1)%wake%lr%mode(1)%freq_in<=0) error stop 'ordinary mode marker mismatch'
      if(lat%ele(2)%wake%lr%mode(1)%freq_in>=0) error stop 'fundamental marker mismatch'
      call reallocate_bunch(one,1)
      dt=0.125e-9_rp
      do i=1,2
        call init_coord(one%particle(1), [0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp], lat%ele(0), element_end=downstream_end$)
        one%particle(1)%charge=q
        call track1_bunch(one,lat%ele(i),err)
        if(err) error stop 'tracking failed'
        call init_coord(one%particle(1), [0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp], lat%ele(0), element_end=downstream_end$)
        one%particle(1)%charge=0
        one%particle(1)%t=dt
        one%particle(1)%vec(5)=-one%particle(1)%beta*c_light*dt
        call track1_bunch(one,lat%ele(i),err)
        if(err) error stop 'tracking failed'
        print '(a,a,a,es25.16)', trim(lat%ele(i)%name), ' witness dp/p = ', '', one%particle(1)%vec(6)
      enddo
      expected=-q*100.0_rp*twopi*1e9_rp*cos(twopi*1e9_rp*dt)/1e9_rp
      print '(a,es25.16)', 'analytic for both = ', expected
    end program
    """
    log = fortran(
        PROGRAM,
        files={
            "fundamental.bmad": """no_digested
    parameter[particle]=electron
    parameter[p0c]=1e9
    parameter[geometry]=open
    bmad_com[absolute_time_tracking]=T
    hom: rfcavity, l=0, rf_frequency=1e9, voltage=0, lr_wake={mode={1e9,100,0,0,0,unpolarized}}
    fund: rfcavity, l=0, rf_frequency=1e9, voltage=0, lr_wake={mode={-1,100,0,0,0,unpolarized}}
    ln: line=(hom,fund)
    use,ln
    """
        },
    ).log
    rows = {
        line.split()[0]: float(line.split("=")[1])
        for line in log.splitlines()
        if "witness dp/p =" in line
    }
    if set(rows) != {"HOM", "FUND"}:
        raise RuntimeError(log)
    expected = -1e-9 * 100 * 2 * math.pi * 1e9 * math.cos(math.pi / 4) / 1e9
    if compare("HOM control delta; resonator at pi/4 delay", rows["HOM"], expected, atol=0, rtol=2e-14):
        raise RuntimeError("Ordinary-mode phase control failed")
    verdict(
        compare(
            "fundamental delta; identical resonator at pi/4 delay",
            rows["FUND"],
            expected,
            atol=0,
            rtol=2e-14,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

