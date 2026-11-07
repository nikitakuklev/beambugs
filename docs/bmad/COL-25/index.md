---
title: "COL-25: Touschek integration discards the tail above τ = 1"
hide:
  - toc
---

# COL-25 — Touschek integration discards the tail above τ = 1

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">extreme-regime</span> [?](../../methodology.md#tags) |
| Source | [touschek_rate1, bmad/multiparticle/touschek_mod.f90:535–546](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/touschek_mod.f90#L535-L546)<br>[Bmad manual, Touschek Scattering (s:touschek), bmad/doc/multiparticle.tex:228–246](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/multiparticle.tex#L228-L246) |

The Touschek rate integral stops at the dimensionless scattering variable τ = 1 instead of extending to infinity. When the bunch rest-frame transverse momenta are relativistic, the omitted tail can dominate the loss rate. At Lorentz factor 10⁵, geometric emittances 10 nm, beta functions 10 m and 20 m, and momentum aperture 3%, the rate is 0.5845 particles/s instead of 4.790 particles/s. This underestimates the loss rate by 87.80% and overestimates the lifetime by a factor of 8.196.

Run `python COL-25.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-25.py"

    [Download COL-25.py](COL-25.py)

    ```python
    """COL-25: Touschek integration discards the tail above τ = 1

    The Touschek rate integral stops at the dimensionless scattering variable τ = 1 instead of extending to infinity. When the bunch rest-frame transverse momenta are relativistic, the omitted tail can dominate the loss rate. At Lorentz factor 10⁵, geometric emittances 10 nm, beta functions 10 m and 20 m, and momentum aperture 3%, the rate is 0.5845 particles/s instead of 4.790 particles/s. This underestimates the loss rate by 87.80% and overestimates the lifetime by a factor of 8.196.

    The script compiles a local-rate driver against BMAD_DIST. Independent quadrature first reproduces the truncated integral as a control, then extends its upper limit until exponential suppression makes the remaining tail negligible.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from scipy.integrate import quad
    from scipy.special import i0e

    PROGRAM = """program probe_touschek
    use bmad
    use touschek_mod, only: touschek_rate1, exp_bessi0
    implicit none
    type(lat_struct) lat
    type(normal_modes_struct) mode
    real(rp) rate, ts(5)
    integer i
    call init_lat(lat,2)
    lat%n_ele_track = 1
    lat%param%particle = electron$
    lat%param%geometry = closed$
    lat%param%n_part = 1.0e10_rp
    do i=0,2
      lat%ele(i)%value(e_tot$) = 1.0e5_rp * m_electron
      lat%ele(i)%a%beta = 10.0_rp
      lat%ele(i)%b%beta = 20.0_rp
    enddo
    mode%a%emittance = 1.0e-8_rp
    mode%b%emittance = 1.0e-8_rp
    mode%sigE_E = 1.0e-3_rp
    mode%sig_z = 0.01_rp
    mode%pz_aperture = 0.03_rp
    call touschek_rate1(mode,rate,lat,ix=1)
    write(*,'(es25.16)') rate
    if (command_argument_count() > 0) then
      mode%pz_aperture=-mode%pz_aperture
      call touschek_rate1(mode,rate,lat,ix=1)
      write(*,'(es25.16)') rate
      lat%param%n_part=2*lat%param%n_part
      call touschek_rate1(mode,rate,lat,ix=1)
      write(*,'(es25.16)') rate
      mode%sig_z=2*mode%sig_z
      call touschek_rate1(mode,rate,lat,ix=1)
      write(*,'(es25.16)') rate
      ts=[0.0_rp,0.1_rp,1.0_rp,2.0_rp,50.0_rp]
      do i=1,5
        write(*,'(es25.16)') exp_bessi0(ts(i),3.0_rp,2.0_rp)
      enddo
    endif
    end program
    """
    gamma = 1e5
    beta2 = 1 - 1 / gamma**2
    ea = eb = 1e-8
    ba, bb = 10.0, 20.0
    tm = beta2 * 0.03**2
    b1 = (ba / ea + bb / eb) / (2 * beta2 * gamma**2)
    b2 = abs(ba / ea - bb / eb) / (2 * beta2 * gamma**2)


    def f(logt):
        t = np.exp(logt)
        arg = t / tm / (1 + t)
        bracket = (
            (2 + 1 / t) ** 2 * (arg - 1)
            + 1
            - np.sqrt(tm * (1 + t) / t)
            - (4 + 1 / t) / (2 * t) * np.log(arg)
        )
        # Clip the tiny unphysical sliver at threshold to match Bmad and isolate
        # the effect of the upper integration bound.
        return max(bracket, 0) * np.sqrt(t / (1 + t)) * np.exp((b2 - b1) * t) * i0e(b2 * t) * t


    trunc = quad(f, np.log(tm), 0, epsabs=1e-9, epsrel=1e-11, limit=500)[0]
    full = quad(f, np.log(tm), np.log(1e4), epsabs=1e-9, epsrel=1e-11, limit=500)[0]
    factor = (
        2.8179403227e-15**2
        * 299792458.0
        * 1e20
        / (8 * np.sqrt(np.pi) * gamma**4 * beta2 * 0.01 * ea * eb)
    )
    actual = float(fortran(PROGRAM).log)
    print(f"B1={b1:.12g} B2={b2:.12g}")
    print(f"production loss rate = {actual:.12g} particles/s")
    print(f"independent integral tau_m..1 = {factor * trunc:.12g} particles/s")
    print(f"independent integral tau_m..infinity = {factor * full:.12g} particles/s")
    print(f"loss-rate underestimate = {100 * (1 - actual / (factor * full)):.5f}%")
    print(f"lifetime overestimate = {factor * full / actual:.8g} times")

    if compare(
        "truncated-integral control (particles/s), tau <= 1", actual, factor * trunc, atol=0, rtol=3e-9
    ):
        raise RuntimeError("The truncated-integral control failed")
    verdict(
        compare(
            "full Touschek rate (particles/s), integral to tau=10000",
            actual,
            factor * full,
            atol=0,
            rtol=3e-9,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

