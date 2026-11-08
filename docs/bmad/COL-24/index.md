---
title: "COL-24: IBS formulas use charge magnitude instead of charge squared in the classical radius"
hide:
  - toc
---

# COL-24 — IBS formulas use charge magnitude instead of charge squared in the classical radius

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-21 on Bmad release [20261020-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261020-0) (`19b8afb`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Issue | [[gh:https://github.com/bmad-sim/bmad-ecosystem/issues/2230]], filed from this evidence |
| Pull request | [[gh:https://github.com/bmad-sim/bmad-ecosystem/pull/2231]], filed from this evidence |
| Source | [ibs_rates_mod.f90:83](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/multiparticle/ibs_rates_mod.f90#L83)<br>[ibs_mod.f90:759](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/multiparticle/ibs_mod.f90#L759)<br>[classical_radius, bmad/code/classical_radius.f90:23](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/classical_radius.f90#L23) |

For multiply charged ions, five IBS rate formulas and the Coulomb-logarithm helper used a classical radius proportional to charge magnitude instead of charge squared. For the Bjorken–Mtingwa formula with a fixed Coulomb logarithm, the rates were underestimated by the squared charge state. For carbon-12⁶⁺, the longitudinal Bjorken–Mtingwa rate was 1.640 × 10⁻⁷ s⁻¹ instead of 5.902 × 10⁻⁶ s⁻¹, a factor of 36. Unit-charge beams have the correct charge scaling.

Run `python COL-24.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-24.py"

    [Download COL-24.py](COL-24.py)

    ```python
    """COL-24: IBS formulas use charge magnitude instead of charge squared in the classical radius

    For multiply charged ions, five IBS rate formulas and the Coulomb-logarithm helper used a classical radius proportional to charge magnitude instead of charge squared. For the Bjorken–Mtingwa formula with a fixed Coulomb logarithm, the rates were underestimated by the squared charge state. For carbon-12⁶⁺, the longitudinal Bjorken–Mtingwa rate was 1.640 × 10⁻⁷ s⁻¹ instead of 5.902 × 10⁻⁶ s⁻¹, a factor of 36. Unit-charge beams have the correct charge scaling.

    The script compiles an electron and carbon-ion driver against BMAD_DIST. It compares Bjorken–Mtingwa rates with independent matrix quadrature using the charge-squared classical radius and requires the electron control to agree.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from scipy.integrate import quad

    PROGRAM = """program probe_rates
    use bmad
    use ibs_mod
    use ibs_rates_mod, only: bjmt1, bane1, mpxx1, mpzt1, cimp1
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: ele
    type(ibs_struct) rates
    integer i, k, ion
    real(rp) rcode, rtrue, big_a, gamma, beta, beta_a, beta_b, sigma_h, a, b, clog
    real(rp) ratios(10)
    character(4) formula(5)

    call init_lat(lat,2)
    lat%n_ele_track = 1
    lat%param%particle = electron$
    lat%param%total_length = 1.0_rp
    ele => lat%ele(1)
    ele%key = drift$
    ele%value(l$) = 1.0_rp
    ele%value(e_tot$) = 2.0e9_rp
    ele%z%sigma = 0.01_rp
    ele%z%sigma_p = 1.0e-3_rp
    ele%a%beta = 10.0_rp
    ele%b%beta = 20.0_rp
    ele%a%emit = 1.0e-9_rp
    ele%b%emit = 1.0e-11_rp
    ele%a%eta = 0.1_rp
    ele%b%eta = 0.01_rp
    clog = 10.0_rp
    formula = ['bjmt','bane','mpxx','mpzt','cimp']
    do i = 1, 2
      ion = species_id('#12C+6')
      if (i == 1) ion = electron$
      lat%param%particle = ion
      gamma = 4000.0_rp
      ele%value(e_tot$) = gamma * mass_of(ion)
      call convert_total_energy_to(ele%value(e_tot$), ion, gamma=gamma, beta=beta)
      rcode = c_light*c_light*e_charge*1.0e-7_rp*abs(charge_of(ion))/mass_of(ion)
      rtrue = classical_radius(ion)
      write(*,'(a,i3,3es23.14)') 'charge, r_code, r_true, squared ratio: ', charge_of(ion), rcode, rtrue, (rtrue/rcode)**2
      do k=1,5
        select case (k)
        case(1)
          call bjmt1(ele,clog,rates,1.0e12_rp)
        case(2)
          call bane1(ele,clog,rates,1.0e12_rp)
        case(3)
          call mpxx1(ele,clog,rates,1.0e12_rp)
        case(4)
          call mpzt1(ele,clog,rates,1.0e12_rp)
        case(5)
          call cimp1(ele,clog,rates,1.0e12_rp)
        end select
        write(*,'(a,a4,3es23.14)') 'rates ', formula(k), rates%inv_Ta,rates%inv_Tb,rates%inv_Tp
      enddo
    enddo

    end program
    """
    out = fortran(PROGRAM).log
    gamma = 4000.0
    beta = np.sqrt(1 - 1 / gamma**2)
    ea, eb, sp, ba, bb, da, db = 1e-9, 1e-11, 1e-3, 10.0, 20.0, 0.1, 0.01
    lp = np.diag([0.0, gamma**2 / sp**2, 0.0])
    lh = np.diag([ba / ea, gamma**2 * da**2 / ba / ea, 0.0])
    lv = np.diag([0.0, gamma**2 * db**2 / bb / eb, bb / eb])
    mat = lp + lh + lv
    # Independent BM quadrature, using generic inverses/determinants and integrating
    # over the full positive axis after lambda=exp(t), t in (-20, 100).
    mass_c = 12 * 931494103.72 - 6 * 510998.95069
    radius = 2.8179403227e-15 * 510998.95069 * 6**2 / mass_c
    big_a = radius**2 * 299792458.0 * 1e12 / (64 * np.pi**2 * beta**3 * gamma**4 * ea * eb * sp)


    def integ(t, li):
        x = np.exp(t)
        a = mat + x * np.eye(3)
        ai = np.linalg.inv(a)
        return np.sqrt(x / np.linalg.det(a)) * (np.trace(li) * np.trace(ai) - 3 * np.trace(li @ ai)) * x


    expected = np.array(
        [
            4 * np.pi * big_a * 10 * quad(integ, -20, 100, args=(li,), epsabs=1e-10, epsrel=1e-11)[0]
            for li in [lh, lv, lp]
        ]
    )
    carbon_line = out.split("squared ratio:   6")[1].splitlines()[1]
    actual = np.fromstring(carbon_line.split("bjmt")[1], sep=" ")
    print("Carbon+6 Bjorken-Mtingwa [Ta^-1,Tb^-1,Tp^-1] (s^-1):")
    print("production:", actual)
    print("independent q^2 radius + matrix quadrature:", expected)
    print("expected/production:", expected / actual)
    electron_actual = np.fromstring(out.split("rates bjmt")[1].splitlines()[0], sep=" ")
    electron_expected = expected * (2.8179403227e-15 / radius) ** 2
    print(
        "electron BM control max relative error:",
        np.max(np.abs(electron_actual / electron_expected - 1)),
    )

    if compare(
        "electron BM control (s^-1), matrix quadrature",
        electron_actual,
        electron_expected,
        atol=0,
        rtol=5e-9,
    ):
        raise RuntimeError("The electron control failed")
    verdict(
        compare(
            "carbon-12 6+ BM rates (s^-1), charge-squared radius and matrix quadrature",
            actual,
            expected,
            atol=0,
            rtol=5e-9,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

