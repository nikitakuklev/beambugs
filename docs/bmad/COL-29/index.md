---
title: "COL-29: CIMP interpolation violates its stated one-percent accuracy"
hide:
  - toc
---

# COL-29 — CIMP interpolation violates its stated one-percent accuracy

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [cimp1 accuracy contract, bmad/multiparticle/ibs_rates_mod.f90:604–614](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/ibs_rates_mod.f90#L604-L614)<br>[CIMP scattering interpolation, bmad/multiparticle/ibs_rates_mod.f90:711–815](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/ibs_rates_mod.f90#L711-L815) |

The CIMP IBS approximation exceeds its stated 1% interpolation accuracy inside its supported domain. Default-real coefficient rounding is amplified by polynomial cancellation, and the original decimal polynomial also misses the accuracy target near the upper endpoint. For the scattering function g at argument 300, the result is 0.07907 instead of 0.08410, a 5.987% deficit; at 2999 it is 0.01003 instead of 0.01188, a 15.56% deficit. Evaluating the original coefficients at high precision still leaves a 1.561% deficit at 2999.

Run `python COL-29.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-29.py"

    [Download COL-29.py](COL-29.py)

    ```python
    """COL-29: CIMP interpolation violates its stated one-percent accuracy

    The CIMP IBS approximation exceeds its stated 1% interpolation accuracy inside its supported domain. Default-real coefficient rounding is amplified by polynomial cancellation, and the original decimal polynomial also misses the accuracy target near the upper endpoint. For the scattering function g at argument 300, the result is 0.07907 instead of 0.08410, a 5.987% deficit; at 2999 it is 0.01003 instead of 0.01188, a 15.56% deficit. Evaluating the original coefficients at high precision still leaves a 1.561% deficit at 2999.

    The script infers the scattering function from production CIMP cooling rates in BMAD_DIST and compares it with Euler-integral quadrature. It also reads BMAD_SOURCE to evaluate the original polynomial coefficients with 70-digit decimal arithmetic.
    """

    import math
    import re
    from decimal import Decimal, localcontext

    from _bmad import compare, fortran, source, verdict
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
    lat%param%particle = electron$
    ele%value(e_tot$) = 4000.0_rp * m_electron
    call convert_total_energy_to(ele%value(e_tot$), electron$, gamma=gamma, beta=beta)
    ele%a%eta = 0.0_rp
    ele%b%eta = 0.0_rp
    ele%a%beta = 10.0_rp
    ele%b%beta = 10.0_rp
    ele%z%sigma_p = 1.0e-6_rp
    ratios = [1.0_rp,2.0_rp,10.0_rp,30.0_rp,100.0_rp,300.0_rp,1000.0_rp,1500.0_rp,2000.0_rp,2999.0_rp]
    write(*,*) 'CIMP: u=b/a, g(u) inferred from horizontal cooling, g(1/u) from vertical cooling'
    do i = 1, size(ratios)
      ele%a%emit = 1.0e-9_rp
      ele%b%emit = ele%a%emit / ratios(i)**2
      call cimp1(ele,clog,rates,1.0e12_rp)
      big_a = classical_radius(electron$)**2*c_light*1.0e12_rp/64.0_rp/pi**2/beta**3/gamma**4/ele%a%emit/ele%b%emit/ele%z%sigma_p
      a = ele%z%sigma_p/gamma*sqrt(ele%a%beta/ele%a%emit)
      b = ele%z%sigma_p/gamma*sqrt(ele%b%beta/ele%b%emit)
      write(*,'(3es23.14)') ratios(i), -rates%inv_Ta/(2*pi**1.5_rp*big_a*clog*a), -rates%inv_Tb/(2*pi**1.5_rp*big_a*clog*b)
    enddo
    end program
    """


    def scattering_function(u: float) -> float:
        """Eq. 34 for u >= 1 via Euler integrals for its two hypergeometric terms."""
        z = (u * u + 1) / (2 * u)
        a = (z - 1) / 2
        p0 = (
            2
            / math.pi
            * quad(
                lambda t: 1 / math.sqrt(1 + a * math.sin(t) ** 2),
                0,
                math.pi / 2,
                epsabs=1e-13,
                epsrel=1e-13,
            )[0]
        )
        pm1 = (
            math.sqrt((z - 1) / (z + 1))
            * 4
            / math.pi
            * quad(
                lambda t: math.cos(t) ** 2 / math.sqrt(1 + a * math.sin(t) ** 2),
                0,
                math.pi / 2,
                epsabs=1e-13,
                epsrel=1e-13,
            )[0]
        )
        return math.sqrt(math.pi / u) * (p0 + 1.5 * pm1)


    output = fortran(PROGRAM).log
    defect = False
    if compare("Euler integral control g(1)", scattering_function(1), math.sqrt(math.pi), atol=2e-14):
        raise RuntimeError("Euler integral control failed")
    for line in output.split("vertical cooling")[1].splitlines():
        if not line.strip():
            continue
        u, actual, _ = map(float, line.split())
        expected = scattering_function(u)
        defect |= compare(
            f"g({u:g}), Euler integral and 1% contract", actual, expected, atol=0, rtol=0.01
        )
        print(f"u={u:g} relative error={actual / expected - 1:.10g}")
    text = (source() / "bmad/multiparticle/ibs_rates_mod.f90").read_text()
    match = re.search(r"\bo = \(/(.*?)/\)", text, re.DOTALL)
    if match is None:
        raise RuntimeError("CIMP coefficient table o was not found")
    with localcontext() as context:
        context.prec = 70
        coefficients = [Decimal(s.strip()) for s in match[1].replace("&", "").split(",")]
        polynomial = sum(
            c * Decimal(2999) ** i * (Decimal("1e-10") if i >= 12 else 1)
            for i, c in enumerate(coefficients)
        )
    print(
        f"g(2999) exact-decimal polynomial={polynomial}; Euler-integral={scattering_function(2999):.16g}"
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

