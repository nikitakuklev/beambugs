---
title: "COL-47: Tail-cut Coulomb logarithms exceed the classical scattering bound"
hide:
  - toc
---

# COL-47 — Tail-cut Coulomb logarithms exceed the classical scattering bound

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [multi_coulomb_log options 2–4](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/ibs_mod.f90#L764-L804)<br>[“Coulomb Logarithm”, equation e:tailcut](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_ring/doc/ibs_ring.tex#L300-L345) |

Selecting a tail-cut Coulomb logarithm with a long radiation-damping time lets the
intrabeam-scattering calculation include collisions beyond its classical scattering limit.
Options 3 and 4 omit the documented maximum of the classical and damping-time minimum impact
parameters; option 2 likewise fails to bound the maximum momentum transfer. At a damping time
of 10⁸ s in the proton example, the respective rate excesses are 14.27%, 10.69% and 12.58%.
Strongly damped cases keep the tail cut inside the classical bound.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)

??? example "Reproducer: COL-47.py"

    [Download COL-47.py](COL-47.py)

    ```python
    """COL-47: Tail-cut Coulomb logarithms exceed the classical scattering bound

    Selecting a tail-cut Coulomb logarithm with a long radiation-damping time lets the
    intrabeam-scattering calculation include collisions beyond its classical scattering limit.
    Options 3 and 4 omit the documented maximum of the classical and damping-time minimum impact
    parameters; option 2 likewise fails to bound the maximum momentum transfer. At a damping time
    of 10⁸ s in the proton example, the respective rate excesses are 14.27%, 10.69% and 12.58%.
    Strongly damped cases keep the tail cut inside the classical bound.

    The script sweeps the radiation-damping time of a proton element and compares options 3 and 4
    with the manual’s lower-impact-parameter bound. The classic log and strongly damped cases are
    controls. The companion script checks the Raubenheimer option against its classical momentum-
    transfer bound.
    """

    import math

    from _bmad import compare, fortran, verdict

    PROGRAM = """! multi_coulomb_log options 1-4 for one proton element (gamma = 100, beta_a = beta_b = 10 m,
    ! emit_a = emit_b = 1e-8 m, sigma_p = 1e-3, no dispersion, line density 1e12 /m) while the
    ! damping time tau_a grows. Columns: tau_a, then the four logs.
    program probe
    use bmad
    use ibs_mod
    implicit none
    type (lat_struct), target :: lat
    type (ele_struct), pointer :: ele
    type (ibs_sim_param_struct) :: cfg
    real(rp) :: clog(4)
    real(rp), parameter :: lambda = 1e12_rp
    integer :: i, k

    call bmad_parser('proton.bmad', lat)
    ele => lat%ele(1)
    ele%a%beta = 10;  ele%a%alpha = 0;  ele%a%gamma = 0.1_rp;  ele%a%eta = 0;  ele%a%etap = 0
    ele%b%beta = 10;  ele%b%alpha = 0;  ele%b%gamma = 0.1_rp;  ele%b%eta = 0;  ele%b%etap = 0
    ele%a%emit = 1e-8_rp;  ele%b%emit = 1e-8_rp;  ele%z%sigma_p = 1e-3_rp
    write(*, '(a,es25.16)') 'RADIUS ', classical_radius(proton$)
    write(*, '(a)') '#      tau_a(s)            clog1                 clog2                 clog3                 clog4'
    do k = -2, 12, 2
      cfg%tau_a = 10.0_rp**k
      do i = 1, 4
        cfg%clog_to_use = i
        call multi_coulomb_log(cfg, ele, clog(i), lambda)
      enddo
      write(*, '(es14.6, 4f22.15)') cfg%tau_a, clog
    enddo
    end program
    """
    log = fortran(
        PROGRAM,
        files={
            "proton.bmad": """! One proton drift at gamma = 100; the probe sets the Twiss and beam sizes.
    parameter[particle] = proton
    parameter[geometry] = open
    beginning[e_tot] = 100 * mass_of(proton)
    d: drift, l = 1
    l1: line = (d)
    use, l1
    """
        },
    ).log
    radius_rows = [line.split()[1] for line in log.splitlines() if line.startswith("RADIUS ")]
    rows = [
        list(map(float, line.split()))
        for line in log.splitlines()
        if len(line.split()) == 5 and line.lstrip()[0].isdigit()
    ]
    if len(radius_rows) != 1 or len(rows) != 8:
        raise RuntimeError(log)
    radius = float(radius_rows[0])
    c, gamma, beta, emit, sig_p, density = 299792458.0, 100.0, 10.0, 1e-8, 1e-3, 1e12
    sigma = math.sqrt(beta * emit)
    volume = (4 * math.pi) ** 1.5 * sigma**2 * gamma
    bmax = min(sigma, (volume / density) ** (1 / 3))
    classic = math.log(gamma**2 * sigma * emit / (radius * beta))
    defect = False
    for tau, log1, log2, log3, log4 in rows:
        if compare(f"tau={tau:g} classic control", log1, classic, atol=2e-13):
            raise RuntimeError("Classic logarithm control failed")
        for option, angle, observed in [
            (3, math.sqrt(emit / beta), log3),
            (4, math.sqrt(2 * emit / beta + (sig_p / gamma) ** 2), log4),
        ]:
            tail = math.sqrt(volume / density / math.pi / (tau / gamma)) / math.sqrt(c * gamma * angle)
            # The documented pi/2 scattering bound is the larger minimum impact parameter.
            minimum = radius / (angle * gamma) ** 2
            expected = math.log(bmax / max(tail, minimum))
            bad = compare(
                f"option {option} tau={tau:g} s; ln(bmax/max(classical,tail))",
                observed,
                expected,
                atol=2e-13,
            )
            if tau <= 1e4 and bad:
                raise RuntimeError("Strong-damping control failed")
            defect |= bad
    verdict(defect)

    ```

??? example "Reproducer: COL-47-option2.py"

    [Download COL-47-option2.py](COL-47-option2.py)

    ```python
    """COL-47: Raubenheimer tail-cut bound

    The script independently integrates the Raubenheimer expression over the production integration interval and caps its maximum momentum transfer at the classical π/2 boundary. It checks dispersive and dispersion-free elements over eight damping times, retaining the separately measured quadrature residual.
    """

    import math

    from _bmad import compare, fortran, verdict
    from scipy.integrate import quad

    PROGRAM = """! multi_coulomb_log options 1 and 2 (Raubenheimer tail cut) for one proton element at gamma 100
    ! while the damping time tau_a grows from 1e-2 s to 1e12 s, line density 1e12 /m:
    !   nodisp -- the COL-47 element: beta_a = beta_b = 10 m, emit_a = emit_b = 1e-8 m, sigma_p = 1e-3;
    !   disp   -- the same with alpha_a = 0.5, eta_a = 1 m, etap_a = 0.05.
    ! Columns: label, beta_a, alpha_a, eta_a, etap_a, beta_b, emit_a, emit_b, sigma_p, lambda, tau_a,
    ! clog1, clog2. The Python oracle rebuilds option 2 from Raubenheimer's Eqs. (12) and (15).
    program probe
    use bmad
    use ibs_mod
    implicit none
    type (lat_struct), target :: lat
    type (ele_struct), pointer :: ele
    type (ibs_sim_param_struct) :: cfg
    integer :: k

    call bmad_parser('proton.bmad', lat)
    write(*, '(a,es25.16)') 'RADIUS ', classical_radius(proton$)
    ele => lat%ele(1)
    do k = -2, 12, 2
      call one('nodisp', 10.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 10.0_rp**k)
    enddo
    do k = -2, 12, 2
      call one('disp', 10.0_rp, 0.5_rp, 1.0_rp, 0.05_rp, 10.0_rp**k)
    enddo

    contains

    subroutine one(label, beta_a, alpha_a, eta_a, etap_a, tau_a)
    character(*) :: label
    real(rp) :: beta_a, alpha_a, eta_a, etap_a, tau_a, clog(2)
    real(rp), parameter :: lambda = 1e12_rp
    integer :: i
    ele%a%beta = beta_a;  ele%a%alpha = alpha_a;  ele%a%gamma = (1 + alpha_a**2) / beta_a
    ele%a%eta = eta_a;  ele%a%etap = etap_a
    ele%b%beta = 10;  ele%b%alpha = 0;  ele%b%gamma = 0.1_rp;  ele%b%eta = 0;  ele%b%etap = 0
    ele%a%emit = 1e-8_rp;  ele%b%emit = 1e-8_rp;  ele%z%sigma_p = 1e-3_rp
    cfg%tau_a = tau_a
    do i = 1, 2
      cfg%clog_to_use = i
      call multi_coulomb_log(cfg, ele, clog(i), lambda)
    enddo
    write(*, '(a, 1x, 10es24.16, 2f22.15)') label, beta_a, alpha_a, eta_a, etap_a, ele%b%beta, ele%a%emit, &
                                          ele%b%emit, ele%z%sigma_p, lambda, tau_a, clog
    end subroutine

    end program
    """
    log = fortran(
        PROGRAM,
        files={
            "proton.bmad": """! One proton drift; the probe sets the Twiss and beam sizes at gamma 100.
    parameter[particle] = proton
    parameter[geometry] = open
    beginning[e_tot] = 100 * mass_of(proton)
    d: drift, l = 1
    l1: line = (d)
    use, l1
    """
        },
    ).log
    radius = [float(line.split()[1]) for line in log.splitlines() if line.startswith("RADIUS ")]
    rows = [line.split() for line in log.splitlines() if line.startswith(("nodisp ", "disp "))]
    if len(radius) != 1 or len(rows) != 16:
        raise RuntimeError(log)
    r, gamma, c = radius[0], 100.0, 299792458.0
    defect = False
    for row in rows:
        label = row[0]
        ba, aa, eta, etap, bb, ea, eb, sp, density, tau, log1, observed = map(float, row[1:])
        g2, sp2 = gamma**2, sp**2
        ha = (eta**2 + (ba * etap + aa * eta) ** 2) / ba
        u = g2 * (ha / ea + 1 / sp2 + ba / g2 / ea + bb / g2 / eb)
        v = g2 * (
            ha * bb / ea / eb + ba * bb / g2 / ea / eb + ba / sp2 / ea + bb / sp2 / eb + eta**2 / ea**2
        )
        w = g2 * (eta**2 * bb / ea**2 / eb + ba * bb / sp2 / ea / eb)

        # Raubenheimer Eq. (15), x=exp(2y). Scaling the integrand prevents a tiny
        # absolute tolerance from terminating integration before relative accuracy is reached.
        def integrand(y, u=u, v=v, w=w):
            return (
                2e10
                * math.exp(y)
                / math.sqrt(math.exp(6 * y) + u * math.exp(4 * y) + v * math.exp(2 * y) + w)
            )

        integral, error = quad(integrand, 0, 40, epsabs=0, epsrel=2e-12)
        if error > abs(integral) * 1e-10:
            raise RuntimeError("Independent quadrature did not converge")
        bbar = gamma * math.sqrt((1 + aa**2) / ba * ea + eb / bb + sp2 / g2)
        qmin = 2 * r / math.sqrt(bb * eb) / bbar
        qstar = math.sqrt(
            tau * density * c * r * r / (4 * math.pi * g2 * ea * eb * sp) * integral / 1e10
        )
        reconstructed = math.log(qstar / qmin)
        cap = math.log(bbar / qmin)
        # The existing quadrature has a separately established <8e-5 log offset.
        # Bound violations are O(1); controls below the cap isolate that offset.
        if abs(observed - reconstructed) > 8e-5:
            raise RuntimeError("Option-2 result differs from its independently integrated formula")
        expected = min(reconstructed, cap)
        print(
            f"{label} tau={tau:g} s: independent integral={integral / 1e10:.16g} quadrature log residual={observed - reconstructed:.8g}"
        )
        bad = compare(
            f"{label} option 2 tau={tau:g} s; ln(min(qstar,Bbar)/qmin)",
            observed,
            expected,
            # Once capped, the expected log is algebraic and has no quadrature error.
            atol=2e-13 if qstar >= bbar else 8e-5,
        )
        if tau <= 1e4 and bad:
            raise RuntimeError("Strong-damping option-2 control failed")
        defect |= bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-24 | ibs-ion-classical-radius (`886607535af8`), source, production, CPU, double, serial |
| Confirmed (tail-cut options 3 and 4) | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed (tail-cut option 2) | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |

