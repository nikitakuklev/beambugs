---
title: "COL-27: Longitudinal IBS equilibrium balances RMS spread instead of variance"
hide:
  - toc
---

# COL-27 — Longitudinal IBS equilibrium balances RMS spread instead of variance

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [ibs_equib_der, bmad/multiparticle/ibs_mod.f90:279–324](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/ibs_mod.f90#L279-L324)<br>[ibs_equib_rlx, bmad/multiparticle/ibs_mod.f90:160–191](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/ibs_mod.f90#L160-L191)<br>[IBS Ring manual, Methods for determining equilibrium emittances, bsim/ibs_ring/doc/ibs_ring.tex:243–277](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_ring/doc/ibs_ring.tex#L243-L277) |

Both IBS equilibrium solvers balance radiation damping against IBS growth using RMS energy spread instead of its variance. For amplitude damping time τ, RMS IBS growth rate r and zero-current RMS spread σ₀, equilibrium requires RMS spread σ = σ₀/√(1 − τr), with r evaluated self-consistently. The solvers instead use σ = σ₀/(1 − τr). In a 2 GeV electron case, the derivative solver returns relative energy spread 0.001221 instead of 0.001139, an excess of 7.241%; the relaxation solver gives the same erroneous balance.

Run `python COL-27.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-27.py"

    [Download COL-27.py](COL-27.py)

    ```python
    """COL-27: Longitudinal IBS equilibrium balances RMS spread instead of variance

    Both IBS equilibrium solvers balance radiation damping against IBS growth using RMS energy spread instead of its variance. For amplitude damping time τ, RMS IBS growth rate r and zero-current RMS spread σ₀, equilibrium requires RMS spread σ = σ₀/√(1 − τr), with r evaluated self-consistently. The solvers instead use σ = σ₀/(1 − τr). In a 2 GeV electron case, the derivative solver returns relative energy spread 0.001221 instead of 0.001139, an excess of 7.241%; the relaxation solver gives the same erroneous balance.

    The script compiles a one-element equilibrium driver against BMAD_DIST and runs both solvers. It independently solves the coupled transverse-emittance and longitudinal-variance equations with Bane quadrature and prints the physical variance residual.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from scipy.integrate import quad
    from scipy.optimize import root

    PROGRAM = """program probe_equilibrium
    use bmad
    use ibs_mod
    implicit none
    type(lat_struct), target :: lat
    type(normal_modes_struct) m0, m1
    type(ibs_sim_param_struct) params
    type(ibs_struct) rates
    integer i,k
    real(rp) tau, residual
    call init_lat(lat,2)
    lat%n_ele_track = 1
    lat%param%particle = electron$
    lat%param%total_length = 100.0_rp
    lat%param%n_part = 1.0e11_rp
    do i=0,2
      lat%ele(i)%key = drift$
      lat%ele(i)%value(l$) = 100.0_rp
      lat%ele(i)%value(e_tot$) = 2.0e9_rp
      lat%ele(i)%a%beta = 10.0_rp
      lat%ele(i)%b%beta = 20.0_rp
      lat%ele(i)%a%gamma = 0.1_rp
      lat%ele(i)%b%gamma = 0.05_rp
      lat%ele(i)%a%eta = 0.1_rp
      lat%ele(i)%b%eta = 0.01_rp
    enddo
    m0%a%emittance = 1.0e-9_rp
    m0%b%emittance = 1.0e-11_rp
    m0%sigE_E = 1.0e-3_rp
    m0%sig_z = 0.01_rp
    m0%z%emittance = m0%sig_z*m0%sigE_E
    tau = 0.01_rp
    m0%a%alpha_damp = lat%param%total_length/c_light/tau
    m0%b%alpha_damp = m0%a%alpha_damp
    m0%z%alpha_damp = m0%a%alpha_damp
    params%clog_to_use = 1
    params%formula = 'bane'
    do k=1,2
      if(k == 1) then
        call ibs_equib_der(lat,params,m0,m1,-1.0_rp)
      else
        call ibs_equib_rlx(lat,params,m0,m1,0.0_rp,[5.0_rp,5.0_rp,5.0_rp],-1.0_rp)
      endif
      lat%ele(1)%a%emit = m1%a%emittance
      lat%ele(1)%b%emit = m1%b%emittance
      lat%ele(1)%z%sigma_p = m1%sigE_E
      lat%ele(1)%z%sigma = m1%sig_z
      call ibs_rates1turn(lat,params,rates,-1.0_rp)
      residual = 1-(m0%sigE_E/m1%sigE_E)**2-tau*rates%inv_Tp
      write(*,'(a,i1)') 'method (1=der,2=rlx): ',k
      write(*,'(a,3es23.14)') 'ea, eb, sigma_p: ',m1%a%emittance,m1%b%emittance,m1%sigE_E
      write(*,'(a,3es23.14)') 'tau*rp, sigma0/sigma, variance residual: ',tau*rates%inv_Tp,m0%sigE_E/m1%sigE_E,residual
    enddo
    end program
    """
    output = fortran(PROGRAM).log

    gamma = 2e9 / 510998.95069
    r = 2.8179403227e-15
    c = 299792458.0
    ea0, eb0, sp0 = 1e-9, 1e-11, 1e-3
    tau = 0.01
    ba, bb, da, db = 10.0, 20.0, 0.1, 0.01
    ha, hb = da * da / ba, db * db / bb


    def rates(v):
        ea, eb, sp = np.exp(v) * [ea0, eb0, sp0]
        sh = 1 / np.sqrt(1 / sp**2 + ha / ea + hb / eb)
        alpha = np.sqrt(ba / ea / (bb / eb))
        g = (
            2
            * np.sqrt(alpha)
            / np.pi
            * quad(
                lambda x: 1 / np.sqrt(1 + x * x) / np.sqrt(alpha * alpha + x * x),
                0,
                np.inf,
                epsabs=1e-11,
                epsrel=1e-11,
            )[0]
        )
        clog = np.log(gamma**2 * np.sqrt(bb * eb + db * db * sp * sp) * ea / r / ba)
        # The solvers keep sigma_z/sigma_p = 10 m, so density = N/(10*sigma_p).
        rp = (
            r
            * r
            * c
            * (1e11 / (10 * sp))
            * clog
            * sh
            * g
            / (16 * gamma**3 * ea**0.75 * eb**0.75 * sp**3 * (ba * bb) ** 0.25)
        )
        return np.array([sp * sp * ha / ea * rp, sp * sp * hb / eb * rp, rp])


    # Amplitude damping of sigma makes variance relax at twice that rate.
    # The longitudinal equilibrium therefore uses 1-(sigma0/sigma)^2, not 1-sigma0/sigma.
    def residual(v):
        ea, eb, sp = np.exp(v) * [ea0, eb0, sp0]
        return np.array([1 - ea0 / ea, 1 - eb0 / eb, 1 - (sp0 / sp) ** 2]) - tau * rates(v)


    sol = root(residual, [0.2, 0.1, 0.1], tol=1e-11)
    assert sol.success, sol.message
    ea, eb, sp = np.exp(sol.x) * [ea0, eb0, sp0]
    print("Independent self-consistent equilibrium from Bane integrals + variance balance:")
    print("ea, eb, sigma_p =", ea, eb, sp)
    print("variance-balance residual:", residual(sol.x))

    production_states = np.array(
        [
            [float(x) for x in line.split(":", 1)[1].split()]
            for line in output.splitlines()
            if line.startswith("ea, eb, sigma_p:")
        ]
    )
    actual = production_states[:, 2]
    production_rate_times_tau = np.array(
        [
            float(line.split(":", 1)[1].split()[0])
            for line in output.splitlines()
            if line.startswith("tau*rp, sigma0/sigma, variance residual:")
        ]
    )
    balance = np.array(
        [
            float(line.split(":", 1)[1].split()[-1])
            for line in output.splitlines()
            if line.startswith("tau*rp, sigma0/sigma, variance residual:")
        ]
    )
    if production_states.shape != (2, 3) or balance.shape != (2,):
        raise RuntimeError("Both equilibrium methods must complete")
    independent_rate_times_tau = np.array(
        [tau * rates(np.log(state / [ea0, eb0, sp0]))[2] for state in production_states]
    )
    # Validate the independent scattering model at both production equilibria before
    # attributing the spread discrepancy to the longitudinal balance equation.
    if compare(
        "Bane rate controls at production equilibria, tau*r",
        production_rate_times_tau,
        independent_rate_times_tau,
        atol=0,
        rtol=2e-11,
    ):
        raise RuntimeError("Independent Bane rates disagree with the production rates")
    print("production physical variance residuals:", balance)
    if np.max(np.abs(residual(sol.x))) > 1e-10:
        raise RuntimeError("The independent equilibrium did not converge")
    verdict(
        compare(
            "der/rlx RMS relative energy spread, variance balance",
            actual,
            np.full(2, sp),
            atol=0,
            rtol=1e-5,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

