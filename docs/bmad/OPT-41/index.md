---
title: "OPT-41: Runge–Kutta integrated matrices reverse physical-time perturbations"
hide:
  - toc
---

# OPT-41 — Runge–Kutta integrated matrices reverse physical-time perturbations

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on Bmad 20261031-0 (`c4c629f`, committed 2026-10-31) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [RK matrix construction, bmad/modules/runge_kutta_mod.f90:426–450](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/runge_kutta_mod.f90#L426-L450)<br>[adjust_start, bmad/modules/runge_kutta_mod.f90:473–482](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/runge_kutta_mod.f90#L473-L482)<br>[z=βc(t_ref−t) contract, bmad/doc/phase-space.tex:81–94](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/phase-space.tex#L81-L94)<br>[mat6 documentation, bmad/doc/methods.tex:253–280](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L253-L280)<br>[selector documentation, bmad/doc/methods.tex:286–314](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L286-L314)<br>[auto dispatch, bmad/code/make_mat6.f90:84–95](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/make_mat6.f90#L84-L95)<br>[valid for the custom-field `elseparator` fixture, bmad/code/valid_mat6_calc_method.f90:122–126](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/valid_mat6_calc_method.f90#L122-L126)<br>[explicit dispatch, bmad/code/make_mat6.f90:133–137](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/make_mat6.f90#L133-L137)<br>[track1 with make_map1, bmad/low_level/make_mat6_bmad.f90:66–76](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/make_mat6_bmad.f90#L66-L76)<br>[shared s-based RK matrix path, bmad/code/track1.f90:238–239](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/track1.f90#L238-L239)<br>[Native RF fields, bmad/code/em_field_calc.f90:404–465](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/em_field_calc.f90#L404-L465)<br>[particle_rf_time, bmad/code/particle_rf_time.f90:74–100](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/particle_rf_time.f90#L74-L100) |

With s-based Runge–Kutta tracking and mat6_calc_method=bmad_standard, the transfer matrix is built from perturbed particles whose time shift has the wrong sign. Bmad's z = −βc(t − t₀), with particle speed β, means a particle displaced by +Δz at fixed s arrives Δz/(βc) earlier; the matrix code makes it arrive that much later. Fields read at absolute time, including native RF with absolute-time tracking, then give wrong longitudinal matrix elements in adaptive and fixed-step tracking. A 1.251 GeV/c proton in a 0.4000 m, 10.00 MV, 374.7 MHz traveling-wave cavity gives M₆₅ = ∂δ_out/∂z_in = −0.05309 m⁻¹ instead of +0.05320 m⁻¹ from separately tracked particles. Default tracking-based matrices, relative-time RF and static fields are correct.

Run `python OPT-41.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: OPT-41.py"

    [Download OPT-41.py](OPT-41.py)

    ```python
    """OPT-41: Runge–Kutta integrated matrices reverse physical-time perturbations

    With s-based Runge–Kutta tracking and mat6_calc_method=bmad_standard, the transfer matrix is built from perturbed particles whose time shift has the wrong sign. Bmad's z = −βc(t − t₀), with particle speed β, means a particle displaced by +Δz at fixed s arrives Δz/(βc) earlier; the matrix code makes it arrive that much later. Fields read at absolute time, including native RF with absolute-time tracking, then give wrong longitudinal matrix elements in adaptive and fixed-step tracking. A 1.251 GeV/c proton in a 0.4000 m, 10.00 MV, 374.7 MHz traveling-wave cavity gives M₆₅ = ∂δ_out/∂z_in = −0.05309 m⁻¹ instead of +0.05320 m⁻¹ from separately tracked particles. Default tracking-based matrices, relative-time RF and static fields are correct.

    Compare public matrix selectors for fixed-step and adaptive Runge–Kutta in a time-dependent longitudinal field and a native traveling-wave cavity. An exact momentum-versus-time solution with independent flight-distance quadrature supplies the custom-field orbit and Jacobian, while separate physical particle perturbations check native RF. Static-field, relative-time and default-matrix controls run in fresh native processes using BMAD_DIST.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from scipy.integrate import quad
    from scipy.optimize import brentq

    LENGTH, MU, Z0, U0, G0 = 0.4, 0.75, 0.015, 1.03, 0.02

    PROGRAM = r"""
    module test_field
    use bmad
    implicit none
    real(rp) slope
    logical relative_clock
    contains
    recursive subroutine field_fn(ele,param,s_rel,orbit,local_ref_frame,field,calc_dfield,err_flag, &
     calc_potential,use_overlap,grid_allow_s_out_of_bounds,rf_time,used_eles)
    type(ele_struct),target::ele
    type(lat_param_struct)::param
    type(coord_struct),intent(in)::orbit
    type(ele_pointer_struct),allocatable,optional::used_eles(:)
    real(rp),intent(in)::s_rel
    real(rp),optional::rf_time
    logical local_ref_frame
    type(em_field_struct)::field
    logical,optional::calc_dfield,err_flag,calc_potential,use_overlap,grid_allow_s_out_of_bounds
    real(rp) tau
    tau=c_light*orbit%t
    if(relative_clock) tau=c_light*particle_rf_time(orbit,ele,.false.,s_rel)
    ! Scalar potential Phi = -P0c*(0.02 + slope*c*t)*s [V]; A = 0.
    field=em_field_struct()
    field%E(3)=ele%value(p0c$)*(0.02_rp+slope*tau)
    if(present(err_flag)) err_flag=.false.
    end subroutine
    end module

    program confirm
    use test_field
    implicit none
    type(lat_struct),target::lat
    type(coord_struct)::a,b,p,m,ep,em
    real(rp)::v(6),h,fd(2,2)
    integer j,k
    logical err
    open(10,file='settings');read(10,*) slope,relative_clock,h;close(10)
    em_field_custom_ptr=>field_fn
    bmad_com%absolute_time_tracking=.not.relative_clock
    bmad_com%rel_tol_adaptive_tracking=1e-12_rp
    bmad_com%abs_tol_adaptive_tracking=1e-14_rp
    ! RK advances past a hard edge by significant_length/100 without integration.
    ! Make that boundary guard negligible at the oracle's sub-picometre tolerance.
    bmad_com%significant_length=1e-13_rp
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) error stop 'parse'
    v=0;v(5)=0.015_rp;v(6)=0.03_rp
    call init_coord(a,v,lat%ele(1),upstream_end$)
    a%t=-a%vec(5)/(a%beta*c_light)
    write(*,'(a,3es25.16)') 'START ',a%beta,a%p0c,mass_of(proton$)
    call track1(a,lat%ele(1),lat%param,b,err_flag=err)
    if(err.or.b%state/=alive$) error stop 'baseline'
    write(*,'(a,4es25.16)') 'END ',b%vec(5:6),c_light*b%t,b%beta
    lat%ele(1)%mat6_calc_method=auto$
    call make_mat6(lat%ele(1),lat%param,a,err_flag=err)
    if(err) error stop 'auto matrix'
    write(*,'(a,4es25.16)') 'AUTO ',lat%ele(1)%mat6(5:6,5),lat%ele(1)%mat6(5:6,6)
    lat%ele(1)%mat6_calc_method=tracking$
    call make_mat6(lat%ele(1),lat%param,a,err_flag=err)
    if(err) error stop 'tracking matrix'
    write(*,'(a,4es25.16)') 'TRACKING ',lat%ele(1)%mat6(5:6,5),lat%ele(1)%mat6(5:6,6)
    lat%ele(1)%mat6_calc_method=bmad_standard$
    if(.not.valid_mat6_calc_method(lat%ele(1),proton$,bmad_standard$)) error stop 'selector'
    call make_mat6(lat%ele(1),lat%param,a,err_flag=err)
    if(err) error stop 'integrated matrix'
    write(*,'(a,4es25.16)') 'INTEGRATED ',lat%ele(1)%mat6(5:6,5),lat%ele(1)%mat6(5:6,6)
    do j=1,2
     k=j+4;p=a;m=a;p%vec(k)=p%vec(k)+h;m%vec(k)=m%vec(k)-h
     call convert_pc_to(p%p0c*(1+p%vec(6)),p%species,beta=p%beta)
     call convert_pc_to(m%p0c*(1+m%vec(6)),m%species,beta=m%beta)
     p%t=-p%vec(5)/(p%beta*c_light);m%t=-m%vec(5)/(m%beta*c_light)
     call track1(p,lat%ele(1),lat%param,ep,err_flag=err)
     if(err.or.ep%state/=alive$) error stop 'positive'
     call track1(m,lat%ele(1),lat%param,em,err_flag=err)
     if(err.or.em%state/=alive$) error stop 'negative'
     fd(:,j)=(ep%vec(5:6)-em%vec(5:6))/(2*h)
    enddo
    write(*,'(a,4es25.16)') 'FD ',fd(:,1),fd(:,2)
    end program
    """

    LAT = """no_digested
    parameter[particle]=proton
    parameter[geometry]=open
    beginning[p0c]=mass_of(proton)/0.75
    e:elseparator,l=0.4,field_calc=custom,tracking_method={method},ds_step={step},mat6_calc_method=tracking,fringe_type=none
    line1:line=(e)
    use,line1
    """


    def oracle(slope):
        """Differentiate the fixed flight-distance constraint analytically."""
        beta_i = U0 / np.hypot(U0, MU)
        tau_i = -Z0 / beta_i

        def momentum(tau):
            return U0 + G0 * (tau - tau_i) + slope * (tau**2 - tau_i**2) / 2

        def beta(tau):
            u = momentum(tau)
            return u / np.hypot(u, MU)

        tau_f = brentq(
            lambda tau: quad(beta, tau_i, tau, epsabs=2e-14, epsrel=2e-14)[0] - LENGTH,
            tau_i + LENGTH,
            tau_i + 2 * LENGTH,
            xtol=5e-15,
            rtol=1e-14,
        )
        uf = momentum(tau_f)
        bf = beta(tau_f)
        j = quad(
            lambda tau: MU**2 / np.hypot(momentum(tau), MU) ** 3,
            tau_i,
            tau_f,
            epsabs=2e-14,
            epsrel=2e-14,
        )[0]
        ai, af = G0 + slope * tau_i, G0 + slope * tau_f
        # Coordinates are (z,delta). At fixed entry reference clock, tau_i=-z/beta_i.
        dui = np.array([0.0, 1.0])
        dti = np.array([-1 / beta_i, Z0 * MU**2 / np.hypot(U0, MU) ** 3 / beta_i**2])
        # Differentiate L=integral(beta d_tau) at fixed spatial exit.
        dtf = ((beta_i + ai * j) * dti - j * dui) / bf
        duf = dui + af * dtf - ai * dti
        reference_tau = LENGTH * np.hypot(1.0, MU)
        # z_f=beta_f*(L/beta0-tau_f); acceleration also changes beta_f.
        dzf = MU**2 / np.hypot(uf, MU) ** 3 * (reference_tau - tau_f) * duf - bf * dtf
        matrix = np.array([dzf, duf])
        end = np.array([bf * (reference_tau - tau_f), uf - 1, tau_f, bf])
        return matrix.ravel(order="F"), end


    def custom_cases():
        defects = []
        configs = [
            ("static", 0.0, False, "fixed_step_runge_kutta", 0.01, 1e-5),
            ("absolute", 0.04, False, "fixed_step_runge_kutta", 0.01, 1e-5),
            ("absolute_fine", 0.04, False, "fixed_step_runge_kutta", 0.005, 3e-6),
            ("absolute_adaptive", 0.04, False, "runge_kutta", 0.005, 3e-6),
            ("relative", 0.04, True, "fixed_step_runge_kutta", 0.005, 3e-6),
        ]
        for name, slope, relative, method, step, h in configs:
            raw = fortran(
                PROGRAM,
                files={
                    "lat.bmad": LAT.format(method=method, step=step),
                    "settings": f"{slope} {'T' if relative else 'F'} {h}",
                },
            ).log
            rows = {
                s.split()[0]: np.array([float(x) for x in s.split()[1:]])
                for s in raw.splitlines()
                if s.startswith(("START ", "END ", "AUTO ", "TRACKING ", "INTEGRATED ", "FD "))
            }
            expected, end = oracle(slope)
            print(f"CASE {name}: oracle=[M55,M65,M56,M66]={expected.tolist()}")
            if compare(
                f"{name} orbit: exact time-polynomial momentum plus flight integral",
                rows["END"],
                end,
                atol=8e-13,
            ):
                raise AssertionError("orbit disagrees with independent oracle")
            for key in ("AUTO", "TRACKING", "FD"):
                if compare(f"{name} {key} matrix", rows[key], expected, atol=4e-10):
                    raise AssertionError("control failed")
            bad = compare(f"{name} INTEGRATED matrix", rows["INTEGRATED"], expected, atol=4e-10)
            if slope and not relative:
                defects.append(bad)
            elif bad:
                raise AssertionError("static/relative integrated matrix control failed")
        return defects


    NATIVE_LAT = """no_digested
    parameter[particle]=proton
    parameter[geometry]=open
    beginning[p0c]=mass_of(proton)/0.75
    e:rfcavity,l=.4,rf_frequency=c_light/.8,voltage=1e7,phi0=.13,cavity_type=traveling_wave, &
     tracking_method={method},mat6_calc_method=tracking,ds_step=.005
    line1:line=(e)
    use,line1
    """


    def native_cases():
        defects = []
        for relative in (False, True):
            for method in ("fixed_step_runge_kutta", "runge_kutta"):
                name = f"native_{'relative' if relative else 'absolute'}_{method}"
                result = fortran(
                    PROGRAM,
                    files={
                        "lat.bmad": NATIVE_LAT.format(method=method),
                        "settings": f"0 {'T' if relative else 'F'} 3e-6",
                    },
                )
                rows = {
                    s.split()[0]: np.array([float(x) for x in s.split()[1:]])
                    for s in result.log.splitlines()
                    if s.startswith(("AUTO ", "TRACKING ", "INTEGRATED ", "FD "))
                }
                for key in ("AUTO", "TRACKING"):
                    if compare(
                        f"{name} {key} versus external particle difference",
                        rows[key],
                        rows["FD"],
                        atol=4e-10,
                    ):
                        raise AssertionError("physical-difference control failed")
                bad = compare(
                    f"{name} INTEGRATED versus external particle difference",
                    rows["INTEGRATED"],
                    rows["FD"],
                    atol=4e-10,
                )
                if relative:
                    if bad:
                        raise AssertionError("relative RF matrix control failed")
                else:
                    defects.append(bad)
        return defects


    if __name__ == "__main__":
        verdict(all(custom_cases() + native_cases()))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `claude-opus-5-5` | high | 2026-11-07 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

