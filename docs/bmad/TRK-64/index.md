---
title: "TRK-64: Legacy standing-wave lcavities lose finite transverse focusing near RF zero crossing"
hide:
  - toc
---

# TRK-64 — Legacy standing-wave lcavities lose finite transverse focusing near RF zero crossing

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [Native dispatch, bmad/low_level/track1_bmad.f90:111–115](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track1_bmad.f90#L111-L115)<br>[standing-wave coefficient branch, bmad/low_level/track_a_lcavity_old.f90:192–201](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity_old.f90#L192-L201)<br>[analytic transverse matrices, same file:231–250](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity_old.f90#L231-L250)<br>[both tracked planes with finite-beta and edge terms, same file:305–345](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity_old.f90#L305-L345)<br>[routine contract, same file:381–391](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity_old.f90#L381-L391)<br>[standing-wave transport contract, bmad/doc/elements.tex:3398–3402](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L3398-L3402)<br>[averaged focusing force, bmad/doc/charged-tracking.tex:1236–1268](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/charged-tracking.tex#L1236-L1268)<br>[Phase units, elements.tex:3285–3286](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L3285-L3286)<br>[manual cavity example, same file:3423–3426](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L3423-L3426)<br>[midpoint phase predictor, track_a_lcavity_old.f90:518–558](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity_old.f90#L518-L558)<br>[Default lcavity attributes, bmad/parsing/set_ele_defaults.f90:204–215](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/set_ele_defaults.f90#L204-L215)<br>[active-length intervals, bmad/low_level/lcavity_rf_step_setup.f90:225–250](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/lcavity_rf_step_setup.f90#L225-L250)<br>[finite ponderomotive kicks, bmad/low_level/track_a_lcavity.f90:429–477](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L429-L477)<br>[Automatic matrix selection, bmad/code/make_mat6.f90:84–95](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/make_mat6.f90#L84-L95)<br>[linear-map preparation, same file:114–137](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/make_mat6.f90#L114-L137)<br>[temporary native tracking for linear matrices, bmad/low_level/make_mat6_bmad.f90:67–78](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/make_mat6_bmad.f90#L67-L78) |

With n_rf_steps=0, a standing-wave lcavity near RF zero crossing overestimates transverse displacement because its small-net-gain branch replaces sin(α)/α by one, where α is the focusing phase. In a 2.300 m, 20.00 MV, 500.0 MHz cavity at zero crossing, a 40.00 MeV total-energy electron launched with normalized horizontal momentum 1.000×10⁻⁴ exits at 230.0 µm instead of 228.8 µm, an error of 1.196 µm (0.5227%). Independent integration of the averaged transverse equation and its constant-energy oscillator limit establish the result. Both transverse planes and legacy transfer matrices are affected; positive-n_rf_steps tracking avoids this branch.

Run `python TRK-64.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: TRK-64.py"

    [Download TRK-64.py](TRK-64.py)

    ```python
    """TRK-64: Legacy standing-wave lcavities lose finite transverse focusing near RF zero crossing

    With n_rf_steps=0, a standing-wave lcavity near RF zero crossing overestimates transverse displacement because its small-net-gain branch replaces sin(α)/α by one, where α is the focusing phase. In a 2.300 m, 20.00 MV, 500.0 MHz cavity at zero crossing, a 40.00 MeV total-energy electron launched with normalized horizontal momentum 1.000×10⁻⁴ exits at 230.0 µm instead of 228.8 µm, an error of 1.196 µm (0.5227%). Independent integration of the averaged transverse equation and its constant-energy oscillator limit establish the result. Both transverse planes and legacy transfer matrices are affected; positive-n_rf_steps tracking avoids this branch.

    Parse deterministic electron cavities with the BMAD_DIST production build and run
    public track1 and make_mat6 clients through the shared runner in fresh processes.
    Compare transverse transport with independent averaged-equation integration and an
    80-digit oscillator matrix exponential, and check zero-field, finite-gain,
    traveling-wave and default stepped controls using their own model.
    """

    from __future__ import annotations

    import math
    import os
    from dataclasses import dataclass

    for _variable in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NPROC", "PJRT_NPROC",
    ):
        os.environ[_variable] = "1"

    import mpmath as mp
    import numpy as np
    from _bmad import fortran, verdict
    from scipy.integrate import solve_ivp

    mp.mp.dps = 80
    MASS = 510998.95069  # Electron rest energy in eV, checked against the native build.
    C = 299792458.0

    NATIVE = r"""
    program confirm_lcavity
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: cav
    type(coord_struct) :: start, finish
    real(rp) :: vec(6)
    integer :: j
    logical :: err
    call bmad_parser('case.bmad', lat, err_flag=err)
    if (err) error stop 'parser failure'
    cav => lat%ele(1)
    open(20,file='input.dat',status='old')
    read(20,*) vec
    close(20)
    call init_coord(start,vec,cav,upstream_end$,electron$)
    call mat_make_unit(cav%mat6)
    if (cav%tracking_method == linear$) then
      call make_mat6(cav,lat%param,start,err_flag=err)
      if (err) error stop 'matrix preparation failure'
    endif
    call track1(start,cav,lat%param,finish,err_flag=err,make_map1=.true.)
    if (err .or. finish%state /= alive$) error stop 'tracking failure'
    write(*,'(a,*(es26.17,1x))') 'META ', mass_of(electron$),cav%value(e_tot_start$),cav%value(e_tot$), &
      cav%value(p0c_start$),cav%value(p0c$),start%beta,finish%beta,cav%value(l_active$)
    write(*,'(a,*(es26.17,1x))') 'START ',start%vec
    write(*,'(a,*(es26.17,1x))') 'END ',finish%vec
    do j=1,6
      write(*,'(a,i1,1x,*(es26.17,1x))') 'MAT',j,cav%mat6(j,:)
    enddo
    end program confirm_lcavity
    """


    @dataclass
    class Case:
        name: str
        energy: float = 40e6
        voltage: float = 20e6
        length: float = 2.3
        frequency: float = 500e6
        phase: float = 0.25
        nrf: int | None = 0
        nstep: int = 1
        wave: str = "standing_wave"
        fringe: str = "full"
        px: float = 0.0
        py: float = 0.0
        is_on: bool = True
        absolute_time: bool = False
        tracking: str = "bmad_standard"


    def native(case):
        nrf = "" if case.nrf is None else f", n_rf_steps={case.nrf}"
        lattice = f"""no_digested
    beginning[e_tot]={case.energy:.17g}
    parameter[particle]=electron
    parameter[geometry]=open
    parameter[absolute_time_tracking]={str(case.absolute_time)[0]}
    c: lcavity, l={case.length:.17g}, voltage={case.voltage:.17g}, rf_frequency={case.frequency:.17g}, &
      phi0={case.phase:.17g}, cavity_type={case.wave}, fringe_type={case.fringe}, &
      num_steps={case.nstep}{nrf}, tracking_method={case.tracking}, is_on={str(case.is_on)[0]}, &
      autoscale_amplitude=F, autoscale_phase=F
    ln: line=(c)
    use, ln
    """
        log = fortran(
            NATIVE,
            files={
                "case.bmad": lattice,
                "input.dat": f"0 {case.px:.17g} 0 {case.py:.17g} 0 0\n",
            },
        ).log
        rows = {}
        for line in log.splitlines():
            if line.startswith(("META ", "START ", "END ", "MAT")):
                label, *values = line.split()
                if label in rows:
                    raise RuntimeError(f"Duplicate native row {label}")
                rows[label] = np.array([float(v) for v in values])
        if set(rows) != {"META", "START", "END", "MAT1", "MAT2", "MAT3", "MAT4", "MAT5", "MAT6"}:
            raise RuntimeError("Incomplete native output")
        if any(not np.isfinite(row).all() for row in rows.values()):
            raise RuntimeError("Nonfinite native output")
        if rows["META"].shape != (8,) or any(
            row.shape != (6,) for label, row in rows.items() if label != "META"
        ):
            raise RuntimeError("Unexpected native output shape")
        if rows["META"][0] != MASS:
            raise RuntimeError("Electron rest energy differs from the oracle constant")
        return rows


    def integrated_map(energy, voltage, length, cosphi, beta0, *, atol=2e-16, rtol=3e-14):
        # q = E*x'/E_start. Then d(x,q)/ds = (q*E_start/E, -G^2*x/(8*E_start*E)).
        # The determinant of this fundamental matrix is unity. The slope matrix
        # instead has determinant E_start/E_end, before beta corrections and edges.
        gradient = voltage / length

        def rhs(s, flattened):
            e = energy + gradient * cosphi * s
            a = np.array([[0, energy / e], [-(gradient**2) / (8 * energy * e), 0]])
            return (a @ flattened.reshape(2, 2)).ravel()

        sol = solve_ivp(
            rhs,
            (0, length),
            np.eye(2).ravel(),
            method="DOP853",
            atol=atol,
            rtol=rtol,
            max_step=length / 32,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        e1 = energy + voltage * cosphi
        p1 = math.sqrt((e1 - MASS) * (e1 + MASS))
        beta1 = p1 / e1
        body = sol.y[:, -1].reshape(2, 2)
        body[1] *= energy / e1
        entrance = np.array([[1, 0], [-gradient * cosphi / (2 * energy), 1]])
        exit_edge = np.array([[1, 0], [gradient * cosphi / (2 * e1), 1]])
        return math.sqrt(beta0 / beta1) * (exit_edge @ body @ entrance), body


    def effective_phase(case, meta):
        # The public legacy tracker uses one midpoint predictor when its phase
        # change is below 1e-4 rad. We independently integrate the trial trajectory
        # to compute its documented finite-slope time correction.
        mass, e0, eref1, p0, pref1, beta0, _ = meta[:7]
        phase0 = 2 * math.pi * case.phase
        co = math.cos(phase0)
        e1 = e0 + case.voltage * co
        p1 = math.sqrt((e1 - mass) * (e1 + mass))
        beta1 = p1 / e1
        _, body = integrated_map(e0, case.voltage, case.length, co, beta0)
        after_body = math.sqrt(beta0 / beta1) * body @ np.array([0.0, 1.0])
        # No entrance kick for a ray launched at x=y=0.
        x1, x2 = case.px, after_body[1] * case.px
        y1, y2 = case.py, after_body[1] * case.py
        cdt = case.length * (e0 + e1) / (p0 + p1)
        cdt_ref = case.length * (e0 + eref1) / (p0 + pref1)
        dz = (x1 * x1 + x2 * x2 + x1 * x2 + y1 * y1 + y2 * y2 + y1 * y2) * cdt / 6
        dphase = 2 * math.pi * case.frequency / C * (cdt + dz - cdt_ref)
        if abs(dphase) >= 1e-4:
            raise RuntimeError("Case requires a phase solver")
        return phase0 + dphase / 2


    def oscillator(energy, voltage, length):
        # Independent matrix exponential of a constant coefficient oscillator.
        e, v, ell = map(mp.mpf, (energy, voltage, length))
        omega = v / (mp.sqrt(8) * e * ell)
        generator = mp.matrix([[0, 1], [-(omega**2), 0]])
        mat = mp.expm(ell * generator)
        return np.array(mat.tolist(), dtype=float)


    def matrix_product_kdk(energy, voltage, length, active_length, steps):
        # The finite-step tracker uses momentum, rather than the legacy total-energy
        # low-beta prescription. Use its exact K/2-D-K/2 contract as the control.
        p = math.sqrt((energy - MASS) * (energy + MASS))
        ds = active_length / steps
        k = (voltage / active_length) ** 2 * ds / (8 * p * p)
        kick = np.array([[1.0, 0], [-k / 2, 1.0]])
        drift = np.array([[1.0, ds], [0, 1.0]])
        outer = np.array([[1.0, (length - active_length) / 2], [0, 1.0]])
        return outer @ np.linalg.matrix_power(kick @ drift @ kick, steps) @ outer


    def main():
        cases = [
            Case("ordinary_axis"),
            Case("ordinary_ray", px=1e-4),
            Case("ordinary_ray_small_positive", px=1e-6),
            Case("ordinary_ray_small_negative", px=-1e-6),
            Case("ordinary_absolute_time", absolute_time=True),
            Case("ordinary_linear_axis", tracking="linear"),
            Case("ordinary_linear_ray", tracking="linear", px=1e-4),
            Case("ordinary_vertical", py=1e-4),
            Case("ordinary_negative_voltage", voltage=-20e6),
            Case("ordinary_other_crossing", phase=0.75),
            Case("ordinary_fringe_none", fringe="none"),
            Case("ordinary_zero_voltage", voltage=0, px=1e-4),
            Case("ordinary_off", is_on=False, px=1e-4),
            Case("ordinary_finite_gain", phase=math.acos(0.02) / (2 * math.pi)),
            Case("switch_positive_inside", phase=math.acos(0.999e-5 / 0.5) / (2 * math.pi)),
            Case("switch_positive_outside", phase=math.acos(1.001e-5 / 0.5) / (2 * math.pi)),
            Case("switch_negative_inside", phase=math.acos(-0.999e-5 / 0.5) / (2 * math.pi)),
            Case("switch_negative_outside", phase=math.acos(-1.001e-5 / 0.5) / (2 * math.pi)),
            Case("ordinary_crest", phase=0),
            Case("ordinary_split2", nstep=2),
            Case("ordinary_split10", nstep=10),
            Case("ordinary_step10", nrf=10),
            Case("ordinary_step100", nrf=100),
            Case("ordinary_default", nrf=None),
            Case("traveling", wave="traveling_wave"),
            Case(
                "discovery_parameters_axis",
                energy=2 * MASS,
                voltage=2 * MASS,
                length=0.4,
                frequency=1e9,
            ),
            Case(
                "discovery_parameters_ray",
                energy=2 * MASS,
                voltage=2 * MASS,
                length=0.4,
                frequency=1e9,
                px=1e-4,
                fringe="none",
            ),
            Case("high_energy_axis", energy=500e6, voltage=20e6, length=1.0, frequency=1.3e9),
            Case("high_energy_ray", energy=500e6, voltage=20e6, length=1.0, frequency=1.3e9, px=1e-4),
        ]
        observations: dict[str, tuple[float, float]] = {}
        defects = 0
        oracle_residual = 0.0
        for case in cases:
            rows = native(case)
            native_mat = np.stack([rows[f"MAT{i}"] for i in range(1, 7)])
            meta = rows["META"]
            energy = meta[1]
            beta0 = meta[5]
            phase = (
                effective_phase(case, meta)
                if case.nrf == 0 and case.nstep == 1 and case.is_on and case.tracking == "bmad_standard"
                else 2 * math.pi * case.phase
            )
            cosphi = math.cos(phase)
            control = case.name in {
                "ordinary_zero_voltage",
                "ordinary_off",
                "ordinary_finite_gain",
                "switch_positive_outside",
                "switch_negative_outside",
                "ordinary_crest",
                "ordinary_step10",
                "ordinary_step100",
                "ordinary_default",
                "traveling",
            }
            if not case.is_on or case.voltage == 0 or case.wave == "traveling_wave":
                expected = np.array([[1.0, case.length], [0, 1.0]])
                oracle = "drift at zero gain"
            elif case.nrf is None or case.nrf > 0:
                expected = matrix_product_kdk(
                    energy, case.voltage, case.length, meta[7], case.nrf or 10
                )
                oracle = "finite-step momentum-normalized K/2-D-K/2 product"
            else:
                expected, _ = integrated_map(energy, case.voltage, case.length, cosphi, beta0)
                oracle = "independent DOP853 averaged-equation integration"
            axis = 3 if case.py else 1
            observed_x = rows["END"][axis - 1]
            expected_x = expected[0, 1] * (case.py or case.px)
            observations[case.name] = (float(observed_x), float(native_mat[0, 1]))
            dx = observed_x - expected_x
            dm12 = native_mat[0, 1] - expected[0, 1]
            block = native_mat[:2, :2]
            if not np.array_equal(block, native_mat[2:4, 2:4]):
                raise RuntimeError("Transverse planes differ in the symmetric cavity")
            tol = 3e-11 if "outside" in case.name else 2e-13
            if control:
                if abs(dm12) > tol or abs(dx) > 3e-15:
                    raise RuntimeError(f"Control failed: {case.name} dm12={dm12} dx={dx}")
            elif abs(dm12) > tol or abs(dx) > 3e-15:
                defects += 1
            if (
                abs(cosphi) < 1e-12
                and case.nrf == 0
                and case.is_on
                and case.nstep == 1
                and case.wave == "standing_wave"
            ):
                analytic = oscillator(energy, case.voltage, case.length)
                residual = float(np.max(np.abs(expected - analytic)))
                oracle_residual = max(oracle_residual, residual)
                if residual > 2e-14:
                    raise RuntimeError("ODE/analytic oracle mismatch")
            print(
                f"{case.name}: M12={native_mat[0, 1]:.17g} expected={expected[0, 1]:.17g} "
                f"dM12={dm12:+.10e} x={observed_x:.17g} expected_x={expected_x:.17g} "
                f"dx={dx:+.10e}; {oracle}; control={control}"
            )
        fd = (
            observations["ordinary_ray_small_positive"][0]
            - observations["ordinary_ray_small_negative"][0]
        ) / 2e-6
        analytic_native = observations["ordinary_axis"][1]
        if abs(fd - analytic_native) > 5e-12:
            raise RuntimeError("Native matrix fails small-ray finite differences")
        print(f"native R12 finite difference, h=1e-6: {fd:.17g}; matrix: {analytic_native:.17g}")
        print(f"zero-crossing ODE vs 80-digit oscillator: max_matrix_error={oracle_residual:.6e}; bound=2e-14")
        print(f"discrepancies={defects}/19 branch cases; controls=10/10 passed")
        verdict(defects > 0)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

