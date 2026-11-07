---
title: "TRK-71: Hard-edge RF fringe spin rotations apply the particle charge twice"
hide:
  - toc
---

# TRK-71 — Hard-edge RF fringe spin rotations apply the particle charge twice

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad 20261031-0 (`c4c629f`, committed 2026-10-31) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [integrated fringe fields, bmad/doc/fringe-fields.tex, bmad/doc/fringe-fields.tex:455–470](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/fringe-fields.tex#L455-L470)<br>[Thomas–BMT equation, bmad/doc/spin.tex, bmad/doc/spin.tex:25–30](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/spin.tex#L25-L30)<br>[orbital fringe kick, bmad/low_level/apply_element_edge_kick.f90, bmad/low_level/apply_element_edge_kick.f90:241–243](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/apply_element_edge_kick.f90#L241-L243)<br>[fringe spin rotations, bmad/low_level/apply_element_edge_kick.f90, bmad/low_level/apply_element_edge_kick.f90:245–255](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/apply_element_edge_kick.f90#L245-L255)<br>[rotate_spin_given_field, bmad/spin/rotate_spin_given_field.f90, bmad/spin/rotate_spin_given_field.f90:49–50](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/rotate_spin_given_field.f90#L49-L50)<br>[charge factor, bmad/spin/spin_omega.f90, bmad/spin/spin_omega.f90:82–86](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/spin_omega.f90#L82-L86)<br>[the cavity gradient, bmad/code/em_field_calc.f90, bmad/code/em_field_calc.f90:427–431](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/em_field_calc.f90#L427-L431)<br>[the e_gun field, bmad/code/em_field_calc.f90, bmad/code/em_field_calc.f90:346–357](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/em_field_calc.f90#L346-L357)<br>[runge_kutta_mod.f90, bmad/modules/runge_kutta_mod.f90:147–147](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/runge_kutta_mod.f90#L147-L147)<br>[time_tracker_mod.f90, bmad/modules/time_tracker_mod.f90:203–204](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/time_tracker_mod.f90#L203-L204)<br>[track1_spin, bmad/spin/track1_spin.f90, bmad/spin/track1_spin.f90:54–62](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/track1_spin.f90#L54-L62)<br>[track1_spin_integration.f90, bmad/spin/track1_spin_integration.f90:56–60](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/track1_spin_integration.f90#L56-L60)<br>[its exit edge, bmad/spin/track1_spin_integration.f90:101–103](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/track1_spin_integration.f90#L101-L103)<br>[spin_fringe_on, bmad/low_level/apply_element_edge_kick.f90, bmad/low_level/apply_element_edge_kick.f90:111–111](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/apply_element_edge_kick.f90#L111-L111)<br>[E_z ∝ sin(k s), bmad/code/em_field_calc.f90, bmad/code/em_field_calc.f90:442–451](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/em_field_calc.f90#L442-L451)<br>[whole half wavelengths, bmad/code/attribute_bookkeeper.f90, bmad/code/attribute_bookkeeper.f90:660–666](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/attribute_bookkeeper.f90#L660-L666)<br>[no entrance edge, bmad/low_level/calc_next_fringe_edge.f90, bmad/low_level/calc_next_fringe_edge.f90:164–167](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/calc_next_fringe_edge.f90#L164-L167)<br>[solenoid fringe, bmad/low_level/apply_element_edge_kick.f90, bmad/low_level/apply_element_edge_kick.f90:347–358](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/apply_element_edge_kick.f90#L347-L358)<br>[fringe spin field, bmad/low_level/track_a_lcavity.f90, bmad/low_level/track_a_lcavity.f90:288–293](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L288-L293)<br>[elseparator longitudinal fringe, bmad/low_level/apply_element_edge_kick.f90, bmad/low_level/apply_element_edge_kick.f90:263–270](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/apply_element_edge_kick.f90#L263-L270) |

With spin tracking, Runge–Kutta, native rfcavity spin integration and n_rf_steps = 0 lcavities multiply hard-edge RF fringe fields by the particle charge before the Thomas–BMT routine applies the charge again. The fringe spin rotation is thus multiplied by the charge number: electrons rotate with the wrong sign, although their orbital kick is correct. A 5.110 MeV electron 1.000 mm off axis at the entrance of a 1.000 MV traveling-wave lcavity rotates by +0.1028 mrad instead of the −0.1028 mrad given by the manual's fringe fields and the Thomas–BMT equation. The DC e_gun exit fringe has the same error. Default standing-wave cavities have no field at their ends; stepped native lcavities use a separate routine.

Run `python TRK-71.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: TRK-71.py"

    [Download TRK-71.py](TRK-71.py)

    ```python
    """TRK-71: Hard-edge RF fringe spin rotations apply the particle charge twice

    With spin tracking, Runge–Kutta, native rfcavity spin integration and n_rf_steps = 0 lcavities multiply hard-edge RF fringe fields by the particle charge before the Thomas–BMT routine applies the charge again. The fringe spin rotation is thus multiplied by the charge number: electrons rotate with the wrong sign, although their orbital kick is correct. A 5.110 MeV electron 1.000 mm off axis at the entrance of a 1.000 MV traveling-wave lcavity rotates by +0.1028 mrad instead of the −0.1028 mrad given by the manual's fringe fields and the Thomas–BMT equation. The DC e_gun exit fringe has the same error. Default standing-wave cavities have no field at their ends; stepped native lcavities use a separate routine.

    Setup: `NPROC`, `PJRT_NPROC`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` all set to 1; the script refuses to run otherwise.

    Track electrons and positrons at Lorentz factor 10, each species as its own reference, 1 mm off axis through one cavity with spin (0,0,1) in fresh native processes. Antisymmetrize the response to ±10 V and subtract spin_fringe_on = False to isolate the fringe spin, then compare it with the Thomas–BMT rotation of the manual's integrated RF fringe fields, whose strength is recovered from the tracked orbital fringe kick. Runge–Kutta, time Runge–Kutta, n_rf_steps = 0, native rfcavity and DC e_gun paths are checked, with a 1 MV case; positron tracking, body spin and the solenoid's magnetic fringe in the same edge routine are passing controls.
    """

    from __future__ import annotations

    import math
    import os
    from dataclasses import dataclass

    from _bmad import compare, fortran, verdict

    PROGRAM = r"""
    program trk71
    use bmad
    implicit none
    type(lat_struct) :: lat
    type(coord_struct) :: start, finish
    real(rp) :: v(6), s(3)
    logical :: err
    read(*,*) v, s
    call bmad_parser('lat.bmad', lat, err_flag=err)
    if (err) error stop 'parse failure'
    bmad_com%spin_tracking_on = .true.
    call init_coord(start, v, lat%ele(1), upstream_end$)
    start%spin = s
    call track1(start, lat%ele(1), lat%param, finish, err_flag=err)
    if (err .or. finish%state /= alive$) error stop 'tracking failure'
    write(*,'(a,4es26.17)') 'CONST ', mass_of(start%species), anomalous_moment_of(start%species), &
      real(charge_of(start%species), rp), hypot(start%p0c, mass_of(start%species))/mass_of(start%species)
    write(*,'(a,8es26.17)') 'ORBIT ', finish%p0c, finish%beta, finish%vec
    write(*,'(a,3es26.17)') 'SPIN ', finish%spin
    end program
    """

    GAMMA = 10.0
    MASS_EV = 510998.95069
    P0C = MASS_EV * math.sqrt(GAMMA**2 - 1)
    BETA = math.sqrt(1 - 1 / GAMMA**2)
    X0 = 1e-3


    @dataclass(frozen=True)
    class Track:
        mass: float
        anomaly: float
        charge: float
        orbit: tuple[float, ...]
        spin: tuple[float, ...]

        @property
        def theta_y(self) -> float:
            # Rotation about y takes S=(0,0,1) to (sin θ, 0, cos θ).
            return math.atan2(self.spin[0], self.spin[2])

        @property
        def theta_x(self) -> float:
            # Rotation about x takes S=(0,0,1) to (0, −sin θ, cos θ).
            return math.atan2(-self.spin[1], self.spin[2])

        @property
        def ux(self) -> float:
            # u_x = P_x/(mc); Bmad stores P_x/P0 with each endpoint's own reference momentum.
            return self.orbit[3] * self.orbit[0] / self.mass


    def track(species: str, element: str, px: float = 0.0) -> Track:
        lattice = f"""no_digested
    parameter[geometry]=open
    parameter[particle]={species}
    parameter[p0c]={P0C:.17g}
    parameter[no_end_marker]=T
    c: {element}
    ln: line=(c)
    use, ln
    """
        log = fortran(
            PROGRAM, files={"lat.bmad": lattice}, stdin=f"{X0} {px} 0 0 0 0  0 0 1\n"
        ).log
        if any(mark in log for mark in ("[ERROR", "[FATAL", "[ABORT", "[WARNING")):
            raise RuntimeError(log)
        rows: dict[str, tuple[float, ...]] = {}
        for line in log.splitlines():
            fields = line.split()
            if fields and fields[0] in ("CONST", "ORBIT", "SPIN"):
                if fields[0] in rows:
                    raise RuntimeError(f"Duplicate output record:\n{log}")
                rows[fields[0]] = tuple(float(value) for value in fields[1:])
        if {key: len(value) for key, value in rows.items()} != {"CONST": 4, "ORBIT": 8, "SPIN": 3} or not all(
            math.isfinite(value) for row in rows.values() for value in row
        ):
            raise RuntimeError(f"Missing or nonfinite output:\n{log}")
        mass, anomaly, charge, gamma = rows["CONST"]
        # The lattice p0c is the e_gun's exit reference, where its only fringe acts; the
        # entrance differs by the gun voltage (2e-5 in γ at 10 V).
        if not math.isclose(gamma, GAMMA, rel_tol=0, abs_tol=3e-5 if element.startswith("e_gun") else 2e-15):
            raise RuntimeError(f"Unexpected entrance Lorentz factor: {gamma}")
        return Track(mass, anomaly, charge, rows["ORBIT"], rows["SPIN"])


    def cavity(key: str, voltage: float, fringe: str, switch: str, method: str, extra: str = "") -> str:
        # 1 m, 1.3 GHz traveling wave: its longitudinal field is finite at both boundaries.
        # The rfcavity waveform is a sine of the phase; phi0 = 0.25 puts its boundary field at a maximum.
        phi0 = 0.25 if key == "rfcavity" else 0.0
        return (
            f"{key}, l=1, voltage={voltage:.17g}, rf_frequency=1.3e9, phi0={phi0}, "
            "cavity_type=traveling_wave, autoscale_amplitude=F, autoscale_phase=F, "
            f"fringe_at={fringe}, spin_fringe_on={switch}, tracking_method={method}, "
            f"spin_tracking_method=tracking{extra}"
        )


    def gun(voltage: float, fringe: str, switch: str) -> str:
        # A DC gun has no entrance edge; its exit fringe carries the uniform field's radial part.
        return (
            f"e_gun, l=1, gradient={voltage:.17g}, autoscale_amplitude=F, autoscale_phase=F, "
            f"fringe_at={fringe}, spin_fringe_on={switch}, tracking_method=runge_kutta, "
            "spin_tracking_method=tracking"
        )


    def odd(build, species: str, observable, voltage: float = 10.0) -> float:
        return (observable(track(species, build(voltage))) - observable(track(species, build(-voltage)))) / 2


    def fringe_spin(species: str, build, end: str) -> float:
        """Odd-in-voltage spin angle with the spin fringe on minus off; the orbit is identical in both."""
        return odd(lambda v: build(v, end, "T"), species, lambda r: r.theta_y) - odd(
            lambda v: build(v, end, "F"), species, lambda r: r.theta_y
        )


    def fringe_kick(species: str, build, end: str) -> tuple[float, Track]:
        """Odd-in-voltage u_x change produced by the orbital fringe at one end."""
        on = odd(lambda v: build(v, end, "T"), species, lambda r: r.ux)
        off = odd(lambda v: build(v, "no_end", "T"), species, lambda r: r.ux)
        return on - off, track(species, build(10.0, "no_end", "T"))


    def rf_oracle(du: float, a: float) -> float:
        # Manual RF fringe integrals at the entrance: ∫E_r ds = −r Ê/4, ∫B_φ ds = r Ê/(4c).
        # Bmad's orbital fringe kick (the manual's ΔP_x c = −q Ê x/2) gives q Ê x = −2 mc² Δu_x.
        # Thomas–BMT with the charge applied once, per unit s at speed βc along z, spin (0,0,1):
        #   Θ_y = −q/(mc² β) [(1/γ + a) c∫B_y + (a + 1/(1+γ)) β (−∫E_x)]
        #       = Δu_x [1/γ + a + β(a + 1/(1+γ))] / (2β).
        return du * (1 / GAMMA + a + BETA * (a + 1 / (1 + GAMMA))) / (2 * BETA)


    def gun_oracle(du: float, a: float) -> float:
        # A DC field has no ∂E_z/∂t, so the hard edge is all electric: ∫E_r ds = −r Ê/2 at the exit
        # with the opposite sign. Thomas–BMT then gives Θ_y = (a + 1/(1+γ)) Δu_x.
        return du * (a + 1 / (1 + GAMMA))


    def main() -> None:
        for variable in (
            "NPROC", "PJRT_NPROC", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        ):
            if os.environ.get(variable) != "1":
                raise RuntimeError(f"{variable} must be 1")
        defect = False

        def lcav(method: str, extra: str = ""):
            return lambda v, end, sw: cavity("lcavity", v, end, sw, method, extra)

        # Each species is its own reference, so both receive the same orbital kick (equal forces):
        # Thomas–BMT then requires equal spin rotations, because q·Ê is the same.
        oracle = {}
        for species in ("electron", "positron"):
            du, ref = fringe_kick(species, lcav("runge_kutta"), "entrance_end")
            oracle[species] = rf_oracle(du, ref.anomaly)
            print(f"{species} lcavity entrance fringe kick, RK: odd Delta u_x={du:.17g} charge={ref.charge:g}")
            observed = fringe_spin(species, lcav("runge_kutta"), "entrance_end")
            label = f"{species} runge_kutta lcavity entrance fringe spin at +/-10 V, rad; Thomas-BMT of manual fringe fields"
            if species == "electron":
                defect |= compare(label, observed, oracle[species], atol=1e-22, rtol=1e-12)
            elif compare("control " + label, observed, oracle[species], atol=1e-22, rtol=1e-12):
                raise RuntimeError("Positron control does not match the oracle")
        if compare("control fringe kick oracle electron vs positron, rad", oracle["electron"], oracle["positron"], atol=0, rtol=1e-14):
            raise RuntimeError("Species received different orbital fringe kicks")

        # Body spin alone (spin fringe off) must also be species independent at equal forces.
        body = {
            s: odd(lambda v: cavity("lcavity", v, "entrance_end", "F", "runge_kutta"), s, lambda r: r.theta_y)
            for s in ("electron", "positron")
        }
        if compare("control runge_kutta lcavity body spin, spin_fringe_on=F, electron vs positron, rad",
                   body["electron"], body["positron"], atol=1e-24, rtol=1e-12):
            raise RuntimeError("Body spin control failed")

        for method, extra in (("time_runge_kutta", ""), ("bmad_standard", ", n_rf_steps=0")):
            observed = fringe_spin("electron", lcav(method, extra), "entrance_end")
            defect |= compare(
                f"electron {method}{extra.replace(', ', ' ')} lcavity entrance fringe spin, rad; same field's Thomas-BMT",
                observed, oracle["electron"], atol=1e-22, rtol=1e-12,
            )

        # Native rfcavity orbits carry no fringe kick, so the RK orbit of the same element supplies Ê.
        def rfcav(method: str):
            return lambda v, end, sw: cavity("rfcavity", v, end, sw, method)

        du, ref = fringe_kick("electron", rfcav("runge_kutta"), "entrance_end")
        observed = fringe_spin("electron", rfcav("bmad_standard"), "entrance_end")
        defect |= compare(
            "electron bmad_standard rfcavity entrance fringe spin, rad; Thomas-BMT from the RK fringe kick",
            observed, rf_oracle(du, ref.anomaly), atol=1e-22, rtol=1e-12,
        )

        for species in ("electron", "positron"):
            du, ref = fringe_kick(species, gun, "exit_end")
            observed = fringe_spin(species, gun, "exit_end")
            label = f"{species} runge_kutta DC e_gun exit fringe spin, rad; all-electric Thomas-BMT from its fringe kick"
            if species == "electron":
                defect |= compare(label, observed, gun_oracle(du, ref.anomaly), atol=1e-22, rtol=1e-10)
            elif compare("control " + label, observed, gun_oracle(du, ref.anomaly), atol=1e-22, rtol=1e-10):
                raise RuntimeError("Positron e_gun control does not match the oracle")

        # Finite field: the entrance fringe sees the same entry state, and Ê scales with the voltage.
        large = track("electron", cavity("lcavity", 1e6, "entrance_end", "T", "runge_kutta")).theta_y - track(
            "electron", cavity("lcavity", 1e6, "entrance_end", "F", "runge_kutta")
        ).theta_y
        defect |= compare(
            "electron runge_kutta 1 MV lcavity entrance fringe spin, rad; 1e5 x the 10 V Thomas-BMT value",
            large, 1e5 * oracle["electron"], atol=0, rtol=1e-7,
        )
        print(f"1 MV observed/Thomas-BMT ratio={large / (1e5 * oracle['electron']):.12f}")

        # Magnetic hard-edge control in the same routine: solenoid fringes always act at both ends,
        # ∫B_r ds = ∓ r B_s/2. With q B_s c = k_s P0 c, entrance and exit give
        # Θ_x = (1 + aγ) k_s (x_in − x_out)/2; an entrance slope of 1 mrad makes x_out ≈ 2 mm.
        ks = 1e-3
        for species in ("electron", "positron"):
            def sol(k: float, sw: str, species: str = species) -> Track:
                return track(species, f"solenoid, l=1, ks={k}, spin_fringe_on={sw}, tracking_method=runge_kutta, "
                             "spin_tracking_method=tracking", px=1e-3)
            records = {(k, sw): sol(k, sw) for k in (ks, -ks) for sw in "TF"}
            observed = ((records[ks, "T"].theta_x - records[ks, "F"].theta_x)
                        - (records[-ks, "T"].theta_x - records[-ks, "F"].theta_x)) / 2
            x_out = (records[ks, "T"].orbit[2] + records[-ks, "T"].orbit[2]) / 2
            expected = (1 + records[ks, "T"].anomaly * GAMMA) * ks * (X0 - x_out) / 2
            # Second-order slope and orbit-rotation terms enter at about 1e-6 relative.
            if compare(f"control {species} runge_kutta solenoid fringe spin at +/-ks, rad; hard-edge Thomas-BMT",
                       observed, expected, atol=0, rtol=5e-6):
                raise RuntimeError("Solenoid fringe control failed")
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed<br><small>rfcavity and DC e_gun execution; equal-force species reproducer</small> | `claude-opus-5-5` | high | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

