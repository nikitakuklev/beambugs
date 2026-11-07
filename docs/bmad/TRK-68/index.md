---
title: "TRK-68: Stepped native lcavities apply each RF energy-kick spin rotation twice"
hide:
  - toc
---

# TRK-68 — Stepped native lcavities apply each RF energy-kick spin rotation twice

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [kick-drift-kick model, bmad/doc/charged-tracking.tex:1162–1181](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/charged-tracking.tex#L1162-L1181)<br>[spin rule, bmad/doc/charged-tracking.tex:1199–1200](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/charged-tracking.tex#L1199-L1200)<br>[Thomas–BMT equation, bmad/doc/spin.tex:9–35](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/spin.tex#L9-L35)<br>[default lcavity spin method, bmad/doc/methods.tex:549–549](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L549-L549)<br>[the tracking spin method, bmad/doc/methods.tex:472–479](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L472-L479)<br>[default of 10, bmad/parsing/set_ele_defaults.f90:213–213](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/set_ele_defaults.f90#L213-L213)<br>[Native dispatch, bmad/low_level/track1_bmad.f90:111–115](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track1_bmad.f90#L111-L115)<br>[track1, bmad/code/track1.f90:232–235](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/track1.f90#L232-L235)<br>[spin gate, bmad/low_level/track_a_lcavity.f90:67–67](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L67-L67)<br>[step weights, bmad/low_level/lcavity_rf_step_setup.f90:225–250](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/lcavity_rf_step_setup.f90#L225-L250)<br>[rotates the spin with it, bmad/low_level/track_a_lcavity.f90:382–387](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L382-L387)<br>[applies the energy kick once, bmad/low_level/track_a_lcavity.f90:397–397](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L397-L397)<br>[converts back to momentum coordinates and the new reference, bmad/low_level/track_a_lcavity.f90:407–408](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L407-L408)<br>[rotates again with the same full field, bmad/low_level/track_a_lcavity.f90:413–416](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L413-L416)<br>[forward kicks, bmad/low_level/track_a_lcavity.f90:141–146](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L141-L146)<br>[reverse kicks, bmad/low_level/track_a_lcavity.f90:184–190](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L184-L190)<br>[zero-length branch, bmad/low_level/track_a_lcavity.f90:87–94](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L87-L94)<br>[step_drift, bmad/low_level/track_a_lcavity.f90:227–236](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L227-L236)<br>[RF fringe, bmad/low_level/track_a_lcavity.f90:288–293](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L288-L293)<br>[after its kick, bmad/low_level/track_a_lcavity.f90:319–322](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L319-L322)<br>[spin_omega, bmad/spin/spin_omega.f90:49–58](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/spin_omega.f90#L49-L58)<br>[its electric term, bmad/spin/spin_omega.f90:72–74](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/spin_omega.f90#L72-L74)<br>[the spatial normalization, bmad/spin/spin_omega.f90:82–86](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/spin_omega.f90#L82-L86)<br>[rotate_spin, bmad/spin/rotate_spin.f90:26–33](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/rotate_spin.f90#L26-L33) |

With tracking_method=bmad_standard, positive n_rf_steps and spin_tracking_method=tracking, an lcavity applies the full longitudinal-electric spin rotation of each energy kick both before and after the kick, instead of half on each side. Refining the steps therefore converges to twice the Thomas–BMT body rotation. A 5.110 MeV electron entering at 1.000 mrad and gaining 1.000 MeV rotates its spin by 0.3037 mrad instead of the 0.1519 mrad obtained by integrating the Thomas–BMT equation over its energy gain. Standing- and traveling-wave cavities are affected with or without fringes, which add their own separate rotations; Runge–Kutta tracking and the n_rf_steps=0 model do not use this path.

Run `python TRK-68.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: TRK-68.py"

    [Download TRK-68.py](TRK-68.py)

    ```python
    """TRK-68: Stepped native lcavities apply each RF energy-kick spin rotation twice

    With tracking_method=bmad_standard, positive n_rf_steps and spin_tracking_method=tracking, an lcavity applies the full longitudinal-electric spin rotation of each energy kick both before and after the kick, instead of half on each side. Refining the steps therefore converges to twice the Thomas–BMT body rotation. A 5.110 MeV electron entering at 1.000 mrad and gaining 1.000 MeV rotates its spin by 0.3037 mrad instead of the 0.1519 mrad obtained by integrating the Thomas–BMT equation over its energy gain. Standing- and traveling-wave cavities are affected with or without fringes, which add their own separate rotations; Runge–Kutta tracking and the n_rf_steps=0 model do not use this path.

    The script uses BMAD_DIST and the shared runner to track electrons with spin (0,0,1) through one lcavity in a fresh native process per case. A 1 MV traveling-wave case without fringes is compared with the closed-form Thomas–BMT integral between the tracked endpoint energies. Symmetric ±10 V cases for both wave types, with and without fringes, compare the linear spin response with one body action plus the fringe rotation implied by the fringe's own transverse kick. Zero-voltage and solenoid-in-cavity controls exercise the same routine's split rotations.
    """

    from __future__ import annotations

    import math

    from _bmad import compare, fortran, verdict

    PROGRAM = r"""
    program trk68
    use bmad
    implicit none
    type(lat_struct) :: lat
    type(coord_struct) :: a, b
    real(rp) :: v(6), s(3)
    logical :: err
    read(*,*) v(2), s
    call bmad_parser('lat.bmad', lat, err_flag=err)
    if (err) error stop 'parse failure'
    bmad_com%spin_tracking_on = .true.
    v(1) = 0; v(3:6) = 0
    call init_coord(a, v, lat%ele(1), upstream_end$)
    a%spin = s
    call track1(a, lat%ele(1), lat%param, b, err_flag=err)
    if (err .or. b%state /= alive$) error stop 'tracking failure'
    write(*,'(a,3es26.17)') 'CONST ', mass_of(a%species), anomalous_moment_of(a%species), &
                             real(charge_of(a%species), rp)
    write(*,'(a,8es26.17)') 'START ', a%p0c, a%beta, a%vec
    write(*,'(a,8es26.17)') 'END   ', b%p0c, b%beta, b%vec
    write(*,'(a,3es26.17)') 'SPIN  ', b%spin
    write(*,'(a,2es26.17)') 'ELE   ', lat%ele(1)%value(l_active$), lat%ele(1)%value(n_rf_steps$)
    end program
    """

    GAMMA = 10.0  # entrance Lorentz factor, total energy 5.110 MeV
    PX = 1e-3  # entrance px = Px/P0, i.e. 1.000 mrad
    LENGTH, FREQUENCY = 1.0, 1.3e9


    def track(voltage: float, steps: int, wave: str, fringe: str, ks: float = 0.0,
              px: float = PX, spin: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> dict:
        p0c = 510998.95069 * math.sqrt(GAMMA**2 - 1)  # entrance reference; the oracle uses Bmad's mass
        lattice = f"""no_digested
    parameter[geometry]=open
    parameter[particle]=electron
    parameter[p0c]={p0c:.17g}
    parameter[no_end_marker]=T
    c: lcavity, l={LENGTH}, voltage={voltage:.17g}, rf_frequency={FREQUENCY}, phi0=0, ks={ks:.17g}, &
     cavity_type={wave}, fringe_at={fringe}, n_rf_steps={steps}, autoscale_amplitude=F, &
     autoscale_phase=F, tracking_method=bmad_standard, spin_tracking_method=tracking
    ln: line=(c)
    use, ln
    """
        log = fortran(PROGRAM, files={"lat.bmad": lattice},
                      stdin=f"{px:.17g} {spin[0]:.17g} {spin[1]:.17g} {spin[2]:.17g}\n").log
        if any(marker in log for marker in ("[ERROR", "[FATAL", "[ABORT")):
            raise RuntimeError(log)
        rows = {line.split()[0]: [float(x) for x in line.split()[1:]]
                for line in log.splitlines() if line[:6] in ("CONST ", "START ", "END   ", "SPIN  ", "ELE   ")}
        if len(rows) != 5 or not all(math.isfinite(x) for row in rows.values() for x in row):
            raise RuntimeError(f"Missing or invalid records:\n{log}")
        mass, anomaly, _ = rows["CONST"]
        if rows["ELE"][1] != steps or (fringe == "both_ends" and rows["ELE"][0] <= 0):
            raise RuntimeError(f"Unexpected element setup: {rows['ELE']}")

        def ends(key: str) -> tuple[float, float]:
            # Energy from the coordinate's own reference and delta; physical Px = px * P0.
            pc = rows[key][0] * (1 + rows[key][7])
            return math.hypot(pc, mass) / mass, rows[key][3] * rows[key][0]

        (g_in, px_in), (g_out, px_out) = ends("START"), ends("END")
        sx, sy, sz = rows["SPIN"]
        return dict(mass=mass, a=anomaly, g_in=g_in, g_out=g_out, px_in=px_in, px_out=px_out,
                    spin=(sx, sy, sz), theta=math.atan2(sx, sz), l_active=rows["ELE"][0])


    def bmt_angle(g_in: float, g_out: float, ux: float, a: float) -> float:
        """Thomas–BMT spin rotation about y for a forward particle in a pure longitudinal E field.

        With E = (0,0,Ez), dS/ds = Ω×S and Ω_y = −(a + 1/(1+γ)) (ux/us) q Ez/mc², where
        ux = Px/mc is constant and us = sqrt(γ²−1−ux²). Since dγ/ds = q Ez/mc²,
        dθ/dγ = −(a + 1/(1+γ)) ux / sqrt(γ² − A²) with A² = 1 + ux², independent of the field profile.
        Substituting γ = A cosh t and u = tanh(t/2) = sqrt((γ−A)/(γ+A)) gives the primitive
        F(γ) = a acosh(γ/A) + (2/ux) atan(u ux/(A+1)), and θ = −ux [F(γ_out) − F(γ_in)].
        """
        big_a = math.sqrt(1 + ux * ux)

        def primitive(g: float) -> float:
            u = math.sqrt((g - big_a) / (g + big_a))
            return a * math.acosh(g / big_a) + (2 / ux) * math.atan(u * ux / (big_a + 1))

        return -ux * (primitive(g_out) - primitive(g_in))


    def main() -> None:
        defect = False

        # Finite acceleration, traveling wave, no fringes: only longitudinal kicks act, so physical
        # Px is constant and the exact BMT angle follows from the tracked endpoint energies.
        # Every rotation here is about y and spin does not act on the orbit, so evaluating each
        # kick's two rotations with half the field would give exactly observed/2. Its residual from
        # the integral (2.1e-8 rad at N=10, 1.4e-10 rad at N=128) is the kick scheme's second-order
        # step error, inside the tolerances.
        for steps, tol in ((10, 1e-7), (128, 1e-9)):
            r = track(1e6, steps, "traveling_wave", "no_end")
            ux = r["px_in"] / r["mass"]
            if compare(f"N={steps} physical Px, eV/c; conserved by longitudinal kicks",
                       r["px_out"], r["px_in"], atol=3e-11, rtol=0):
                raise RuntimeError("transverse momentum is not conserved")
            one = bmt_angle(r["g_in"], r["g_out"], ux, r["a"])
            defect |= compare(f"N={steps} gamma {r['g_in']:.17g} -> {r['g_out']:.17g} spin angle, rad; "
                              "closed-form Thomas–BMT integral", r["theta"], one, atol=tol)
            print(f"N={steps} observed/BMT={r['theta'] / one:.15g}; half-field equivalent "
                  f"observed/2 - BMT={r['theta'] / 2 - one:.6g} rad")

        # Weak field: the odd part of the response to ±V is linear in V up to O(V³). At linear order
        # every kick has the entrance coefficient, so one body action is exactly
        # B = −(a + 1/(1+γ)) (ux/us) ΔE_odd/mc², with ΔE_odd the tracked odd energy gain. The native
        # fringe rotates spin with the same radial field that gives its transverse kick, so its
        # rotation is F = (a + 1/(1+γ)) Δux_odd from the tracked change of physical Px.
        # Ponderomotive standing-wave kicks are quadratic in V and cancel in the odd part. The
        # binary64 resolution of the 5.110 MeV total energy (about 1e-9 eV) limits the 10 V energy
        # gain, and so B, to about 1e-10 relative.
        for wave in ("traveling_wave", "standing_wave"):
            for fringe in ("no_end", "both_ends"):
                plus, minus = (track(sign * 10.0, 10, wave, fringe) for sign in (1, -1))
                mass, a, g = plus["mass"], plus["a"], plus["g_in"]
                ux = plus["px_in"] / mass
                coef = a + 1 / (1 + g)
                theta = (plus["theta"] - minus["theta"]) / 2
                de = (plus["g_out"] - minus["g_out"]) / 2  # in units of mc²
                fringe_rot = coef * (plus["px_out"] - minus["px_out"]) / 2 / mass
                body = -coef * ux / math.sqrt(g * g - 1 - ux * ux) * de
                defect |= compare(f"{wave} fringe_at={fringe} odd body spin response at 10 V, rad; "
                                  "one Thomas–BMT action", theta - fringe_rot, body, atol=0, rtol=1e-9)
                print(f"{wave} fringe_at={fringe}: fringe rotation={fringe_rot:.6g} rad; "
                      f"(observed - fringe)/one body action={(theta - fringe_rot) / body:.15g}")

        # Controls with the same routine. Zero voltage: no electric spin action at all.
        r = track(0.0, 10, "traveling_wave", "both_ends")
        if compare("V=0 spin; unchanged", r["spin"], (0.0, 0.0, 1.0), atol=0, rtol=0):
            raise RuntimeError("zero-voltage control failed")
        # Solenoid field inside the same cavity, V=0, on axis: each step's drift applies two half
        # rotations of 0.5*ds*Bs. BMT for an axial particle in Bz gives Ω_z = −(1+a) ks, so spin
        # (1,0,0) ends at (cos φ, sin φ, 0) with φ = −(1+a) ks L.
        ks = 0.6
        r = track(0.0, 10, "traveling_wave", "both_ends", ks=ks, px=0.0, spin=(1.0, 0.0, 0.0))
        phi = -(1 + r["a"]) * ks * LENGTH
        if compare("V=0, ks=0.6/m on-axis spin; Thomas–BMT solenoid rotation",
                   r["spin"], (math.cos(phi), math.sin(phi), 0.0), atol=2e-15):
            raise RuntimeError("solenoid split-rotation control failed")

        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed<br><small>fringe-separated linear response for both wave types</small> | `claude-opus-5-5` | high | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

