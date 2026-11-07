---
title: "TRK-69: Native tracking ignores the selected spin method in stepped lcavities, patches, crab cavities and beambeam elements"
hide:
  - toc
---

# TRK-69 — Native tracking ignores the selected spin method in stepped lcavities, patches, crab cavities and beambeam elements

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad 20261031-0 (`c4c629f`, committed 2026-10-31) |
| Source | [spin_tracking_method definitions, bmad/doc/methods.tex:449–452](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L449-L452)<br>[bmad/doc/methods.tex:458–459](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L458-L459)<br>[bmad/doc/methods.tex:467–470](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L467-L470)<br>[beambeam, bmad/doc/methods.tex:534–534](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L534-L534)<br>[crab_cavity, bmad/doc/methods.tex:537–537](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L537-L537)<br>[lcavity, bmad/doc/methods.tex:549–549](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L549-L549)<br>[patch, bmad/doc/methods.tex:553–553](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L553-L553)<br>[crab_cavity, bmad/spin/valid_spin_tracking_method.f90:51–55](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/valid_spin_tracking_method.f90#L51-L55)<br>[patch, bmad/spin/valid_spin_tracking_method.f90:72–76](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/valid_spin_tracking_method.f90#L72-L76)<br>[lcavity and beambeam, bmad/spin/valid_spin_tracking_method.f90:101–105](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/valid_spin_tracking_method.f90#L101-L105)<br>[default of 10, bmad/parsing/set_ele_defaults.f90:213–213](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/set_ele_defaults.f90#L213-L213)<br>[native dispatch, bmad/low_level/track1_bmad.f90:111–116](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track1_bmad.f90#L111-L116)<br>[track1, bmad/code/track1.f90:232–235](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/track1.f90#L232-L235)<br>[the spin dispatcher call, bmad/code/track1.f90:313–313](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/track1.f90#L313-L313)<br>[off, bmad/spin/track1_spin.f90:68–70](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/track1_spin.f90#L68-L70)<br>[custom, bmad/spin/track1_spin.f90:86–94](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/track1_spin.f90#L86-L94)<br>[symp_lie_ptc, bmad/spin/track1_spin.f90:96–100](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/track1_spin.f90#L96-L100)<br>[stepped lcavity's spin gate, bmad/low_level/track_a_lcavity.f90:67–67](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L67-L67)<br>[crab_cavity rotation, bmad/low_level/track_a_crab_cavity.f90:121–131](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_crab_cavity.f90#L121-L131)<br>[beambeam rotation, bmad/low_level/track_a_beambeam.f90:279–290](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_beambeam.f90#L279-L290)<br>[track1_bmad calls track_a_patch, bmad/low_level/track1_bmad.f90:178–179](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track1_bmad.f90#L178-L179)<br>[documented default, bmad/low_level/track_a_patch.f90:16–17](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_patch.f90#L16-L17)<br>[entrance-side, bmad/low_level/track_a_patch.f90:77–79](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_patch.f90#L77-L79)<br>[exit-side, bmad/low_level/track_a_patch.f90:99–101](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_patch.f90#L99-L101)<br>[ele_to_fibre, bmad/ptc/ele_to_fibre.f90:384–407](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/ele_to_fibre.f90#L384-L407) |

With tracking_method=bmad_standard, Bmad never runs the selected spin_tracking_method for an lcavity with positive n_rf_steps (default 10), a patch, a crab_cavity or a beambeam element; only each element's built-in spin code acts. An lcavity with symp_lie_ptc or custom spin tracking returns the spin unchanged: an electron with Lorentz factor γ = 10 entering at 1.000 mrad and gaining 1.000 MeV keeps spin (0,0,1), while PTC spin tracking of the same cavity at n_rf_steps = 0 gives 0.1518 mrad, matching the 0.1519 mrad Thomas–BMT integral. A patch with symp_lie_ptc drops its 0.3000 rad spin-frame rotation, and crab_cavity and beambeam elements with spin_tracking_method=off still rotate the spin. Custom spin routines are never called for these elements.

Run `python TRK-69.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: TRK-69.py"

    [Download TRK-69.py](TRK-69.py)

    ```python
    """TRK-69: Native tracking ignores the selected spin method in stepped lcavities, patches, crab cavities and beambeam elements

    With tracking_method=bmad_standard, Bmad never runs the selected spin_tracking_method for an lcavity with positive n_rf_steps (default 10), a patch, a crab_cavity or a beambeam element; only each element's built-in spin code acts. An lcavity with symp_lie_ptc or custom spin tracking returns the spin unchanged: an electron with Lorentz factor γ = 10 entering at 1.000 mrad and gaining 1.000 MeV keeps spin (0,0,1), while PTC spin tracking of the same cavity at n_rf_steps = 0 gives 0.1518 mrad, matching the 0.1519 mrad Thomas–BMT integral. A patch with symp_lie_ptc drops its 0.3000 rad spin-frame rotation, and crab_cavity and beambeam elements with spin_tracking_method=off still rotate the spin. Custom spin routines are never called for these elements.

    The script uses BMAD_DIST and the shared runner to track one electron with spin through a single element with track1, one fresh native process per case, with a counting custom spin routine installed. For symp_lie_ptc it also calls the PTC tracker that the selector names directly on the same element and start orbit, and checks that value against the closed-form Thomas–BMT integral (lcavity) or the patch frame rotation. The off selector is compared with the unchanged entrance spin and the custom selector with one call of the routine. Controls track the lcavity at n_rf_steps=0, the patch with the tracking selector and an rfcavity with the custom selector.
    """

    from __future__ import annotations

    import math

    from _bmad import compare, fortran, verdict

    PROGRAM = r"""
    module hook_mod
    use bmad
    implicit none
    integer :: n_calls = 0
    contains
    ! Custom spin routine: counts its calls and marks the spin so a call is visible.
    subroutine count_spin(start_orb, ele, param, end_orb, err_flag, make_quaternion)
      type (coord_struct) :: start_orb, end_orb
      type (ele_struct) :: ele
      type (lat_param_struct) :: param
      logical err_flag
      logical, optional :: make_quaternion
      n_calls = n_calls + 1
      end_orb%spin = [0.0_rp, 1.0_rp, 0.0_rp]
      err_flag = .false.
    end subroutine
    end module

    program trk69
    use hook_mod
    implicit none
    type(lat_struct) :: lat
    type(coord_struct) :: a, b, c
    real(rp) :: v(6), s(3)
    integer :: direct_ptc
    logical :: err
    read(*,*) v(1), v(2), s, direct_ptc
    call bmad_parser('lat.bmad', lat, err_flag=err)
    if (err) error stop 'parse failure'
    track1_spin_custom_ptr => count_spin
    bmad_com%spin_tracking_on = .true.
    v(3:6) = 0
    call init_coord(a, v, lat%ele(1), upstream_end$)
    a%spin = s
    call track1(a, lat%ele(1), lat%param, b, err_flag=err)
    if (err .or. b%state /= alive$) error stop 'tracking failure'
    write(*,'(a,3es26.17)') 'CONST ', mass_of(a%species), anomalous_moment_of(a%species), a%p0c
    write(*,'(a,3es26.17)') 'END   ', b%p0c, b%vec(2), b%vec(6)
    write(*,'(a,3es26.17)') 'SPIN  ', b%spin
    write(*,'(a,2es26.17)') 'ELE   ', lat%ele(1)%value(n_rf_steps$), real(n_calls, rp)
    if (direct_ptc == 1) then
      ! The tracker that track1_spin calls for spin_tracking_method = symp_lie_ptc.
      c = a
      call track1_symp_lie_ptc(c, lat%ele(1), lat%param)
      write(*,'(a,3es26.17)') 'PTC   ', c%spin
    endif
    end program
    """

    GAMMA = 10.0  # entrance Lorentz factor, total energy 5.110 MeV
    P0C = 510998.95069 * math.sqrt(GAMMA**2 - 1)
    LCAV = ("lcavity, l=1, voltage=1e6, rf_frequency=1.3e9, phi0=0, cavity_type=traveling_wave, "
            "autoscale_amplitude=F, autoscale_phase=F, fringe_at=no_end, ds_step=0.01, "
            "tracking_method=bmad_standard, spin_tracking_method={sm}{steps}")
    PATCH = "patch, tilt=0.3, tracking_method=bmad_standard, spin_tracking_method={sm}"
    CRAB = ("crab_cavity, l=0.5, voltage=2e5, rf_frequency=1.3e9, phi0=0.1, "
            "tracking_method=bmad_standard, spin_tracking_method={sm}")
    BEAMBEAM = ("beambeam, sig_x=1e-4, sig_y=1e-4, sig_z=1e-3, n_slice=1, n_particle=1e10, "
                "tracking_method=bmad_standard, spin_tracking_method={sm}")
    RFCAV = ("rfcavity, l=1, voltage=1e6, rf_frequency=1.3e9, phi0=0.1, "
             "tracking_method=bmad_standard, spin_tracking_method={sm}")


    def track(element: str, x: float = 0.0, px: float = 0.0,
              spin: tuple[float, float, float] = (0.0, 0.0, 1.0), ptc: bool = False) -> dict:
        lattice = f"""no_digested
    parameter[geometry]=open
    parameter[particle]=electron
    parameter[p0c]={P0C:.17g}
    parameter[no_end_marker]=T
    c: {element}
    ln: line=(c)
    use, ln
    """
        stdin = " ".join(f"{t:.17g}" for t in (x, px, *spin)) + f" {int(ptc)}\n"
        log = fortran(PROGRAM, files={"lat.bmad": lattice}, stdin=stdin).log
        if any(marker in log for marker in ("[ERROR", "[FATAL", "[ABORT")):
            raise RuntimeError(log)
        keys = ("CONST ", "END   ", "SPIN  ", "ELE   ", "PTC   ")
        rows = {line.split()[0]: [float(t) for t in line.split()[1:]]
                for line in log.splitlines() if line[:6] in keys}
        if len(rows) != 4 + ptc or not all(math.isfinite(t) for r in rows.values() for t in r):
            raise RuntimeError(f"Missing or invalid records:\n{log}")
        return rows


    def theta(s: list[float]) -> float:
        """Spin rotation about y from (0,0,1): atan2(Sx, Sz)."""
        return math.atan2(s[0], s[2])


    def bmt_angle(g_in: float, g_out: float, ux: float, a: float) -> float:
        """Closed-form Thomas–BMT rotation about y for transport through a pure longitudinal E field.

        Px is constant (ux = Px/mc), and dθ/dγ = −(a + 1/(1+γ)) ux/√(γ² − A²), A² = 1 + ux².
        With γ = A cosh t and u = tanh(t/2): θ = −ux [F(γ_out) − F(γ_in)],
        F(γ) = a acosh(γ/A) + (2/ux) atan(u ux/(A+1)). Independent of the field's s-profile.
        """
        big_a = math.sqrt(1 + ux * ux)

        def prim(g: float) -> float:
            u = math.sqrt((g - big_a) / (g + big_a))
            return a * math.acosh(g / big_a) + (2 / ux) * math.atan(u * ux / (big_a + 1))

        return -ux * (prim(g_out) - prim(g_in))


    def control(label: str, observed, expected, *, atol: float, rtol: float = 0.0) -> None:
        if compare(label, observed, expected, atol=atol, rtol=rtol):
            raise RuntimeError(f"Control or oracle failed: {label}")


    def calls(r: dict) -> int:
        return int(r["ELE"][1])


    def main() -> None:
        defect = False
        px = 1e-3  # entrance px = Px/P0, i.e. 1.000 mrad

        # Stepped lcavity at the default n_rf_steps with the PTC selector. With bmad_standard
        # orbit tracking, ele_to_fibre builds the same longitudinal-field PTC cavity for any
        # n_rf_steps, so a direct PTC call on this element is the selector's own result.
        r = track(LCAV.format(sm="symp_lie_ptc", steps=""), px=px, ptc=True)
        mass, anomaly, p0 = r["CONST"]
        pc_out = r["END"][0] * (1 + r["END"][2])
        g_out = math.hypot(pc_out, mass) / mass
        bmt = bmt_angle(GAMMA, g_out, px * p0 / mass, anomaly)
        print(f"lcavity n_rf_steps={r['ELE'][0]:g} (default), gamma 10 -> {g_out:.15g}, "
              f"physical Px out/in={r['END'][1] * r['END'][0] / (px * p0):.15f}")
        control("lcavity PTC tracker on the same element, rad; closed-form Thomas–BMT integral",
                theta(r["PTC"]), bmt, atol=0, rtol=2e-4)
        defect |= compare("lcavity N=10 spin=symp_lie_ptc track1 spin angle, rad; PTC tracker "
                          "on the same element", theta(r["SPIN"]), theta(r["PTC"]), atol=1e-12)

        r = track(LCAV.format(sm="custom", steps=", n_rf_steps=10"), px=px)
        defect |= compare("lcavity N=10 spin=custom routine calls; one per track1",
                          calls(r), 1, atol=0)

        r0 = track(LCAV.format(sm="symp_lie_ptc", steps=", n_rf_steps=0"), px=px, ptc=True)
        control("control lcavity N=0 spin=symp_lie_ptc track1 spin angle, rad; PTC tracker",
                theta(r0["SPIN"]), theta(r0["PTC"]), atol=1e-15)
        r0 = track(LCAV.format(sm="custom", steps=", n_rf_steps=0"), px=px)
        control("control lcavity N=0 spin=custom routine calls", calls(r0), 1, atol=0)

        # Patch: tilt rotates the exit frame by 0.3 rad about z, so the spin's components
        # transform with the inverse rotation: (1,0,0) -> (cos 0.3, -sin 0.3, 0).
        frame = [math.cos(0.3), -math.sin(0.3), 0.0]
        r = track(PATCH.format(sm="symp_lie_ptc"), spin=(1.0, 0.0, 0.0), ptc=True)
        control("patch tilt=0.3 PTC tracker spin; frame rotation", r["PTC"], frame, atol=1e-15)
        defect |= compare("patch tilt=0.3 spin=symp_lie_ptc track1 spin; frame rotation",
                          r["SPIN"], frame, atol=1e-15)
        r = track(PATCH.format(sm="custom"), spin=(1.0, 0.0, 0.0))
        defect |= compare("patch spin=custom routine calls; one per track1", calls(r), 1, atol=0)
        r = track(PATCH.format(sm="tracking"), spin=(1.0, 0.0, 0.0))
        control("control patch tilt=0.3 spin=tracking track1 spin; frame rotation",
                r["SPIN"], frame, atol=1e-15)

        # Crab cavity and beambeam off axis: off means the exit spin equals the entrance spin.
        for name, element, x in (("crab_cavity", CRAB, 1e-3), ("beambeam", BEAMBEAM, 1e-4)):
            r = track(element.format(sm="off"), x=x)
            print(f"{name} x={x:g} m spin=off: track1 rotation about y={theta(r['SPIN']):.10g} rad")
            defect |= compare(f"{name} spin=off track1 spin; unchanged entrance spin",
                              r["SPIN"], [0.0, 0.0, 1.0], atol=1e-15)
            r = track(element.format(sm="custom"), x=x)
            defect |= compare(f"{name} spin=custom routine calls; one per track1",
                              calls(r), 1, atol=0)

        r = track(RFCAV.format(sm="custom"), px=px)
        control("control rfcavity spin=custom routine calls", calls(r), 1, atol=0)
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `claude-opus-5-5` | high | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed<br><small>direct PTC oracle on the stepped cavity and beambeam execution</small> | `claude-opus-5-5` | high | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

