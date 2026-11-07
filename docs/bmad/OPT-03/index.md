---
title: "OPT-03: Radiation quadrature aliases complete wiggler periods"
hide:
  - toc
---

# OPT-03 — Radiation quadrature aliases complete wiggler periods

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`rad1_damp_and_stoc_mats`, bmad/modules/rad_6d_mod.f90:494–507](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/rad_6d_mod.f90#L494-L507)<br>[lines 540–551](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/rad_6d_mod.f90#L540-L551)<br>[lines 384–405](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/rad_6d_mod.f90#L384-L405)<br>[`rad_damp_and_stoc_mats`, rad_6d_mod.f90:335–338](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/rad_6d_mod.f90#L335-L338)<br>[`tracking_rad_map_setup`, bmad/low_level/tracking_rad_map_setup.f90:54](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/tracking_rad_map_setup.f90#L54)<br>[bmad/modules/radiation_mod.f90:238–246](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/radiation_mod.f90#L238-L246) |

For a planar wiggler with 8 or 16 complete periods, the radiation-matrix quadrature samples only field peaks through refinement level five and accepts an aliased integral. At both tested tolerances, a 3.2 m, 16-period, 1 T wiggler at 3 GeV gives the radiation integral I₂ = 0.03196 m⁻¹ and relative energy loss 1.215×10⁻⁵. These are twice I₂ = L gₘₐₓ²/2 = 0.01598 m⁻¹ and the corresponding loss 6.074×10⁻⁶, where L is the length and gₘₐₓ is the peak curvature. The conventional radiation-integral routine agrees with the analytic I₂.

Run `python OPT-03.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: OPT-03.py"

    [Download OPT-03.py](OPT-03.py)

    ```python
    """OPT-03: Radiation quadrature aliases complete wiggler periods

    For a planar wiggler with 8 or 16 complete periods, the radiation-matrix quadrature samples only
    field peaks through refinement level five and accepts an aliased integral. At both tested
    tolerances, a 3.2 m, 16-period, 1 T wiggler at 3 GeV gives the radiation integral I₂ = 0.03196
    m⁻¹ and relative energy loss 1.215×10⁻⁵. These are twice I₂ = L gₘₐₓ²/2 = 0.01598 m⁻¹ and the
    corresponding loss 6.074×10⁻⁶, where L is the length and gₘₐₓ is the peak curvature. The
    conventional radiation-integral routine agrees with the analytic I₂.

    The script compiles a Fortran program that parses a 3.2 m planar wiggler lattice inline, with 8,
    16, 24 and 32 periods, calls `rad1_damp_and_stoc_mats` at tolerances 10⁻⁴ and 10⁻⁸, and prints its I2, its I3 and the
    energy-loss entry of its damping vector, together with the conventional
    `radiation_integrals` I2 and the analytic values from the element's g_max. The conventional routine
    is the control: it must agree with the analytic value.
    """

    from _bmad import compare, fortran, verdict

    LATTICE = """parameter[geometry] = open
    parameter[e_tot] = 3e9
    parameter[no_end_marker] = T
    beginning[beta_a] = 10
    beginning[beta_b] = 12
    w: wiggler, l = 3.2, n_period = {n_period}, b_max = 1, field_calc = planar_model, tracking_method = symp_lie_bmad, num_steps = 512
    ln: line = (w)
    use, ln
    """

    PROGRAM = """
    program wigprobe
    use bmad
    use rad_6d_mod
    implicit none
    type(lat_struct), target :: lat
    type(coord_struct) o1, o2
    type(rad_map_struct) rm
    type(rad_int1_struct) ri
    type(normal_modes_struct) mode
    type(coord_struct), allocatable :: orb(:)
    type(ele_struct), pointer :: ele
    real(rp) tol
    integer i, icache, stat
    logical err
    call bmad_parser('lat.bmad', lat)
    ele => lat%ele(1)
    call init_coord(o1, [0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp], ele, upstream_end$)
    call make_mat6(ele, lat%param, o1, o2, err)
    if (err) error stop 'make_mat6 failed'
    call twiss_and_track(lat, orb, stat)
    if (stat /= ok$) error stop 'twiss_and_track failed'
    icache = -2
    call radiation_integrals(lat, orb, mode, icache)
    print '(a,es25.16e3)', 'RESULT conventional ', mode%synch_int(2)
    ! Planar wiggler: g = g_max cos(k s), so the average of g^2 over whole periods is g_max^2/2.
    print '(a,es25.16e3)', 'RESULT analytic ', ele%value(l$) * ele%value(g_max$)**2 / 2
    ! The average of |cos|^3 over a period is 4/(3 pi).
    print '(a,es25.16e3)', 'RESULT analytic3 ', 4 * ele%value(l$) * ele%value(g_max$)**3 / (3 * pi)
    print '(a,es25.16e3)', 'RESULT lossfactor ', 2.0_rp/3 * classical_radius(electron$) * (ele%value(e_tot$) / mass_of(electron$))**3
    do i = 1, 2
      tol = 10.0_rp**(-4*i)
      call rad1_damp_and_stoc_mats(ele, .true., o1, o2, rm, tol, tol, err, lat%ele(0), ri)
      if (err) error stop 'rad1_damp_and_stoc_mats failed'
      print '(a,i0,es25.16e3)', 'RESULT tol1e-', 4*i, ri%i2
      print '(a,i0,es25.16e3)', 'RESULT damp1e-', 4*i, rm%xfer_damp_vec(6)
      print '(a,i0,es25.16e3)', 'RESULT i3tol1e-', 4*i, ri%i3
    enddo
    end program
    """

    KEYS = {"conventional", "analytic", "analytic3", "lossfactor"} | {
        f"{kind}1e-{n}" for kind in ("tol", "damp", "i3tol") for n in (4, 8)
    }
    # A quadrature that resolves the period is far inside 1% for this smooth integrand.
    RTOL = 0.01


    def wiggler(n_period: int) -> bool:
        """Run one wiggler; True when the quadrature misses the analytic I2 or energy loss."""
        lattice = LATTICE.format(n_period=n_period)
        log = fortran(PROGRAM, files={"lat.bmad": lattice}).log
        values = {}
        for line in log.splitlines():
            words = line.split()
            if len(words) == 3 and words[0] == "RESULT":
                values[words[1]] = float(words[2])
        if set(values) != KEYS:
            raise RuntimeError(f"Missing results:\n{log}")
        # The conventional integration resolves the periods; it agrees with the analytic value
        # to 1e-15 relative. 1e-12 relative leaves room for its step-size discretization.
        if compare(
            f"{n_period} periods: radiation_integrals I2 (1/m), expected L*g_max^2/2",
            values["conventional"],
            values["analytic"],
            atol=0,
            rtol=1e-12,
        ):
            raise RuntimeError("The conventional control disagrees with the analytic I2.")
        defect = False
        for tol in ("1e-4", "1e-8"):
            defect |= compare(
                f"{n_period} periods: rad1_damp_and_stoc_mats I2 (1/m) at tolerance {tol}, "
                "expected L*g_max^2/2",
                values[f"tol{tol}"],
                values["analytic"],
                atol=0,
                rtol=RTOL,
            )
            # On axis the damping vector's pz entry is the relative energy loss
            # -(2/3) r_e gamma^3 I2, U0/E for an ultrarelativistic electron.
            defect |= compare(
                f"{n_period} periods: rad1_damp_and_stoc_mats xfer_damp_vec(6) at tolerance {tol}, "
                "expected -(2/3) r_e gamma^3 L g_max^2/2",
                values[f"damp{tol}"],
                -values["lossfactor"] * values["analytic"],
                atol=0,
                rtol=RTOL,
            )
            # I3 = integral of |g|^3, the integrand of the stochastic matrix.
            defect |= compare(
                f"{n_period} periods: rad1_damp_and_stoc_mats I3 (1/m^2) at tolerance {tol}, "
                "expected 4 L g_max^3/(3 pi)",
                values[f"i3tol{tol}"],
                values["analytic3"],
                atol=0,
                rtol=RTOL,
            )
        return defect


    # The first five levels sample multiples of L/16. With 8 or 16 periods these are whole or half
    # periods, where g^2 = g_max^2, so every level returns 2x the integral and the extrapolation
    # error is zero, whatever the tolerance.
    defects = [wiggler(n) for n in (16, 8, 24, 32)]
    verdict(any(defects))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

