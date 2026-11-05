---
title: "TRK-17: BendKick applies a straight dipole's other multipoles unrotated in the dipole's frame"
hide:
  - toc
---

# TRK-17 — `BendKick` applies a straight dipole's other multipoles unrotated in the dipole's frame

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-24 on BeamTracking main (`d8e45fd`) |
| Issue | [[gh:https://github.com/bmad-sim/BeamTracking.jl/issues/376]], filed from this evidence |
| Pull request | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/377]], filed from this evidence |
| Source | [ext/BeamTrackingBeamlinesExt/symplectic_bl.jl:169–209](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L169-L209)<br>[ext/BeamTrackingBeamlinesExt/symplectic_bl.jl:174–175](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L174-L175)<br>[ext/BeamTrackingBeamlinesExt/symplectic_bl.jl:189–195](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L189-L195)<br>[ext/BeamTrackingBeamlinesExt/symplectic_bl.jl:202](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L202)<br>[src/kernels/bend_kick.jl:37–39](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/bend_kick.jl#L37-L39)<br>[src/kernels/bend_kick.jl:54](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/bend_kick.jl#L54)<br>[src/kernels/bend_kick.jl:41–43](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/bend_kick.jl#L41-L43)<br>[src/kernels/bend_kick.jl:64–66](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/bend_kick.jl#L64-L66)<br>[src/kernels/bend_kick.jl:45–48](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/bend_kick.jl#L45-L48)<br>[src/kernels/bend_kick.jl:60–62](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/bend_kick.jl#L60-L62)<br>[src/kernels/bend_kick.jl:50–52](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/bend_kick.jl#L50-L52)<br>[src/kernels/bend_kick.jl:56–58](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/bend_kick.jl#L56-L58)<br>[ext/BeamTrackingBeamlinesExt/symplectic_bl.jl:140–167](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L140-L167)<br>[ext/BeamTrackingBeamlinesExt/symplectic_bl.jl:160](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L160)<br>[ext/BeamTrackingBeamlinesExt/utils_bl.jl:77–81](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/utils_bl.jl#L77-L81)<br>[ext/BeamTrackingBeamlinesExt/utils_bl.jl:115–116](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/utils_bl.jl#L115-L116)<br>[src/kernels/radiation.jl:133–138](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/radiation.jl#L133-L138) |

With an explicit `BendKick` method, a tilted, skew or negative straight dipole applies its other multipoles in the wrong frame, changing orbit and spin transport. Rotation covariance (a tilted magnet must equal the untilted one between coordinate rotations) and independent Lorentz-force integration establish the correct map. For a proton with reference momentum p₀c = 2mc², twice its rest energy, in a 0.2000 m dipole of normalized strength 0.3000 m⁻¹ with sextupole strength 0.8000 m⁻³, both tilted by 0.3700 rad, the spin quaternion is off by up to 6.122 × 10⁻⁵ per component instead of 0. The default method avoids this mixed-multipole orbital error, but tilted or skew pure straight dipoles get incorrect radiation damping on the default path; curved bends are unaffected.

??? example "Reproducer: TRK-17.jl"

    [Download TRK-17.jl](TRK-17.jl)

    ```julia
    # TRK-17: `BendKick` applies a straight dipole's other multipoles unrotated in the dipole's frame
    #
    # With an explicit `BendKick` method, a tilted, skew or negative straight dipole applies its other
    # multipoles in the wrong frame, changing orbit and spin transport. Rotation covariance (a tilted
    # magnet must equal the untilted one between coordinate rotations) and independent Lorentz-force
    # integration establish the correct map. For a proton with reference momentum p₀c = 2mc², twice its
    # rest energy, in a 0.2000 m dipole of normalized strength 0.3000 m⁻¹ with sextupole strength 0.8000
    # m⁻³, both tilted by 0.3700 rad, the spin quaternion is off by up to 6.122 × 10⁻⁵ per component
    # instead of 0. The default method avoids this mixed-multipole orbital error, but tilted or skew
    # pure straight dipoles get incorrect radiation damping on the default path; curved bends are
    # unaffected.
    #
    # The script tests rotation covariance, orbit and spin, for a proton with p₀c = 2mc² through 0.2 m
    # straight elements under `BendKick` with fringes off: a dipole plus sextupole tilted as a whole by
    # 0.37 rad against the untilted magnet between coordinate rotations, and a negative dipole against
    # the π-rotated positive one; a stored zero sextupole and `Symplectic` tracking are controls. A
    # recheck compares converged `BendKick` and `Symplectic` with `RungeKutta` integration of the
    # Lorentz force, and with `RungeKutta` tracking of the magnet the frame error predicts. Further
    # checks rebuild the kernel call with in-frame coefficients, confirm that a curved bend is
    # covariant, and measure the damping error of a tilted pure dipole on the default path.
    using SciBmad
    const BT = SciBmad.BeamTracking
    const SA = SciBmad.SA
    const EXT = Base.get_extension(BT, :BeamTrackingBeamlinesExt)

    const sp = Species("proton"); const R = 2 * massof(sp) / C_LIGHT          # P0c = 2 M
    const v0 = [0.013, 0.021, -0.009, -0.017, 0.025, 0.08]; const t = 0.37
    function trk(ele, v; spin=true, s=sp, R=R)
        b = Bunch(v=reshape(copy(v), 1, 6), spin=spin, species=s, p_over_q_ref=R)
        track!(b, Beamline([ele], species_ref=s, p_over_q_ref=R); use_explicit_SIMD=false)
        vec(copy(b.coords.v)), spin ? vec(copy(b.coords.q)) : nothing
    end
    rot(v, a) = (c = cos(a); s = sin(a); w = copy(v); w[1:4] = [c * v[1] - s * v[3], c * v[2] - s * v[4], s * v[1] + c * v[3], s * v[2] + c * v[4]]; w)
    rotq(q, a) = (c = cos(a); s = sin(a); [q[1], c * q[2] - s * q[3], s * q[2] + c * q[3], q[4]])
    md(a, b) = maximum(abs, a - b)
    function cov(tilted, untilted, a; v=v0, kw...)              # residual of the covariance contract
        o, q = trk(tilted, v; kw...); ou, qu = trk(untilted, rot(v, -a); kw...)
        md(o, rot(ou, a)), q === nothing ? NaN : md(q, rotq(qu, a))
    end

    # thick_bdipole(::BendKick) rotates the coordinates by phi = atan2(Ks0, Kn0) so that the dipole
    # becomes a normal dipole |K0|, but passes the unrotated kn, ks to multipole_kick!, rotate_spin! and
    # deterministic_radiation_multipole! inside that frame. There the order-m coefficient must be (kn +
    # i ks) exp(-i m phi), the rotation get_strengths applies for a per-multipole tilt; using (kn + i
    # ks) itself acts in the element frame as the order-m multipole times exp(+i m phi). Covariance
    # contract: a magnet whose multipoles are all tilted by t equals rotating the coordinates by -t,
    # tracking the untilted magnet and rotating back by t, in orbit and spin. Fringes are off (NoEnd) so
    # that only the body is tested.
    bk = BendKick(fringe_at=Fringe.NoEnd)                        # order 4, one step
    res = Dict{String,Tuple{Float64,Float64}}()
    println("Rotation covariance, L = 0.2, expected residual 0 (rounding ~1e-16):")
    for (name, k, a) in (("Kn0 = 0.3, Kn2 = 0.0, tilts t (control)", (Kn0=0.3, Kn2=0.0), t),
                         ("Kn0 = 0.3, Kn2 = 0.8, tilts t", (Kn0=0.3, Kn2=0.8), t))
        res[name] = cov(LineElement(L=0.2; k..., tilt0=t, tilt2=t, tracking_method=bk), LineElement(L=0.2; k..., tracking_method=bk), a)
        println("  BendKick   ", rpad(name, 40), " orbit ", res[name][1], "  spin quaternion ", res[name][2])
    end
    # Tilting by pi flips the odd orders: (Kn0, Kn2) = (-0.3, -0.8) is (0.3, 0.8) rotated by pi.
    name = "(-0.3, -0.8) = pi-rotated (0.3, 0.8)"
    res[name] = cov(LineElement(L=0.2, Kn0=-0.3, Kn2=-0.8, tracking_method=bk), LineElement(L=0.2, Kn0=0.3, Kn2=0.8, tracking_method=bk), pi)
    println("  BendKick   ", rpad(name, 40), " orbit ", res[name][1], "  spin quaternion ", res[name][2])
    sy = cov(LineElement(L=0.2, Kn0=0.3, Kn2=0.8, tilt0=t, tilt2=t, tracking_method=Symplectic(fringe_at=Fringe.NoEnd)),
             LineElement(L=0.2, Kn0=0.3, Kn2=0.8, tracking_method=Symplectic(fringe_at=Fringe.NoEnd)), t)
    println("  Symplectic ", rpad("Kn0 = 0.3, Kn2 = 0.8, tilts t", 40), " orbit ", sy[1], "  spin quaternion ", sy[2])

    # --- Independent recheck: RungeKutta integrates the Lorentz force of the element-frame multipole field
    # (rk4_kernel!, BeamTracking.multipole_field); no frame rotation, no splitting. Converged BendKick and
    # Symplectic (order 4, 400 steps) are compared with it. Prediction from the fault: BendKick tracks the
    # magnet whose order-m coefficient is multiplied by exp(i m phi), phi = atan2(Ks0, Kn0):
    #   tilt0 = tilt2 = t (phi = -t)  ->  sextupole tilt 2t;   Kn0 < 0 (phi = pi)  ->  Kn2 -> -Kn2, Kn1 unchanged;
    #   Ks0 > 0 (phi = pi/2)          ->  Kn1 -> -Kn1;         Kn0 > 0 (phi = 0)   ->  unchanged.
    bkc = BendKick(n_steps=400, fringe_at=Fringe.NoEnd); syc = Symplectic(n_steps=400, fringe_at=Fringe.NoEnd)
    rk = RungeKutta(n_steps=4000)
    o(k, tm) = trk(LineElement(L=0.2; k..., tracking_method=tm), v0; spin=false)[1]
    rc = Dict{String,Vector{Float64}}()
    for (name, k, kpred) in (("Kn0 = Kn2 tilts t", (Kn0=0.3, Kn2=0.8, tilt0=t, tilt2=t), (Kn0=0.3, Kn2=0.8, tilt0=t, tilt2=2t)),
                             ("Kn0 = -0.3, Kn2 = 0.8", (Kn0=-0.3, Kn2=0.8), (Kn0=-0.3, Kn2=-0.8)),
                             ("Ks0 = 0.3, Kn1 = 0.5", (Ks0=0.3, Kn1=0.5), (Ks0=0.3, Kn1=-0.5)),
                             ("Kn0 = -0.3, Kn1 = 0.5", (Kn0=-0.3, Kn1=0.5), (Kn0=-0.3, Kn1=0.5)),
                             ("Kn0 = 0.3, Kn2 = 0.8", (Kn0=0.3, Kn2=0.8), (Kn0=0.3, Kn2=0.8)))
        b, s, r, p = o(k, bkc), o(k, syc), o(k, rk), o(kpred, rk)
        rc[name] = [md(b, r), md(s, r), md(b, p)]
        println("recheck: ", rpad(name, 32), " |BendKick - RK| = ", rc[name][1], "  |Symplectic - RK| = ", rc[name][2],
                "  |BendKick - RK(predicted magnet)| = ", rc[name][3])
    end

    # --- Mechanism and fix: one bkb_multipole! call equals BendKick(order 2, one step). Feeding it the
    # in-frame coefficients restores covariance.
    L = 0.2; tm2 = BendKick(order=2, fringe_at=Fringe.NoEnd)
    ele = LineElement(L=L, Kn0=0.3, Kn2=0.8, tilt0=t, tilt2=t, tracking_method=tm2)
    kn, ks = EXT.get_strengths(ele.BMultipoleParams, L, R)
    phi = atan(ks[1], kn[1]); mm = SA[1, 3]
    knr = @. kn * cos(mm * phi) + ks * sin(mm * phi); ksr = @. -kn * sin(mm * phi) + ks * cos(mm * phi)
    println("builder (kn, ks) = ", (kn, ks), "; in the dipole frame = ", (knr, ksr), "  (untilted: (0.3, 0.8), (0, 0))")
    tilde_m, _, beta_0 = BT.drift_params(sp, R)
    function kernel(knx, ksx)
        b = Bunch(v=reshape(copy(v0), 1, 6), spin=true, species=sp, p_over_q_ref=R)
        BT.bkb_multipole!(1, b.coords, 0.0, nothing, tilde_m, beta_0, gyromagnetic_anomaly(sp), 0,
                          BT.rot_quaternion(0, 0, phi), BT.inv_rot_quaternion(0, 0, phi), hypot(kn[1], ks[1]), mm, knx, ksx, L)
        vec(copy(b.coords.v)), vec(copy(b.coords.q))
    end
    pub = trk(ele, v0); ref = trk(LineElement(L=L, Kn0=0.3, Kn2=0.8, tracking_method=tm2), rot(v0, -t))
    ref = (rot(ref[1], t), rotq(ref[2], t)); kas = kernel(kn, ks); kfx = kernel(knr, ksr)
    println("kernel: |public - kernel(builder kn, ks)| orbit ", md(pub[1], kas[1]), " spin ", md(pub[2], kas[2]))
    println("kernel: |public - covariant reference|    orbit ", md(pub[1], ref[1]), " spin ", md(pub[2], ref[2]))
    println("kernel: |kernel(in-frame) - reference|    orbit ", md(kfx[1], ref[1]), " spin ", md(kfx[2], ref[2]))

    # --- Scope. (a) Curved bends rotate by -tilt_ref and treat kn, ks as body-frame strengths
    # (symplectic_bl.jl:443-475): covariance holds under the default method. (b) The default path for a
    # tilted pure straight dipole (thick_pure_bdipole, :140-167) passes the unrotated (kn, ks) to the
    # in-frame radiation damping kick: covariance fails by a small amount (5 GeV electron).
    c1 = cov(SBend(L=0.5, g=0.2, Kn0=0.2, Kn2=0.8, tilt2=0.1, tilt_ref=t), SBend(L=0.5, g=0.2, Kn0=0.2, Kn2=0.8, tilt2=0.1), t)
    println("scope: SciBmadStandard SBend tilt_ref = t with Kn2 (curved, body frame): orbit ", c1[1], " spin ", c1[2])
    el = Species("electron"); Re = BT.E_to_R(el, 5e9); ve = [0.001, 0.02, -0.001, -0.015, 0.0, 0.0]
    for rad in (false, true)
        tm = SciBmadStandard(radiation_damping_on=rad)
        c = cov(LineElement(L=1.0, Kn0=0.1, tilt0=t, tracking_method=tm), LineElement(L=1.0, Kn0=0.1, tracking_method=tm), t; v=ve, s=el, R=Re)
        res["damping=$rad"] = c
        println("scope: SciBmadStandard pure dipole tilt0 = t, radiation_damping_on = $rad: orbit ", c[1], " spin ", c[2],
                "  (pz after = ", trk(LineElement(L=1.0, Kn0=0.1, tilt0=t, tracking_method=tm), ve; s=el, R=Re)[1][6], ")")
    end
    # Damping kernel rebuild (Symplectic(order 2, one step, NoEnd) = one bkb_multipole! call): element-frame
    # (kn, ks) as the builder passes them, against the in-frame dipole (|K0|, 0).
    let L = 1.0, tm = Symplectic(order=2, fringe_at=Fringe.NoEnd, radiation_damping_on=true)
        ele = LineElement(L=L, Kn0=0.1, tilt0=t, tracking_method=tm)
        knv, ksv = EXT.get_strengths(ele.BMultipoleParams, L, Re); kn1, ks1 = knv[1], ksv[1]
        ph = atan(ks1, kn1); tm_e, _, b0_e = BT.drift_params(el, Re)
        function kern(knx, ksx)
            b = Bunch(v=reshape(copy(ve), 1, 6), spin=true, species=el, p_over_q_ref=Re)
            BT.bkb_multipole!(1, b.coords, 0.0, (chargeof(el), massof(el), massof(el) / tm_e / b0_e), tm_e, b0_e,
                              gyromagnetic_anomaly(el), 0, BT.rot_quaternion(0, 0, ph), BT.inv_rot_quaternion(0, 0, ph),
                              hypot(kn1, ks1), SA[1], SA[knx], SA[ksx], L)
            vec(copy(b.coords.v))
        end
        pubd = trk(ele, ve; s=el, R=Re)[1]
        refd = rot(trk(LineElement(L=L, Kn0=0.1, tracking_method=tm), rot(ve, -t); s=el, R=Re)[1], t)
        println("scope: damping kernel: |public - kernel(builder kn, ks)| ", md(pubd, kern(kn1, ks1)),
                "  |kernel(|K0|, 0) - reference| ", md(kern(hypot(kn1, ks1), 0.0), refd))
    end

    bad = res["Kn0 = 0.3, Kn2 = 0.8, tilts t"][1] > 1e-6 && res["(-0.3, -0.8) = pi-rotated (0.3, 0.8)"][1] > 1e-6 &&
          res["Kn0 = 0.3, Kn2 = 0.0, tilts t (control)"][1] < 1e-15 && sy[1] < 1e-15 &&
          all(rc[n][1] > 1e-6 && rc[n][3] < 1e-13 for n in ("Kn0 = Kn2 tilts t", "Kn0 = -0.3, Kn2 = 0.8", "Ks0 = 0.3, Kn1 = 0.5")) &&
          all(rc[n][1] < 1e-13 for n in ("Kn0 = -0.3, Kn1 = 0.5", "Kn0 = 0.3, Kn2 = 0.8")) &&
          md(kfx[1], ref[1]) < 1e-15 && res["damping=true"][1] > 1e-12
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-23 | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | xhigh | 2026-10-23 | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-23 | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |

