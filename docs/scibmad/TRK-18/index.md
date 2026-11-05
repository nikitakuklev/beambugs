---
title: "TRK-18: Builders skip the frame rotation at a nominally zero angle and drop its derivative"
hide:
  - toc
---

# TRK-18 — Builders skip the frame rotation at a nominally zero angle and drop its derivative

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P1 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-24 on BeamTracking main (`d8e45fd`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [symplectic_bl.jl:147](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L147)<br>[symplectic_bl.jl:189](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L189)<br>[symplectic_bl.jl:235](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L235)<br>[symplectic_bl.jl:264](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L264)<br>[symplectic_bl.jl:302](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L302)<br>[symplectic_bl.jl:399](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L399)<br>[symplectic_bl.jl:421](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L421)<br>[symplectic_bl.jl:455](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L455)<br>[symplectic_bl.jl:554](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L554)<br>[symplectic_bl.jl:595](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L595)<br>[src/kernels/bend_kick.jl:37–39](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/bend_kick.jl#L37-L39)<br>[src/kernels/fringe.jl:13](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/fringe.jl#L13)<br>[src/kernels/fringe.jl:17](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/fringe.jl#L17)<br>[src/kernels/fringe.jl:63](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/fringe.jl#L63)<br>[src/kernels/fringe.jl:67](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/fringe.jl#L67)<br>[ext/BeamTrackingBeamlinesExt/utils_bl.jl:115–116](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/utils_bl.jl#L115-L116)<br>[src/operators.jl:116–118](https://github.com/bmad-sim/GTPSA.jl/blob/a5790a7deaa62c13d3edc168d5f98b15e8a3c490/src/operators.jl#L116-L118) |

Differentiating tracking with respect to a tilt, skew strength or bend-face angle whose nominal value is zero can silently discard its first-order response. Finite differences and the equivalent alignment rotation establish the response: for a 5.000 GeV electron in a 0.2000 m quadrupole of normalized strength 0.5000 m⁻², the derivative of the exit normalized vertical momentum with respect to `tilt1` is 0 instead of −2.988 × 10⁻³ rad⁻¹ under both GTPSA and ForwardDiff. GTPSA, the truncated power series package behind SciBmad's Taylor maps, loses these derivatives consistently; ForwardDiff's failure depends on the sign of the differentiated variable. Source inspection extends the affected tests to combined-function and implicit bend builders; ordinary floating-point tracking and alignment tilts are unaffected.

??? example "Reproducer: TRK-18.jl"

    [Download TRK-18.jl](TRK-18.jl)

    ```julia
    # TRK-18: Builders skip the frame rotation at a nominally zero angle and drop its derivative
    #
    # Differentiating tracking with respect to a tilt, skew strength or bend-face angle whose nominal
    # value is zero can silently discard its first-order response. Finite differences and the equivalent
    # alignment rotation establish the response: for a 5.000 GeV electron in a 0.2000 m quadrupole of
    # normalized strength 0.5000 m⁻², the derivative of the exit normalized vertical momentum with
    # respect to `tilt1` is 0 instead of −2.988 × 10⁻³ rad⁻¹ under both GTPSA and ForwardDiff. GTPSA,
    # the truncated power series package behind SciBmad's Taylor maps, loses these derivatives
    # consistently; ForwardDiff's failure depends on the sign of the differentiated variable. Source
    # inspection extends the affected tests to combined-function and implicit bend builders; ordinary
    # floating-point tracking and alignment tilts are unaffected.
    #
    # The script tracks a 5 GeV electron through one element at a time (default method unless stated)
    # and differentiates the six exit coordinates with respect to an angle a at a = 0 with ForwardDiff
    # and GTPSA. The angle is a quadrupole `tilt1` of either sign or skew strength, a dipole `tilt0`, a
    # `BendKick` dipole-plus-sextupole `tilt0`, an `SBend` `tilt_ref` or `e1`, or a separator `etilt0`.
    # Each derivative is compared with a four-point finite difference of floating-point tracking; the
    # quadrupole's alignment tilt is the control, and ForwardDiff at a = 10⁻⁹ shows that the derivative
    # is lost only at exactly zero.
    using SciBmad
    using SciBmad: BeamTracking, ForwardDiff
    using SciBmad.GTPSA
    const sp = Species("electron")
    const R = BeamTracking.E_to_R(sp, 5e9)
    const v0 = [0.013 0.021 -0.009 -0.017 0.025 0.08]
    function trackv(ele, z)
        b = Bunch(v0 .+ z, species=sp, p_over_q_ref=R)
        track!(b, Beamline([ele], species_ref=sp, p_over_q_ref=R))
        return vec(b.coords.v)
    end
    const X = vars(Descriptor(1, 1))[1]
    fd4(f, h) = (f(-2h) - 8f(-h) + 8f(h) - f(2h)) / (12h)    # independent: Float tracking only
    bad = false
    println(rpad("element, angle a", 44), rpad("max|FD|", 12), rpad("|ForwardDiff - FD|", 20), rpad("|GTPSA - FD|", 16), "|ForwardDiff at 1e-9 - FD|")
    # The builders construct the frame-rotation quaternion only if !(angle ≈ 0) (symplectic_bl.jl lines
    # 147, 189, 235, 264, 302, 399, 421, 455, 554, 595; the labels give the line exercised), and the
    # bend fringe applies e1, e2 only if !(e ≈ 0) (fringe.jl lines 13, 17, 63, 67). GTPSA's isapprox
    # compares only the scalar part, so a knob on a nominally zero angle always takes the no-rotation
    # branch; a ForwardDiff Dual with value ±0 takes it for one sign of its partial. The angle of a
    # straight multipole is atan2(ks, kn) of the tilted strengths, and for tilt1 = a the test drops the
    # derivative.
    for (name, mk, defect) in (
            ("Quadrupole, tilt1 = a  (:264)", a -> Quadrupole(L=0.2, Kn1=0.5 + zero(a), tilt1=a), true),
            ("Quadrupole, tilt1 = -a (:264)", a -> Quadrupole(L=0.2, Kn1=0.5 + zero(a), tilt1=-a), true),
            ("Quadrupole, Kn1 = 0.5, Ks1 = a  (:264)", a -> Quadrupole(L=0.2, Kn1=0.5 + zero(a), Ks1=a), true),
            ("Quadrupole, alignment tilt = a (control)", a -> Quadrupole(L=0.2, Kn1=0.5 + zero(a), tilt=a), false),
            ("Kn0 = 0.01, tilt0 = a  (:147)", a -> LineElement(L=0.2, Kn0=0.01 + zero(a), tilt0=a), true),
            ("Kn0 + Kn2, BendKick NoEnd, tilt0 = a (:189)", a -> LineElement(L=0.2, Kn0=0.3 + zero(a), Kn2=0.8, tilt0=a,
                                                                               tracking_method=BendKick(fringe_at=Fringe.NoEnd)), true),
            ("SBend, tilt_ref = a  (:421)", a -> SBend(L=0.5, g=0.02, Kn0=0.02 + zero(a), tilt_ref=a), true),
            ("SBend, e1 = a  (fringe.jl:13,67)", a -> SBend(L=0.5, g=0.02, Kn0=0.02 + zero(a), e1=a), true),
            ("En0 = 1e6 V/m, etilt0 = a  (:595)", a -> LineElement(L=0.2, En0=1e6 + zero(a), etilt0=a), true))
        f(a) = trackv(mk(a), zero(a))
        fd = fd4(f, 1e-4)
        ad = ForwardDiff.derivative(f, 0.0)
        tp = map(t -> t[1], f(X))
        ad9 = ForwardDiff.derivative(a -> f(a + 1e-9), 0.0)
        ea, et, e9 = maximum(abs, ad - fd), maximum(abs, tp - fd), maximum(abs, ad9 - fd)
        defect && (global bad |= max(ea, et) > 1e-6 * maximum(abs, fd))
        println(rpad(name, 44), rpad(round(maximum(abs, fd); sigdigits=6), 12), rpad(round(ea; sigdigits=4), 20),
                rpad(round(et; sigdigits=4), 16), round(e9; sigdigits=4))
    end
    q1 = trackv(Quadrupole(L=0.2, Kn1=0.5, tilt1=0.01), 0.0); q2 = trackv(Quadrupole(L=0.2, Kn1=0.5, tilt=0.01), 0.0)
    println("recheck: Float tilt1 = 0.01 and alignment tilt = 0.01 agree to ", maximum(abs, q1 - q2))
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-23 | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-23 | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |

