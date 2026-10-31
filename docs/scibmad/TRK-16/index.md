---
title: "TRK-16: sincu/sinhcu use the two-term series for arguments up to 0.004 in magnitude in Float64"
hide:
  - toc
---

# TRK-16 — `sincu`/`sinhcu` use the two-term series for arguments up to 0.004 in magnitude in Float64

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on BeamTracking main (`0bf3585`) |
| Source | [src/utils/math_simd.jl:30–38](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/math_simd.jl#L30-L38)<br>[src/utils/math_simd.jl:47–55](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/math_simd.jl#L47-L55)<br>[src/utils/math_simd.jl:35](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/math_simd.jl#L35)<br>[src/utils/math_simd.jl:52](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/math_simd.jl#L52)<br>[src/utils/math_simd.jl:97](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/math_simd.jl#L97)<br>[src/kernels/bend_kick.jl:114](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/bend_kick.jl#L114)<br>[src/kernels/bend_kick.jl:141](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/bend_kick.jl#L141)<br>[src/kernels/solenoid_kick.jl:74](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/solenoid_kick.jl#L74)<br>[src/kernels/elsep.jl:43](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/elsep.jl#L43)<br>[src/utils/coord_transforms.jl:80](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/coord_transforms.jl#L80)<br>[src/utils/math_simd.jl:24–29](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/math_simd.jl#L24-L29)<br>[src/utils/math_simd.jl:91–96](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/math_simd.jl#L91-L96)<br>[src/utils/math_simd.jl:33](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/math_simd.jl#L33)<br>[src/utils/math_simd.jl:50](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/math_simd.jl#L50) |

In Float64 the helpers `sincu` (sin x / x) and `sinhcu` (sinh x / x) use the two-term series 1 ∓ x²/6 for |x| up to 0.004000, although their docstrings promise accuracy near the origin, which limits accuracy for weak elements. High-precision references and the omitted x⁴/120 term establish the error: at x = 0.003900 each helper is low by 1.928 × 10⁻¹², and the derivative of `sincu` has relative error 1.521 × 10⁻⁶. In a 0.7300 m `Exact()` bend of angle 0.002920 rad, the longitudinal coordinate differs from an independent high-precision circular trajectory by 4.221 × 10⁻¹³ m. Both signs of x are affected; the errors limit tight convergence comparisons and are negligible for ordinary tracking.

??? example "Reproducer: TRK-16.jl"

    [Download TRK-16.jl](TRK-16.jl)

    ```julia
    # TRK-16: `sincu`/`sinhcu` use the two-term series for arguments up to 0.004 in magnitude in Float64
    #
    # In Float64 the helpers `sincu` (sin x / x) and `sinhcu` (sinh x / x) use the two-term series 1 ∓
    # x²/6 for |x| up to 0.004000, although their docstrings promise accuracy near the origin, which
    # limits accuracy for weak elements. High-precision references and the omitted x⁴/120 term establish
    # the error: at x = 0.003900 each helper is low by 1.928 × 10⁻¹², and the derivative of `sincu` has
    # relative error 1.521 × 10⁻⁶. In a 0.7300 m `Exact()` bend of angle 0.002920 rad, the longitudinal
    # coordinate differs from an independent high-precision circular trajectory by 4.221 × 10⁻¹³ m. Both
    # signs of x are affected; the errors limit tight convergence comparisons and are negligible for
    # ordinary tracking.
    #
    # The script compares `sincu`, `sinhcu` and `one_cos_norm` with 256-bit `BigFloat` values at
    # arguments inside and just outside the series window, and the ForwardDiff derivative of `sincu`
    # with the exact derivative. It then tracks a proton with p₀c = 2mc² through a 0.73 m `Exact()`
    # `SBend` at bend angles inside and outside the window, and compares the exit coordinates with a
    # circular trajectory intersected with the exit face, computed in `BigFloat` without BeamTracking
    # code. Arguments outside the window are the controls.
    using SciBmad
    const BT = SciBmad.BeamTracking
    const FD = SciBmad.ForwardDiff
    setprecision(BigFloat, 256)
    # The series 1 -+ x^2/6 drops x^4/120, which is below double rounding only for |x| < (120 eps)^(1/4)
    # = 4.0e-4, the threshold the same functions use for Float16 and Float32; the Float64 threshold is
    # 0.004, so between 4e-4 and 4e-3 the relative error reaches 2.1e-12. Callers: exact_bend!
    # (sincu(theta), sincu(theta'/2)), the exact solenoid, the separator body (sinhcu(k_E L/(2 ps))),
    # the curved alignment transform, and one_cos_norm(x) = sincu(x/2)^2/2, documented as (1 - cos
    # x)/x^2 to machine precision.
    for x in (3e-4, 0.0039, 0.0041)
        es = Float64(BT.sincu(x) - sin(big(x)) / big(x)); eh = Float64(BT.sinhcu(x) - sinh(big(x)) / big(x))
        println("x = $x: sincu - exact = $es, sinhcu - exact = $eh, predicted series error -x^4/120 = ", x < 0.004 ? -x^4 / 120 : 0.0)
    end
    x = 0.0078
    println("one_cos_norm($x) relative error = ", Float64((BT.one_cos_norm(x) - (1 - cos(big(x))) / big(x)^2) / ((1 - cos(big(x))) / big(x)^2)),
            "  (docstring: machine precision)")
    x = 0.0039; bx = big(x)
    dexact = (bx * cos(bx) - sin(bx)) / bx^2
    println("ForwardDiff d sincu/dx at $x: relative error = ", Float64((FD.derivative(BT.sincu, x) - dexact) / dexact), "  (series derivative -x/3 drops x^3/30)")
    # Exact() bend body (Exact has no fringe, TRK-03, so e1 = e2 = 0 gives the pure body) against the circle.
    # Oracle: the particle moves on a circle of horizontal radius P_h/Kn0, P_h = sqrt((1+pz)^2 - py^2),
    # intersected with the exit face, the ray from the reference centre of curvature at angle g L.
    function circle(v, g, L; m=0.5)
        x, px, y, py, z, pz = big.(v); g = big(g); L = big(L); rho = 1 / g; th = g * L
        P = 1 + pz; Ph = sqrt(P^2 - py^2); R = Ph / g                      # Kn0 = g
        pos = [rho + x, big(0)]; d = [px, sqrt(Ph^2 - px^2)] ./ Ph          # radial (X), tangential (Z)
        C = pos .+ R .* [-d[2], d[1]]                                        # left normal: bends towards the centre
        u = [cos(th), sin(th)]; uc = u[1] * C[1] + u[2] * C[2]
        s = uc + sqrt(uc^2 - (C[1]^2 + C[2]^2) + R^2); Q = s .* u
        a = pos .- C; b = Q .- C
        dpsi = atan(a[1] * b[2] - a[2] * b[1], a[1] * b[1] + a[2] * b[2]); Sh = R * dpsi
        d1 = [-b[2], b[1]] ./ R
        S = Sh * P / Ph; beta = P / sqrt(P^2 + m^2); beta0 = 1 / sqrt(1 + m^2)
        Float64.([s - rho, Ph * (d1[1] * u[1] + d1[2] * u[2]), y + py / Ph * Sh, py, z + beta * L / beta0 - S, pz])
    end
    sp = Species("proton"); pc = 2 * massof(sp)            # m/P0 = 0.5
    v = [0.013, 0.021, -0.009, -0.017, 0.025, 0.08]
    function trk(g, L)
        bl = Beamline([SBend(L=L, g=g, tracking_method=Exact())], species_ref=sp, pc_ref=pc)
        b = Bunch(reshape(copy(v), 1, 6), species=sp, p_over_q_ref=pc / (C_LIGHT * chargeof(sp)))
        track!(b, bl); vec(b.v)
    end
    res = Dict{Float64,Float64}()
    for (g, L) in ((0.004, 0.73), (0.1, 0.73), (0.0004, 0.73))
        o = trk(g, L); e = circle(v, g, L); res[g] = maximum(abs.(o .- e))
        println("Exact SBend theta = $(g*L): max|track - circle| = ", res[g], "  per coordinate ", o .- e)
    end
    bad = abs(BT.sincu(0.0039) - sin(big(0.0039)) / big(0.0039)) > 1e-12 && res[0.004] > 1e-13 && res[0.1] < 1e-15
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | BeamTracking 0.8.7 (`d8e45fdc25db`), pinned clones as installed (SciBmad 5f0fb83ac889, BeamTracking d8e45fdc25db, Beamlines 35fe0162c72d), Julia 1.13.2, CPU, Float64, one thread |

