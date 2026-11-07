---
title: "COL-02: Stochastic radiation in a tilted bend or separator is computed in the wrong frame"
hide:
  - toc
---

# COL-02 — Stochastic radiation in a tilted bend or separator is computed in the wrong frame

| | |
|---|---|
| Code | [SciBmad](../index.md) · Collective effects and spin |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-22 on BeamTracking release [v0.8.7](https://github.com/bmad-sim/BeamTracking.jl/tree/v0.8.7) (`65e343f`) |
| Issue | [[gh:https://github.com/bmad-sim/BeamTracking.jl/issues/378]], filed from this evidence |
| Pull request | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/379]], filed from this evidence |
| Source | [`BeamTracking.jl/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl:415,425-426,436`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L415-L436)<br>[`symplectic_bl.jl:448,459-460,470`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L448-L470)<br>[`symplectic_bl.jl:545,558-559,568`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L545-L568)<br>[`symplectic_bl.jl:594-600,610`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L594-L610)<br>[`BeamTracking.jl/src/kernels/radiation.jl:242-246,269-270`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/radiation.jl#L242-L270)<br>[`BeamTracking.jl/src/kernels/implicit.jl:763-777`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/implicit.jl#L763-L777)<br>[`BeamTracking.jl/src/kernels/elsep.jl:1-4`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/elsep.jl#L1-L4)<br>[`115-123`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/elsep.jl#L115-L123)<br>[`BeamTracking.jl/src/kernels/integrators.jl:11-12,17-18,25-26`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/integrators.jl#L11-L26)<br>[`radiation.jl:181`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/radiation.jl#L181)<br>[`208-227`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/radiation.jl#L208-L227)<br>[`BeamTracking.jl/src/kernels/bend_kick.jl:37-39`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/bend_kick.jl#L37-L39)<br>[`68-69`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/bend_kick.jl#L68-L69)<br>[`implicit.jl:16-17,30-31`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/implicit.jl#L16-L31)<br>[`general_bl.jl:102-104`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/general_bl.jl#L102-L104)<br>[`rungekutta.jl:53-55`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/rungekutta.jl#L53-L55) |

With `radiation_fluctuations_on`, the random radiation kick in a tilted bend or a skew electrostatic separator is computed in a different rotated frame from the element's deterministic transport, distorting the energy kick and recoil direction. A tilted element must match the untilted one with coordinates rotated in and out under the same random seed; a direct kick calculation confirms the required frame. For a 3.000 GeV electron at x = 3.000 cm, y = 2.000 cm in a 0.5000 m bend of curvature 0.1000 m⁻¹ tilted by 0.4000 rad, the stochastic relative-momentum kick is −3.298 × 10⁻⁶ instead of −3.300 × 10⁻⁶. Tilted magnetic and four-potential bends and skew separators show it; on-axis energy kicks remain correct, but recoil directions can still be wrong.

Run `julia COL-02.jl`.

??? example "Reproducer: COL-02.jl"

    [Download COL-02.jl](COL-02.jl)

    ```julia
    # COL-02: Stochastic radiation in a tilted bend or separator is computed in the wrong frame
    #
    # With `radiation_fluctuations_on`, the random radiation kick in a tilted bend or a skew
    # electrostatic separator is computed in a different rotated frame from the element's deterministic
    # transport, distorting the energy kick and recoil direction. A tilted element must match the
    # untilted one with coordinates rotated in and out under the same random seed; a direct kick
    # calculation confirms the required frame. For a 3.000 GeV electron at x = 3.000 cm, y = 2.000 cm in
    # a 0.5000 m bend of curvature 0.1000 m⁻¹ tilted by 0.4000 rad, the stochastic relative-momentum
    # kick is −3.298 × 10⁻⁶ instead of −3.300 × 10⁻⁶. Tilted magnetic and four-potential bends and skew
    # separators show it; on-axis energy kicks remain correct, but recoil directions can still be wrong.
    #
    # The script tracks one electron through the tilted `SBend` and through the untilted bend with its
    # coordinates rotated in and out by BeamTracking's own frame rotation, under the same random seed,
    # once with fluctuations off (a control that must agree) and once with them on. It then calls the
    # stochastic kick directly, with the tilt argument the bend builder passes and with the one matching
    # the deterministic frame, and compares the energy-kick ratio with the analytic
    # √((1+g·x_used)/(1+g·x_right)), where g is the curvature and x the horizontal position in each
    # frame.
    using SciBmad, Random, LinearAlgebra
    const BT = SciBmad.BeamTracking

    sp = Species("electron"); E = 3e9; t = 0.4; g = 0.1; L = 0.5
    v0 = [0.03 0.0 0.02 0.0 0.0 0.0]
    tm(fl) = Symplectic(order = 2, n_steps = 1, radiation_damping_on = false, radiation_fluctuations_on = fl)

    function rotate(v, q)                         # apply BeamTracking's own frame rotation
        c = Bunch(v = copy(v)).coords
        BT.rotation!(1, c, q, 0)
        return c.v
    end
    # W is the coordinate rotation of the deterministic bend kernel, rot_quaternion(0, 0, -tilt_ref).
    # Rotation covariance: a bend with tilt_ref = t must act as W^-1 o (untilted bend) o W, for the
    # stochastic part as well as the deterministic part.
    win = BT.rot_quaternion(0, 0, -t); wout = BT.inv_rot_quaternion(0, 0, -t)

    function trk(v, tilt, fl, seed)
        line = Beamline([SBend(g = g, L = L, tilt_ref = tilt, tracking_method = tm(fl))], species_ref = sp, E_ref = E)
        b = Bunch(v = copy(v), species = sp, p_over_q_ref = line.p_over_q_ref)
        Random.seed!(seed)
        track!(b, line; use_explicit_SIMD = false)
        return b.coords.v
    end

    for fl in (false, true)
        vA = trk(v0, t, fl, 7)                                  # tilted bend
        vB = rotate(trk(rotate(v0, win), 0.0, fl, 7), wout)     # rotate in, untilted bend, rotate out
        println("fluctuations=", fl, "  tilted   = ", vec(vA))
        println("fluctuations=", fl, "  covariant= ", vec(vB))
        println("fluctuations=", fl, "  max |difference| = ", maximum(abs, vA .- vB))
    end
    # Size of the stochastic part, for scale:
    vdet = trk(v0, t, false, 7); vA = trk(v0, t, true, 7)
    vB = rotate(trk(rotate(v0, win), 0.0, true, 7), wout)
    println("stochastic dpz tilted = ", vA[6] - vdet[6], "  covariant = ", vB[6] - vdet[6],
            "  ratio = ", (vA[6] - vdet[6]) / (vB[6] - vdet[6]))
    println("stochastic dpx tilted = ", vA[2] - vdet[2], "  covariant = ", vB[2] - vdet[2])
    println("stochastic dpy tilted = ", vA[4] - vdet[4], "  covariant = ", vB[4] - vdet[4])
    # Independent recheck: one stochastic kick called with the argument the bend builder passes
    # (ntilt = -tilt_ref) versus the argument that reproduces the deterministic frame, compared
    # with the analytic ratio sqrt(h_used/h_right), h = 1 + g x_local (radiation.jl:181,211,218).
    x, y = v0[1], v0[3]; m = massof(sp)
    xr = rotate(v0, win)[1]; xw = rotate(v0, BT.rot_quaternion(0, 0, t))[1]
    println("recheck: local x deterministic frame = ", xr, " (cos t x + sin t y = ", cos(t) * x + sin(t) * y,
            "); frame used by stochastic kick = ", xw, " (cos t x - sin t y = ", cos(t) * x - sin(t) * y, ")")
    # Up to BeamTracking d8e45fd the kernel takes a tilt angle and rotates by rot_quaternion(0, 0, -tilt);
    # from the fix of BeamTracking#379 on it takes that rotation and its inverse, w and w_inv, instead.
    function kick(v, targ)
        c = Bunch(v = copy(v)).coords; Random.seed!(5)
        a = (1, c, 0.0, BT.CPU(), -1.0, m, E, g)
        z = (SciBmad.SA[1], SciBmad.SA[g], SciBmad.SA[0.0], L / 2)
        frame = applicable(BT.stochastic_radiation!, a..., targ, z...) ? (targ,) :
                (BT.rot_quaternion(0, 0, -targ), BT.inv_rot_quaternion(0, 0, -targ))
        BT.stochastic_radiation!(a..., frame..., z...)
        return c.v
    end
    println("recheck: dpz(builder arg)/dpz(deterministic frame) = ", kick(v0, -t)[6] / kick(v0, t)[6],
            "   analytic sqrt((1+g x_used)/(1+g x_right)) = ", sqrt((1 + g * xw) / (1 + g * xr)))
    on = zeros(1, 6)
    println("recheck: on-axis recoil (px,py) builder arg = ", kick(on, -t)[[2, 4]], "  deterministic frame = ", kick(on, t)[[2, 4]])
    ok = maximum(abs, vA .- vB) > 1e3 * eps() * maximum(abs, vB .- vdet)
    println(ok ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), source, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), source, Julia 1.13.2, CPU, Float64, serial |

