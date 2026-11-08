---
title: "TRK-03: Exact() bend ignores e1, e2 and fringe_at"
hide:
  - toc
---

# TRK-03 — `Exact()` bend ignores `e1`, `e2` and `fringe_at`

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on BeamTracking commit [`0bf3585`](https://github.com/bmad-sim/BeamTracking.jl/commit/0bf3585fdc9c), committed 2026-10-28 |
| Source | [ext/BeamTrackingBeamlinesExt/exact_bl.jl:12–31](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/exact_bl.jl#L12-L31)<br>[src/kernels/bend_kick.jl:187–195](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/bend_kick.jl#L187-L195)<br>[ext/BeamTrackingBeamlinesExt/exact_bl.jl:34–42](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/exact_bl.jl#L34-L42)<br>[src/bend.jl:7–9](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/bend.jl#L7-L9)<br>[src/bend.jl:23–24](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/bend.jl#L23-L24)<br>[src/tracking_methods.jl:69–74](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/tracking_methods.jl#L69-L74)<br>[bmad/modules/fringe_mod.f90:183–207](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/modules/fringe_mod.f90#L183-L207) |

Tracking an `SBend` with `Exact()` ignores the entrance and exit face angles `e1` and `e2` and the fringe selector `fringe_at`, so the bend has no edge focusing. The linear map must include a hard-edge kick at each enabled face, as the analytic edge matrices and `Symplectic(order=2)` tracking confirm. For a 0.7300 m bend with curvature 0.2000 m⁻¹ and face angles 0.1100 and 0.1300 rad, the exit normalized horizontal momentum pₓ changes with entrance position x at −0.02910 m⁻¹ instead of +0.01905 m⁻¹.

Run `julia TRK-03.jl`.

??? example "Reproducer: TRK-03.jl"

    [Download TRK-03.jl](TRK-03.jl)

    ```julia
    # TRK-03: `Exact()` bend ignores `e1`, `e2` and `fringe_at`
    #
    # Tracking an `SBend` with `Exact()` ignores the entrance and exit face angles `e1` and `e2` and the
    # fringe selector `fringe_at`, so the bend has no edge focusing. The linear map must include a
    # hard-edge kick at each enabled face, as the analytic edge matrices and `Symplectic(order=2)`
    # tracking confirm. For a 0.7300 m bend with curvature 0.2000 m⁻¹ and face angles 0.1100 and 0.1300
    # rad, the exit normalized horizontal momentum pₓ changes with entrance position x at −0.02910 m⁻¹
    # instead of +0.01905 m⁻¹.
    #
    # The script computes 6×6 transfer matrices with ForwardDiff for a proton with p₀c = 2mc² through
    # `SBend(L=0.73, g=0.2, e1=0.11, e2=0.13)` and compares their horizontal and vertical blocks with
    # the analytic product of entrance edge kick, sector body and exit edge kick. It also checks that
    # the `Exact()` matrix does not change when both face angles are set to zero, while
    # `Symplectic(order=2)` matches the analytic matrix as the control. A recheck at finite amplitude
    # shows that `Exact()` equals the fringe-free `Symplectic(order=2, fringe_at=NoEnd)` body.
    using SciBmad
    using SciBmad: ForwardDiff
    using LinearAlgebra

    const sp = Species("proton"); const pc = 2 * massof(sp)
    const L = 0.73; const g = 0.2; const e1 = 0.11; const e2 = 0.13

    function jac(tm; e1 = e1, e2 = e2)
      ele = SBend(L = L, g = g, e1 = e1, e2 = e2, tracking_method = tm)
      bl = Beamline([ele], species_ref = sp, pc_ref = pc)
      f = x -> begin
        b = Bunch(reshape(copy(x), 1, 6), species = sp, p_over_q_ref = pc / (C_LIGHT * chargeof(sp)))
        track!(b, bl; use_explicit_SIMD = false)
        vec(b.v)
      end
      return ForwardDiff.jacobian(f, zeros(6))
    end

    # A hard-edge face of angle e with Kn0 = g kicks dpx = +g tan(e) x, dpy = -g tan(e) y at first order
    # (Fortran Bmad's zero-fringe-integral edge kick), so the linear x and y maps are E(e2) * Sector *
    # E(e1).
    th = g * L
    Sx = [cos(th) sin(th)/g; -g*sin(th) cos(th)]; Sy = [1 L; 0 1]
    Ex(e) = [1 0; g*tan(e) 1]; Ey(e) = [1 0; -g*tan(e) 1]
    Mx = Ex(e2) * Sx * Ex(e1); My = Ey(e2) * Sy * Ey(e1)

    Jexact = jac(Exact()); Jexact0 = jac(Exact(); e1 = 0.0, e2 = 0.0)
    Jsym   = jac(Symplectic(order = 2))
    println("Exact(): x-plane R = ", Jexact[1:2, 1:2])
    println("expected x-plane   = ", Mx)
    println("Exact(): max |R - expected| (x,y)        = ", max(maximum(abs, Jexact[1:2,1:2] - Mx), maximum(abs, Jexact[3:4,3:4] - My)))
    println("Exact(): max |R - sector-only| (x,y)     = ", max(maximum(abs, Jexact[1:2,1:2] - Sx), maximum(abs, Jexact[3:4,3:4] - Sy)))
    println("Exact(): max |J(e1,e2) - J(0,0)| (6x6)    = ", maximum(abs, Jexact - Jexact0))
    println("Symplectic(order=2): max |R - expected|  = ", max(maximum(abs, Jsym[1:2,1:2] - Mx), maximum(abs, Jsym[3:4,3:4] - My)))
    bad = maximum(abs, Jexact - Jexact0) == 0 && maximum(abs, Jexact[1:2,1:2] - Mx) > 1e-3
    # --- Independent recheck: Exact() equals the fringe-free Symplectic body at finite amplitude ---
    let v = [0.013 0.021 -0.009 -0.017 0.025 0.08]
      o(tm) = (b = Bunch(copy(v), species = sp, p_over_q_ref = pc / C_LIGHT);
               track!(b, Beamline([SBend(L = L, g = g, e1 = e1, e2 = e2, tracking_method = tm)], species_ref = sp, pc_ref = pc); use_explicit_SIMD = false);
               vec(b.v))
      println("recheck: max|Exact - Symplectic(2, NoEnd)|    = ", maximum(abs, o(Exact()) - o(Symplectic(order = 2, fringe_at = Fringe.NoEnd))))
      println("recheck: max|Exact - Symplectic(2, BothEnds)| = ", maximum(abs, o(Exact()) - o(Symplectic(order = 2))))
    end
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-22</span> | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |
| Found | <span class="bb-nowrap">`gpt-6-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-22</span> | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-22</span> | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | <span class="bb-nowrap">`gpt-6-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-22</span> | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-10-22</span> | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |

