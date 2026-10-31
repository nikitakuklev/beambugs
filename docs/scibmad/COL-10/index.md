---
title: "COL-10: IBS statistics are taken before the element's entrance alignment and aperture"
hide:
  - toc
---

# COL-10 — IBS statistics are taken before the element's entrance alignment and aperture

| | |
|---|---|
| Code | [SciBmad](../index.md) · Collective effects and spin |
| Priority | P1 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-23 on BeamTracking main (`d8e45fd`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`BeamTracking.jl/ext/BeamTrackingBeamlinesExt/unpack_bl.jl:139-160`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/unpack_bl.jl#L139-L160)<br>[`BeamTracking.jl/ext/BeamTrackingBeamlinesExt/general_bl.jl:127`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/general_bl.jl#L127)<br>[`BeamTracking.jl/src/kernels/ibs_kick.jl:37-42`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/ibs_kick.jl#L37-L42)<br>[`unpack_bl.jl:345`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/unpack_bl.jl#L345)<br>[`ibs_kick.jl:37-42`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/ibs_kick.jl#L37-L42) |

The intrabeam scattering (IBS) kick takes its bunch statistics before the element's entrance alignment and aperture act, so it mixes two coordinate frames and counts particles the entrance aperture removes. Offsetting a drift cannot change collisions within the bunch, as translating the bunch itself confirms. For 200 protons at 1.000 GeV with horizontal beam size 100.0 µm, offsetting a 1.000 m drift horizontally by 10.00 µm changes the largest IBS kick by 6.131% of its aligned value instead of zero; an outlier removed by the drift's entrance aperture changes it by 66.02% of the live-only value. Horizontal offsets and entrance losses are reproduced; source inspection places the other alignment parameters on the same entrance path.

??? example "Reproducer: COL-10.jl"

    [Download COL-10.jl](COL-10.jl)

    ```julia
    # COL-10: IBS statistics are taken before the element's entrance alignment and aperture
    #
    # The intrabeam scattering (IBS) kick takes its bunch statistics before the element's entrance
    # alignment and aperture act, so it mixes two coordinate frames and counts particles the entrance
    # aperture removes. Offsetting a drift cannot change collisions within the bunch, as translating the
    # bunch itself confirms. For 200 protons at 1.000 GeV with horizontal beam size 100.0 µm, offsetting
    # a 1.000 m drift horizontally by 10.00 µm changes the largest IBS kick by 6.131% of its aligned
    # value instead of zero; an outlier removed by the drift's entrance aperture changes it by 66.02% of
    # the live-only value. Horizontal offsets and entrance losses are reproduced; source inspection
    # places the other alignment parameters on the same entrance path.
    #
    # The script tracks 200 Gaussian-distributed protons, each representing 10⁴ particles, through a 1 m
    # drift with deterministic IBS damping only, and compares the momentum kicks for horizontal drift
    # offsets of 0.1, 1 and 10 beam sizes with those of the aligned drift. A second case adds an outlier
    # at 20 beam sizes that the drift's own entrance aperture removes and compares with the bunch
    # without it. Controls translate the bunch instead of the drift, which leaves the kicks unchanged,
    # and remove the aperture, which leaves the live kicks identical.
    using SciBmad, Random
    const BT = SciBmad.BeamTracking

    Random.seed!(1)
    n = 200
    sig = [1e-4 1e-5 1e-4 1e-5 1e-3 1e-4]
    v0 = randn(n, 6) .* sig
    sp = Species("proton")
    const W = 1e4                               # macroparticle weight
    ibs = Symplectic(ibs_damping_on = true, ibs_fluctuations_on = false)
    drift(; kw...) = LineElement(; L = 1.0, tracking_method = ibs, kw...)
    line(eles...) = Beamline(collect(eles), species_ref = sp, E_ref = 1.0e9)
    aperture = (x1_limit = -1e-3, x2_limit = 1e-3, y1_limit = -1e-3, y2_limit = 1e-3)  # 10 sigma

    function run(v, ln; state = fill(BT.STATE_ALIVE, size(v, 1)))
        b = Bunch(v = copy(v), state = copy(state), weight = W, species = sp, p_over_q_ref = ln.p_over_q_ref)
        track!(b, ln; use_explicit_SIMD = false)
        return b
    end
    dp(b, v) = b.coords.v[1:n, [2, 4, 6]] .- v[1:n, [2, 4, 6]]   # IBS momentum change of the first n rows
    rel(a, r) = maximum(abs, a .- r) / maximum(abs, r)

    ref = dp(run(v0, line(drift())), v0)
    println("max |kick| of the aligned drift = ", maximum(abs, ref))
    println("(a) element offset; expected 0:")
    rel_offset = Dict{Float64,Float64}()
    for dx in (1e-5, 1e-4, 1e-3)
        rel_offset[dx] = rel(dp(run(v0, line(drift(x_offset = dx))), v0), ref)
        println("    x_offset = $dx (", dx / sig[1], " sigma_x): max |kick - aligned| / max |kick| = ", rel_offset[dx])
    end
    # Independent recheck: move the bunch instead of the element. The kick only sees X - <X>,
    # so it must not change, and it does not, so the offset dependence comes from comparing
    # body-frame coordinates with lab-frame statistics.
    vs = v0 .+ [1e-4 0 0 0 0 0]
    println("    recheck: bunch shifted by 1e-4 through the aligned drift: rel = ", rel(dp(run(vs, line(drift())), vs), ref))

    println("(b) outlier removed by the drift's own entrance aperture; expected 0:")
    vo = vcat(v0, 20 .* sig)                   # outlier alive at the entrance, outside the aperture
    bA = run(vo, line(drift(; aperture...)))
    rel_aperture = rel(dp(bA, vo), dp(run(v0, line(drift(; aperture...))), v0))
    println("    outlier state after tracking = ", bA.coords.state[end], " (alive = ", BT.STATE_ALIVE, ")")
    println("    max |kick - kick without outlier| / max |kick| = ", rel_aperture)
    # Independent recheck: the same bunch through the drift without the aperture, where the
    # outlier stays alive. Identical live kicks mean the statistics never see the aperture.
    println("    recheck: live kicks with vs without the aperture: max |difference| = ",
            maximum(abs, dp(bA, vo) .- dp(run(vo, line(drift())), vo)))
    println(rel_offset[1e-5] > 1e-6 && rel_aperture > 1e-6 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-10-23 | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed (same-element entrance-aperture timing) | `gpt-6-astra` | xhigh | 2026-10-23 | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |

