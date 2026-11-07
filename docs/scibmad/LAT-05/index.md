---
title: "LAT-05: 0 - BatchParam(array) returns the array unnegated"
hide:
  - toc
---

# LAT-05 — `0 - BatchParam(array)` returns the array unnegated

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-22 on BeamTracking release [v0.8.7](https://github.com/bmad-sim/BeamTracking.jl/tree/v0.8.7) (`65e343f`) |
| Issue | [[gh:https://github.com/bmad-sim/BeamTracking.jl/issues/374]], filed from this evidence |
| Pull request | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/375]], filed from this evidence |
| Source | [`BeamTracking.jl/src/batch.jl:102-103`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/batch.jl#L102-L103)<br>[`BeamTracking.jl/src/batch.jl:131`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/batch.jl#L131) |

Subtracting an array-backed `BatchParam` from scalar zero, as in `0 - c.K1`, returns the batch unnegated: `0 - BatchParam([1, -2])` gives [1, −2] instead of [−1, 2], while unary minus and lane-wise subtraction give the correct signs. For a 1 GeV electron starting at x = 1 mm with other coordinates zero, a 0.5 m quadrupole using `Kn1 = 0 - K1` with K1 = [0.3, 0.6] m⁻² focuses horizontally instead of defocusing. After a 1 m drift, x is [0.8146, 0.6334] mm instead of [1.190, 1.383] mm. The negation `-c.K1` works.

Run `julia LAT-05.jl`.

??? example "Reproducer: LAT-05.jl"

    [Download LAT-05.jl](LAT-05.jl)

    ```julia
    # LAT-05: `0 - BatchParam(array)` returns the array unnegated
    #
    # Subtracting an array-backed `BatchParam` from scalar zero, as in `0 - c.K1`, returns the batch
    # unnegated: `0 - BatchParam([1, -2])` gives [1, −2] instead of [−1, 2], while unary minus and
    # lane-wise subtraction give the correct signs. For a 1 GeV electron starting at x = 1 mm with other
    # coordinates zero, a 0.5 m quadrupole using `Kn1 = 0 - K1` with K1 = [0.3, 0.6] m⁻² focuses
    # horizontally instead of defocusing. After a 1 m drift, x is [0.8146, 0.6334] mm instead of [1.190,
    # 1.383] mm. The negation `-c.K1` works.
    #
    # The script compares `0 - BatchParam([1.0, -2.0])` with the lane-wise result and prints three
    # controls that negate correctly: a tiny nonzero scalar minus the batch, unary minus and a zero
    # batch minus the batch. It then tracks 1 GeV electrons through a 0.5 m quadrupole and a 1 m drift
    # over the two-lane batch K1 = [0.3, 0.6] m⁻², with `Kn1` written as `0 - K1` and as `-K1`; the
    # final x must agree.
    using SciBmad
    bp = BatchParam([1.0, -2.0])
    # _batch_addsub returns the batch unchanged for a scalar left operand ≈ 0, for both + and -.
    obs = (0 - bp).batch
    exp = [0 - 1.0, 0 - (-2.0)]        # lane-wise subtraction
    println("observed (0 - BatchParam([1,-2])).batch = ", obs)
    println("expected lane-wise 0 - b_i               = ", exp)
    println("controls: (1e-300 - bp).batch = ", (1e-300 - bp).batch, ", (-bp).batch = ", (-bp).batch,
            ", (BatchParam([0.0,0.0]) - bp).batch = ", (BatchParam([0.0, 0.0]) - bp).batch)

    # Consequence in tracking: two FODO-like quads whose strengths are written as 0 - K and -K must agree.
    function xfinal(f)
        q = Quadrupole(L=0.5, Kn1=DefExpr(f))
        bl = Beamline([q, Drift(L=1.0)], species_ref=Species("electron"), E_ref=1e9)
        bl.context.K1 = BatchParam([0.3, 0.6])
        res = track(bl; v0=repeat([1e-3 0 0 0 0 0], 2, 1))
        return res.v[:, 1, end]
    end
    xa = xfinal(c -> 0 - c.K1); xb = xfinal(c -> -c.K1)
    println("tracked x with Kn1 = 0 - K1: ", xa, "; with Kn1 = -K1: ", xb)
    bad = obs != exp || xa != xb
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |

