---
title: "OPT-06: Summary alphac (momentum compaction) equals the slip factor etac"
hide:
  - toc
---

# OPT-06 — Summary `alphac` (momentum compaction) equals the slip factor `etac`

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P1 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on SciBmad main (`ae9423d`) |
| Issue | [[gh:https://github.com/bmad-sim/SciBmad.jl/issues/152]], filed from this evidence |
| Pull request | [[gh:https://github.com/bmad-sim/SciBmad.jl/pull/153]], filed from this evidence |
| Source | [`src/twiss/twiss.jl:744-745`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss.jl#L744-L745)<br>[`twiss_operators.jl:596-668`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss_operators.jl#L596-L668)<br>[`twiss.jl:352`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss.jl#L352)<br>[`bmad/code/momentum_compaction.f90:47-59`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/code/momentum_compaction.f90#L47-L59)<br>[`tao_evaluate_a_datum.f90:1718`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/tao/code/tao_evaluate_a_datum.f90#L1718)<br>[`:2771`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/tao/code/tao_evaluate_a_datum.f90#L2771) |

The `twiss` summary value `alphac`, labelled the momentum compaction α_p = (1/C) dC/dδ (C the orbit length, δ the relative momentum offset), is computed from the coordinate slip, like the slip factor `etac`. For coasting beams on momentum whose closed orbit has zero longitudinal advance, `alphac` equals `etac` and is low by 1/γ² (γ the Lorentz factor). For 1 GeV/c protons it is −0.4682 instead of 0 in a straight cell and −0.2238 instead of 0.2443 in a cell with bends, as analytic transport and tracking establish. `etac` also loses velocity and closed-orbit time corrections: its straight-cell order-2 δ coefficient is 0.7470 instead of 1.185, and nonzero closed-orbit longitudinal advance corrupts its value. Protons and ions are most affected; ultrarelativistic electron errors are small.

??? example "Reproducer: OPT-06.jl"

    [Download OPT-06.jl](OPT-06.jl)

    ```julia
    # OPT-06: Summary `alphac` (momentum compaction) equals the slip factor `etac`
    #
    # The `twiss` summary value `alphac`, labelled the momentum compaction α_p = (1/C) dC/dδ (C the
    # orbit length, δ the relative momentum offset), is computed from the coordinate slip, like the slip
    # factor `etac`. For coasting beams on momentum whose closed orbit has zero longitudinal advance,
    # `alphac` equals `etac` and is low by 1/γ² (γ the Lorentz factor). For 1 GeV/c protons it is
    # −0.4682 instead of 0 in a straight cell and −0.2238 instead of 0.2443 in a cell with bends, as
    # analytic transport and tracking establish. `etac` also loses velocity and closed-orbit time
    # corrections: its straight-cell order-2 δ coefficient is 0.7470 instead of 1.185, and nonzero
    # closed-orbit longitudinal advance corrupts its value. Protons and ions are most affected;
    # ultrarelativistic electron errors are small.
    #
    # The script runs coasting `twiss` (RF off) for 1 GeV/c protons on a straight cell of thin
    # quadrupoles and drifts and on a cell of thick quadrupoles and two bends, and reads `alphac` and
    # `etac`. It compares them with Bmad's formulas η = −(M₅₆ + M₅,₁:₄·D₀)/C and α_p = η + 1/γ²,
    # evaluated from the ForwardDiff one-turn Jacobian M of `track`, with the closed-orbit dispersion D₀
    # from (I − M₄)D₀ = M[1:4,6]. In the straight cell the path length does not depend on δ, so
    # analytically α_p = 0 and η = −1/γ².
    using SciBmad, LinearAlgebra
    const FD = SciBmad.ForwardDiff
    num(x) = x isa AmplitudeDependentValue ? getterm(x) : x
    sp = Species("proton"); pc = 1e9
    mp = 938.27208943e6; γ2 = 1 + (pc / mp)^2   # proton rest energy [eV]; γ² = 1 + (βγ)²

    # α_p = (1/C) dC/dδ and η = (1/T) dT/dδ = α_p − 1/γ² when the closed orbit has zero z advance. Bmad:
    # R56_closed = M[5,6] + M[5,1:4]·D₀ with (I − M₄) D₀ = M[1:4,6], η = −R56_closed/C.
    function expected(bl, C)
        M = FD.jacobian(v -> vec(track(bl, v0 = reshape(v, 1, 6)).v[1, :, end]), zeros(6))
        D0 = (I - M[1:4, 1:4]) \ M[1:4, 6]
        η = -(M[5, 6] + dot(M[5, 1:4], D0)) / C
        return η, η + 1 / γ2
    end

    qf = Multipole(Kn1L = 0.8); qd = Multipole(Kn1L = -0.7); d = Drift(L = 1.0)
    straight = Beamline([qf, d, qd, d], species_ref = sp, pc_ref = pc)
    qf2 = Quadrupole(Kn1 = 0.36, L = 0.5); qd2 = Quadrupole(Kn1 = -0.36, L = 0.5); b = SBend(L = 1.2, angle = pi / 32); d2 = Drift(L = 0.1)
    bent = Beamline([qf2, d2, b, d2, qd2, d2, b, d2], species_ref = sp, pc_ref = pc)

    bad = false
    for (name, bl, C) in (("straight", straight, 2.0), ("bends", bent, 2 * 0.5 + 4 * 0.1 + 2 * 1.2))
        tw = twiss(bl, rf_on = false, cols = ["beta1"])
        η, αp = expected(bl, C)
        println(name, ": observed alphac=", num(tw.alphac), " etac=", num(tw.etac))
        println(name, ": expected alphac=", αp, " etac=", η, "  (1/γ²=", 1 / γ2, ", C=", tw.s[end], ")")
        global bad |= abs(num(tw.alphac) - αp) > 1e-9
    end
    println("analytic straight cell: alphac=0, etac=-1/γ²=", -1 / γ2)
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release tag clones, Julia 1.13.2 guide wrapper, CPU, Float64 |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release tag clones, Julia 1.13.2 guide wrapper, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release tag clones, Julia 1.13.2 guide wrapper, CPU, Float64 |

