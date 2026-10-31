---
title: "OPT-11: The spin Lie bracket uses the result field instead of the second input"
hide:
  - toc
---

# OPT-11 — The spin Lie bracket uses the result field instead of the second input

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on NonlinearNormalForm main (`7e40539`) |
| Source | [`NonlinearNormalForm.jl/src/vectorfield.jl:237-248`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/vectorfield.jl#L237-L248)<br>[`vectorfield.jl:196`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/vectorfield.jl#L196)<br>[`vectorfield.jl:258-299`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/vectorfield.jl#L258-L299)<br>[`forest/code/Ci_tpsa.f90:3424`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/forest/code/Ci_tpsa.f90#L3424) |

For vector fields with spin, written as (orbital part, spin quaternion), NonlinearNormalForm's in-place Lie bracket `lb!` uses the result field G instead of the second input H in its spin part, so the bracket is not antisymmetric. The commutator of the package's own Lie operators, and the bracket in Forest's FPP, require the term H·∇f where the code computes G·∇f. For F = (0, x·i) and H = (∂ₓ, 0), the spin component [F,H].q₁ is 0 instead of −1, and for generic polynomial fields the spin part is off by 5.0 in norm while the orbital part is exact. `lb!` is unexported, and its only caller, the correction step of `log`, changes only how fast `log` converges, so no public output is known to change.

??? example "Reproducer: OPT-11.jl"

    [Download OPT-11.jl](OPT-11.jl)

    ```julia
    # OPT-11: The spin Lie bracket uses the result field instead of the second input
    #
    # For vector fields with spin, written as (orbital part, spin quaternion), NonlinearNormalForm's
    # in-place Lie bracket `lb!` uses the result field G instead of the second input H in its spin part,
    # so the bracket is not antisymmetric. The commutator of the package's own Lie operators, and the
    # bracket in Forest's FPP, require the term H·∇f where the code computes G·∇f. For F = (0, x·i) and
    # H = (∂ₓ, 0), the spin component [F,H].q₁ is 0 instead of −1, and for generic polynomial fields the
    # spin part is off by 5.0 in norm while the orbital part is exact. `lb!` is unexported, and its only
    # caller, the correction step of `log`, changes only how fast `log` converges, so no public output
    # is known to change.
    #
    # The script works with `Descriptor(2, 4)` vector fields. Test 1 brackets F = (0, x·i) with H = (∂ₓ,
    # 0) in both orders and compares the spin component q₁ with the exact −1 and +1. Test 2 takes
    # generic polynomial fields, forms the commutator L_F L_H − L_H L_F by applying NNF's own in-place
    # Lie operator `mul!` to the identity spin map, and compares its orbital and spin parts with `lb!`;
    # it also shows that replacing G·∇f by H·∇f makes the spin part agree.
    using SciBmad
    const NNF = NonlinearNormalForm
    d = Descriptor(2, 4); x, p = vars(d); z = zero(x); o = one(x)
    vf(v, q) = VectorField(v = v, q = q)
    br(F, H) = (G = zero(F); NNF.lb!(G, F, H); G)
    qs(G) = [G.q.q0, G.q.q1, G.q.q2, G.q.q3]

    # Test 1: the exact [F,H] spin part is −H·∇f = −i, and [H,F] = +i.
    F = vf([z, z], [z, x, z, z]); H = vf([o, z], [z, z, z, z])
    gfh, ghf = br(F, H), br(H, F)
    println("test 1: [F,H].q1 = ", scalar(gfh.q.q1), " (exact -1);  [H,F].q1 = ", scalar(ghf.q.q1),
            " (exact +1);  antisymmetry residual = ", scalar(gfh.q.q1 + ghf.q.q1))

    F2 = vf([p^2, -x], [z, x * p, z, 0.3x]); H2 = vf([o + x, p], [z, z, p, x^2])
    G = br(F2, H2)
    m0 = DAMap(v = [x, p], q = [o, z, z, z], nv = 2)
    # A field (F, f) acts on a map (m, q) as L_(F,f)(m,q) = (F·∇m, F·∇q + q f). The commutator L_F L_H −
    # L_H L_F is L_(G,g) with G = F·∇H − H·∇F and g = h f − f h + F·∇h − H·∇f, as in FPP's liebra.
    L(F, m) = (r = zero(m); NNF.mul!(r, F, m); r)   # in-place Lie operator; `F * m` is ambiguous (OPT-13)
    C1 = L(F2, L(H2, m0)); C2 = L(H2, L(F2, m0))
    cq = [C1.q.q0 - C2.q.q0, C1.q.q1 - C2.q.q1, C1.q.q2 - C2.q.q2, C1.q.q3 - C2.q.q3]
    cv = [C1.v[i] - C2.v[i] for i in 1:2]
    eorb = maximum(GTPSA.normTPS(cv[i] - G.v[i]) for i in 1:2)
    espin = maximum(GTPSA.normTPS(cq[i] - qs(G)[i]) for i in 1:4)
    asym = maximum(GTPSA.normTPS(qs(G)[i] + qs(br(H2, F2))[i]) for i in 1:4)
    println("test 2: |orbital lb! - commutator| = ", eorb, "  |spin lb! - commutator| = ", espin,
            "  spin antisymmetry residual = ", asym)
    # Corrected spin part: lb! result + G·∇f - H·∇f (replace the G derivative by the H derivative).
    const TI = SciBmad.TI
    fg(V, t) = (r = zero(t); TI.fgrad!(r, V, t); r)
    fq = qs(F2); gfix = [qs(G)[i] + fg(G.v, fq[i]) - fg(H2.v, fq[i]) for i in 1:4]
    println("        |corrected spin - commutator| = ", maximum(GTPSA.normTPS(cq[i] - gfix[i]) for i in 1:4))
    println(abs(scalar(gfh.q.q1) + 1) > 1e-12 || espin > 1e-12 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | NonlinearNormalForm 0.5.3 (`7e4053985736`), release tag clones, Julia 1.13.2 guide wrapper, CPU, Float64; SciBmad 0.6.0 (62399f930847) for the twiss path |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | NonlinearNormalForm 0.5.3 (`7e4053985736`), release tag clones, Julia 1.13.2 guide wrapper, CPU, Float64; SciBmad 0.6.0 (62399f930847) for the twiss path |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | NonlinearNormalForm 0.5.3 (`7e4053985736`), release tag clones, Julia 1.13.2 guide wrapper, CPU, Float64; SciBmad 0.6.0 (62399f930847) for the twiss path |

