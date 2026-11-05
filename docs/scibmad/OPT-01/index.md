---
title: "OPT-01: Coasting twiss ignores delta0; a_initial runs ignore delta0 and v0"
hide:
  - toc
---

# OPT-01 — Coasting `twiss` ignores `delta0`; `a_initial` runs ignore `delta0` and `v0`

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on SciBmad main (`ae9423d`) |
| Source | [`src/twiss/twiss.jl:466-474`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss.jl#L466-L474)<br>[`twiss.jl:499-503`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss.jl#L499-L503)<br>[`twiss_operators.jl:531`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss_operators.jl#L531)<br>[`twiss.jl:241-245`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss.jl#L241-L245)<br>[`twiss.jl:108-113`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss.jl#L108-L113)<br>[`twiss.jl:242`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss.jl#L242)<br>[`twiss.jl:272`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss.jl#L272) |

For a coasting beam (RF off), `twiss(delta0=δ₀)` finds the closed orbit at the momentum offset δ₀ but tracks the map from δ = 0, and a run started from an initial normalizing map `a_initial` ignores both `delta0` and `v0`. Thin-lens transfer matrices and Jacobians of tracking at δ₀ give the correct optics: for 1 GeV/c electrons in a thin-quadrupole cell at δ₀ = 0.01, `twiss` returns the tune q₁ = 0.1436 and β₁ = 3.441 m instead of 0.1422 and 3.421 m, and `pz` = 0. In a checked ring with dispersion the tabulated orbit does not close and the spin tune is taken at δ = 0. Periodic optics with RF on are unaffected, but `a_initial` still bypasses `v0`.

??? example "Reproducer: OPT-01.jl"

    [Download OPT-01.jl](OPT-01.jl)

    ```julia
    # OPT-01: Coasting `twiss` ignores `delta0`; `a_initial` runs ignore `delta0` and `v0`
    #
    # For a coasting beam (RF off), `twiss(delta0=δ₀)` finds the closed orbit at the momentum offset δ₀
    # but tracks the map from δ = 0, and a run started from an initial normalizing map `a_initial`
    # ignores both `delta0` and `v0`. Thin-lens transfer matrices and Jacobians of tracking at δ₀ give
    # the correct optics: for 1 GeV/c electrons in a thin-quadrupole cell at δ₀ = 0.01, `twiss` returns
    # the tune q₁ = 0.1436 and β₁ = 3.441 m instead of 0.1422 and 3.421 m, and `pz` = 0. In a checked
    # ring with dispersion the tabulated orbit does not close and the spin tune is taken at δ = 0.
    # Periodic optics with RF on are unaffected, but `a_initial` still bypasses `v0`.
    #
    # The script builds a periodic cell of thin quadrupoles and 1 m drifts for 1 GeV/c electrons, whose
    # closed orbit is zero at every δ, and runs coasting `twiss` at δ₀ = ±0.01. It compares q₁, β₁ and
    # `pz` with thin-lens matrices that use the exact-drift chromatic length L/(1+δ), and checks those
    # against the ForwardDiff Jacobian of `track` at δ₀. It then starts an open-line `twiss` from the
    # periodic normalizing map as `a_initial`, requesting δ₀ = 0.01 and a 1 mm x offset in `v0`, and
    # checks the first-row x and `pz`.
    using SciBmad, LinearAlgebra
    const FD = SciBmad.ForwardDiff
    num(x) = x isa AmplitudeDependentValue ? getterm(x) : x

    # The closed orbit of this cell is x = px = y = py = 0 at every δ, so only δ itself changes the
    # optics.
    qf = Multipole(Kn1L = 0.8); qd = Multipole(Kn1L = -0.7); d = Drift(L = 1.0)
    bl = Beamline([qf, d, qd, d], species_ref = Species("electron"), pc_ref = 1e9)

    # Thin-lens matrices in normalized px, kicks normalized to p0, drifts with the exact-drift chromatic
    # length L/(1+δ).
    D(δ) = [1.0 1 / (1 + δ); 0 1]
    K(k) = [1.0 0; -k 1]
    M(δ) = D(δ) * K(-0.7) * D(δ) * K(0.8)              # horizontal, start of qf
    tunebeta(m) = (mu = acos(tr(m) / 2); (mu / 2pi, m[1, 2] / sin(mu)))

    bad = false
    for δ in (-0.01, 0.01)
        tw = twiss(bl, rf_on = false, delta0 = δ, cols = ["beta1", "pz"])
        qa, ba = tunebeta(M(δ))
        jac = FD.jacobian(v -> vec(track(bl, v0 = reshape(v, 1, 6)).v[1, :, end]), [0, 0, 0, 0, 0, δ])
        qt, bt = tunebeta(jac[1:2, 1:2])
        println("delta0=$δ  observed q1=", num(tw.q1), " beta1=", tw.beta1[1], " pz=", tw.pz[1])
        println("            analytic q1=", qa, " beta1=", ba, " pz=", δ)
        println("            tracking-Jacobian q1=", qt, " beta1=", bt,
                "  |analytic-tracking|=", max(abs(qa - qt), abs(ba - bt)))
        global bad |= abs(num(tw.q1) - qa) > 1e-6 || abs(tw.pz[1] - δ) > 1e-12
    end
    q00, _ = tunebeta(M(0.0)); println("on-momentum analytic q1=", q00)

    # a_initial branch: start an open-line Twiss from the periodic normalizing map,
    # requesting δ=0.01 and a 1 mm x offset of the expansion origin.
    tw0 = twiss(bl, rf_on = false, cols = ["a"], as_taylor_series = true)
    a = tw0.a[1]
    tw1 = twiss(bl, rf_on = false, a_initial = a, delta0 = 0.01, v0 = [1e-3 0.0 0.0 0.0 0.0 0.01], cols = ["x", "pz", "beta1"])
    println("a_initial + delta0=0.01, v0 x=1e-3: observed x[1]=", tw1.x[1], " pz[1]=", tw1.pz[1],
            " beta1[end]=", tw1.beta1[end], "; expected x[1]=1e-3, pz[1]=0.01")
    bad |= abs(tw1.pz[1] - 0.01) > 1e-12 || abs(tw1.x[1] - 1e-3) > 1e-12
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Found | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |

