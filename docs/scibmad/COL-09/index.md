---
title: "COL-09: AGS spin-ramp example: plotted time span and crossing-rate factor"
hide:
  - toc
---

# COL-09 — AGS spin-ramp example: plotted time span and crossing-rate factor

| | |
|---|---|
| Code | [SciBmad](../index.md) · Collective effects and spin |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-31 on SciBmad main (`ae9423d`) |
| Source | [`SciBmad.jl/examples/julia/ags-spin-ramp.ipynb:333`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/examples/julia/ags-spin-ramp.ipynb#L333)<br>[`457`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/examples/julia/ags-spin-ramp.ipynb#L457)<br>[`599-600`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/examples/julia/ags-spin-ramp.ipynb#L599-L600)<br>[`ags-spin-ramp.ipynb:193`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/examples/julia/ags-spin-ramp.ipynb#L193)<br>[`ags-spin-ramp.ipynb:243`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/examples/julia/ags-spin-ramp.ipynb#L243)<br>[`ags-spin-ramp.ipynb:263`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/examples/julia/ags-spin-ramp.ipynb#L263)<br>[`ags-spin-ramp.ipynb:315`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/examples/julia/ags-spin-ramp.ipynb#L315)<br>[`ags-spin-ramp.ipynb:333`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/examples/julia/ags-spin-ramp.ipynb#L333)<br>[`ags-spin-ramp.ipynb:457`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/examples/julia/ags-spin-ramp.ipynb#L457)<br>[`ags-spin-ramp.ipynb:599-600`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/examples/julia/ags-spin-ramp.ipynb#L599-L600) |

The Julia AGS spin-ramp example notebook compresses its plotted time axis and omits a factor 1/(2π) from the resonance-crossing rate, so the resonance strength it infers is wrong. The notebook defines the Froissart–Stora crossing rate α as the change in spin tune Gγ (anomaly times Lorentz factor) per turn divided by 2π, and inverting the Froissart–Stora formula for synthetic data establishes the expected strength. A spin-tune ramp of 2.000 over 1000 turns is plotted as 2.000 × 10⁻⁴, and α is taken as 0.002000 instead of 0.0003183; synthetic data with resonance strength 0.01000 yield 0.02507. Tracking is unaffected, but readers reusing the plotted rate or the inferred strength inherit factors of 6.283 and 2.507, respectively.

??? example "Reproducer: COL-09.jl"

    [Download COL-09.jl](COL-09.jl)

    ```julia
    # COL-09: AGS spin-ramp example: plotted time span and crossing-rate factor
    #
    # The Julia AGS spin-ramp example notebook compresses its plotted time axis and omits a factor
    # 1/(2π) from the resonance-crossing rate, so the resonance strength it infers is wrong. The
    # notebook defines the Froissart–Stora crossing rate α as the change in spin tune Gγ (anomaly times
    # Lorentz factor) per turn divided by 2π, and inverting the Froissart–Stora formula for synthetic
    # data establishes the expected strength. A spin-tune ramp of 2.000 over 1000 turns is plotted as
    # 2.000 × 10⁻⁴, and α is taken as 0.002000 instead of 0.0003183; synthetic data with resonance
    # strength 0.01000 yield 0.02507. Tracking is unaffected, but readers reusing the plotted rate or
    # the inferred strength inherit factors of 6.283 and 2.507, respectively.
    #
    # The script reads the notebook from the installed SciBmad package, checks that its time-axis, ramp,
    # crossing-rate and resonance-strength lines are present verbatim, and evaluates their arithmetic
    # against the notebook's own definition of α. It synthesizes a final vertical polarization from a
    # known resonance strength with the Froissart–Stora formula and recovers the strength with the
    # notebook's formula. As a control, it confirms that `track` with `n_turns = 3` saves four entries,
    # one per turn.
    nb = read(joinpath(dirname(Base.find_package("SciBmad")), "..", "examples", "julia", "ags-spin-ramp.ipynb"), String)
    # Notebook lines (in the .ipynb JSON):
    #   :193  alpha = (change in G*gamma per turn)/(2 pi)            (Froissart-Stora convention)
    #   :263  ring.E_ref = E_start + dE_dt*Time(), dE_dt = (E_end-E_start)/(1000 trev): dGgamma = 2 in 1000 turns
    #   :315  track(...; n_turns = 1000), save_every_n_turns = 1 -> saved turns 0:1000 at t = k*trev
    #   :333  T = approx_rev_time .* (0:1000) ./ 10000         (times used to label the saved turns)
    #   :457  epsilon = sqrt(2*(2/1000)/pi*log(2/(1+s_y[end])))  (uses alpha = 2/1000)
    #   :599-600 the alpha axis is 2/N
    lines = ["T = approx_rev_time .* (0:1000) ./ 10000;", "ring.E_ref = E_start + dE_dt*Time();",
             "epsilon = sqrt(2*(2/1000)/pi*log(2/(1+s_y[end])))", "scatter!(2 ./ turns, final_s_y)",
             "\\\\alpha = (\\\\text{Change in \$G\\\\gamma\$ per turn})/(2\\\\pi)"]
    for l in lines
        println("present in shipped notebook: ", occursin(l, nb), "  ", l)
    end

    trev = 1.0                                 # any value; results are ratios
    dGg_dt = 2 / (1000trev)                    # G*gamma ramp rate implied by dE_dt
    T = trev .* (0:1000) ./ 10000
    println("G*gamma span on the plotted axis = ", dGg_dt * (T[end] - T[1]), "   tracked span = ", dGg_dt * 1000trev)

    alpha_def = (2 / 1000) / (2pi)             # notebook's own definition
    alpha_used = 2 / 1000
    println("alpha by definition = ", alpha_def, "   alpha used = ", alpha_used, "   ratio = ", alpha_used / alpha_def)
    # Froissart-Stora: S_y = 2 exp(-pi eps^2 / (2 alpha)) - 1. Synthesize S_y from a known eps:
    eps_true = 0.01
    Sy = 2exp(-pi * eps_true^2 / (2alpha_def)) - 1
    eps_nb = sqrt(2 * (2 / 1000) / pi * log(2 / (1 + Sy)))
    println("eps recovered by the notebook formula = ", eps_nb, "   true eps = ", eps_true,
            "   ratio = ", eps_nb / eps_true, "   sqrt(2 pi) = ", sqrt(2pi))
    # Independent recheck: track saves turns 0:n_turns (n_turns+1 entries) at save_every_n_turns = 1,
    # so saved entry k is at t = k*trev, not k*trev/10000.
    using SciBmad
    r = track(Beamline([Drift(L = 1.0)], species_ref = Species("electron"), E_ref = 1e9), n_turns = 3)
    println("recheck: n_turns = 3 -> saved entries = ", size(r.v, 3))
    ok = all(l -> occursin(l, nb), lines) && abs(dGg_dt * T[end] - 2) > 1 && abs(eps_nb / eps_true - 1) > 0.1
    println(ok ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), source, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), source, Julia 1.13.2, CPU, Float64, serial |

