---
title: "OPT-09: An invalid getterm throws UndefVarError: s"
hide:
  - toc
---

# OPT-09 — An invalid `getterm` throws `UndefVarError: s`

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P5 |
| Severity | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-22 on SciBmad 0.6.0 (`62399f9`); not reproduced on SciBmad main (`ae9423d`), 2026-10-31 |
| Issue | [[gh:https://github.com/bmad-sim/SciBmad.jl/issues/150]], filed from this evidence |
| Pull request | [[gh:https://github.com/bmad-sim/SciBmad.jl/pull/151]], filed from this evidence<br>[[gh:https://github.com/bmad-sim/SciBmad.jl/pull/158]], fix |
| Source | [`src/adv.jl:28-30`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/adv.jl#L28-L30) |

Requesting a coefficient beyond the truncation order, such as `getterm(q1, delta=3)` after `twiss(chrom=2)`, throws `UndefVarError: s` instead of the intended error message, which names the truncation order as the problem. GTPSA's own validity test confirms that the request is out of range, so failing is correct. Only the message is misleading, and no value is affected.

??? example "Reproducer: OPT-09.jl"

    [Download OPT-09.jl](OPT-09.jl)

    ```julia
    # OPT-09: An invalid `getterm` throws `UndefVarError: s`
    #
    # Requesting a coefficient beyond the truncation order, such as `getterm(q1, delta=3)` after
    # `twiss(chrom=2)`, throws `UndefVarError: s` instead of the intended error message, which names the
    # truncation order as the problem. GTPSA's own validity test confirms that the request is out of
    # range, so failing is correct. Only the message is misleading, and no value is affected.
    #
    # The script runs `twiss(chrom=2)` on a periodic cell of thin quadrupoles and 1 m drifts (1 GeV/c
    # electrons, RF off), reads the valid term `getterm(q1, delta=1)`, and asks GTPSA's
    # `mad_desc_isvalidm` whether the descriptor holds δ³. It then calls `getterm(q1, delta=3)` and
    # reports the type and message of the error it throws.
    using SciBmad
    qf = Multipole(Kn1L = 0.8); qd = Multipole(Kn1L = -0.7); d = Drift(L = 1.0)
    bl = Beamline([qf, d, qd, d], species_ref = Species("electron"), pc_ref = 1e9)
    tw = twiss(bl, rf_on = false, chrom = 2)
    println("q1 type: ", typeof(tw.q1), "  getterm(q1, delta=1)=", getterm(tw.q1, delta = 1))
    tps = tw.q1.taylor_series
    mono = UInt8[0, 0, 0, 0, 0, 3]
    println("independent: mad_desc_isvalidm(delta^3) = ", GTPSA.mad_desc_isvalidm(GTPSA.getdesc(tps).desc, Cint(6), mono))
    # The guard that rejects this monomial builds its message from a variable `s` that is not defined in
    # that function.
    msg = try
        getterm(tw.q1, delta = 3); "no error"
    catch err
        string(typeof(err), ": ", first(split(sprint(showerror, err), '\n')))
    end
    println("getterm(q1, delta=3) -> ", msg)
    println(occursin("UndefVarError", msg) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release tag clones, Julia 1.13.2 guide wrapper, CPU, Float64 |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release tag clones, Julia 1.13.2 guide wrapper, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release tag clones, Julia 1.13.2 guide wrapper, CPU, Float64 |

