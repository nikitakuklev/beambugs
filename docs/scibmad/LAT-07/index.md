---
title: "LAT-07: GTPSA special functions of a DefExpr discard the evaluation Context"
hide:
  - toc
---

# LAT-07 — GTPSA special functions of a `DefExpr` discard the evaluation Context

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on Beamlines release [v0.11.0](https://github.com/bmad-sim/Beamlines.jl/tree/v0.11.0) (`35fe016`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Pull request | [[gh:https://github.com/bmad-sim/Beamlines.jl/pull/171]], fix |
| Source | [`Beamlines.jl/ext/BeamlinesGTPSAExt/BeamlinesGTPSAExt.jl:16-20`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/ext/BeamlinesGTPSAExt/BeamlinesGTPSAExt.jl#L16-L20)<br>[`Beamlines.jl/src/defexpr.jl:132-160`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/defexpr.jl#L132-L160)<br>[`Beamlines.jl/src/defexpr.jl:94-99`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/defexpr.jl#L94-L99)<br>[`Beamlines.jl/src/context.jl:88-97`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/context.jl#L88-L97) |

Applying `sincu` (sin x/x) to a deferred expression (`DefExpr`) discards the Context supplied when the expression is evaluated and looks up variables in the global Contexts instead. With `d` reading a variable `a`, evaluating `sincu(d)` in `Context(a=0.3)` raises “Variable a is not defined” when no global `a` exists. A global `a = 0.1` instead produces 0.9983 without warning, rather than the correct sin(0.3)/0.3 = 0.9851, and a Beamline's `Kn1` built this way takes that wrong value; the generic `sin` correctly uses the supplied Context. Source inspection extends the failure to the other functions in the GTPSA extension list, including `sinhc`, `erf` and `wf`.

Run `julia LAT-07.jl`.

??? example "Reproducer: LAT-07.jl"

    [Download LAT-07.jl](LAT-07.jl)

    ```julia
    # LAT-07: GTPSA special functions of a `DefExpr` discard the evaluation Context
    #
    # Applying `sincu` (sin x/x) to a deferred expression (`DefExpr`) discards the Context supplied when
    # the expression is evaluated and looks up variables in the global Contexts instead. With `d`
    # reading a variable `a`, evaluating `sincu(d)` in `Context(a=0.3)` raises “Variable a is not
    # defined” when no global `a` exists. A global `a = 0.1` instead produces 0.9983 without warning,
    # rather than the correct sin(0.3)/0.3 = 0.9851, and a Beamline's `Kn1` built this way takes that
    # wrong value; the generic `sin` correctly uses the supplied Context. Source inspection extends the
    # failure to the other functions in the GTPSA extension list, including `sinhc`, `erf` and `wf`.
    #
    # The script evaluates `sincu(DefExpr(c -> c.a))` in `Context(a=0.3)`, first with no global `a` and
    # then after adding a global Context with `a = 0.1`, and reads the `Kn1` of a quadrupole built from
    # the same expression in a Beamline whose Context has `a = 0.3`. Each must equal sincu(0.3); `sin`
    # of the same expression is printed as a control.
    using SciBmad
    d = DefExpr(c -> c.a)
    exp = GTPSA.sincu(0.3)
    println("expected sincu(0.3) = ", exp, " (generic control: sin(d)(Context(a=0.3)) = ", sin(d)(Context(a=0.3)), ", sin(0.3) = ", sin(0.3), ")")
    # GTPSA.f(d::DefExpr) = DefExpr(() -> f(d())): the zero-argument lambda is wrapped as (c=NULL_CONTEXT) -> f(),
    # so d is evaluated in NULL_CONTEXT, which falls back to GLOBAL_CONTEXTS; generic functions pass c through.
    r1 = try
        GTPSA.sincu(d)(Context(a=0.3))
    catch e
        "ERROR: " * first(split(sprint(showerror, e), '\n'))
    end
    println("(a) no global a:   sincu(d)(Context(a=0.3)) = ", r1)
    push!(Beamlines.GLOBAL_CONTEXTS, Context(a=0.1))
    r2 = GTPSA.sincu(d)(Context(a=0.3))
    println("(b) global a=0.1:  sincu(d)(Context(a=0.3)) = ", r2, "  (= sincu(0.1) = ", GTPSA.sincu(0.1), ")")
    # Same through a lattice: a quadrupole strength defined in the Beamline's Context
    q = Quadrupole(L=0.5, Kn1=GTPSA.sincu(DefExpr(c -> c.a)))
    bl = Beamline([q], species_ref=Species("electron"), E_ref=1e9, context=Context(a=0.3))
    r3 = bl.line[1].Kn1
    println("(c) Beamline context a=0.3, global a=0.1: Kn1 = ", r3)
    bad = !(r1 isa Number && r1 == exp) || r2 != exp || r3 != exp
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |

