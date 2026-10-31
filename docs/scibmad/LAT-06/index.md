---
title: "LAT-06: Multipole tilt assignment needing type promotion errors"
hide:
  - toc
---

# LAT-06 — Multipole tilt assignment needing type promotion errors

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on Beamlines main (`35fe016`) |
| Source | [`Beamlines.jl/src/bmultipole.jl:151-158`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/bmultipole.jl#L151-L158)<br>[`Beamlines.jl/src/emultipole.jl:135-141`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/emultipole.jl#L135-L141)<br>[`Beamlines.jl/src/element.jl:486-493`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/element.jl#L486-L493)<br>[`element.jl:493`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/element.jl#L493)<br>[`bmultipole.jl:129`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/bmultipole.jl#L129)<br>[`:159-165`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/bmultipole.jl#L159-L165) |

Assigning a multipole tilt that needs a wider number type than the element's stored strengths raises `Incorrect value for BMultipoleParams` when that multipole order already exists. `tilt1 = 0.1` on the Float32 `Quadrupole(Kn1=1f0)` fails, as does a ForwardDiff `Dual` tilt on an ordinary Float64 quadrupole, and the electric `etilt1` fails the same way. The assignment should store the promoted value, as it does for an order the element does not yet have (`tilt2 = 0.1` returns 0.1) and for an assignment that needs no promotion. Derivatives with respect to multipole tilts therefore cannot be taken by assigning a `Dual` tilt; the failure is loud and produces no wrong values.

??? example "Reproducer: LAT-06.jl"

    [Download LAT-06.jl](LAT-06.jl)

    ```julia
    # LAT-06: Multipole tilt assignment needing type promotion errors
    #
    # Assigning a multipole tilt that needs a wider number type than the element's stored strengths
    # raises `Incorrect value for BMultipoleParams` when that multipole order already exists. `tilt1 =
    # 0.1` on the Float32 `Quadrupole(Kn1=1f0)` fails, as does a ForwardDiff `Dual` tilt on an ordinary
    # Float64 quadrupole, and the electric `etilt1` fails the same way. The assignment should store the
    # promoted value, as it does for an order the element does not yet have (`tilt2 = 0.1` returns 0.1)
    # and for an assignment that needs no promotion. Derivatives with respect to multipole tilts
    # therefore cannot be taken by assigning a `Dual` tilt; the failure is loud and produces no wrong
    # values.
    #
    # The script attempts three promoting tilt assignments to an existing order: `tilt1 = 0.1` on a
    # Float32 quadrupole, a `Dual` `tilt1` on a Float64 quadrupole and `etilt1 = 0.1` on a Float32
    # electric multipole. Two controls must succeed: a tilt for a new order on the Float32 quadrupole
    # and a tilt that needs no promotion.
    using SciBmad
    const FD = SciBmad.ForwardDiff
    function attempt(label, f, expected)
        try
            val = f()
            println(rpad(label, 44), " observed ", val, "  expected ", expected)
            return val == expected
        catch e
            println(rpad(label, 44), " observed ERROR: ", first(split(sprint(showerror, e), '\n')), "  expected ", expected)
            return false
        end
    end
    # For an existing order, param_replace returns the scalar from setproperty! instead of the promoted
    # parameter group, and the element then stores that scalar as the group, which fails validation.
    ok = Bool[]
    push!(ok, attempt("Quadrupole(Kn1=1f0); tilt1 = 0.1", () -> (e = Quadrupole(Kn1=1f0); e.tilt1 = 0.1; e.tilt1), 0.1))
    push!(ok, attempt("Quadrupole(Kn1=0.36); tilt1 = Dual(0.1,1)", () -> (e = Quadrupole(Kn1=0.36); d = FD.Dual(0.1, 1.0); e.tilt1 = d; e.tilt1 == d), true))
    push!(ok, attempt("LineElement(En1=1f0); etilt1 = 0.1", () -> (e = LineElement(En1=1f0); e.etilt1 = 0.1; e.etilt1), 0.1))
    println("controls (promotion through the add-new-order branch, bmultipole.jl:159-165):")
    push!(ok, attempt("Quadrupole(Kn1=1f0); tilt2 = 0.1 (new order)", () -> (e = Quadrupole(Kn1=1f0); e.tilt2 = 0.1; e.tilt2), 0.1))
    push!(ok, attempt("Quadrupole(Kn1=1.0); tilt1 = 0.1 (no promotion)", () -> (e = Quadrupole(Kn1=1.0); e.tilt1 = 0.1; e.tilt1), 0.1))
    println(all(ok[4:5]) && !all(ok[1:3]) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU |

