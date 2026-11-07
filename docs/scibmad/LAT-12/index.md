---
title: "LAT-12: writebl fails for any electric multipole"
hide:
  - toc
---

# LAT-12 — `writebl` fails for any electric multipole

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on Beamlines release [v0.11.0](https://github.com/bmad-sim/Beamlines.jl/tree/v0.11.0) (`35fe016`) |
| Source | [`Beamlines.jl/src/io.jl:98`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/io.jl#L98)<br>[`Beamlines.jl/src/io.jl:13-14`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/io.jl#L13-L14)<br>[`Beamlines.jl/src/emultipole.jl:180-185`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/emultipole.jl#L180-L185)<br>[`:55`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/emultipole.jl#L55)<br>[`Beamlines.jl/src/io.jl:1-4`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/io.jl#L1-L4) |

`writebl`, the Beamlines function that writes a lattice as Julia code and is documented as internal and under development, raises a `FieldError` for any element with an electric multipole, because it reads a field `tilt` where the electric multipole field is named `etilt`. The electric multipole's own `show` uses `etilt`, and the magnetic branch, which reads its own `tilt` field, writes `Kn1=0.2,tilt1=0.1` for a control element. An element with `En0 = 1` fails instead of producing text containing `En0=1.0`. The failure is loud.

Run `julia LAT-12.jl`.

??? example "Reproducer: LAT-12.jl"

    [Download LAT-12.jl](LAT-12.jl)

    ```julia
    # LAT-12: `writebl` fails for any electric multipole
    #
    # `writebl`, the Beamlines function that writes a lattice as Julia code and is documented as
    # internal and under development, raises a `FieldError` for any element with an electric multipole,
    # because it reads a field `tilt` where the electric multipole field is named `etilt`. The electric
    # multipole's own `show` uses `etilt`, and the magnetic branch, which reads its own `tilt` field,
    # writes `Kn1=0.2,tilt1=0.1` for a control element. An element with `En0 = 1` fails instead of
    # producing text containing `En0=1.0`. The failure is loud.
    #
    # The script calls `writebl` on a Beamline holding one element with `En0 = 1.0` and expects a
    # `Beamline` expression containing `En0=1.0`. It also prints the field names of the electric
    # multipole, and writes a magnetic element with `Kn1` and `tilt1` as a control.
    using SciBmad
    bl = Beamline([LineElement(L=1.0, En0=1.0)], species_ref=Species("proton"), pc_ref=1e9)
    # writebl reads `em.tilt` from each electric multipole, whose field is named `etilt`.
    r = try
        io = IOBuffer(); Beamlines.writebl(io, bl); String(take!(io))
    catch e
        "ERROR: " * first(split(sprint(showerror, e), '\n'))
    end
    println("observed writebl(En0 element) -> ", r)
    em = first(bl.line[1].EMultipoleParams)
    println("EMultipole item properties: ", propertynames(em))
    blm = Beamline([LineElement(L=1.0, Kn1=0.2, tilt1=0.1)], species_ref=Species("proton"), pc_ref=1e9)
    io = IOBuffer(); Beamlines.writebl(io, blm); println("control magnetic writebl -> ", String(take!(io)))
    println("expected: a Julia Beamline expression containing En0=1.0")
    println(startswith(r, "ERROR") ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |

