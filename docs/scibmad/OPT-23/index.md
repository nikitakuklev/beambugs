---
title: "OPT-23: twiss with h columns and spin=true ends the process (division by a zero quaternion)"
hide:
  - toc
---

# OPT-23 — `twiss` with `h` columns and `spin=true` ends the process (division by a zero quaternion)

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-31 on SciBmad main (`ae9423d`) |
| Source | [`src/twiss/twiss_operators.jl:920-977`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss_operators.jl#L920-L977)<br>[`:940`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss_operators.jl#L940)<br>[`:974`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss_operators.jl#L974)<br>[`map.jl:614`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/map.jl#L614)<br>[`ext/NonlinearNormalFormGTPSAExt.jl:29-38`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/ext/NonlinearNormalFormGTPSAExt.jl#L29-L38)<br>[`:1013`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss_operators.jl#L1013)<br>[`:1018`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss_operators.jl#L1018) |

On rings where spin optics succeeds, requesting `h<ijkl>` columns (resonance driving terms) with `spin=true` ends the Julia process while `twiss` builds the Hamiltonian. The temporary linear map built there has a zero quaternion, and inverting it divides by zero in the native library, which bypasses Julia exception handling. For the 18 GeV electron quickstart FODO at order 2, the call prints `invalid domain div` and exits with status 1. Giving that map the identity quaternion lets the checked orbital coefficients agree with `spin=false` to 1.990e-14, as the decoupling of orbital and spin motion requires. Closed-lattice requests that reach this step are affected; open-lattice `h` is unimplemented, and spin normalization has separate failure conditions.

??? example "Reproducer: OPT-23.jl"

    [Download OPT-23.jl](OPT-23.jl)

    ```julia
    # OPT-23: `twiss` with `h` columns and `spin=true` ends the process (division by a zero quaternion)
    #
    # On rings where spin optics succeeds, requesting `h<ijkl>` columns (resonance driving terms) with
    # `spin=true` ends the Julia process while `twiss` builds the Hamiltonian. The temporary linear map
    # built there has a zero quaternion, and inverting it divides by zero in the native library, which
    # bypasses Julia exception handling. For the 18 GeV electron quickstart FODO at order 2, the call
    # prints `invalid domain div` and exits with status 1. Giving that map the identity quaternion lets
    # the checked orbital coefficients agree with `spin=false` to 1.990e-14, as the decoupling of
    # orbital and spin motion requires. Closed-lattice requests that reach this step are affected;
    # open-lattice `h` is unimplemented, and spin normalization has separate failure conditions.
    #
    # The script requests columns `h3000`, `h2100`, `h1020` and `h1002` with `spin=true` on the
    # quickstart FODO, with the quaternion inverse wrapped to raise a Julia error at |q|² = 0, so the
    # failing call is located instead of ending the process. It then re-evaluates the internal
    # linear-map builder with the identity quaternion added and compares the `h` columns with those of
    # `spin=false`. Set `RAW` to 1 for the plain call, which ends the process with the native diagnostic
    # and exit status 1.
    using SciBmad, LinearAlgebra
    const NNF = SciBmad.NNF; const TI = SciBmad.TI
    @elements begin
      qf = Quadrupole(Kn1=0.37, L=0.5); sf = Sextupole(Kn2=1.2, L=0.2); d = Drift(L=0.1)
      b = SBend(L=1.2, angle=pi/132); qd = Quadrupole(Kn1=-0.36, L=0.5); sd = Sextupole(Kn2=-1.2, L=0.2)
    end
    fodo = Beamline([qf, sf, b, d, qd, sd, b, d], species_ref=Species("electron"), pc_ref=18e9)  # quickstart FODO
    cols = ["h3000", "h2100", "h1020", "h1002"]
    if get(ENV, "RAW", "0") == "1"
      println("plain twiss(cols=h..., spin=true, order=2):"); flush(stdout)
      twiss(fodo, cols=cols, spin=true, order=2)
      println("returned (NOT REPRODUCED)")
      exit(0)
    end
    # Default mode: raise a Julia error where the native library would divide by |q|² = 0, so the
    # backtrace locates the call.
    const QT = NNF.Quaternion{<:GTPSA.TPS}
    nrm2(q1) = q1.q0 * q1.q0 + q1.q1 * q1.q1 + q1.q2 * q1.q2 + q1.q3 * q1.q3
    chk(n) = iszero(TI.scalar(n)) && error("quaternion inverse with |q|^2 = 0")
    function NNF.inv!(q::QT, q1::QT)
      n = nrm2(q1); chk(n)
      TI.copy_tps!(q.q1, -q1.q1/n); TI.copy_tps!(q.q2, -q1.q2/n); TI.copy_tps!(q.q3, -q1.q3/n); TI.copy_tps!(q.q0, q1.q0/n)
      return
    end
    init = TI.InitGTPSA{GTPSA.Dynamic,Descriptor}(dynamic_descriptor = Descriptor(6, 2))
    m0 = SciBmad._twiss_make_identity(zeros(1, 6), init, Val(false), Val(true))
    println("zero(map with spin): quaternion scalar parts = ", [TI.scalar(getfield(zero(m0).q, k)) for k in 1:4])
    failed = false
    try
      twiss(fodo, cols=cols, spin=true, order=2)
    catch err
      global failed = true
      fr = [f for f in stacktrace(catch_backtrace()) if occursin("NonlinearNormalForm", string(f.file)) || occursin("SciBmad", string(f.file))]
      println("twiss(h columns, spin=true): ", sprint(showerror, err), "  at ",
              join(["$(f.func) $(basename(string(f.file))):$(f.line)" for f in fr[1:min(end, 3)]], " <- "))
    end
    # Re-evaluate _make_ri_lin! with r_lin given the identity quaternion; the h columns with spin must
    # then equal those without, since the orbital normal form does not depend on spin.
    src = read(joinpath(dirname(pathof(SciBmad)), "twiss", "twiss_operators.jl"), String)
    i = findfirst("@inline function _make_ri_lin!(twi, cache)", src); j = findnext("\n  return ri_lin\nend", src, last(i))
    Core.eval(SciBmad, Meta.parse(replace(src[first(i):last(j)],
      "r_lin = zero(r)\n" => "r_lin = zero(r)\n  isnothing(r_lin.q) || NNF.setquat!(r_lin.q, q=I)\n")))
    tws = twiss(fodo, cols=cols, spin=true, order=2); tw0 = twiss(fodo, cols=cols, spin=false, order=2)
    println("independent: r_lin with identity quaternion: max |h(spin=true) - h(spin=false)| = ",
            maximum(maximum(abs.(getproperty(tws, Symbol(c)) .- getproperty(tw0, Symbol(c)))) for c in cols),
            ", max |h| = ", maximum(maximum(abs.(getproperty(tw0, Symbol(c)))) for c in cols))
    println(failed ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-10-23 | SciBmad main (`5f0fb83ac889`), pinned source clones and main worktrees, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | SciBmad 0.6.0 (`5f0fb83ac889`), pinned clones as installed (SciBmad 5f0fb83ac889, NonlinearNormalForm 7e4053985736, GTPSA a5790a7deaa6, BeamTracking d8e45fdc25db, Beamlines 35fe0162c72d), Julia 1.13.2, CPU, Float64, one thread |

