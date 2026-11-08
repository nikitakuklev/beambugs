---
title: "LAT-26: Bmad SciBmad writer turns thin reference bends into ordinary dipole kicks"
hide:
  - toc
---

# LAT-26 — Bmad SciBmad writer turns thin reference bends into ordinary dipole kicks

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on bmad-ecosystem release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`bmad/output/write_lattice_scibmad_format.f90:115–118`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_scibmad_format.f90#L115-L118)<br>[`bmad/output/write_lattice_scibmad_format.f90:647–670`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_scibmad_format.f90#L647-L670)<br>[`bmad/output/write_lattice_scibmad_format.f90:10–13`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_scibmad_format.f90#L10-L13)<br>[`bmad/output/write_lattice_scibmad_format.f90:94–96`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_scibmad_format.f90#L94-L96)<br>[`bmad/output/write_lattice_scibmad_format.f90:490`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_scibmad_format.f90#L490)<br>[`util_programs/bmad_to_scibmad/bmad_to_scibmad.f90:53–59`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/bmad_to_scibmad/bmad_to_scibmad.f90#L53-L59)<br>[`bmad/doc/elements.tex:3860–3878`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L3860-L3878)<br>[`bmad/modules/multipole_mod.f90:248–254`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/multipole_mod.f90#L248-L254)<br>[`ext/BeamTrackingBeamlinesExt/symplectic_bl.jl:31–44`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L31-L44)<br>[`src/kernels/multipole.jl:28–35`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/multipole.jl#L28-L35)<br>[`ext/BeamTrackingBeamlinesExt/unpack_bl.jl:514–523`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/unpack_bl.jl#L514-L523) |

Exporting a zero-length Bmad multipole with K0L_status=bends_reference to SciBmad silently turns its reference bend into an ordinary dipole kick. For integrated angle θ, the source map requires Δpₓ=θδ and Δz=−θx, where x is horizontal position and δ relative momentum deviation. With θ=1.000 mrad, x=1.000 mm and δ=0.01000, the imported element gives Δpₓ=−0.001000 and Δz=0 instead of +0.00001000 and −1.000 μm, and kicks the design ray. Native tracking and an independently integrated quadratic Hamiltonian agree. Both dipole signs, tilt and an added quadrupole reproduce the error; straight-reference import passes.

Run `python LAT-26.py` with [_scibmad.py](../helpers/_scibmad.py) in the same directory.

??? example "Reproducer: LAT-26.py"

    [Download LAT-26.py](LAT-26.py)

    ```python
    """LAT-26: Bmad SciBmad writer turns thin reference bends into ordinary dipole kicks

    Exporting a zero-length Bmad multipole with K0L_status=bends_reference to SciBmad silently turns its reference bend into an ordinary dipole kick. For integrated angle θ, the source map requires Δpₓ=θδ and Δz=−θx, where x is horizontal position and δ relative momentum deviation. With θ=1.000 mrad, x=1.000 mm and δ=0.01000, the imported element gives Δpₓ=−0.001000 and Δz=0 instead of +0.00001000 and −1.000 μm, and kicks the design ray. Native tracking and an independently integrated quadratic Hamiltonian agree. Both dipole signs, tilt and an added quadrupole reproduce the error; straight-reference import passes.

    Setup: `SCIBMAD_JULIA` set to a Julia launcher, an executable that starts Julia in the SciBmad environment; `BMAD_SOURCE` set to the root of a bmad-ecosystem `20261031-0` build, whose `production/bin` holds `bmad_to_scibmad` and `tao`; and the run pinned to one CPU (`taskset -c <n>`), which the embedded Julia program checks.

    The script exports inline Bmad decks, tracks the generated Julia on scalar and full SIMD batches, and compares six-vectors and Jacobians with a quadratic Hamiltonian matrix exponential. Native Bmad reference-bend tracking, ordinary steering and zero dipole strength provide controls.
    """

    import os
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from _scibmad import run_julia

    _PROGRAM = r'''
    using SciBmad, LinearAlgebra
    const BT = SciBmad.BeamTracking
    const ART = pwd()
    const BIN = joinpath(ENV["BMAD_SOURCE"],"production","bin")
    const CPU=strip(match(r"Cpus_allowed_list:\s*(.*)",read("/proc/self/status",String)).captures[1])
    @assert occursin(r"^\d+$",CPU) "Launch pinned to one CPU, for example taskset -c 5"
    @assert Threads.nthreads()==1 && BLAS.get_num_threads()==1
    const SP = Species("proton")
    const PC = 1e9
    const J = kron(Matrix{Float64}(I,3,3), [0.0 1.0;-1.0 0.0])
    const V = [0.001,0.002,-0.003,0.004,0.005,0.01]

    function oracle(theta, tilt, k1, mode)
        # Integrate the linear Hamilton equations as a matrix exponential. The
        # reference frame rotation contributes +theta*(1+delta) to p_b, whereas
        # the magnetic impulse is -theta, leaving +theta*delta.
        # Canonical completion is H=-theta*x_b*delta. Ordinary steering has H=theta*x_b.
        c,s = cos(tilt),sin(tilt)
        u = [c,0.0,s,0.0,0.0,0.0]
        w = [-s,0.0,c,0.0,0.0,0.0]
        hess = k1*(u*u'-w*w')
        linear=zeros(6)
        if mode=="bends_reference"
            hess[:,6] .-= theta*u
            hess[6,:] .-= theta*u
        else
            linear .+= theta*u
        end
        generator=zeros(7,7)
        generator[1:6,1:6] = J*hess
        generator[1:6,7] = J*linear
        exp(generator)
    end
    function imported(line,v;simd=false)
        width = simd ? Int(BT.pick_vector_width(Float64)) : 1
        simd && @assert width > 1
        b=Bunch(repeat(reshape(copy(v),1,6),width,1),species=SP,p_over_q_ref=PC/C_LIGHT)
        simd && @assert b.v isa BT.SIMD.FastContiguousArray
        track!(b,line;use_explicit_SIMD=simd)
        @assert all(b.state .== BT.STATE_ALIVE) && all(isfinite,b.v)
        firstrow=vec(copy(b.v[1:1,:]))
        @assert all(b.v[j,:] == firstrow for j in 1:width)
        firstrow
    end
    function native(deck,v,tag)
        start=join(["particle_start[$key]=$(v[i])" for (i,key) in enumerate(("x","px","y","py","z","pz"))],"\n")
        path=joinpath(ART,tag*".bmad")
        write(path,"no_digested\n"*start*"\n"*deck)
        command=joinpath(ART,"orbit.tao")
        write(command,"pipe ele:orbit 1\nquit\n")
        out=read(Cmd(`$(BIN*"/tao") -noplot -noinit -lat $path -startup $command`;dir=ART),String)
        write(joinpath(ART,tag*".log"),out)
        @assert occursin("state;STR;F;Alive",out) && !occursin(r"\[(ERROR|FATAL|ABORT|SEVERE)",out)
        [parse(Float64,match(Regex("(?m)^"*key*";REAL;F;\\s*([-+0-9.Ee]+)"),out).captures[1]) for key in ("x","px","y","py","z","pz")]
    end
    function main()
        defect = false
        println("ENV Julia=",VERSION," SciBmad=",pkgversion(SciBmad)," BeamTracking=",pkgversion(BT)," Beamlines=",pkgversion(SciBmad.Beamlines)," CPU=",CPU," numerical_threads=1")
        println("EXPLICIT_SIMD lanes=",Int(BT.pick_vector_width(Float64))," particles per comparison; scalar remainder absent")
        for (tag,theta,tilt,k1,mode) in (
            ("reference",.001,0.0,0.0,"bends_reference"),
            ("negative_tilt",-.001,.17,0.0,"bends_reference"),
            ("mixed",.001,.17,.3,"bends_reference"),
            ("straight",.001,.17,0.0,"straight_reference"),
            ("zero",0.0,.17,.3,"bends_reference"))
            deck="""
    parameter[particle] = proton
    parameter[p0c] = 1e9
    parameter[geometry] = open
    beginning[beta_a] = 10
    beginning[beta_b] = 10
    m: multipole, k0l=$theta, k0l_status=$mode, tilt=$tilt, k1l=$k1
    lat: line=(m)
    use, lat
    """
            input=joinpath(ART,tag*".bmad")
            output=joinpath(ART,tag*".jl")
            write(input,deck)
            text=read(Cmd(`$(BIN*"/bmad_to_scibmad") $input $output`;dir=ART),String)
            write(joinpath(ART,tag*"-export.log"),text)
            @assert !occursin(r"\[(WARN|ERROR|FATAL|ABORT|SEVERE)",text)
            source=read(output,String)
            println("EXPORT ",tag," ",join(filter(l->occursin("Multipole(",l),split(source,'\n'))," "))
            mod=Module();Core.eval(mod,:(using SciBmad))
            Base.include_string(mod,replace(source,r"^using .*$"m=>""))
            line=Core.eval(mod,:lat).branches[1].beamlines[1]
            transfer=oracle(theta,tilt,k1,mode)
            rays=[zeros(6),V,-V]
            if tag=="reference"
                append!(rays,[1e-4*Matrix{Float64}(I,6,6)[:,i] for i in 1:6])
            end
            err_native=0.0
            err_import=0.0
            for (i,v) in enumerate(rays)
                expected=(transfer*vcat(v,1.0))[1:6]
                got=imported(line,v)
                simd=imported(line,v;simd=true)
                @assert maximum(abs,simd-got) < 3e-16
                if mode=="bends_reference"
                    own=native(deck,v,tag*"-ray"*string(i))
                    err_native=max(err_native,maximum(abs,own-expected))
                    @assert maximum(abs,own-expected)<3e-16
                end
                err_import=max(err_import,maximum(abs,got-expected))
                if tag=="reference" && i<=3
                    println("RAY ",i," initial=",v," imported=",got," expected=",expected)
                end
            end
            h=1e-4
            jac=hcat([(imported(line,V+h*I(6)[:,i])-imported(line,V-h*I(6)[:,i]))/(2h) for i=1:6]...)
            err_jac=maximum(abs,jac-transfer[1:6,1:6])
            println("CASE ",tag," native_error=",err_native," import_error=",err_import,
                    " jacobian_error=",err_jac," oracle=quadratic_Hamiltonian_exponential"," R26=",jac[2,6]," expected_R26=",transfer[2,6],
                    " R51=",jac[5,1]," expected_R51=",transfer[5,1])
            if mode=="straight_reference" || theta==0
                @assert err_import<3e-16 && err_jac<2e-14
            else
                defect |= err_import>3e-16 || err_jac>2e-14
            end
        end
        thin=Beamline([SBend(L=0.0,g_ref=0.1,Kn0L=.001)],species_ref=SP,pc_ref=PC)
        try
            imported(thin,zeros(6))
            error("expected thin-reference tracking refusal")
        catch ex
            msg=sprint(showerror,ex)
            @assert occursin("Undefined for tracking method",msg)
            println("NATIVE_CAPABILITY_BOUNDARY ",msg)
        end
        println(defect ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
    end
    main()
    '''


    def main() -> None:
        root = os.environ.get("BMAD_SOURCE")
        if not root:
            raise RuntimeError("Set BMAD_SOURCE to the root of the Bmad build")
        source = Path(root).expanduser().resolve(strict=True)
        for name in ("bmad_to_scibmad", "tao"):
            binary = source / "production" / "bin" / name
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise RuntimeError(f"Executable required: {binary}")
        os.environ["BMAD_SOURCE"] = str(source)
        # Keep generated decks beside the reproducer, with automatic cleanup.
        with TemporaryDirectory(prefix="LAT-26-", dir=Path(__file__).resolve().parent) as work:
            run_julia(_PROGRAM, cwd=Path(work))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | bmad-ecosystem 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), production bmad_to_scibmad and Tao 20261031-0; SciBmad 5f0fb83ac889, BeamTracking d8e45fdc25db, Beamlines 35fe0162c72d; Julia 1.13.2, CPU, Float64, one thread |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | bmad-ecosystem 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), production bmad_to_scibmad and Tao 20261031-0; SciBmad 5f0fb83ac889, BeamTracking d8e45fdc25db, Beamlines 35fe0162c72d; Julia 1.13.2, CPU, Float64, one thread |

