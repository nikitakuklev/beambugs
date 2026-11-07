"""COL-17: Bend radiation omits the field wedges at inclined faces

With radiation enabled in a dipole with inclined hard-edge faces, tracking applies energy loss and quantum excitation only in the curved sector body and omits the field wedges between each physical face and the sector's end plane. In a 1.500 m, π/32 rad rectangular bend for 3.000 GeV/c electrons, rays displaced horizontally by ±10.00 mm lose momenta differing by the equivalent of 9.600 eV. Their field-filled trajectories are translations of each other along the parallel faces, and independent physical-face integration confirms equal losses. Along the nonradiating orbit, the excitation variance corresponds to field paths of 1.500 m ± 0.9825 mm instead of 1.500 m. Refining the integration, raising its order or subdividing the body preserves the discrepancy; perpendicular faces are unaffected.

Setup: `SCIBMAD_JULIA` set to a Julia launcher, an executable that starts Julia in the SciBmad environment; the Python script runs its embedded Julia program through it.

The script tracks fixed rays through a rectangular bend and a perpendicular-face control and compares the native mean-loss and excitation-variance path lengths, evaluated along the nonradiating orbit, with independent Cartesian circle/face intersections. It also tests translation symmetry of the public radiating loss, integration refinement and marker-separated partitions that keep only the physical ends. The excitation recorder sums the native Gaussian variances and returns zero random kicks, so no seed is needed.
"""

from _scibmad import run_julia

_PROGRAM = r'''
using SciBmad, LinearAlgebra
const BT=SciBmad.BeamTracking
const SP=Species("electron")
const PC=3e9
const MC=massof(SP)
const R=-PC/C_LIGHT
const MU=MC/PC
const E0=hypot(PC,MC)
const BETA0=PC/E0
const L=1.5
const TH=pi/32
const G=TH/L
const A=2/(3*4pi*BT.EPS_0)*E0^3/MC^4*G^2
const D=55/(24sqrt(3)*4pi*BT.EPS_0)*BT.H_BAR*C_LIGHT*E0^5/MC^7*abs(G)^3
const BAD=Ref(false)
function check(label,got,want,tol,derivation;control=false)
    err=maximum(abs,got.-want); ok=isfinite(err) && err<=tol
    println(label," observed=",got," expected=",want," max_error=",err," atol=",tol," derivation=",derivation)
    flush(stdout)
    control && !ok && error("control failed: $label")
    BAD[] |= !ok
end
coords(v)=Bunch(reshape(copy(v),1,6)).coords
function fringe(c,e,sign)
    BT.fringe!(1,c,nothing,MU,BETA0,G,nothing,nothing,e,e,Val(sign))
end
# Independent Cartesian geometry. At entrance the physical plane is Z=X*tan(e).
# Exit plane: n dot (r-r_ref)=0, n=e_s+tan(e)*e_x. Uniform vertical B gives
# alpha(ell)=alpha0-G*ell/P and a circle of radius P/G. Vacuum extrapolations
# connect the input/output observation planes to the physical magnet faces.
function oracle(v,e)
    P=1+v[6]; @assert v[3]==v[4]==0
    ps=sqrt(P^2-v[2]^2); ux=v[2]/P; uz=ps/P
    ellin=v[1]*tan(e)/(uz-ux*tan(e))
    xi=v[1]+ux*ellin; zi=uz*ellin
    alpha=atan(v[2],ps); rho=P/G
    rx=(cos(TH)-1)/G; rz=sin(TH)/G
    nx=-sin(TH)+tan(e)*cos(TH); nz=cos(TH)+tan(e)*sin(TH)
    ray(ell)=(xi+rho*(cos(alpha-G*ell/P)-cos(alpha)), zi+rho*(sin(alpha)-sin(alpha-G*ell/P)))
    ell=L
    for j=1:12
        x,z=ray(ell); a=alpha-G*ell/P
        ell-=(nx*(x-rx)+nz*(z-rz))/(nx*sin(a)+nz*cos(a))
    end
    x,z=ray(ell); a=alpha-G*ell/P
    xe=cos(TH)*(x-rx)+sin(TH)*(z-rz)
    se=-sin(TH)*(x-rx)+cos(TH)*(z-rz)
    outangle=a+TH; ellout=-se/cos(outangle)
    beta=P/hypot(P,MU)
    out=[xe+sin(outangle)*ellout,P*sin(outangle),0.0,0.0,
         v[5]+L*beta/BETA0-ellin-ell-ellout,v[6]]
    return ell,out
end
# Probe the native deterministic coefficient on exact nonradiating body orbits.
function frozen_mean(v,e,n)
    c0=coords(v); fringe(c0,e,1); start=vec(copy(c0.v))
    ds=L/n; weight=0.0
    for j=0:n
        c=coords(start)
        j>0 && BT.exact_bend!(1,c,G*(j*ds),G,G,MU,BETA0,nothing,j*ds)
        P=1+c.v[6]
        probe=vec(copy(c.v)); probe[2]/=P; probe[4]/=P; probe[6]=0.0
        h=(j==0 || j==n) ? ds/2 : ds
        r=coords(probe)
        BT.deterministic_radiation_multipole!(1,r,-1.0,MC,E0,G,SciBmad.SA[1],SciBmad.SA[G],SciBmad.SA[0.0],h)
        # Solve P_out=1/(1+A*dl) for the length using its small stored delta.
        weight+=-r.v[6]/(1+r.v[6])/A
    end
    return weight
end
# Record the actual Gaussian variance parameters, with no random trajectory kicks.
# This measures frozen diffusion coefficients, not a full propagated covariance.
mutable struct VarianceRecorder
    variance::Float64
    calls::Int
end
function BT.gaussian_random(rec::VarianceRecorder, sigma1, sigma2)
    rec.variance += sigma1^2
    rec.calls += 1
    return zero(sigma1),zero(sigma2)
end
function frozen_variance(v,e,n)
    rec=VarianceRecorder(0.0,0)
    c=coords(v)
    params=(nothing,MU,BETA0,nothing,G,nothing,nothing,G,SciBmad.SA[1],SciBmad.SA[G],SciBmad.SA[0.0])
    # Untilted radiation: a scalar tilt of 0 in BeamTracking d8e45fd; from 0bf3585 on, the
    # kernel takes the rotation pair (w, w_inv), and nothing means no rotation.
    mm,kn,ks=SciBmad.SA[1],SciBmad.SA[G],SciBmad.SA[0.0]
    photon=applicable(BT.stochastic_radiation!,1,c,0.0,rec,-1.0,MC,E0,G,0.0,mm,kn,ks,L) ?
        (rec,-1.0,MC,E0,G,0.0,mm,kn,ks) : (rec,-1.0,MC,E0,G,nothing,nothing,mm,kn,ks)
    edge=(nothing,MU,BETA0,G,nothing,nothing,e,e)
    BT.order_two_integrator!(1,c,BT.bkb_multipole!,params,photon,L/n,n,edge,Val(true),Val(true),Val(true),L)
    @assert rec.calls==n+1 && c.v[6]==0.0
    return rec.variance/D
end
function line(n,e;pieces=1,rad=false)
    @assert n%pieces==0
    eles=LineElement[]
    for i=1:pieces
        fr=pieces==1 ? Fringe.BothEnds : i==1 ? Fringe.EntranceEnd : i==pieces ? Fringe.ExitEnd : Fringe.NoEnd
        push!(eles,SBend(L=L/pieces,g_ref=G,Kn0=G,e1=e,e2=e,
              tracking_method=Symplectic(order=2,n_steps=n÷pieces,fringe_at=fr,radiation_damping_on=rad)))
        i<pieces && push!(eles,Marker())
    end
    Beamline(eles,species_ref=SP,pc_ref=PC)
end
function trk(bl,v)
    b=Bunch(reshape(copy(v),1,6),species=SP,p_over_q_ref=R)
    track!(b,bl;use_explicit_SIMD=false)
    @assert all(b.state .== BT.STATE_ALIVE) && all(isfinite,b.v)
    vec(b.v)
end
function main()
    BLAS.set_num_threads(1)
    @assert Threads.nthreads()==1 && BLAS.get_num_threads()==1
    println("ENV Julia=",VERSION," SciBmad=",pkgversion(SciBmad)," BeamTracking=",pkgversion(BT)," Beamlines=",pkgversion(SciBmad.Beamlines)," numerical_threads=1")
    println("COEFFICIENTS A=",A," D=",D," derivation=unit_charge_ultrarelativistic_uniform_field")
    rectangular_lengths=Float64[]
    for (label,e) in (("perpendicular",0.0),("rectangular",TH/2))
        geometry_error=0.0
        for x in (-0.01,0.0,0.01)
            v=[x,0.0,0.0,0.0,0.0,0.0]
            ell,out=oracle(v,e)
            geometry_error=max(geometry_error,maximum(abs,trk(line(1,e),v)-out))
            means=[frozen_mean(v,e,n) for n in (64,128,256)]
            variances=[frozen_variance(v,e,n) for n in (64,128,256)]
            mean=(4means[3]-means[2])/3
            variance=(4variances[3]-variances[2])/3
            refinement=abs(variance-(4variances[2]-variances[1])/3)
            @assert refinement<5e-13
            control=e==0.0 || x==0.0
            check("MEAN $label x=$x",mean,ell,5e-13,"Cartesian circle between physical faces; Richardson body coefficient";control=control)
            check("DIFFUSION $label x=$x",variance,ell,5e-13,"sum native sigma^2/D; Richardson physical-field-length comparison";control=control)
            e!=0.0 && push!(rectangular_lengths,mean)
        end
        @assert geometry_error<8e-15
        println("GEOMETRY_CONTROL ",label," max_coordinate_error=",geometry_error," expected=0 atol=8e-15 derivation=Cartesian_circle_and_face_intersections")
    end
    slope=(rectangular_lengths[3]-rectangular_lengths[1])/(0.02L)
    check("RECTANGULAR_X_SLOPE_per_m",slope,0.0,3e-10,"parallel physical faces give translation-invariant field length")
    for n in (128,256,512)
        e=TH/2
        rays=[[x,0.0,0.0,0.0,0.0,0.0] for x in (-0.01,0.01)]
        losses=[-trk(line(n,e;rad=true),v)[6] for v in rays]
        difference=(losses[2]-losses[1])*PC
        println("PUBLIC n=",n," loss_minus=",losses[1]," loss_plus=",losses[2]," difference_eV=",difference," expected_difference_eV=0 derivation=translation_symmetry_of_parallel_faces")
        BAD[] |= abs(difference)>1e-6
        if n==512
            for pieces in (2,4)
                residual=maximum(maximum(abs,trk(line(n,e;pieces=pieces,rad=true),v)-trk(line(n,e;rad=true),v)) for v in rays)
                @assert residual<5e-15
                println("PARTITION_CONTROL pieces=",pieces," max_coordinate_difference=",residual," expected=0 atol=5e-15 derivation=same_body_steps_and_only_outer_physical_faces")
            end
        end
    end
    println(BAD[] ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
end
main()
'''


if __name__ == "__main__":
    run_julia(_PROGRAM)
