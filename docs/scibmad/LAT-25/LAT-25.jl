# LAT-25: Simultaneous magnetic-strength convention changes rescale the partner twice
#
# A magnetic strength can be set as a field (B) or normalized to rigidity (K), integrated or per
# unit length. Reassigning a nonzero strength its current value through a name that changes both
# choices at once rescales the unassigned normal or skew partner by an extra L or 1/L, where L is
# the element length; this self-assignment must preserve the field, as two successive single-change
# self-assignments do. For a proton dipole with L = 0.25 m, rigidity R = 7.3 T·m, `Bn0` = −0.21 T
# and `Bs0` = 0.13 T, assigning `Kn0L` its own value keeps the kick Δpx = 7.192×10⁻³ but reduces Δpy
# from 4.452×10⁻³, the uniform-field Lorentz-force value, to 1.113×10⁻³. A zero partner or L = 1 m
# hides the error.
#
# The script sweeps 3,456 strength assignments over element lengths, signed rigidities, multipole
# orders, all 16 pairs of source and destination conventions, either assigned component and three
# assigned values, and checks the stored fields against the intended conversion with the extra
# factor in the 864 simultaneous-change cases only. It then tracks 16 protons through a 0.25 m
# dipole after a `Kn0L` or `Ks0L` self-assignment and compares the kicks with the uniform-field
# Lorentz kicks Δpx = −`Bn0`·L/R and Δpy = `Bs0`·L/R, which the same dipole built directly in that
# convention reproduces; further dipole and round-trip cases follow.
using SciBmad, Printf, LinearAlgebra
BLAS.set_num_threads(1)

key(order, normal, normalized, integrated) = Symbol(normalized ? "K" : "B", normal ? "n" : "s", order, integrated ? "L" : "")
function magnet(L, R, order, normalized, integrated; n=-0.21, s=0.13)
    e = Multipole(L=L, tracking_method=Exact(fringe_at=Fringe.NoEnd))
    f = (normalized ? inv(R) : 1.0) * (integrated ? L : 1.0)
    setproperty!(e, key(order, true, normalized, integrated), n*f)
    setproperty!(e, key(order, false, normalized, integrated), s*f)
    bl = Beamline([e], species_ref=Species(R < 0 ? "electron" : "proton"), p_over_q_ref=R)
    return bl, bl.line[1]
end

function main()
    simultaneous = 0; controls = 0; worst = 0.0
    for L in (0.25, 1.0, 2.75), R in (0.6, 7.3, -4.2), order in (0, 1, 3, 7),
        sn in (false, true), si in (false, true), tn in (false, true), ti in (false, true),
        normal in (false, true), alpha in (-1.5, 0.25, 1.0)
        bl, e = magnet(L, R, order, sn, si)
        target = key(order, normal, tn, ti)
        setproperty!(e, target, alpha*getproperty(e, target))
        # Normalization switches deliberately rescale both coefficients; a
        # pure integrated-status switch preserves the unassigned field.
        en = -0.21 * (normal || sn != tn ? alpha : 1.0)
        es = 0.13 * (!normal || sn != tn ? alpha : 1.0)
        # A normalization change sets the partner to old_other*value/old_val, and a simultaneous
        # integration change multiplies it by another L or 1/L: the extra factor modelled here.
        both = sn != tn && si != ti
        extra = both ? (ti ? L : inv(L)) : 1.0
        expected_n = en * (normal ? 1.0 : extra)
        expected_s = es * (normal ? extra : 1.0)
        actual_n = getproperty(e, key(order, true, false, false))
        actual_s = getproperty(e, key(order, false, false, false))
        residual = max(abs(actual_n-expected_n), abs(actual_s-expected_s))
        worst = max(worst, residual)
        @assert residual < 2e-15
        @assert e.BMultipoleParams.normalized[1] == tn
        @assert e.BMultipoleParams.integrated[1] == ti
        both ? (simultaneous += 1) : (controls += 1)
    end
    @printf("coefficient sweep simultaneous=%d controls=%d max_model_residual=%.17g\n", simultaneous, controls, worst)

    # Repeated self-assignments must be an identity representation round trip.
    bl, e = magnet(0.25, 7.3, 1, false, false)
    for k in (:Kn1L, :Bn1, :Kn1L, :Bn1)
        setproperty!(e, k, getproperty(e, k))
        @printf("roundtrip key=%s Bn1=%.17g Bs1=%.17g\n", k, e.Bn1, e.Bs1)
    end
    @assert abs(e.Bs1-0.13) < 1e-15
    # Inverse length factors cancel over a complete round trip, while each
    # intermediate representation has the wrong field. Inspect intermediates.

    # Reproduce the docstring's own sequence and inspect its stated ratio.
    doc = LineElement(Kn1=0.1, Ks1=0.2, L=0.4)
    doc.Kn1L = 0.3
    oldratio = doc.Ks1L/doc.Kn1L
    doc.Bn1 = 0.5
    @printf("doc example prior_ratio=%.17g final_ratio=%.17g Bs1=%.17g ratio_preserving_Bs1=%.17g printed_doc_Bs1=%.17g\n", oldratio, doc.Bs1/doc.Bn1, doc.Bs1, 0.5oldratio, 0.2*0.5/0.3)

    # A separate uniform-field Lorentz-force oracle tests physical impact.
    L = 0.25; R = 7.3
    v = zeros(16,6)
    for i in 1:16
        v[i,:] = [0.002sin(i), 0.003cos(i), 0.004cos(i), 0.005sin(i), 0.006sin(2i), 0.02cos(2i)]
    end
    for simd in (false, true), normal in (false, true)
        direct, e = magnet(L, R, 0, false, false)
        control, ec = magnet(L, R, 0, true, true)
        target = key(0, normal, true, true)
        setproperty!(e, target, getproperty(e, target))
        b = Bunch(v=copy(v), species=Species("proton"), p_over_q_ref=R)
        c = Bunch(v=copy(v), species=Species("proton"), p_over_q_ref=R)
        track!(b, direct; use_explicit_SIMD=simd)
        track!(c, control; use_explicit_SIMD=simd)
        @assert all(b.state .== BeamTracking.STATE_ALIVE)
        @assert all(c.state .== BeamTracking.STATE_ALIVE)
        expected = v[:,[2,4]] .+ permutedims([0.21L/R, 0.13L/R])
        control_error = maximum(abs, c.v[:,[2,4]] - expected)
        error = maximum(abs, b.v[:,[2,4]] - expected)
        @assert control_error < 2e-17
        @assert error > 0.003
        @printf("orbit simd=%s assigned_normal=%s dpx=%.17g dpy=%.17g expected_dpx=%.17g expected_dpy=%.17g max_error=%.17g oracle_control_error=%.17g\n", simd, normal, b.v[1,2]-v[1,2], b.v[1,4]-v[1,4], 0.21L/R, 0.13L/R, error, control_error)
    end
    staged_controls()
    println("DEFECT REPRODUCED")
end
const SP=Species("proton")
const R=5.0
const V=hcat([0.003sin(i) for i in 1:12], [0.004cos(i) for i in 1:12],
             [0.003cos(2i) for i in 1:12], [0.004sin(2i) for i in 1:12],
             [0.01sin(3i) for i in 1:12], [0.05cos(3i) for i in 1:12])
function line(L,normalized,integrated)
    factor=(normalized ? 1/R : 1.0)*(integrated ? L : 1.0)
    nkey=Symbol(normalized ? "Kn0" : "Bn0",integrated ? "L" : "")
    skey=Symbol(normalized ? "Ks0" : "Bs0",integrated ? "L" : "")
    e=Multipole(L=L,tracking_method=Exact(fringe_at=Fringe.NoEnd))
    setproperty!(e,nkey,0.3factor)
    setproperty!(e,skey,0.4factor)
    Beamline([e],species_ref=SP,p_over_q_ref=R)
end
function tracked(bl,simd)
    b=Bunch(v=copy(V),species=SP,p_over_q_ref=R)
    track!(b,bl;use_explicit_SIMD=simd)
    @assert all(b.state .== BeamTracking.STATE_ALIVE)
    b.v
end
function staged_controls()
for simd in (false,true), L in (0.4,1.0,2.0), normalized in (false,true), integrated in (false,true), normal in (true,false)
    direct=line(L,normalized,integrated)
    staged=line(L,normalized,integrated)
    convertall=line(L,normalized,integrated)
    e=direct.line[1]
    key=Symbol(normal ? (!normalized ? "Kn0" : "Bn0") : (!normalized ? "Ks0" : "Bs0"), !integrated ? "L" : "")
    value=getproperty(e,key)
    setproperty!(e,key,value)
    es=staged.line[1]
    midkey=Symbol(normal ? (!normalized ? "Kn0" : "Bn0") : (!normalized ? "Ks0" : "Bs0"), integrated ? "L" : "")
    setproperty!(es,midkey,getproperty(es,midkey))
    setproperty!(es,key,getproperty(es,key))
    convertall.line[1].BM_independent=[(order=1,normalized=!normalized,integrated=!integrated)]
    expected=V[:,[2,4]].+permutedims([-0.3L/R,0.4L/R])
    actual=tracked(direct,simd)
    staged_v=tracked(staged,simd)
    all_v=tracked(convertall,simd)
    expected_other=normal ? 0.4L/R : 0.3L/R
    actual_other=normal ? e.Ks0L : e.Kn0L
    ratio=integrated ? 1/L : L
    @assert abs(actual_other/expected_other-ratio)<=8eps(ratio)
    @assert maximum(abs,staged_v[:,[2,4]]-expected)<2e-16
    @assert maximum(abs,all_v[:,[2,4]]-expected)<2e-16
    error=maximum(abs,actual[:,[2,4]]-expected)
    @assert L==1.0 ? error<2e-16 : error>0.01
    @printf("simd=%s L=%.1f source_normalized=%s source_integrated=%s setter=%s value=%.17g partner_expected=%.17g partner_actual=%.17g ratio=%.17g max_kick_error=%.17g staged_error=%.3g BM_independent_error=%.3g\n",simd,L,normalized,integrated,key,value,expected_other,actual_other,actual_other/expected_other,error,maximum(abs,staged_v[:,[2,4]]-expected),maximum(abs,all_v[:,[2,4]]-expected))
    if !simd && L==0.4 && !normalized && !integrated && normal
        for i in 1:12
            @printf("particle=%d actual_dpx=%.17g expected_dpx=%.17g actual_dpy=%.17g expected_dpy=%.17g\n",i,actual[i,2]-V[i,2],-0.3L/R,actual[i,4]-V[i,4],0.4L/R)
        end
    end
end
end

main()
