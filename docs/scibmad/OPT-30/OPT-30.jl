# OPT-30: Automatic damping detection gives nonperiodic periodic Twiss functions
#
# With `damping` and the initial normalizer `a_initial` omitted, `twiss` decides whether the optics
# are damped by checking the normalizer's symplecticity instead of the transport map's, and misses
# damping in a supported user transport map. For M = A·diag(ρᵢRᵢ)·A⁻¹ with symplectic Courant–Snyder
# A, rotations Rᵢ and amplitude factors ρᵢ, the matched beta functions must be periodic. With β =
# (2, 3, 4) m and ρ = (0.93, 0.87, 0.81), automatic selection instead changes β₁ from 2.000 to 1.730
# m and β₂ from 3.000 to 2.271 m and omits damping summaries. Explicit `damping=true` restores
# periodicity and the analytic decrements −log ρᵢ. The undamped control passes; realistic
# radiation-lattice impact was not measured.
#
# The script wraps a six-dimensional linear map with β = (2, 3, 4) m, α = (0.2, −0.1, 0.3), ρ =
# (0.93, 0.87, 0.81) and rotation angles (0.37, 0.59, 0.23) in a user transport element, runs
# `twiss` with `damping` omitted and with `damping=true`, and compares the entrance and exit beta
# functions and the damping decrements with the analytic values. The same map without damping is a
# control, and the printed symplectic defects of M and A separate the two matrices that the
# detection confuses.
using SciBmad, LinearAlgebra, Printf
BLAS.set_num_threads(1)
const NNF = SciBmad.NonlinearNormalForm
rotation(t) = [cos(t) sin(t); -sin(t) cos(t)]
function linear_transport(v, q, matrix)
    (ntuple(i -> sum(matrix[i,j]*v[j] for j in 1:6), 6), q)
end
function main()
    betas = [2., 3., 4.]
    alphas = [.2, -.1, .3]
    radii = [.93, .87, .81]
    A = zeros(6,6)
    R = zeros(6,6)
    for i in 1:3
        rows = 2i-1:2i
        A[rows,rows] = [sqrt(betas[i]) 0; -alphas[i]/sqrt(betas[i]) 1/sqrt(betas[i])]
        R[rows,rows] = radii[i]*rotation([.37,.59,.23][i])
    end
    # M = A diag(ρᵢ R(θᵢ)) A⁻¹ with symplectic A: the matched beta functions are periodic and the
    # amplitude decrements are −log ρᵢ, although M itself is not symplectic. Automatic detection
    # instead gives β_exit = ρᵢ² β_entrance.
    M = A*R/A
    @printf("map_symplectic_defect=%.16g known_A_symplectic_defect=%.3g\n", norm(NNF.checksymp(M)), norm(NNF.checksymp(A)))
    for (tag, matrix) in (("damped", M), ("undamped", A*(R/Diagonal(repeat(radii,inner=2)))/A))
        ring = Beamline([LineElement(L=1., transport_map=linear_transport, transport_map_params=matrix)], species_ref=Species("electron"), pc_ref=1e9)
        for damping in (nothing, true)
            tw = twiss(ring; damping, cols=["beta1","beta2","alpha1","alpha2"])
            @printf("%s damping=%s beta1=%.16g -> %.16g beta2=%.16g -> %.16g damp1_present=%s\n", tag, string(damping), tw.beta1[1], tw.beta1[end], tw.beta2[1], tw.beta2[end], string(:damp1 in propertynames(tw)))
            @assert maximum(abs.([tw.beta1[1],tw.beta2[1]]-betas[1:2])) < 2e-13
            if tag == "damped" && isnothing(damping)
                @assert maximum(abs.([tw.beta1[end],tw.beta2[end]]-betas[1:2].*radii[1:2].^2)) < 2e-13
                @assert !(:damp1 in propertynames(tw))
            else
                @assert maximum(abs.([tw.beta1[end],tw.beta2[end]]-betas[1:2])) < 2e-13
            end
            if damping === true && tag == "damped"
                actual = [tw.damp1,tw.damp2,tw.damp3]
                @printf("decrements=%s expected=%s max_error=%.3g\n", string(actual), string(-log.(radii)), maximum(abs.(actual+log.(radii))))
                @assert maximum(abs.(actual+log.(radii))) < 2e-14
            end
        end
    end
    println("DEFECT REPRODUCED")
end
main()
