# OPT-31: Degenerate damped modes corrupt eigenvectors and equilibrium covariance
#
# For damped maps with nonsymplectic coupling and repeated eigenvalues, NonlinearNormalForm mixes a
# conjugate eigenvector into the degenerate eigenspace. `mat_eigen` then returns vectors that fail
# the eigenvector equation, and `equilibrium_moments` violates Σ = MΣMᵀ + Q, with Σ the equilibrium
# covariance and Q the noise. A direct Kronecker solve, a convergent covariance sum and a supplied
# analytic normalizer independently establish the expected result. For damping factor 0.87 in each
# mode, angles (0.61, 0.61, 1.13) rad and Q = 2×10⁻⁸·I, the covariance has 22.56% relative error,
# its fixed-point residual is 0.8035‖Q‖, and the eigenvector residual is 0.3538. Separating the
# repeated angles by 10⁻⁵ rad restores accuracy; realistic-ring prevalence was not established.
#
# The script builds M = T·diag(0.87 R(θᵢ))·T⁻¹ with T the identity except T₁₃ = 0.375 and T₄₂ =
# 0.125, a stable diagonalizable map. It checks `mat_eigen` against M·V = V·Λ and compares
# `equilibrium_moments` with the covariance sum over 256 turns, the direct Kronecker solve and the
# result for the known normalizer T, then repeats with the repeated angles separated by 10⁻⁵.
# Finally it applies the degenerate-pair orthogonalization `orthogonalize_degenerate_pairs!` alone
# to exact eigenvectors built from T and (1, i)/√2.
using SciBmad, LinearAlgebra, Printf
const NNF = SciBmad.NonlinearNormalForm
BLAS.set_num_threads(1)
rotation(t) = [cos(t) sin(t); -sin(t) cos(t)]
function main()
    vars6 = collect(vars(Descriptor(6,1)))
    T = Matrix{Float64}(I,6,6)
    T[1,3] = .375
    T[4,2] = .125
    Q = Matrix{Float64}(I,6,6)*2e-8
    for gap in (0., 1e-5)
        R = zeros(6,6)
        for (i,theta) in enumerate((.61,.61+gap,1.13))
            R[2i-1:2i,2i-1:2i] = .87*rotation(theta)
        end
        M = T*R/T
        # Independent reference is the convergent covariance sum, not an eigensystem.
        sigma_ref = copy(Q)
        term = copy(Q)
        for n in 1:256
            term = M*term*M'
            sigma_ref += term
        end
        direct = reshape((I-kron(M,M))\vec(Q),6,6)
        reference_residual = norm(sigma_ref-M*sigma_ref*M'-Q)/norm(Q)
        @assert reference_residual < 8e-15
        @assert norm(direct-sigma_ref)/norm(sigma_ref) < 5e-15
        eig = NNF.mat_eigen(M)
        eig_reference = eigen(M)
        e0 = norm(M*eig_reference.vectors-eig_reference.vectors*Diagonal(eig_reference.values))
        e1 = norm(M*eig.vectors-eig.vectors*Diagonal(eig.values))
        map = DAMap(nv=6,v=vars6,v_matrix=M,s=Q)
        computed = equilibrium_moments(map)
        known_a = DAMap(nv=6,v=vars6,v_matrix=T,s=zeros(6,6))
        known_sigma = equilibrium_moments(map,known_a)
        @printf("gap=%g radius=%.16g diagonalizable_condition=%.6g raw_eigen_residual=%.3g nnf_eigen_residual=%.16g\n",gap,maximum(abs.(eig.values)),cond(T),e0,e1)
        @printf("covariance_relative_error=%.16g Lyapunov_residual_over_Q=%.16g series_residual_over_Q=%.3g direct_vs_series=%.3g known_A_relative_error=%.3g\n",norm(computed-sigma_ref)/norm(sigma_ref),norm(computed-M*computed*M'-Q)/norm(Q),reference_residual,norm(direct-sigma_ref)/norm(sigma_ref),norm(known_sigma-sigma_ref)/norm(sigma_ref))
        @assert norm(known_sigma-sigma_ref)/norm(sigma_ref) < 8e-15
        if gap == 0
            @assert e1 > .01
            @assert norm(computed-M*computed*M'-Q)/norm(Q) > .1
        else
            @assert e1 < 8e-14
            @assert norm(computed-sigma_ref)/norm(sigma_ref) < 8e-14
        end
    end
    # Isolate the orthogonalizer using explicitly known, exact eigenvectors.
    # For R(theta), (1,i)/sqrt(2) is the +theta eigenvector.
    R = zeros(6,6)
    V = zeros(ComplexF64,6,6)
    values = zeros(ComplexF64,6)
    for (i,theta) in enumerate((.61,.61,1.13))
        R[2i-1:2i,2i-1:2i] = .87*rotation(theta)
        V[2i-1,2i-1] = 1/sqrt(2)
        V[2i,2i-1] = im/sqrt(2)
        V[:,2i] = conj.(V[:,2i-1])
        values[2i-1] = .87*exp(im*theta)
        values[2i] = conj(values[2i-1])
    end
    M = T*R/T
    # Exact eigenvectors of M = T R T⁻¹ are T times those of R; the orthogonalizer must keep them
    # eigenvectors.
    V = T*V
    before = norm(M*V-V*Diagonal(values))
    NNF.orthogonalize_degenerate_pairs!(V,values)
    after = norm(M*V-V*Diagonal(values))
    @printf("isolated_exact_eigenvectors_before=%.3g after=%.16g\n",before,after)
    @assert before < 2e-15 && after > .01
    println("DEFECT REPRODUCED")
end
main()
