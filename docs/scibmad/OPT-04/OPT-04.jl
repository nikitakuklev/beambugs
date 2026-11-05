# OPT-04: Linear De Moivre `B`/`E` use columns of A⁻¹ instead of rows
#
# In its linear branch (expansion order 1), `twiss` builds the De Moivre matrices `B` and `E`, which
# split the one-turn matrix and the beam moments by mode, from columns of A⁻¹, the inverse
# normalizing matrix, instead of its rows. In an uncoupled plane the documented definition gives E₁
# = [β −α; −α γ], and the corresponding B₁ matches a tracked one-turn Jacobian to 1.8e-15. For a
# thin-quadrupole cell with β = 3.441 m and α = −1.376, the returned E₁[1,2] and E₁[2,1] are 0 and
# 1.776 instead of 1.376, and E₁[2,2] is 0.2906 m⁻¹ instead of 0.8412 m⁻¹. The horizontal beam-size
# term used by `dynamic_aperture` stays correct in uncoupled lattices; by source algebra, its
# vertical mode-1 term and coupled lattices are wrong.
#
# The script runs order-1 `twiss` on a periodic cell of thin quadrupoles and 1 m drifts (1 GeV/c
# electrons, RF off) and compares the returned B₁ and E₁ with A J₁ A⁻¹ and A P₁ Aᵀ built from the
# returned normalizing matrix A, and with their Twiss forms [α β; −γ −α] and [β −α; −α γ]. As an
# independent check it forms (M − cos μ I)/sin μ from the ForwardDiff one-turn Jacobian M of
# `track`. It also calls NNF's `compute_de_moivre`, which implements the correct formula, on the
# coasting normalizing map and prints the error it raises.
using SciBmad, LinearAlgebra
const FD = SciBmad.ForwardDiff
qf = Multipole(Kn1L = 0.8); qd = Multipole(Kn1L = -0.7); d = Drift(L = 1.0)
bl = Beamline([qf, d, qd, d], species_ref = Species("electron"), pc_ref = 1e9)

tw = twiss(bl, rf_on = false, at = [1], cols = ["beta1", "alpha1", "a1", "B1", "E1"])
β, α = tw.beta1[1], tw.alpha1[1]; γ = (1 + α^2) / β
A = tw.a1[1][1:4, 1:4]
B = tw.B1[1]; E = tw.E1[1]
# De Moivre definitions: with A the linear normalizing matrix (Floquet to lab), B_k = A J_k A⁻¹ and
# E_k = −B_k S = A P_k Aᵀ, so the one-turn matrix is M = Σ_k (cos μ_k H_k + sin μ_k B_k) and E_k is
# the symmetric mode-k moment matrix.
P1 = Diagonal([1.0, 1, 0, 0]); J = kron(I(2), [0.0 1; -1 0])
Bx = A * J * P1 * inv(A); Ex = A * P1 * A'
println("returned B1[1:2,1:2] = ", B[1:2, 1:2])
println("A J1 A^-1[1:2,1:2]   = ", Bx[1:2, 1:2], "   Twiss form [α β; -γ -α] = ", [α β; -γ -α])
println("returned E1[1:2,1:2] = ", E[1:2, 1:2])
println("A P1 A^T[1:2,1:2]    = ", Ex[1:2, 1:2], "   Twiss form [β -α; -α γ] = ", [β -α; -α γ])
println("max|B1 - A J1 A^-1| = ", maximum(abs, B - Bx), "   max|E1 - A P1 A^T| = ", maximum(abs, E - Ex),
        "   E1 asymmetry = ", maximum(abs, E - E'))

M = FD.jacobian(v -> vec(track(bl, v0 = reshape(v, 1, 6)).v[1, :, end]), zeros(6))[1:2, 1:2]
μ = acos(tr(M) / 2); μ *= sign(M[1, 2])
Bt = (M - cos(μ) * I) / sin(μ)
println("independent: (M - cos μ I)/sin μ = ", Bt, "   max diff to A J1 A^-1: ", maximum(abs, Bt - Bx[1:2, 1:2]))
# Related fix-path caveat: NNF's (unexported) compute_de_moivre, which implements the right
# formula, fails for a coasting map in the default Vector storage (6x6 VARS_CPARAM Jacobian
# times 4x4 jp_mat, NNF src/de_moivre.jl:48-52); its StaticArray method uses HVARS.
const NNF = NonlinearNormalForm
a1map = twiss(bl, rf_on = false, at = [1], cols = ["a1"], as_taylor_series = true).a1[1]
cd = try NNF.compute_de_moivre(a1map); "ok" catch err; first(split(sprint(showerror, err), '\n')) end
println("NNF.compute_de_moivre(coasting a1, Vector storage) -> ", cd)
println(maximum(abs, E - Ex) > 1e-9 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
