# OPT-05: Map-valued `E1..E3` throw `UndefVarError: nj_mat`
#
# Requesting the De Moivre matrix `E1` from coasting `twiss` above order 1, such as
# `twiss(chrom=2)`, throws `UndefVarError: nj_mat` instead of returning the documented matrix,
# because its map-valued branch reads undefined variables. For a thin-quadrupole test cell, both
# −B₁S from the working `B1` output (S the symplectic form) and the Twiss form [β −α; −α γ] give E₁
# = [3.441 1.376; 1.376 0.8412], with β in m and γ in m⁻¹. By source inspection, `E1` and `E2` fail
# in higher-order coasting runs, and `E1`–`E3` in higher-order RF runs with knob parameters. Order-1
# optics and RF optics without knobs take a separate linear branch; `E3` is explicitly unavailable
# for coasting beams.
#
# The script runs `twiss(chrom=2)` without RF on a periodic cell of thin quadrupoles and 1 m drifts
# (1 GeV/c electrons), reads `B1` from the map-valued branch together with `beta1` and `alpha1`, and
# builds the expected E₁ both as −B₁S and from the Twiss form [β −α; −α γ]. It then requests `E1` in
# the same configuration and reports the error it throws.
using SciBmad, LinearAlgebra
qf = Multipole(Kn1L = 0.8); qd = Multipole(Kn1L = -0.7); d = Drift(L = 1.0)
bl = Beamline([qf, d, qd, d], species_ref = Species("electron"), pc_ref = 1e9)

tb = twiss(bl, rf_on = false, chrom = 2, at = [1], cols = ["beta1", "alpha1", "B1"])
# Expected E₁ = B₁ ∘ (−S), at δ = 0 the matrix −B₁S, with B₁ from the map-valued B branch.
B = tb.B1[1]; S = kron(I(2), [0.0 1; -1 0])
β, α = tb.beta1[1], tb.alpha1[1]
println("chrom=2 B1 (map branch) [1:2,1:2] = ", B[1:2, 1:2], "  -> expected E1 = -B1*S: ", (-B*S)[1:2, 1:2])
println("independent Twiss form [β -α; -α γ] = ", [β -α; -α (1 + α^2)/β])
msg = try
    twiss(bl, rf_on = false, chrom = 2, at = [1], cols = ["E1"])
    "no error"
catch err
    sprint(showerror, err)
end
println("twiss(chrom=2, cols=[\"E1\"]) -> ", first(split(msg, '\n')))
println(occursin("nj_mat", msg) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
