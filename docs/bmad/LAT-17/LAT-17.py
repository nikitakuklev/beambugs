"""LAT-17: The manual calls a momentum spread an energy spread

Initializing an ordinary Gaussian bunch with sig_pz sets σδ, the RMS fractional momentum spread, although the manual calls it fractional energy spread. At finite speed these differ: σE/E₀ = β₀² σδ to first order about the reference particle. Here E₀ is reference energy, σE is RMS energy spread, β₀ is reference speed divided by light speed, and δ is momentum deviation divided by reference momentum. For an electron reference momentum of 1 MeV/c and sig_pz = 0.002, the generated σδ is 0.002, while the relative energy spread is smaller. The longitudinal covariance and mass-shell relation establish the momentum convention.

The script initializes the deterministic Gaussian case with seed 314159 and recomputes charge-weighted moments from emitted particles. It compares the longitudinal covariance with the requested momentum covariance and reconstructs energies from the mass shell. BMAD_SOURCE supplies the conflicting manual sentence.
"""

import io

import numpy as np
from _bmad import compare, fortran, verdict
from _s3_12_particle_contracts import INIT, document_claim, rows

result = fortran(INIT, ["pseudo"], outputs=["particles.dat"])
data = np.loadtxt(io.BytesIO(result.outputs["particles.dat"]))
if data.shape != (2048, 9) or not np.isfinite(data).all() or not (data[:, 6] > 0).all():
    raise RuntimeError("Expected 2048 finite particles with positive charge weights")
mass = rows(result.log)["init"][4]
w = data[:, 6].astype(np.longdouble)
w /= w.sum()
center = w @ data[:, :6]
d = data[:, :6] - center
cov = (d.T * w) @ d
# A momentum chirp d(delta)/dz gives Cov(z,delta)=chirp*Var(z).
sig_z, sig_delta, chirp = 0.004, 0.002, 0.2
expected = np.array([[sig_z**2, chirp * sig_z**2], [chirp * sig_z**2, sig_delta**2]])
if compare(
    "covariance (z,delta), momentum/chirp expectation",
    cov[4:, 4:],
    expected,
    atol=2e-18,
    rtol=2e-12,
):
    raise RuntimeError("Momentum covariance control failed")
pc = 1e6 * (1 + data[:, 5])
energy = np.hypot(pc, mass)
np.testing.assert_allclose(data[:, 7], pc / energy, rtol=3e-15)
# E^2=(pc)^2+(mc^2)^2 gives dE/E=beta^2 dp/p at the reference particle.
e_rel = energy / np.hypot(1e6, mass)
sigma_e = np.sqrt(w @ (e_rel - w @ e_rel) ** 2)
beta0_sq = 1e12 / (1e12 + mass**2)
print(
    f"sigma_delta={float(np.sqrt(cov[5, 5])):.16g}; sigma_E/E0={float(sigma_e):.16g}; beta0^2={beta0_sq:.16g}; first-order at delta=0={beta0_sq * 0.002:.16g}"
)
verdict(
    document_claim("bmad/doc/beam-init.tex", r"\vn{%sig_pz} is the fractional energy spread dE/E.")
)
