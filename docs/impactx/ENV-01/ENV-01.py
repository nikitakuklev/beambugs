"""ENV-01: eigenemittance diagnostics corrupt small or repeated modes."""

import os

for variable in (
    "NPROC",
    "PJRT_NPROC",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
from impactx import Envelope, ImpactX, Map6x6

sim = ImpactX()
sim.particle_shape = 1
sim.space_charge = False
sim.diagnostics = False
sim.eigenemittances = True
sim.init_grids()
ref = sim.beam.ref
ref.set_charge_qe(1).set_mass_MeV(1000).set_kin_energy_MeV(1000)
bg = ref.beta_gamma
print(f"gamma={ref.gamma:.17g} beta_gamma={bg:.17g}")
j = np.kron(np.eye(3), [[0.0, 1.0], [-1.0, 0.0]])


def compare(label, moments, expected):
    projected = np.array([moments[f"emittance_{axis}n"] for axis in "xyt"])
    eigen = np.sort([moments[f"emittance_{i}"] for i in (1, 2, 3)])
    np.testing.assert_allclose(np.sort(projected), expected, rtol=2e-14, atol=0)
    good = np.all(np.isfinite(eigen)) and np.max(abs(eigen / expected - 1)) < 1e-11
    print(
        f"{label}: expected={expected.tolist()} projected={projected.tolist()} eigen={eigen.tolist()} valid={good}"
    )
    return good


try:
    for emittances in ([1e-6, 2e-6, 3e-6], [1e-8, 2e-8, 1e-3], [1e-6, 1e-6, 1e-3]):
        expected = np.array(emittances)
        # Covariances in (x,px,y,py,t,pt); p's are normalized to reference P0.
        diagonal = np.repeat(expected, 2)
        diagonal[1::2] /= bg**2
        matrix = Map6x6()
        raw = matrix.to_numpy()
        raw[:] = np.diag(diagonal)
        normalized = raw.copy()
        normalized[1::2, :] *= bg
        normalized[:, 1::2] *= bg
        assert np.linalg.eigvalsh(normalized).min() > 0
        # For SPD Sigma, i*sqrt(Sigma)*J*sqrt(Sigma) is Hermitian and has
        # eigenvalues +/- epsilon.  This avoids a characteristic polynomial.
        square_root = np.diag(np.sqrt(np.diag(normalized)))
        oracle = np.linalg.eigvalsh(1j * square_root @ j @ square_root)[3:]
        np.testing.assert_allclose(oracle, expected, rtol=5e-15, atol=0)
        print(
            f"SPD condition_number={np.linalg.cond(normalized):.8g} hermitian_oracle={oracle.tolist()}"
        )
        env_good = compare("envelope", Envelope(matrix, 0.0).beam_moments(ref), expected)

        # Twelve equal-weight particles with exactly diagonal sample covariance:
        # each coordinate has two entries +/-sqrt(6*variance), all others zero.
        points = np.concatenate((np.diag(np.sqrt(6 * diagonal)), -np.diag(np.sqrt(6 * diagonal))))
        sim.beam.clear_particles()
        sim.beam.add_n_particles(
            x=points[:, 0].copy(),
            px=points[:, 1].copy(),
            y=points[:, 2].copy(),
            py=points[:, 3].copy(),
            t=points[:, 4].copy(),
            pt=points[:, 5].copy(),
            qm=ref.qm_ratio_SI,
            bunch_charge=1e-15,
        )
        particles_good = compare("particles", sim.beam.beam_moments(), expected)
        if emittances[-1] == 3e-6:
            assert env_good and particles_good
        else:
            assert not env_good and not particles_good
    print("DEFECT REPRODUCED")
finally:
    sim.finalize()
