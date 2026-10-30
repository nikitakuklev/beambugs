"""IO-01: A tracer erases weighted beam size and emittance.

Weighted two-pass oracle and permutation/scale controls.
"""

import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NPROC",
    "PJRT_NPROC",
):
    os.environ[name] = "1"

import numpy as np
from impactx import ImpactX

COORDS = (
    "position_x",
    "momentum_x",
    "position_y",
    "momentum_y",
    "position_t",
    "momentum_t",
)


class Oracle:
    def __init__(self):
        self.sim = ImpactX()
        self.sim.particle_shape = 1
        self.sim.space_charge = False
        self.sim.diagnostics = False
        self.sim.slice_step_diagnostics = False
        self.sim.verbose = 0
        self.sim.init_grids()
        self.ref = self.sim.beam.ref

    def load(self, u, weights=None, species="proton", gamma=2.0, anomaly=None, spin=None):
        self.sim.beam.clear_particles()
        self.ref.reset()
        self.ref.set_species(species)
        self.ref.set_kin_energy_MeV((gamma - 1) * self.ref.mass_MeV)
        if anomaly is not None:
            self.ref.set_gyromagnetic_anomaly(anomaly)
        self.sim.spin = spin is not None
        u = np.atleast_2d(np.asarray(u, dtype=float))
        kwargs = dict(zip(("x", "px", "y", "py", "t", "pt"), u.T.copy()))
        kwargs["qm"] = self.ref.qm_ratio_SI
        kwargs["w"] = np.ones(len(u)) if weights is None else np.asarray(weights, dtype=float)
        if spin is not None:
            kwargs.update({k: np.full(len(u), v) for k, v in zip(("sx", "sy", "sz"), spin)})
        self.sim.beam.add_n_particles(**kwargs)

    def push(self, element):
        self.sim.lattice.clear()
        self.sim.lattice.append(element)
        self.sim.track_particles()
        return self.sim.beam.to_df(local=True)

    def close(self):
        self.sim.finalize()


o = Oracle()
try:
    for sigma in (1e-5, 1e-7, 1e-9):
        for tracer_weight in (0.0, 1e-24):
            u = np.zeros((5, 6))
            u[0, 0] = 1
            u[1:, 0] = sigma * np.array([-1, -1, 1, 1])
            u[1:, 1] = 1e-3 * np.array([-1, 1, -1, 1])
            weights = np.array([tracer_weight, 1, 1, 1, 1])
            # Center on the weighted mean directly. No subtraction of O(1) second moments.
            center = np.sum(weights[:, None] * u, axis=0) / np.sum(weights)
            deviations = u - center
            cov = (deviations.T * weights) @ deviations / weights.sum()
            expected_sigma = np.sqrt(cov[0, 0])
            expected_emit = np.sqrt(cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2)
            for label, order in (
                ("tracer_first", [0, 1, 2, 3, 4]),
                ("tracer_last", [1, 2, 3, 4, 0]),
            ):
                o.load(u[order], weights[order])
                m = o.sim.beam.beam_moments()
                print(
                    f"sigma={sigma:g} tracer_weight={tracer_weight:g} {label}: "
                    f"sigma_x={m['sigma_x']:.15g} expected={expected_sigma:.15g} "
                    f"emit_x={m['emittance_x']:.15g} expected={expected_emit:.15g}",
                    flush=True,
                )
                if label == "tracer_last":
                    np.testing.assert_allclose(m["sigma_x"], expected_sigma, rtol=2e-14, atol=0)
                    np.testing.assert_allclose(m["emittance_x"], expected_emit, rtol=2e-14, atol=0)
                if sigma == 1e-9 and label == "tracer_first":
                    assert m["sigma_x"] == 0 and m["emittance_x"] == 0
    print("DEFECT REPRODUCED")
finally:
    o.close()
