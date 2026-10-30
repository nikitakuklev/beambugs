"""TRK-13: Hminus has the wrong magnetic-anomaly sign.

Check magnetic moment implied by set_species, then longitudinal precession.
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
from impactx import ImpactX, RefPart, elements

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
c, magnetic_field, length = 299792458.0, 0.1, 0.1
proton = RefPart().set_species("proton")
anion = RefPart().set_species("Hminus")
ratio = anion.mass_MeV / proton.mass_MeV
bare = -(1 + proton.gyromagnetic_anomaly) * ratio - 1
# Kilich & Pachucki, PRA 112, 042813 (2025), sigma=24.243189(9) ppm.
shield = 24.243189e-6
shielded = -(1 + proton.gyromagnetic_anomaly) * ratio * (1 - shield) - 1
print(
    f"mass_ratio={ratio:.15g} stored_a={anion.gyromagnetic_anomaly:.15g} "
    f"bare_a={bare:.15g} shielded_a={shielded:.15g}"
)
implied_mu_ratio = (
    (1 + anion.gyromagnetic_anomaly)
    * anion.qm_ratio_SI
    / ((1 + proton.gyromagnetic_anomaly) * proton.qm_ratio_SI)
)
print(f"implied_mu_Hminus_over_mu_proton={implied_mu_ratio:.15g}; expected={1 - shield:.15g}")
assert implied_mu_ratio < 0
try:
    for label, species, anomaly in (
        ("proton", "proton", None),
        ("builtin", "Hminus", None),
        ("bare_override", "Hminus", bare),
        ("shielded_override", "Hminus", shielded),
    ):
        o.load(np.zeros(6), species=species, anomaly=anomaly, spin=(1, 0, 0))
        bz = o.ref.qm_ratio_SI * magnetic_field / c
        expected = -(1 + o.ref.gyromagnetic_anomaly) * bz * length / o.ref.beta_gamma
        df = o.push(elements.ChrAcc(ds=length, ez=0, bz=bz))
        angle = np.arctan2(df.spin_y.iloc[0], df.spin_x.iloc[0])
        np.testing.assert_allclose(angle, expected, rtol=2e-14, atol=1e-16)
        print(f"{label}: spin_angle={angle:.15g} BMT={expected:.15g}", flush=True)
    print("DEFECT REPRODUCED")
finally:
    o.close()
