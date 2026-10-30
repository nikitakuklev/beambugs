"""TRK-14: Weak nonzero ChrAcc acceleration loses precision.

Integrate the documented transverse-paraxial Hamiltonian at high precision.
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

import mpmath as mp
import numpy as np
from impactx import ImpactX, elements

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


mp.mp.dps = 65
o = Oracle()
u = np.array([0.01, 0.02, -0.015, 0.01, 0.03, -0.02])
length = 0.4


def expected(ez, gamma):
    x, px, y, py, t, pt = map(mp.mpf, u)
    g0, strength, ds = map(mp.mpf, (gamma, ez, length))
    bg = mp.sqrt(g0 * g0 - 1)
    gp0 = g0 - bg * pt
    ptrans2 = bg**2 * (px**2 + py**2)

    def p(s):
        return mp.sqrt((gp0 + strength * s) ** 2 - 1)

    def pref(s):
        return mp.sqrt((g0 + strength * s) ** 2 - 1)

    integral = mp.quad(lambda s: 1 / p(s), [0, ds])
    dt = mp.quad(
        lambda s: (
            (gp0 + strength * s) / p(s) * (1 + ptrans2 / (2 * p(s) ** 2))
            - (g0 + strength * s) / pref(s)
        ),
        [0, ds],
    )
    reftime = mp.quad(lambda s: (g0 + strength * s) / pref(s), [0, ds])
    renorm = bg / pref(ds)
    return np.array(
        list(
            map(
                float,
                (
                    x + bg * px * integral,
                    px * renorm,
                    y + bg * py * integral,
                    py * renorm,
                    t + dt,
                    pt * renorm,
                ),
            )
        )
    ), float(reftime)


try:
    for ez in (0.0, 1e-6, 1e-10, -1e-10, 1e-14, -1e-14, 1e-16):
        o.load(u)
        want, want_ref = expected(ez, o.ref.gamma)
        df = o.push(elements.ChrAcc(ds=length, ez=ez, bz=0))
        got = df[list(COORDS)].to_numpy()[0]
        print(
            f"ez={ez:g} x={got[0]:.15g} expected_x={want[0]:.15g} "
            f"t={got[4]:.15g} expected_t={want[4]:.15g} "
            f"ref_ct={o.ref.t:.15g} expected_ref_ct={want_ref:.15g} "
            f"max_map_error={np.max(abs(got - want)):.9g}",
            flush=True,
        )
        if ez == 0:
            np.testing.assert_allclose(got, want, rtol=0, atol=5e-16)
            np.testing.assert_allclose(o.ref.t, want_ref, rtol=0, atol=2e-16)
        if abs(ez) == 1e-14:
            assert abs(got[0] - want[0]) > 1e-5
            assert abs(o.ref.t - want_ref) > 1e-4
    print("DEFECT REPRODUCED")
finally:
    o.close()
