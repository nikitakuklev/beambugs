"""TRK-12: `Buncher` uses the wrong transverse fields for spin transport

With spin tracking enabled (experimental in ImpactX), `Buncher` replaces the transverse RF fields
with an effective magnetic field that reproduces the orbital kick but not the Thomas–BMT spin
precession. Maxwell's near-axis fields of a localized axisymmetric RF field give a spin rotation
about the vertical axis of −Ax/[2(γ + 1)] for a particle at offset x, with A = `k`·`V`, independent
of the magnetic anomaly; a 65-digit quadrature confirms it. For a proton at γ = 2 in
`Buncher(V=0.01, k=2)`, the spin rotation per metre of offset is +0.01529 rad/m instead of
−0.003333 rad/m, wrong in sign and magnitude. The orbital kick is correct; the spin error persists
for electrons, at γ = 10 and at low voltage.

The script tracks two particles at x = ±offset with initial spin (0, 0, 1) through `Buncher(k=2)`
and takes the spin-rotation and `px` slopes per metre of offset. It compares them with the
Maxwell–Thomas–BMT and Maxwell–Lorentz responses, integrated at 65 digits over a cosine-squared
field profile and checked against the closed forms, for protons at γ = 2 with `V` = 0.01 and 1e-6,
electrons at γ = 2 and protons at γ = 10, at offsets of 1 mm and 10 µm. It also confirms that the
observed spin slope equals A(1 + aγ)/[2(γ² − 1)], with a the species anomaly, the value the
effective field produces.
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
V, k, half_length = 0.01, 2.0, 0.031


def maxwell_response(gamma, anomaly, voltage):
    g = mp.mpf(gamma)
    bg = mp.sqrt(g * g - 1)
    beta = bg / g
    a = mp.mpf(anomaly)
    rf_k, rf_v, h = map(mp.mpf, (k, voltage, half_length))

    # Ez(z,ct)=C*cos(pi*z/(2*h))^2*sin(k*ct), |z|<=h.
    # Its integral has voltage bg*V. Maxwell gives Er=-r*d_z(Ez)/2,
    # Bphi=r*d_ct(Ez)/2, in the normalized units used by ImpactX.
    def profile(z):
        return mp.cos(mp.pi * z / (2 * h)) ** 2

    transit = mp.quad(lambda z: profile(z) * mp.cos(rf_k * z / beta), [-h, h])
    amplitude = bg * rf_v / transit

    def electric(z):
        derivative = -mp.pi / (2 * h) * mp.sin(mp.pi * z / h)
        return -amplitude * derivative * mp.sin(rf_k * z / beta) / 2

    def magnetic(z):
        return amplitude * profile(z) * rf_k * mp.cos(rf_k * z / beta) / 2

    field_spin = float(
        mp.quad(
            lambda z: -(a + 1 / g) * magnetic(z) / beta + (a + 1 / (g + 1)) * electric(z),
            [-h, h],
        )
    )
    field_orbit = float(mp.quad(lambda z: (electric(z) / beta - magnetic(z)) / bg, [-h, h]))
    return field_spin, field_orbit


try:
    for species, gamma, voltage in (
        ("proton", 2.0, V),
        ("proton", 2.0, 1e-6),
        ("electron", 2.0, V),
        ("proton", 10.0, V),
    ):
        for offset in (1e-3, 1e-5):
            u = np.zeros((2, 6))
            u[:, 0] = [-offset, offset]
            o.load(u, species=species, gamma=gamma, spin=(0, 0, 1))
            field_spin, field_orbit = maxwell_response(gamma, o.ref.gyromagnetic_anomaly, voltage)
            expected = -k * voltage / (2 * (gamma + 1))
            expected_orbit = k * voltage / (2 * (gamma**2 - 1))
            np.testing.assert_allclose(field_spin, expected, rtol=2e-15, atol=0)
            np.testing.assert_allclose(field_orbit, expected_orbit, rtol=2e-15, atol=0)
            df = o.push(elements.Buncher(V=voltage, k=k))
            angles = np.arctan2(df.spin_x.to_numpy(), df.spin_z.to_numpy())
            actual = (angles[1] - angles[0]) / (2 * offset)
            orbit = np.diff(df.momentum_x.to_numpy())[0] / (2 * offset)
            effective_b = (
                k * voltage * (1 + o.ref.gyromagnetic_anomaly * gamma) / (2 * (gamma**2 - 1))
            )
            np.testing.assert_allclose(actual, effective_b, rtol=1e-10, atol=1e-16)
            np.testing.assert_allclose(orbit, expected_orbit, rtol=2e-14, atol=1e-17)
            assert abs(actual - expected) > abs(expected) * 0.5
            print(
                f"{species} gamma={gamma:g} V={voltage:g} offset={offset:g} spin_slope={actual:.15g} "
                f"Maxwell_BMT={field_spin:.15g} orbit={orbit:.15g} "
                f"Maxwell_Lorentz={field_orbit:.15g}",
                flush=True,
            )
    print("Manual declares spin tracking unsupported")
    print("DEFECT REPRODUCED")
finally:
    o.close()
