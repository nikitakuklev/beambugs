"""TRK-15: Weak `ExactSbend` bends lose precision through cancellation

`ExactSbend` with a very small nonzero bend angle `phi` loses precision, because its position update
subtracts large bend-radius terms and nearly equal longitudinal momenta. A 65-digit calculation of
the Lorentz trajectory to the exit plane gives the correct exit coordinates, which approach the
exact drift as the angle goes to zero. For a proton at γ = 2 entering `ExactSbend(ds=0.4, phi=1e-12,
B=0)` (`phi` in degrees) at x = 0.01 m with `px` = 0.02, `x` exits at 0.02190 m instead of
0.01782 m, an error of 4.075 mm; at `phi` = 1e-10 degrees the error is 0.02487 mm. Bends of exactly
zero and of 10° are correct, and both curvature signs fail.

The script tracks a proton at γ = 2 with all six coordinates nonzero through `ExactSbend(ds=0.4,
B=0)` with `phi` = 0, 10, 1e-3, 1e-6, 1e-8, 1e-10 and ±1e-12 degrees, and compares the exit
coordinates with a 65-digit integration of the motion in the uniform field, in the entrance
laboratory frame, up to the tilted exit plane. The 0° and 10° bends are the controls and must agree
to 3e-15.
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


# In a uniform vertical field h = qB/P0 = phi/L, dpx/dZ = -h exactly. The exit plane is
# -sin(phi)*X + cos(phi)*Z = L*sinc(phi). Integrating dX/dZ, dY/dZ and d(ct)/dZ avoids both
# enormous radii and subtracting momenta.
def lorentz_to_exit(phi, gamma):
    x, px, y, py, t, pt = map(mp.mpf, u)
    ds, phi, gamma = map(mp.mpf, (length, phi, gamma))
    h = phi / ds
    ibeta = gamma / mp.sqrt(gamma**2 - 1)
    energy = ibeta - pt
    pperp2 = 1 - 2 * ibeta * pt + pt**2 - py**2

    def pz(z):
        return mp.sqrt(pperp2 - (px - h * z) ** 2)

    def lab_x(z):
        return x + mp.quad(lambda v: (px - h * v) / pz(v), [0, z])

    sine, cosine = mp.sin(phi), mp.cos(phi)
    plane_location = ds * mp.sinc(phi)
    zf = mp.findroot(
        lambda z: -sine * lab_x(z) + cosine * z - plane_location,
        (ds * mp.mpf("0.9"), ds * mp.mpf("1.1")),
    )
    xf = cosine * lab_x(zf) + sine * zf - ds * phi / 2 * mp.sinc(phi / 2) ** 2
    travel_integral = mp.quad(lambda z: 1 / pz(z), [0, zf])
    return np.array(
        list(
            map(
                float,
                (
                    xf,
                    cosine * (px - h * zf) + sine * pz(zf),
                    y + py * travel_integral,
                    py,
                    t + energy * travel_integral - ds * ibeta,
                    pt,
                ),
            )
        )
    )


try:
    for angle in (0, 10, 1e-3, 1e-6, 1e-8, 1e-10, 1e-12, -1e-12):
        o.load(u)
        element = elements.ExactSbend(ds=length, phi=angle, B=0)
        want = lorentz_to_exit(element.phi, o.ref.gamma)
        df = o.push(element)
        got = df[list(COORDS)].to_numpy()[0]
        err = np.max(abs(got - want))
        print(
            f"phi_deg={angle:g} x={got[0]:.15g} expected_x={want[0]:.15g} "
            f"t={got[4]:.15g} expected_t={want[4]:.15g} "
            f"x_error_mm={1000 * (got[0] - want[0]):.12g} max_error={err:.9g}",
            flush=True,
        )
        if angle in (0, 10):
            np.testing.assert_allclose(got, want, rtol=0, atol=3e-15)
        if abs(angle) == 1e-12:
            assert abs(got[0] - want[0]) > 1e-3
    print("DEFECT REPRODUCED")
finally:
    o.close()
