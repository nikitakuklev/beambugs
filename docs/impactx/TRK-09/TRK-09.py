"""TRK-09: ExactCFbend higher-order multipole kicks lose all precision at small curvature.

The curvilinear decapole kick at y = 0 is k4*x^4/24 + O(k4 x^5/rc); the code evaluates it as a
difference of O(1) polynomial and log terms in rho = 1 + x/rc, multiplied by rc^4*k4.
Expected: the decapole changes px by -ds*k4*x^4/24 relative to the same bend without it.
"""
import numpy as np
from impactx import ImpactX, elements

COLS = ["position_x", "momentum_x", "position_y", "momentum_y", "position_t", "momentum_t"]

sim = ImpactX()
sim.particle_shape = 1
sim.space_charge = False
sim.diagnostics = False
sim.slice_step_diagnostics = False
sim.init_grids()


def track(lattice, coords, kin_MeV=938.27208816):
    """Track proton(s) with (x, px, y, py, t, pt) through a lattice; return survivors."""
    beam = sim.beam
    beam.clear_particles()
    ref = beam.ref
    ref.reset()
    ref.set_species("proton").set_kin_energy_MeV(kin_MeV)
    u = np.atleast_2d(np.asarray(coords, dtype=float))
    beam.add_n_particles(
        x=u[:, 0].copy(), px=u[:, 1].copy(), y=u[:, 2].copy(),
        py=u[:, 3].copy(), t=u[:, 4].copy(), pt=u[:, 5].copy(),
        qm=ref.qm_ratio_SI, bunch_charge=1e-12,
    )
    sim.lattice.clear()
    sim.lattice.extend(lattice)
    sim.track_particles()
    df = beam.to_df(local=True)
    return np.zeros((0, 6)) if df is None else df[COLS].to_numpy(copy=True)


ds, x, k4 = 1e-4, 1e-3, 1e4
try:
    base = dict(ds=ds, k_skew=[0.0] * 5, int_order=2, mapsteps=1)
    with_k4 = track([elements.ExactCFbend(k_normal=[1e-3, 0, 0, 0, k4], **base)], [x, 0, 0, 0, 0, 0])[0, 1]
    without = track([elements.ExactCFbend(k_normal=[1e-3, 0, 0, 0, 0], **base)], [x, 0, 0, 0, 0, 0])[0, 1]
    expected = -ds * k4 * x**4 / 24
    print(f"rc = 1000 m: decapole px change observed {with_k4 - without:.4g}, expected {expected:.4g}")
    print("DEFECT REPRODUCED" if abs(with_k4 - without - expected) > 1e-3 * abs(expected) else "NOT REPRODUCED")
finally:
    sim.finalize()
