"""TRK-08: Sol with ks = 0 returns NaN positions instead of a drift.

sin(ks L/2)/(ks/2) -> L as ks -> 0, so a switched-off solenoid is the linear drift."""
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


u = [0.013, 0.021, -0.017, 0.008, 0.015, -0.03]
try:
    out = track([elements.Sol(ds=0.7, ks=0.0)], u)[0]
    exp = track([elements.Drift(ds=0.7)], u)[0]
    print(f"observed {np.round(out, 4).tolist()}\nexpected {np.round(exp, 4).tolist()}")
    print("DEFECT REPRODUCED" if not np.allclose(out, exp, atol=1e-12) else "NOT REPRODUCED")
finally:
    sim.finalize()
