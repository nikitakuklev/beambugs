"""TRK-05: periodic PolygonAperture folds coordinates with abs() and mirrors asymmetric polygons.

Both test points lie inside the central period [-0.5, 0.5), so repeat_x = 1 must not change
which of them is inside the rectangle x in [-0.3, 0.1], y in [-0.2, 0.2]."""
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


vx = [-0.3, 0.1, 0.1, -0.3, -0.3]
vy = [-0.2, -0.2, 0.2, 0.2, -0.2]
pts = [[-0.2, 0, 0, 0, 0, 0], [0.2, 0, 0, 0, 0, 0]]
try:
    out = track([elements.PolygonAperture(vertices_x=vx, vertices_y=vy, repeat_x=1.0)], pts)
    print(f"surviving x observed {out[:, 0].tolist()}, expected [-0.2]")
    print("DEFECT REPRODUCED" if out[:, 0].tolist() != [-0.2] else "NOT REPRODUCED")
finally:
    sim.finalize()
