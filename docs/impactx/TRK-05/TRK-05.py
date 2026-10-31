"""TRK-05: Periodic `PolygonAperture` mirrors asymmetric polygons

A `PolygonAperture` with a nonzero period `repeat_x` (or `repeat_y`) folds coordinates into the
central cell through their absolute value, which mirrors every negative coordinate. For a polygon
that is not symmetric about x = 0 (or y = 0), particles inside the aperture are removed and
particles whose mirror image lies inside it pass, whereas repeating the aperture with a period must
leave a point already in the central period unchanged. For a rectangle with x ∈ [−0.3, 0.1] m and y
∈ [−0.2, 0.2] m and `repeat_x=1`, a particle at x = −0.2 m, inside the rectangle, is removed; with
`repeat_x=0` it survives. Polygons symmetric about the folded axis are unaffected.

The script tracks particles at x = −0.2 m (inside the rectangle) and x = 0.2 m (outside) through the
rectangular `PolygonAperture` with `repeat_x=1`. Both points lie in the central period [−0.5, 0.5)
m, so only the particle at x = −0.2 m may survive.
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


vx = [-0.3, 0.1, 0.1, -0.3, -0.3]
vy = [-0.2, -0.2, 0.2, 0.2, -0.2]
pts = [[-0.2, 0, 0, 0, 0, 0], [0.2, 0, 0, 0, 0, 0]]
try:
    out = track([elements.PolygonAperture(vertices_x=vx, vertices_y=vy, repeat_x=1.0)], pts)
    print(f"surviving x observed {out[:, 0].tolist()}, expected [-0.2]")
    print("DEFECT REPRODUCED" if out[:, 0].tolist() != [-0.2] else "NOT REPRODUCED")
finally:
    sim.finalize()
