"""TRK-06: linear CFbend returns NaN when a focusing frequency is zero.

k = 0 is a sector bend (equal to Sbend); k = -1/rc^2 cancels horizontal focusing. Both maps are
finite: sin(w L)/w -> L, (1 - cos)/w^2 -> L^2/2. A neighbouring k gives the limit. Exactly zero
gives NaN; a rounding residue (h = 0.3: k = -h^2 leaves gx = -1.4e-17) gives finite wrong x and t."""
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
bad = False
try:
    for rc, k, ref_el in [
        (3.0, 0.0, elements.Sbend(ds=0.7, rc=3.0)),
        (2.0, -0.25, elements.CFbend(ds=0.7, rc=2.0, k=-0.25 + 1e-9)),
        (1 / 0.3, -(0.3**2), elements.CFbend(ds=0.7, rc=1 / 0.3, k=-(0.3**2) + 1e-9)),
    ]:
        out = track([elements.CFbend(ds=0.7, rc=rc, k=k)], u)[0]
        exp = track([ref_el], u)[0]
        print(f"rc={rc:.4g} k={k:.4g}: observed {np.round(out, 4).tolist()}\n          expected {np.round(exp, 4).tolist()}")
        bad |= not np.allclose(out, exp, atol=1e-8)
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
finally:
    sim.finalize()
