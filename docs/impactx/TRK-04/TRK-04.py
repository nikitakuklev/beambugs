"""TRK-04: soft-edge elements followed by their reverse() do not return the beam in a lattice.

reverse() is documented to undo the original element. Tracking [element, reversed copy] must
return every particle to its initial coordinates. Constant profiles exclude profile parity."""
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
profile = dict(ds=0.7, cos_coefficients=[2.0], sin_coefficients=[0.0], mapsteps=100)
cases = [
    ("SoftQuadrupole", dict(gscale=0.8)),
    ("SoftSolenoid", dict(bscale=0.8)),
    ("RFCavity", dict(escale=0.2, freq=1e8, phase=0.0)),
]
bad = False
try:
    for name, kw in cases:
        el = getattr(elements, name)(**profile, **kw)
        inv = el.copy()
        inv.reverse()
        out = track([el, inv], u)
        err = np.max(np.abs(out[0] - u))
        print(f"{name}: max |final - initial| = {err:.4g}, expected 0")
        bad |= err > 1e-12
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
finally:
    sim.finalize()
