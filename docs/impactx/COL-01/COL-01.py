"""COL-01: PIC 2.5D longitudinal space-charge kick depends on a hidden 1 m length scale.

Scale every length (bunch, mesh, kick length) by a at fixed charge and energy. Coulomb fields
scale as 1/a^2, so a*(kick) must be unchanged for every component. The transverse kicks pass;
the longitudinal kick multiplies the absolute 2D potential, whose log reference is 1 m.
"""
import numpy as np
from impactx import ImpactX, elements

sim = ImpactX()
sim.verbose = 0
sim.tiny_profiler = False
sim.diagnostics = False
sim.space_charge = "2p5D"
sim.poisson_solver = "fft"
sim.particle_shape = 2
sim.strang_split = False
sim.n_cell = [64, 64, 8]
sim.space_charge_num_longitudinal_bins = 65
sim.init_grids()

rng = np.random.default_rng(1)
xyz = rng.normal(size=(16384, 3)) * [1e-4, 1e-4, 1e-3]  # sigma x, y, t in m
xyz -= xyz.mean(axis=0)


def scaled_kicks(a):
    beam = sim.beam
    beam.clear_particles()
    ref = beam.ref
    ref.reset()
    ref.set_species("electron").set_kin_energy_MeV(4.59899055)  # gamma = 10
    z = np.zeros(len(xyz))
    beam.add_n_particles(x=a * xyz[:, 0], y=a * xyz[:, 1], t=a * xyz[:, 2], px=z, py=z, pt=z,
                         qm=ref.qm_ratio_SI, bunch_charge=1e-12)
    kick_only = elements.Programmable(ds=1e-3 * a)  # one collective kick, no transport
    kick_only.push = lambda pc, step, period: None
    sim.lattice.clear()
    sim.lattice.append(kick_only)
    sim.track_particles()
    df = beam.to_df(local=True).reset_index(drop=True).sort_values("idcpu")
    return a * df[["momentum_x", "momentum_t"]].to_numpy()


try:
    k1, k2 = scaled_kicks(1.0), scaled_kicks(2.0)
    rel = np.linalg.norm(k2 - k1, axis=0) / np.linalg.norm(k1, axis=0)
    print(f"relative change of a*kick under a=1 -> 2: px {rel[0]:.4g}, pt {rel[1]:.4g}; expected 0 for both")
    print("DEFECT REPRODUCED" if rel[1] > 1e-6 and rel[0] < 1e-9 else "NOT REPRODUCED")
finally:
    sim.finalize()
