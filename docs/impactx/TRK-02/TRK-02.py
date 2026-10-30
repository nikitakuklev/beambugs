"""TRK-02: RFCavity spin transport rotates spin without a field and with a wrong energy factor.

Thomas-BMT: dS/ds = 0 where the field vanishes. At the entrance of a cavity with on-axis field
escale*ez (ez = cos_coefficients[0]/2, the Fourier a0/2 term; phase 0), a particle with px precesses at
|dSx/ds| = escale*ez*px*(G + 1/(gamma+1)) (longitudinal E-field term).
"""
import numpy as np
from impactx import ImpactX, elements

sim = ImpactX()
sim.particle_shape = 1
sim.space_charge = False
sim.diagnostics = False
sim.slice_step_diagnostics = False
sim.spin = True
sim.init_grids()


def spin_after(cavity, coords):
    beam = sim.beam
    beam.clear_particles()
    ref = beam.ref
    ref.reset()
    ref.set_species("proton").set_kin_energy_MeV(938.27208816)
    x, px, y, py, t, pt = (np.array([c]) for c in coords)
    beam.add_n_particles(x=x, px=px, y=y, py=py, t=t, pt=pt, qm=ref.qm_ratio_SI,
                         bunch_charge=1e-12, sx=np.array([0.0]), sy=np.array([0.0]), sz=np.array([1.0]))
    sim.lattice.clear()
    sim.lattice.append(cavity)
    sim.track_particles()
    df = beam.to_df(local=True)
    return df[["spin_x", "spin_y", "spin_z"]].to_numpy()[0], ref


try:
    u = [0.013, 0.021, -0.017, 0.008, 0.015, -0.03]
    s, _ = spin_after(elements.RFCavity(ds=0.7, escale=0.2, freq=1e8, phase=0.0, cos_coefficients=[0.0],
                                        sin_coefficients=[0.0], mapsteps=100), u)
    print(f"zero on-axis field: spin observed {np.round(s, 6).tolist()}, expected [0, 0, 1]")
    zero_field_bad = np.max(np.abs(s - [0, 0, 1])) > 1e-12

    L, px = 1e-7, 1e-3
    s, ref = spin_after(elements.RFCavity(ds=L, escale=0.2, freq=1e8, phase=0.0, cos_coefficients=[2.0],
                                          sin_coefficients=[0.0], mapsteps=10), [0, px, 0, 0, 0, 0])
    g = ref.gamma
    expected = 0.2 * (2.0 / 2) * px * (ref.gyromagnetic_anomaly + 1 / (g + 1))
    ratio = abs(s[0] / L) / expected
    print(f"entrance |dSx/ds| observed {abs(s[0] / L):.4g}, expected {expected:.4g}; ratio {ratio:.4g} "
          f"(sqrt((g^2-1)/(g^2+1)) = {np.sqrt((g**2 - 1) / (g**2 + 1)):.4g})")
    print("DEFECT REPRODUCED" if zero_field_bad or abs(ratio - 1) > 1e-4 else "NOT REPRODUCED")
finally:
    sim.finalize()
