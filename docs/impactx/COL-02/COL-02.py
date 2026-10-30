"""COL-02: incoherent synchrotron radiation uses the electron classical radius for every species.

Classical loss in a bend of radius R over ds (ultrarelativistic): dP/P = (2/3) r_q (beta gamma)^3 ds / R^2,
with the species radius r_q = q^2 / (4 pi eps0 m c^2). For a proton r_e / r_p = m_p / m_e = 1836.
"""
import numpy as np
from impactx import ImpactX, elements
from scipy.constants import c, epsilon_0

sim = ImpactX()
sim.verbose = 0
sim.tiny_profiler = False
sim.diagnostics = False
sim.space_charge = False
sim.isr = True
sim.isr_order = 1
sim.isr_on_ref_part = True
sim.init_grids()
try:
    ref = sim.beam.ref
    ref.set_species("proton").set_kin_energy_MeV(7e6)  # 7 TeV
    z = np.zeros(4)
    sim.beam.add_n_particles(x=z, y=z, t=z, px=z, py=z, pt=z, qm=ref.qm_ratio_SI, bunch_charge=1e-15)
    R, ds = 2800.0, 100.0
    bg = ref.beta_gamma
    r_p = ref.charge**2 / (4 * np.pi * epsilon_0 * ref.mass * c**2)
    expected = 2 / 3 * r_p * bg**3 * ds / R**2
    sim.lattice.append(elements.ExactSbend(ds=ds, phi=np.degrees(ds / R)))
    sim.track_particles()
    observed = (bg - ref.beta_gamma) / bg
    print(f"proton reference dP/P observed {observed:.4g}, expected {expected:.4g}, ratio {observed / expected:.4g}")
    print("DEFECT REPRODUCED" if abs(observed / expected - 1) > 1e-3 else "NOT REPRODUCED")
finally:
    sim.finalize()
