"""LAT-12: Explicit false HDF5 reference changes retain the old beta

Reading an HDF5 beam with conserve_momentum set to false preserves normalized momenta while changing their reference, but computes β, the speed divided by the speed of light, from the old reference. For an electron with relative momentum deviation δ = 0.1 and a reference change from 1 to 1.7 MeV/c, β remains 0.9069 instead of 0.9646 at the final 1.87 MeV/c. The time-derived longitudinal coordinate z stays 0.01000 m instead of 0.01064 m. Explicit true conserves physical momentum and returns the correct β.

The script reads the same electron at a changed reference with each explicit policy in a fresh process. It validates the input mass shell, computes β from the final momentum and reconstructs z from the stored time offset. Physical-momentum conservation supplies the true-policy control.
"""

import numpy as np
from _bmad import compare, verdict
from _s3_11_particle_io import policy

defect = False
for mode in ("true", "false"):
    result = policy(mode)
    before, after = result["input"], result["output"]
    if compare(
        f"{mode} input coordinates/reference control",
        before[:7],
        [0.001, 0.04, -0.002, -0.02, 0.01, 0.1, 1e6],
        atol=2e-15,
    ):
        raise RuntimeError("Input control failed")
    initial_pc = before[6] * (1 + before[5])
    initial_beta = initial_pc / np.hypot(initial_pc, before[9])
    if compare(f"{mode} input beta; initial mass shell", before[7], initial_beta, atol=2e-15):
        raise RuntimeError("Input mass-shell control failed")
    expected_momenta = (
        [0.04 / 1.7, -0.02 / 1.7, 1.1 / 1.7 - 1] if mode == "true" else [0.04, -0.02, 0.1]
    )
    if compare(
        f"{mode} normalized momenta control", after[[1, 3, 5]], expected_momenta, atol=2e-15
    ):
        raise RuntimeError("Momentum policy control failed")
    if compare(
        f"{mode} reference and absolute time control",
        after[[6, 8]],
        [1.7e6, 4.1e-9],
        atol=0,
        rtol=2e-15,
    ):
        raise RuntimeError("Reference/time control failed")
    # The file stores dt = -z_in/(beta_in*c). The same dt requires z_out/z_in = beta_out/beta_in.
    pc = after[6] * (1 + after[5])
    beta = pc / np.hypot(pc, before[9])
    z = before[4] * beta / initial_beta
    mismatch = compare(f"{mode}: beta from final mass shell", after[7], beta, atol=2e-15)
    mismatch |= compare(f"{mode}: z (m) from stored time offset", after[4], z, atol=2e-17)
    if mode == "true" and mismatch:
        raise RuntimeError("Conserved-momentum control failed")
    defect |= mismatch
verdict(defect)
