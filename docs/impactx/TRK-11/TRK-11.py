"""TRK-11: integer factorial overflow corrupts multipole normalization.

With zero entrance transverse momentum, one second-order drift/kick/drift step
evaluates its kick at the entrance x.  The exit momentum therefore has an exact
discrete-map expectation, without treating a finite magnet as a thin limit.
"""

import ctypes
import math
import os

for variable in (
    "NPROC",
    "PJRT_NPROC",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
from impactx import ImpactX, elements

sim = ImpactX()
sim.particle_shape = 1
sim.space_charge = False
sim.diagnostics = False
sim.slice_step_diagnostics = False
sim.init_grids()


def push(element, points):
    sim.beam.clear_particles()
    reference = sim.beam.ref
    reference.reset()
    reference.set_charge_qe(1).set_mass_MeV(1000).set_kin_energy_MeV(1000)
    zeros = np.zeros_like(points)
    sim.beam.add_n_particles(
        x=points.copy(),
        y=zeros.copy(),
        t=zeros.copy(),
        px=zeros.copy(),
        py=zeros.copy(),
        pt=zeros.copy(),
        qm=reference.qm_ratio_SI,
        bunch_charge=1e-15,
    )
    sim.lattice.clear()
    sim.lattice.append(element)
    sim.track_particles()
    return sim.beam.to_df(local=True)[["momentum_x", "momentum_y"]].to_numpy(copy=True)


try:
    print(f"native C int width={ctypes.sizeof(ctypes.c_int) * 8}")
    for exponent in (12, 13, 17):
        order = exponent + 1
        amplitude = 1e-3 if exponent < 17 else 1e-8
        x = 0.25
        length = 0.2
        # x = 1/4 is exact in binary; Python's factorial is integer-exact.
        integrated = math.factorial(exponent) * amplitude / x**exponent
        wrapped = ctypes.c_int(math.factorial(exponent)).value
        overflow_factor = math.factorial(exponent) / wrapped
        print(
            f"order={order} exponent={exponent} true_factorial={math.factorial(exponent)} wrapped_factorial={wrapped} ratio={overflow_factor:.15g}"
        )
        for component in ("normal", "skew"):
            normal = integrated if component == "normal" else 0.0
            skew = integrated if component == "skew" else 0.0
            thick_normal = [0.0] * exponent + [normal / length]
            thick_skew = [0.0] * exponent + [skew / length]
            models = [
                (
                    "thin",
                    elements.Multipole(multipole=order, K_normal=normal, K_skew=skew),
                ),
                (
                    "thick",
                    elements.ExactMultipole(
                        ds=length,
                        k_normal=thick_normal,
                        k_skew=thick_skew,
                        int_order=2,
                        mapsteps=1,
                        nslice=1,
                    ),
                ),
            ]
            positions = np.array([x, x / 2])
            expected = (
                np.array([-normal, skew])
                * positions[:, None] ** exponent
                / math.factorial(exponent)
            )
            for name, element in models:
                observed = push(element, positions)
                np.testing.assert_allclose(
                    observed, expected * overflow_factor, rtol=2e-14, atol=1e-23
                )
                if exponent == 12:
                    np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=1e-23)
                else:
                    assert not np.allclose(observed, expected, rtol=1e-12, atol=1e-23)
                index = 0 if component == "normal" else 1
                print(
                    f"{name} {component} order={order}: expected={expected[0, index]:.15g} actual={observed[0, index]:.15g} ratio={observed[0, index] / expected[0, index]:.15g}"
                )
    print("DEFECT REPRODUCED")
finally:
    sim.finalize()
