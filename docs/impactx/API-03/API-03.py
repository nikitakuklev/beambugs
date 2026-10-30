"""API-03: An inert insertion must preserve an exact sector bend's reference arc."""

import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NPROC",
    "PJRT_NPROC",
):
    os.environ[name] = "1"

from math import atan2, cos, degrees, radians, sin

import numpy as np
from impactx import RefPart, elements, push

LENGTH = 1.0
ANGLE = 20.0
theta = radians(ANGLE)
radius = LENGTH / theta
expected = np.array([radius * (cos(theta) - 1), radius * sin(theta), theta, LENGTH])


def track(lattice):
    ref = RefPart().set_species("proton")
    ref.set_kin_energy_MeV(1000)
    for el in lattice:
        push(ref, el)
    return np.array([ref.x, ref.z, atan2(-ref.px, ref.pz), ref.s])


original = elements.KnownElementsList([elements.ExactSbend(ds=LENGTH, phi=ANGLE)])
baseline = track(original)
np.testing.assert_allclose(baseline, expected, rtol=0, atol=2e-15)
print(f"original: angle_deg={degrees(baseline[2]):.15g} x={baseline[0]:.15g} z={baseline[1]:.15g}")
for spacing in (0.5, 0.3):
    split = elements.transformation.insert_element_every_ds(
        original, ds=spacing, element=elements.Marker("inert")
    )
    pieces = [el for el in split if isinstance(el, elements.ExactSbend)]
    assert original[0].ds == LENGTH and abs(original[0].phi - theta) < 1e-14
    assert abs(sum(el.ds for el in pieces) - LENGTH) < 1e-15
    observed = track(split)
    np.testing.assert_allclose(observed[2], len(pieces) * theta, rtol=0, atol=2e-15)
    print(
        f"spacing={spacing}: piece_lengths={[el.ds for el in pieces]} piece_angles_deg={[degrees(el.phi) for el in pieces]}"
    )
    print(
        f"  observed_angle_deg={degrees(observed[2]):.15g} expected_angle_deg={ANGLE:g} s={observed[3]:.15g}"
    )
    assert np.max(np.abs(observed - expected)) > 0.1
    # A sector with the same field has the same curvature in every segment.
    for el in pieces:
        # Constructor phi is degrees; this property's documented units are radians.
        el.phi = theta * el.ds / LENGTH
    repaired = track(split)
    np.testing.assert_allclose(repaired, expected, rtol=0, atol=2e-15)
    print(
        f"  angle-scaled control: angle_deg={degrees(repaired[2]):.15g} max_arc_error={np.max(np.abs(repaired - expected)):.3g}"
    )
print("DEFECT REPRODUCED")
