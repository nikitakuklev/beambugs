"""OPT-34: Transverse normalized beam emittance depends on reference momentum

Computing transverse normalized emittance for a beam with zero physical dispersion
multiplies by the mean particle momentum even though transverse momenta are normalized to
reference momentum. This adds an extra factor 1 + ⟨δ⟩, where δ is the fractional momentum
offset, and changes the result when only the reference momentum is redefined. An electron
beam with physical horizontal normalized emittance 5.479 × 10⁻⁷ m is reported as 5.892 ×
10⁻⁷ m when reference momentum is reduced to 93%; a mean offset of 0.025 causes a 2.5% error
at unchanged reference momentum.

The script constructs weighted hypercube beams with zero physical dispersion and changes only
their reference momentum. It checks particle invariants and computes normalized emittance
directly from physical-momentum covariance before comparing projected and mode outputs.
BMAD_DIST selects the build.
"""

import itertools

import numpy as np
from _bmad import compare, verdict
from _dispersion_and_moments import moment_cases, moments, require, twiss

signs = np.array(list(itertools.product([-1.0, 1.0], repeat=6)))
# The cubic weight changes particle charges but leaves every first and second moment unchanged.
weights = 1 + 0.2 * signs[:, 0] * signs[:, 2] * signs[:, 4]
require(moments(signs, weights)[1], np.eye(6))
normal = np.zeros((6, 6))
for j, args in enumerate([(5.5, -0.65, 7e-7), (2.7, 0.25, 1.3e-6), (8, -0.2, 4e-8)]):
    normal[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = twiss(*args)
cases = []
for species, pbase, delta, ratio in itertools.product(
    ["electron", "positron", "proton"], [4e5, 9e8], [0, 0.025], [0.93, 1, 1.07, 1.31]
):
    points = signs @ normal.T + [0.0009, 0.00003, -0.0013, -0.00002, 0.003, delta]
    cases.append(
        dict(
            species=species,
            pbase=pbase,
            pnew=pbase * ratio,
            points=points,
            ratio=ratio,
            delta=delta,
        )
    )
for ratio in [0.9, 1, 1.1]:
    points = signs @ normal.T
    points[:, 4] = 0
    points[:, 5] = 0.015
    cases.append(
        dict(
            species="electron", pbase=4e5, pnew=4e5 * ratio, points=points, ratio=ratio, delta=0.015
        )
    )
rows = moment_cases(cases, weights)
defect = False
law_residual = 0.0
for i, case in enumerate(cases, 1):
    mass, *invariants = rows[f"meta{i}"]
    require(invariants, [0, 0, 0, 0], atol=4e-16, rtol=0)
    _, cov = moments(case["points"], weights)
    # Physical normalized emittance is the determinant in (x, Px/(mc)), independent of P0.
    expected = (
        case["pbase"]
        / mass
        * np.array([np.sqrt(np.linalg.det(cov[j : j + 2, j : j + 2])) for j in (0, 2)])
    )
    observed = rows[f"emit{i}"][[1, 3]]
    predicted_factor = (1 + float(case["delta"])) / float(case["ratio"])
    law_residual = max(law_residual, float(np.max(np.abs(observed / expected - predicted_factor))))
    if case["ratio"] == 1 and case["delta"] == 0:
        require(observed, expected, atol=0, rtol=1e-13)
    # Eigenmode ordering is x,y for this uncoupled family; eigenanalysis has ~1e-7 relative resolution.
    require(rows[f"emit{i}"][[4, 5]], observed, atol=0, rtol=2e-7)
    defect |= compare(
        f"{case['species']} P0c={case['pbase']:g} eV ratio={case['ratio']:g} mean_delta={case['delta']:g}, transverse norm_emit (m)",
        observed,
        expected,
        atol=0,
        rtol=1e-13,
    )
print(f"maximum residual of extra-factor law (1+mean_delta)/ratio: {law_residual:.17g}")
verdict(defect)
