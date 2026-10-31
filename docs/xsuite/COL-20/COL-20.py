"""COL-20: Zero horizontal dispersion invariant creates a NaN B&M rate

`BjorkenMtingwaIBS` divides by the horizontal dispersion invariant Hₓ, which is zero where both the
dispersion `dx` and its slope `dpx` vanish, so one such sampled location makes the horizontal growth
rate Kₓ NaN. The collision-matrix integral is regular there for positive emittances and momentum
spread, and independent Carlson-integral and Nagaitsev evaluations give finite rates. For protons at
reference speed β₀ = 0.45 (in units of c) with zero dispersion, Kₓ is NaN instead of −2.872 × 10⁻³
s⁻¹; in a three-segment line with only one zero-invariant location, it is NaN instead of −7.397 ×
10⁻⁴ s⁻¹. NumPy emits division warnings and the other planes stay finite; a nonzero `dpx` avoids the
singularity even when `dx` = 0.

The script uses the shared helper `_ibs.py`, which builds periodic `LineSegmentMap` optics, checks
them with tracked off-momentum closed orbits and computes the collision-matrix rates independently
of Xfields, for protons at β₀ = 0.45. Five cases cover zero dispersion, a tiny `dx` = 10⁻⁹ m
continuity control, `dx` = 0 with `dpx` = 0.11, zero horizontal with nonzero vertical dispersion,
and a three-segment line in which only one location has Hₓ = 0. At zero dispersion it also checks
the reference rates against the rest-frame kinetic-energy invariant. It requires the installed
prebuilt kernels (compilation is forbidden).
"""

from _ibs import PARAMS, local_oracle, run_case
from _prebuilt import identity, kernels, np

identity()
print("ORACLE Hx=Dx^2/betx+betx*(Dpx+alfx*Dx/betx)^2; Hx=0 iff Dx=Dpx=0")
print("Positive collision matrix; exact eigenvalue/Carlson integral gives finite amplitude rates")
print("Energy invariant sum((epsx/betx,epsy/bety,sigma_delta^2/gamma^2)*K)=0 at zero dispersion")
for label, optics in [
    ("zero", dict(dx=0.0, dpx=0.0)),
    ("zero_dx_nonzero_prime", dict(dx=0.0, dpx=0.11)),
    ("near_zero", dict(dx=1e-9, dpx=0.0)),
    ("zero_with_vertical", dict(dx=0.0, dpx=0.0, dy=0.5, dpy=-0.04)),
    ("one_zero_location", dict(dx=0.0, dpx=0.0, varying=True)),
]:
    record, tw = run_case(label, 0.45, params=PARAMS, **optics)
    if label in {"zero", "zero_with_vertical", "one_zero_location"}:
        assert np.isnan(record["actual"][0])
        assert np.all(np.isfinite(record["oracle"]))
        assert any("divide" in message for message in record["warnings"])
    else:
        assert np.all(np.isfinite(record["actual"]))
    if label in {"zero", "near_zero"}:
        start = 1 if label == "zero" else 0
        np.testing.assert_allclose(
            record["actual"][start:], record["oracle"][start:], rtol=1e-9, atol=2e-13
        )
    if label == "zero":
        weights = np.array(
            [
                PARAMS["gemitt_x"] / tw.betx[0],
                PARAMS["gemitt_y"] / tw.bety[0],
                PARAMS["sigma_delta"] ** 2 / tw.gamma0**2,
            ]
        )
        residual = float(np.dot(weights, record["oracle"]))
        print("ZERO_ENERGY_RESIDUAL", residual)
        print("ZERO_MATRIX_EIGENVALUES", local_oracle(tw, 0)[1].tolist())
        assert abs(residual) < 1e-23

kernels()
print("DEFECT REPRODUCED")
