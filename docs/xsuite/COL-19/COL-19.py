"""COL-19: B&M applies an extra beta factor to momentum dispersion

`BjorkenMtingwaIBS`, the Bjorken–Mtingwa model of intrabeam-scattering growth rates, multiplies the
dispersions and their slopes in both planes by the relativistic reference speed β₀. Xtrack supplies
them as derivatives with respect to the relative momentum deviation δ, which the manual's collision
matrices use unscaled; integrating the collision matrix with Carlson elliptic integrals gives the
correct rates and agrees with Xfields' Nagaitsev model at zero vertical dispersion. For protons at
β₀ = 0.45 with beta function βₓ = 8 m, dispersion Dₓ = 1.7 m and slope Dₓ′ = 0.11, the longitudinal
cooling rate is −2.890 × 10⁻³ s⁻¹ instead of −2.596 × 10⁻³ s⁻¹, 11.36% too large in magnitude; at
β₀ = 0.75 the error is 22.43%.

The shared helper `_ibs.py` builds each periodic 59 m segment from `LineSegmentMap` elements,
confirms with tracked off-momentum closed orbits that the Twiss dispersion is dx/dδ, and computes
the collision-matrix rates independently of Xfields at the same Coulomb logarithm. The script runs
six cases (β₀ = 0.45, 0.75 and 0.999, plus reversed, vertical, varying and very small dispersion)
and checks that Xfields' B&M rates equal that calculation with the dispersions multiplied by β₀ and
differ from it without the factor. With very small dispersion the factor no longer matters, so there
the rates must also match the unscaled calculation. It requires the installed prebuilt kernels
(compilation is forbidden).
"""

from _ibs import PARAMS, run_case
from _prebuilt import identity, kernels, np

identity()
print("ORACLE L=Lx+Ly+Lz; phi=Dprime+alpha*D/beta; H=D^2/beta+beta*phi^2")
print("Rest-frame momentum delta=gamma*u_parallel; L uses gamma*D_delta")
print("Eigenvalue/Carlson integral: J_i=2*(sum(R)-3*R_i)/(3*sqrt(det(L)))")
print("Amplitude rates C/2*trace(L_u*J); shared Coulomb log held fixed")

for label, beta, optics in [
    ("low", 0.45, {}),
    ("mid", 0.75, dict(dx=-1.7, dpx=-0.11)),
    ("high", 0.999, {}),
    ("vertical", 0.45, dict(dy=0.5, dpy=-0.04)),
    ("varying", 0.75, dict(varying=True)),
    ("tiny", 0.45, dict(dx=1e-7, dpx=0)),
]:
    record, _ = run_case(label, beta, params=PARAMS, **optics)
    np.testing.assert_allclose(
        record["actual"], record["beta_scaled_oracle"], rtol=3e-8, atol=2e-13
    )
    if label == "tiny":
        np.testing.assert_allclose(record["actual"], record["oracle"], rtol=1e-9, atol=2e-13)
    else:
        assert np.max(abs(np.array(record["relative_error"]))) > 1e-4

kernels()
print("DEFECT REPRODUCED")
