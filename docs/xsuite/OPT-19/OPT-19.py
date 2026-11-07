"""OPT-19: Normalized-emittance covariance uses the initialization rigidity

Computing beam covariance from normalized emittances after a reference-momentum change uses the initialization rigidity at every row. The locally normalized transverse basis requires conversion with the local rigidity. A pure rebase from 650.0 MeV/c to 1.170 GeV/c predicts horizontal position variance 1.193 × 10⁻⁵ m² instead of 6.626 × 10⁻⁶ m², although particle positions do not change. Analytic covariance transport and an independently tracked ensemble agree on the correct value. Internal reference-energy changes in a segment also fail; fixed-reference controls pass.

The script constructs a transverse covariance from two normalized emittances and non-unit entrance ellipses, then tracks an eight-point symmetric quadrature with exactly that population covariance. It compares the full transverse covariance against analytic transport in both open Twiss modes through standalone and internal reference changes, including simultaneous segment acceleration. Initial rows, unchanged reference and a fixed-reference cavity gain are controls.
"""

from _prebuilt import CTX, identity, kernels, np, xt
from _reference_momentum import MASS, NEMITT, P0, SIGMA0, cases, twiss, twiss_modes

identity()
failed = 0
for kind, ratio, _t, line, p, expected in cases():
    # Eight signed columns of 2*Cholesky(Sigma) have population covariance Sigma.
    root = np.linalg.cholesky(SIGMA0)
    rays = np.concatenate((2 * root, -2 * root), axis=1)
    keys = ("x", "px", "y", "py")
    cloud = xt.Particles(_context=CTX, p0c=P0, mass0=MASS, **dict(zip(keys, rays)))
    line.track(cloud)
    assert np.all(cloud.state == 1)
    points = np.array([getattr(cloud, key) for key in keys])
    tracked = points @ points.T / 8
    oracle = expected[:4, :4] @ SIGMA0 @ expected[:4, :4].T
    np.testing.assert_allclose(tracked, oracle, atol=3e-21, rtol=3e-14)
    for mode in twiss_modes(kind):
        tw = twiss(line, p, mode)
        sigma = tw.get_beam_covariance(nemitt_x=NEMITT[0], nemitt_y=NEMITT[1]).Sigma[:, :4, :4]
        np.testing.assert_allclose(sigma[0], SIGMA0, atol=3e-20, rtol=3e-12)
        ok = np.allclose(sigma[-1], oracle, atol=3e-20, rtol=3e-12)
        if ratio == 1:
            np.testing.assert_allclose(sigma[-1], oracle, atol=3e-20, rtol=3e-12)
        elif not ok:
            # Replacing entrance rigidity with local rigidity divides by a.
            np.testing.assert_allclose(sigma[-1] / ratio, oracle, atol=3e-20, rtol=3e-12)
        failed += not ok
        print(
            f"{kind} a={ratio:g} mode={mode}: observed_Sigma11={sigma[-1, 0, 0]:.15g} expected_Sigma11={oracle[0, 0]:.15g} tracked_Sigma11={tracked[0, 0]:.15g} quadrature_max_error={np.max(np.abs(tracked - oracle)):.3g} derivation=Sigma_out=A*Sigma_in*A.T,epsilon=epsilon_n*mass0/p0c(local) physical_match={ok}"
        )
kernels()
print("DEFECT REPRODUCED" if failed else "NOT REPRODUCED")
