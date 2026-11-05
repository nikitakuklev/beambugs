"""Periodic LineSegmentMap optics and an independent reference for analytical IBS growth rates.

The reference integrates the Gaussian collision matrix exactly with Carlson elliptic integrals.
"""

import warnings
from typing import TypedDict

from _prebuilt import CTX, np, xt
from scipy.constants import c
from scipy.special import elliprd
from xfields.ibs import BjorkenMtingwaIBS


class BeamParams(TypedDict):
    gemitt_x: float
    gemitt_y: float
    sigma_delta: float
    bunch_length: float
    total_beam_intensity: int


PARAMS: BeamParams = dict(
    gemitt_x=3e-6,
    gemitt_y=0.8e-6,
    sigma_delta=7e-4,
    bunch_length=0.35,
    total_beam_intensity=4_000_000_000,
)


def make_line(beta, *, dx=1.7, dpx=0.11, dy=0.0, dpy=0.0, varying=False):
    base = dict(betx=8.0, bety=14.0, alfx=0.45, alfy=-0.25, dx=dx, dpx=dpx, dy=dy, dpy=dpy)
    optics = [base]
    lengths = [59.0]
    if varying:
        optics += [
            dict(betx=17.0, bety=6.0, alfx=-0.3, alfy=0.6, dx=0.8, dpx=-0.07, dy=0.4, dpy=0.035),
            dict(betx=11.0, bety=9.0, alfx=0.1, alfy=-0.2, dx=2.2, dpx=0.03, dy=-0.2, dpy=-0.015),
        ]
        lengths = [13.0, 29.0, 17.0]
    elements = []
    for i, length in enumerate(lengths):
        end = optics[(i + 1) % len(optics)]
        kwargs = {key: [value, end[key]] for key, value in optics[i].items()}
        elements.append(
            xt.LineSegmentMap(length=length, qx=0.12, qy=0.09, longitudinal_mode="frozen", **kwargs)
        )
    line = xt.Line(elements=elements)
    gamma = 1 / np.sqrt(1 - beta**2)
    line.particle_ref = xt.Particles(
        _context=CTX, mass0=xt.PROTON_MASS_EV, p0c=xt.PROTON_MASS_EV * beta * gamma
    )
    line.build_tracker(_context=CTX)
    tw = line.twiss(method="4d")
    for key, value in base.items():
        np.testing.assert_allclose(getattr(tw, key)[0], value, rtol=2e-9, atol=1e-13)
    # Confirm every dispersion column against tracked closed orbits at 18 offsets.
    offsets = np.linspace(-8e-4, 8e-4, 18)
    particles = xt.Particles(
        _context=CTX,
        mass0=xt.PROTON_MASS_EV,
        p0c=line.particle_ref.p0c[0],
        delta=offsets,
        x=dx * offsets,
        px=dpx * offsets,
        y=dy * offsets,
        py=dpy * offsets,
    )
    max_orbit_error = 0.0
    for element, endpoint in zip(elements, optics[1:] + optics[:1], strict=True):
        element.track(particles)
        for coord, disp in [("x", "dx"), ("px", "dpx"), ("y", "dy"), ("py", "dpy")]:
            err = float(np.max(abs(getattr(particles, coord) - endpoint[disp] * offsets)))
            max_orbit_error = max(max_orbit_error, err)
            assert err < 2e-15
    return tw, max_orbit_error


def local_oracle(tw, row, *, dispersion_scale=1.0, params: BeamParams = PARAMS):
    """Integrate the matrix collision kernel analytically after diagonalizing L.

    For eigenvalues l_i, put c_i=1/l_i and R_i=c_i RD(c_j,c_k,c_i).
    Substituting lambda=1/t in the matrix integral gives
    J_i = 2 (sum(R)-3 R_i)/(3 sqrt(det L)). The amplitude rate is
    C/2 tr(L_u J). No B&M polynomial coefficients or quadrature are used.
    """
    g = tw.gamma0
    parts = []
    for axis, emit, trans in [("x", params["gemitt_x"], 0), ("y", params["gemitt_y"], 2)]:
        beta = tw["bet" + axis][row]
        alpha = tw["alf" + axis][row]
        disp = dispersion_scale * tw["d" + axis][row]
        prime = dispersion_scale * tw["dp" + axis][row]
        phi = prime + alpha * disp / beta
        block = np.zeros((3, 3))
        block[trans, trans] = beta / emit
        block[1, 1] = g * g * (disp * disp / beta + beta * phi * phi) / emit
        block[trans, 1] = block[1, trans] = -g * beta * phi / emit
        parts.append(block)
    parts.append(np.diag([0.0, g * g / params["sigma_delta"] ** 2, 0.0]))
    matrix = sum(parts)
    vals, vecs = np.linalg.eigh(matrix)
    assert np.min(vals) > 0
    inv = 1 / vals
    rs = np.array([inv[i] * elliprd(inv[(i + 1) % 3], inv[(i + 2) % 3], inv[i]) for i in range(3)])
    js = 2 * (sum(rs) - 3 * rs) / (3 * np.sqrt(np.prod(vals)))
    kernel = (vecs * js) @ vecs.T
    contractions = np.array([np.trace(part @ kernel) for part in parts])
    return contractions, vals


def oracle(tw, *, dispersion_scale=1.0, params: BeamParams = PARAMS, log=None, left_average=False):
    if log is None:
        log = BjorkenMtingwaIBS(tw).coulomb_log(**params)
    r0 = tw.particle_on_co.get_classical_particle_radius0()
    prefactor = (
        params["total_beam_intensity"]
        * r0
        * r0
        * c
        * log
        / (
            16
            * np.pi
            * tw.beta0**3
            * tw.gamma0**4
            * params["gemitt_x"]
            * params["gemitt_y"]
            * params["sigma_delta"]
            * params["bunch_length"]
        )
    )
    rows = [
        local_oracle(tw, i, dispersion_scale=dispersion_scale, params=params)[0]
        for i in range(len(tw.s))
    ]
    if left_average:
        return (
            prefactor
            * np.sum(np.array(rows[:-1]) * np.diff(tw.s)[:, None], axis=0)
            / tw.line_length
        )
    return prefactor * np.trapezoid(rows, tw.s, axis=0) / tw.line_length


def run_case(label, beta, *, params: BeamParams = PARAMS, **optics):
    tw, orbit_error = make_line(beta, **optics)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = np.array(tw.get_ibs_growth_rates(formalism="B&M", **params).as_tuple())
    expected = oracle(tw, params=params)
    scaled = oracle(tw, dispersion_scale=tw.beta0, params=params)
    logs = BjorkenMtingwaIBS(tw).coulomb_log(**params)
    rec = dict(
        label=label,
        beta=beta,
        optics=optics,
        parameters=params,
        actual=actual.tolist(),
        oracle=expected.tolist(),
        beta_scaled_oracle=scaled.tolist(),
        relative_error=((actual - expected) / np.maximum(abs(expected), 1e-25)).tolist(),
        orbit_max_abs_error=orbit_error,
        coulomb_log=logs,
        warnings=sorted({str(w.message) for w in caught}),
    )
    rec["twiss"] = {
        name: tw[name].tolist()
        for name in ("s", "betx", "bety", "alfx", "alfy", "dx", "dpx", "dy", "dpy")
    }
    if np.all(np.isfinite(actual)):
        coefficient_error = 0.0
        bm = BjorkenMtingwaIBS(tw)
        args = (params["gemitt_x"], params["gemitt_y"], params["sigma_delta"])
        aa, bb, cc = (np.asarray(getattr(bm, "_" + key)(*args)) for key in ("a", "b", "c"))
        plane_a = [np.asarray(getattr(bm, "_a" + key)(*args)) for key in ("x", "y", "z")]
        plane_b = [np.asarray(getattr(bm, "_b" + key)(*args)) for key in ("x", "y", "z")]
        _, bracket_x, bracket_y, bracket_z = bm._constants(**params)
        for row in range(len(tw.s)):
            # Rebuild quadratic forms independently; compare the integrand pointwise.
            g = tw.gamma0
            parts = []
            for axis, emit, trans in [("x", args[0], 0), ("y", args[1], 2)]:
                beta_u = tw["bet" + axis][row]
                disp = tw.beta0 * tw["d" + axis][row]
                phi = tw.beta0 * tw["dp" + axis][row] + tw["alf" + axis][row] * disp / beta_u
                part = np.zeros((3, 3))
                part[trans, trans] = beta_u / emit
                part[trans, 1] = part[1, trans] = -g * beta_u * phi / emit
                part[1, 1] = g * g * (disp * disp / beta_u + beta_u * phi * phi) / emit
                parts.append(part)
            parts.append(np.diag([0.0, g * g / args[2] ** 2, 0.0]))
            for lam in [1e3, 1e6, 1e9]:
                total = sum(parts)
                matrix = total + lam * np.eye(3)
                inverse = np.linalg.inv(matrix)
                # The isotropic part cancels in the trace contraction. Evaluate
                # inv(M)-I/(lambda+mu) via its resolvent identity to avoid
                # subtracting nearly equal traces at large lambda.
                mu = np.trace(total) / 3
                anisotropic_inverse = -(total - mu * np.eye(3)) @ inverse / (lam + mu)
                for u, bracket in enumerate(
                    (np.asarray(bracket_x)[row], np.asarray(bracket_y)[row], bracket_z)
                ):
                    poly = (
                        bracket
                        * np.sqrt(lam)
                        * (plane_a[u][row] * lam + plane_b[u][row])
                        / (lam**3 + aa[row] * lam**2 + bb[row] * lam + cc[row]) ** 1.5
                    )
                    tensor = (
                        np.sqrt(lam)
                        * (
                            np.trace(parts[u]) * np.trace(anisotropic_inverse)
                            - 3 * np.trace(parts[u] @ anisotropic_inverse)
                        )
                        / np.sqrt(np.linalg.det(matrix))
                    )
                    coefficient_error = max(
                        coefficient_error, float(abs(poly - tensor) / max(abs(tensor), 1e-100))
                    )
                    np.testing.assert_allclose(poly, tensor, rtol=2e-12, atol=1e-25)
        rec["scaled_integrand_max_relative_error"] = coefficient_error
    if not optics.get("varying", False) and optics.get("dy", 0) == 0 and optics.get("dpy", 0) == 0:
        nag = np.array(tw.get_ibs_growth_rates(formalism="Nagaitsev", **params).as_tuple())
        # Nagaitsev uses left-endpoint averaging; B&M linearly interpolates the rate.
        nag_oracle = oracle(tw, params=params, left_average=True)
        np.testing.assert_allclose(nag, nag_oracle, rtol=3e-12, atol=2e-17)
        rec["nagaitsev"] = nag.tolist()
        rec["nagaitsev_left_average_oracle"] = nag_oracle.tolist()
    print("CASE", rec)
    print(f"RATES {label} actual={actual.tolist()} expected={expected.tolist()}")
    print(f"ORBIT {label} max_abs_error={orbit_error:.17g}")
    return rec, tw
