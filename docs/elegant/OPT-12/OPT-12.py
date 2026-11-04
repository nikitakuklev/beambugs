"""OPT-12: Unmatched chromatic derivatives use exit dispersion instead of entrance dispersion

With `twiss_output matched=0`, Elegant uses exit dispersion where the chain rule requires entrance
dispersion to compute chromatic beta functions and phase advances. For a thin sextupole of
integrated strength 1 m⁻² followed by a 1 m drift, take entrance beta functions βₓ = 3 m and
βᵧ = 2 m, alpha αₓ = 0.4, horizontal dispersion ηₓ = 0 and slope ηₓ′ = 0.2. The analytic drift
Jacobian and tracking give zero derivatives with respect to relative momentum offset δ, but the
output gives dβₓ/dδ = −1.040 m, dβᵧ/dδ = 0.8000 m, dνₓ/dδ = 0.01231 and dνᵧ/dδ = −0.01273, where ν
is phase advance in cycles. Users matching chromatic functions in dispersive transport lines are
affected.

The script transports the entrance optics through the sextupole and drift in serial Elegant (and
with `--mpi` in 3-rank Pelegant), and compares the four written derivatives with those of the beta
functions and phase advances reconstructed from tracked particle pairs around the dispersive
reference orbit at δ = -0.01, 0 and 0.01. Two controls, a zero-strength sextupole and a zero
entrance dispersion slope, remove the product of dispersion and quadratic focusing that the
contraction needs; both must give zero.
"""

import sys

import numpy as np
from _elegant import modes, particles, run, verdict

P0 = 1000.0  # reference momentum, units of m_e c
ENTRANCE = {"x": (3.0, 0.4), "y": (2.0, 0.0)}  # (beta, alpha) at the entrance, in m
ETA0 = 0.0  # entrance dispersion (m); the slope is varied between cases
DRIFT = 1.0  # m
DELTAS = (-0.01, 0.0, 0.01)
PROBE = 1e-5  # half-step of the Jacobian probes
COPIES = 2  # copies of each particle, so that each MPI worker tracks some
ATOL = 1e-10


def commands(etap):
    return (
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, default_order=2,\n"
        " output=out.sdds &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&twiss_output filename=twiss.sdds, concat_order=2, matched=0,\n"
        f" beta_x={ENTRANCE['x'][0]!r}, alpha_x={ENTRANCE['x'][1]!r},\n"
        f" beta_y={ENTRANCE['y'][0]!r}, alpha_y={ENTRANCE['y'][1]!r},\n"
        f" eta_x={ETA0!r}, etap_x={etap!r} &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )


def probe_beam(etap):
    """Eight particles per delta, at +-PROBE in x, xp, y and yp around the reference orbit.

    The reference orbit of momentum P0*(1 + delta) starts at (eta*delta, eta'*delta) in x and xp,
    the entrance dispersion times delta, and on axis in y.
    """
    rows = []
    for delta in DELTAS:
        for axis in range(4):
            for sign in (1, -1):
                coordinates = np.zeros(4)
                coordinates[axis] = sign * PROBE
                coordinates[:2] += (ETA0 * delta, etap * delta)
                rows.append((*coordinates, P0 * (1 + delta)))
    x, xp, y, yp, p = np.repeat(np.array(rows), COPIES, axis=0).T
    return {"x": x, "xp": xp, "y": y, "yp": yp, "p": p, "particleID": np.arange(1, len(x) + 1)}


def tracked_optics(out):
    """Exit beta function and phase advance (in cycles) of each plane at each delta.

    Symmetric differences about the reference orbit give the transfer matrix J at each delta.
    With the entrance beta0 and alpha0 and gamma0 = (1 + alpha0^2)/beta0,
    beta1 = J11^2 beta0 - 2 J11 J12 alpha0 + J12^2 gamma0 and the phase advance is
    atan2(J12, J11 beta0 - J12 alpha0) / (2 pi).
    """
    coordinates = np.column_stack([out[name] for name in ("x", "xp", "y", "yp")])
    if not np.array_equal(coordinates, np.repeat(coordinates[::COPIES], COPIES, axis=0)):
        sys.exit("Identical particles left the element differently.")
    # Index: delta, probed axis (x, xp, y, yp), sign (+, -), output coordinate.
    exit_ = coordinates[::COPIES].reshape(len(DELTAS), 4, 2, 4)
    jacobian = (exit_[:, :, 0] - exit_[:, :, 1]) / (2 * PROBE)  # [delta, probed axis, output]
    optics = {}
    for plane, (beta0, alpha0) in ENTRANCE.items():
        first = 0 if plane == "x" else 2  # index of the plane's position coordinate
        j11 = jacobian[:, first, first]  # d position_out / d position_in
        j12 = jacobian[:, first + 1, first]  # d position_out / d slope_in
        gamma0 = (1 + alpha0**2) / beta0
        beta = j11**2 * beta0 - 2 * j11 * j12 * alpha0 + j12**2 * gamma0
        phase = np.arctan2(j12, j11 * beta0 - j12 * alpha0) / (2 * np.pi)
        optics[plane] = (beta, phase)
    return optics


def case(mode, knl, etap):
    """Written (dbetax/dp, dbetay/dp, dnux/dp, dnuy/dp), the tracked derivatives and exit betas."""
    lattice = f"S: MULT,ORDER=2,KNL={knl!r}\nD: DRIF,L={DRIFT!r}\nBL: LINE=(S,D)\n"
    result = run(
        {"lattice.lte": lattice, "run.ele": commands(etap)},
        mode,
        {"beam.sdds": probe_beam(etap)},
        outputs=("twiss.sdds", "out.sdds"),
    )
    twiss = result["twiss.sdds"]
    written = [float(twiss[name][0]) for name in ("dbetax/dp", "dbetay/dp", "dnux/dp", "dnuy/dp")]
    optics = tracked_optics(particles(result["out.sdds"]))
    span = DELTAS[2] - DELTAS[0]
    tracked = [(optics[p][k][2] - optics[p][k][0]) / span for k in (0, 1) for p in "xy"]
    return written, tracked, [optics[p][0][1] for p in "xy"]


def fmt(values):
    return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


def compare(label, observed, expected):
    """Print one case; return True when observed and expected disagree."""
    observed, expected = np.asarray(observed), np.asarray(expected)
    if not (np.isfinite(observed).all() and np.isfinite(expected).all()):
        sys.exit(f"{label}: nonfinite value, the comparison is not valid.")
    bad = bool(np.any(np.abs(observed - expected) > ATOL))
    print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
    return bad


# The reference orbit of any momentum crosses the sextupole at x = eta*delta = 0, where a thin
# sextupole kick (proportional to x^2 or x*y) and its derivatives (proportional to x) vanish. The
# map is therefore the drift [[1, L], [0, 1]] in each plane for every delta, so the exit beta
# function beta0 - 2 L alpha0 + L^2 gamma0 and the phase advance atan2(L, beta0 - L alpha0)/(2 pi)
# do not depend on delta: all four derivatives are zero.
EXPECTED_BETA = [
    b - 2 * DRIFT * a + DRIFT**2 * (1 + a**2) / b for b, a in (ENTRANCE["x"], ENTRANCE["y"])
]

defect = False
for mode in modes():
    for label, knl, etap in (
        ("control KNL=0", 0.0, 0.2),
        ("control etap_x=0", 1.0, 0.0),
        ("KNL=1, etap_x=0.2", 1.0, 0.2),
    ):
        written, tracked, exit_beta = case(mode, knl, etap)
        if compare(f"{mode} {label}: tracked exit beta x, y", exit_beta, EXPECTED_BETA):
            sys.exit("The tracked beta function disagrees with the analytic map: invalid oracle.")
        if compare(f"{mode} {label}: tracked dbeta/ddelta x, y, dnu/ddelta x, y", tracked, [0] * 4):
            sys.exit("The tracked derivatives are not zero: invalid oracle.")
        bad = compare(
            f"{mode} {label}: written dbetax/dp, dbetay/dp, dnux/dp, dnuy/dp", written, [0] * 4
        )
        if knl == 0.0 or etap == 0.0:
            if bad:
                sys.exit("A control has nonzero derivatives: the comparison is not valid.")
            continue
        defect |= bad
        # Elegant contracts the second-order map T with the exit dispersion eta_exit = eta0 + L
        # eta0' instead of the entrance dispersion. For the combined x map on y = 0,
        # x1 = x + L x' - L KNL x^2 / 2 (T111 = -L KNL/2, R11 = 1, R12 = L),
        # dR11/ddelta = 2 T111 eta_exit; with dR12/ddelta = 0,
        # d beta1/d delta = 2 (dR11/ddelta) (beta0 - L alpha0) = -1.04 m.
        eta_exit = ETA0 + DRIFT * etap
        predicted = 2 * (-DRIFT * knl * eta_exit) * (ENTRANCE["x"][0] - DRIFT * ENTRANCE["x"][1])
        print(f"{mode} {label}: exit-dispersion contraction predicts dbetax/dp={predicted:.16g}")
print(f"comparison: absolute tolerance {ATOL:g}")
verdict(defect)
