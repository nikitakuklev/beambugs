"""TRK-10: CSBEND edge model 3 is documented as symplectic but its finite-gap kick is not six-dimensionally symplectic

The manual recommends CSBEND edge model 3 for applications requiring symplecticity, without
restricting the claim to transverse motion at fixed energy. With a finite pole gap, its
momentum-dependent vertical kick omits the time shift needed for a six-dimensional canonical map. In a 0.9 m, 0.23 rad bend, a particle with relative momentum offset δ = 0.15 and vertical
offset y = 20 mm receives no exit-edge path shift, while canonical completion requires
Δs = −2.222×10⁻⁷ m. The measured six-dimensional symplecticity residual is 2.224×10⁻⁵, despite a
transverse residual below 3.5×10⁻¹³. The completion is not a measured physical fringe delay.

The script tracks 13 electrons, a central particle and a finite-difference star around it in the
canonical coordinates (x, p_x/p0, y, p_y/p0, ct, -E/(p0 c)), through a CSBEND with the exit edge
off and with exit edge model 3. The Jacobian of the exit edge alone, from the paired runs, is
tested against the symplectic form, and the path shift that the kick's generating potential
requires is compared with the observed one. Controls are zero gap and zero vertical offset, where
the mixed symplecticity defect vanishes, the transverse block at fixed energy, and the Jacobian
completed with the required shift.
"""

import sys

import numpy as np
from _elegant import modes, particles, run, verdict

C = 299792458.0  # m/s, c_mks of SDDS
P0, LENGTH, ANGLE, EDGE, FINT = 20.0, 0.9, 0.23, 0.13, 0.6  # m_e c, m, rad, rad
STEP = 2e-5  # finite-difference step in the canonical coordinates
W = 1.15  # 1 + delta of the central particle
X, XP = 0.006, 0.025  # central particle: x (m) and slope xp (yp = 0)
# (label, HGAP in m, central y in m)
CASES = [
    ("base", 0.018, 0.02),
    ("double gap", 0.036, 0.02),
    ("zero gap", 0.0, 0.02),
    ("zero y", 0.018, 0.0),
]
# Symplectic form for the pairs (x, px), (y, py), (ct, eta) with eta = -E/(p0 c).
FORM = np.kron(np.eye(3), np.array([[0.0, 1.0], [-1.0, 0.0]]))
TRANSVERSE = FORM[:4, :4]


def canonical(out):
    """Rows (x, px/p0, y, py/p0, ct, -E/(p0 c)) with the exact momentum and energy conversions."""
    p = out["p"]
    f = p / P0 / np.sqrt(1 + out["xp"] ** 2 + out["yp"] ** 2)
    return np.column_stack(
        [out["x"], f * out["xp"], out["y"], f * out["yp"], C * out["t"], -np.hypot(p, 1) / P0]
    )


def jacobian(values):
    """Central differences of the star: rows 1.. are center +/- STEP along axis 0..5 in turn."""
    return ((values[1::2] - values[2::2]) / (2 * STEP)).T


def star(y):
    """Central particle and +/- STEP along each canonical axis, as a beam for run()."""
    # eta = -sqrt(w^2 + 1/p0^2) is minus the energy over p0 c for momentum w p0 (units of m_e c).
    center = np.array([X, W * XP / np.sqrt(1 + XP**2), y, 0.0, 0.0, -np.hypot(W, 1 / P0)])
    points = np.vstack(
        [center, *[center + s * STEP * np.eye(6)[a] for a in range(6) for s in (1, -1)]]
    )
    p = P0 * np.sqrt(points[:, 5] ** 2 - 1 / P0**2)
    ps = np.sqrt((p / P0) ** 2 - points[:, 1] ** 2 - points[:, 3] ** 2)
    return {
        "x": points[:, 0],
        "xp": points[:, 1] / ps,
        "y": points[:, 2],
        "yp": points[:, 3] / ps,
        "t": points[:, 4] / C,
        "p": p,
    }


def track(mode, gap, beam, exit_model):
    lattice = (
        f"B: CSBEND,L={LENGTH},ANGLE={ANGLE},E2={EDGE},HGAP={gap!r},FINT={FINT},"
        f"EDGE1_EFFECTS=0,EDGE2_EFFECTS={exit_model},N_SLICES=192,INTEGRATION_ORDER=4,"
        "REFERENCE_CORRECTION=0,SYNCH_RAD=0,ISR=0\n"
        "BL: LINE=(B)\n"
    )
    commands = (
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0}, output=out.sdds, "
        "default_order=2 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track soft_failure=0 &end\n"
    )
    out = particles(
        run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
    )
    if len(out["p"]) != len(beam["p"]):
        sys.exit(f"{mode}: particles were lost")
    if not all(np.all(np.isfinite(v)) for v in out.values()):
        sys.exit(f"{mode}: nonfinite output coordinates")
    return out


def residual(j, form):
    return float(np.max(np.abs(j.T @ form @ j - form)))


defect = False
for mode in modes():
    for label, gap, y in CASES:
        beam = star(y)
        off, on = track(mode, gap, beam, 0), track(mode, gap, beam, 3)
        c_off, c_on = canonical(off), canonical(on)
        # The exit edge alone: J_on = J_edge J_off, so J_edge = J_on inverse(J_off).
        j_off = jacobian(c_off)
        j_edge = jacobian(c_on) @ np.linalg.inv(j_off)
        # Edge kick of model 3 (csbend.c, applySimpleDipoleEdgeKick): Delta py = -a(w) y with
        # a = h tan(E2 - psi/w), h = angle/L, w = 1 + delta, and the finite-gap angle
        # psi = 2 HGAP FINT h (1 + sin^2 E2)/cos E2. It derives from V = a(w) y^2/2, which depends on
        # energy, so canonical conjugacy of (ct, eta) requires a shift of ct: with
        # d(eta)/dw = -beta, i.e. dw/d(eta) = -1/beta, for eta = -sqrt(w^2 + 1/p0^2) = -E/(p0 c):
        # Delta ct = -a'(w) y^2/(2 beta), i.e. a path shift Delta s = beta Delta ct = -a'(w) y^2/2
        # with a' = h psi sec^2(E2 - psi/w)/w^2.
        w = off["p"] / P0
        beta = off["p"] / np.hypot(off["p"], 1)
        h = ANGLE / LENGTH
        psi = 2 * gap * FINT * h * (1 + np.sin(EDGE) ** 2) / np.cos(EDGE)
        a = h * np.tan(EDGE - psi / w)
        a_prime = h * psi / (np.cos(EDGE - psi / w) ** 2 * w**2)
        expected_ds = -a_prime * off["y"] ** 2 / 2
        observed_ds = (on["t"] - off["t"]) * C * beta
        # The mixed derivative of the missing time term gives the 6D residual |a' y/beta|.
        predicted = float(abs(a_prime[0] * off["y"][0] / beta[0]))
        measured = residual(j_edge, FORM)
        # Controls: the kick is the transverse impulse -a y; the transverse block at fixed energy
        # is symplectic; adding the predicted shift to ct restores 6D symplecticity.
        kick_error = np.max(np.abs((c_on[:, 3] - c_off[:, 3]) + a * off["y"]))
        transverse = residual(j_edge[:4, :4], TRANSVERSE)
        # Offline completion with the shift the edge lacks: zero where the edge already applies it.
        missing_ds = expected_ds - observed_ds
        completed = c_on.copy()
        completed[:, 4] += missing_ds / beta
        j_completed = jacobian(completed) @ np.linalg.inv(j_off)
        completed_residual = residual(j_completed, FORM)
        arrays = (c_off, c_on, completed, j_off, j_edge, j_completed, expected_ds, observed_ds)
        scalars = (predicted, measured, transverse, completed_residual, kick_error)
        if not (all(np.all(np.isfinite(v)) for v in arrays) and np.all(np.isfinite(scalars))):
            sys.exit(f"{mode} {label}: nonfinite coordinates, Jacobian or residual")
        print(
            f"{mode} {label}: observed Delta_s={float(observed_ds[0])!r} "
            f"expected Delta_s={float(expected_ds[0]) + 0.0!r} m; observed 6D residual={measured:.10g} "
            f"expected={predicted:.10g}; transverse residual={transverse:.3g}; "
            f"completed residual={completed_residual:.3g}"
        )
        if kick_error > 2e-14:
            sys.exit(f"{mode} {label}: the edge is not the transverse kick alone")
        if transverse > 1e-10 or completed_residual > 1e-10:
            sys.exit(f"{mode} {label}: the transverse or completed control is not symplectic")
        if np.max(np.abs(missing_ds)) <= 2e-15:
            # The edge applies the required shift (zero where the required shift is zero).
            if measured > 1e-10:
                sys.exit(f"{mode} {label}: the edge applies the shift but is not symplectic")
            continue
        if np.max(np.abs(observed_ds)) > 2e-15:
            sys.exit(f"{mode} {label}: the edge applies a shift other than the required one")
        if abs(measured - predicted) > 1e-10:
            sys.exit(f"{mode} {label}: the 6D residual is not the predicted one")
        if label == "base":
            defect |= measured > 1e-5 and abs(expected_ds[0]) > 1e-8
print("comparison: residuals to 1e-10, kick to 2e-14 and Delta_s to 2e-15")
verdict(defect)
