---
title: "TRK-13: CSBEND spin tracking passes canonical momenta to the slope-based spin update"
hide:
  - toc
---

# TRK-13 — CSBEND spin tracking passes canonical momenta to the slope-based spin update

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`integrate_csbend_ordn`, csbend.c:1930–1935](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1930-L1935)<br>[`updateSpinQuaternionLocalFS`, spinUpdate.c:47–125](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/spinUpdate.c#L47-L125)<br>[`convertToDipoleCanonicalCoordinates`, csbend.c:1733–1741](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1733-L1741)<br>[radiation branch, csbend.c:1938–1950](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1938-L1950)<br>[spin_tracking, elegant.tex:7291–7292](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7291-L7292)<br>[CSBEND among spin-aware elements, elegant.tex:11191–11193](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L11191-L11193) |

With spin tracking enabled, CSBEND's unexpanded-Hamiltonian integrator passes canonical momenta to a spin routine that expects slopes, giving incorrect electron spin precession. For initial spin (0, 0, 1) in a planar 1.2 m, 0.22 rad bend at reference momentum p₀ = 800 mₑc, relative momentum offset δ = 0.4 and entrance slope x′ = 0.16, exact circular motion and Thomas–BMT predict an exit spin angle of −0.1552 rad relative to the exit reference axis. Tracking converges to −0.1605 rad, an error of 5.297 mrad that slice refinement cannot remove. The on-axis, on-momentum control agrees.

Run `python TRK-13.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-13.py"

    [Download TRK-13.py](TRK-13.py)

    ```python
    """TRK-13: CSBEND spin tracking passes canonical momenta to the slope-based spin update

    With spin tracking enabled, CSBEND's unexpanded-Hamiltonian integrator passes canonical momenta to
    a spin routine that expects slopes, giving incorrect electron spin precession. For initial spin
    (0, 0, 1) in a planar 1.2 m, 0.22 rad bend at reference momentum p₀ = 800 mₑc, relative momentum
    offset δ = 0.4 and entrance slope x′ = 0.16, exact circular motion and Thomas–BMT predict an exit
    spin angle of −0.1552 rad relative to the exit reference axis. Tracking converges to −0.1605 rad,
    an error of 5.297 mrad that slice refinement cannot remove. The on-axis, on-momentum control
    agrees.

    The script tracks 12 electrons with spin (0, 0, 1) through a pure CSBEND at several slice counts and
    compares each final spin angle with the exact uniform-field result: the Thomas–BMT precession along
    the exact circle, cut by the exit plane. The first-order sampling error of the spin update is
    removed by Richardson extrapolation in the slice count, and the extrapolated angle is also compared
    with the limit of the same integral computed with canonical momenta in place of slopes. An on-axis,
    on-momentum particle is the control.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from scipy.integrate import quad

    C = 299792458.0  # m/s, c_mks of SDDS
    A_E = 0.00115965218128  # electron anomalous magnetic moment, particleAnomalousMagneticMoment
    P0, LENGTH, ANGLE = 800.0, 1.2, 0.22  # m_e c, m, rad
    SLICES = (64, 256, 1024, 4096)
    TOL = 1e-9  # rad; Richardson residual and quadrature are far below this
    H = ANGLE / LENGTH  # curvature of the reference orbit

    # 12 electrons: three momentum offsets times four entrance slopes, zero initial spin tilt.
    DELTA = np.repeat([-0.3, 0.0, 0.4], 4)
    XP = np.tile([0.0, 0.04, 0.08, 0.16], 3)
    P = P0 * (1 + DELTA)
    N = len(P)
    CONTROL, SMALL, LARGE = 4, 5, 11  # rows: on-axis; delta = 0, xp = 0.04; delta = 0.4, xp = 0.16


    def exact():
        """Exit orbit (x, xp, c*t) and spin angle from the exact uniform-field circle.

        A particle of relative momentum w = 1 + delta moves on a circle of curvature k = h/w. Starting
        at angle alpha = atan(xp) to the reference direction, after turning through `turn` it is at
        lab direction alpha - turn. The exit plane makes angle ANGLE with the entrance plane, so the
        exit slope angle phi satisfies sin(phi) = sin(alpha + ANGLE) - k sin(ANGLE)/h; the turn is
        alpha + ANGLE - phi. The field is perpendicular to the velocity, so the Thomas-BMT equation
        rotates the spin by -(1 + a*gamma) times the particle's turn in the lab frame; the exit
        reference frame adds ANGLE.
        """
        k = H / (1 + DELTA)
        alpha = np.arctan(XP)
        phi = np.arcsin(np.sin(alpha + ANGLE) - k * np.sin(ANGLE) / H)
        turn = alpha + ANGLE - phi
        distance = turn / k
        lab_x = (np.cos(alpha - turn) - np.cos(alpha)) / k
        lab_z = (np.sin(alpha) - np.sin(alpha - turn)) / k
        x = (lab_x - (np.cos(ANGLE) - 1) / H) * np.cos(ANGLE) + (lab_z - np.sin(ANGLE) / H) * np.sin(
            ANGLE
        )
        ct = distance * np.sqrt(1 + P * P) / P  # c*t = path/beta, beta = p/sqrt(1 + p^2)
        spin = ANGLE - (1 + A_E * np.sqrt(1 + P * P)) * turn
        return x, np.tan(phi), ct, spin


    def canonical_limit():
        """Spin angle when the dipole-canonical momentum Q = w*sin(phi) replaces the slope.

        The spin update multiplies the precession rate by dl/ds = sqrt((1 + h x)^2 + xp^2 + yp^2) and
        divides by w. Here xp is the argument it is given: Q. The integral is taken along the exact
        orbit of `exact`, so only the wrong argument differs.
        """
        result = []
        for p, w, a in zip(P, 1 + DELTA, np.arctan(XP), strict=True):

            def integrand(theta, w=w, a=a):
                # Orbit at reference arc angle theta (s = theta/h): same circle as in exact().
                phi = np.arcsin(np.sin(a + theta) - np.sin(theta) / w)
                turn = a + theta - phi
                x = (np.cos(a - turn) - np.cos(a)) * np.cos(theta) * w / H
                x += (np.sin(a) - np.sin(a - turn)) * np.sin(theta) * w / H
                x -= (1 - np.cos(theta)) / H
                return np.sqrt((1 + H * x) ** 2 + (w * np.sin(phi)) ** 2)

            integral = quad(integrand, 0, ANGLE, epsabs=1e-13, epsrel=1e-13)[0]
            result.append(ANGLE - (1 + A_E * np.sqrt(1 + p * p)) / w * integral)
        return np.array(result)


    def track(mode, slices):
        lattice = (
            f"B: CSBEND,L={LENGTH!r},ANGLE={ANGLE!r},N_SLICES={slices},INTEGRATION_ORDER=4,"
            "EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,SYNCH_RAD=0,ISR=0,REFERENCE_CORRECTION=0\n"
            "BL: LINE=(B)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds,"
            " spin_tracking=1 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        zero = np.zeros(N)
        beam = {
            "xp": XP,
            "p": P,
            "spx": zero,
            "spy": zero,
            "spz": np.ones(N),
            "particleID": np.arange(1, N + 1),
        }
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if len(out["p"]) != N:
            sys.exit(f"{mode}: {N - len(out['p'])} particles lost")
        return out


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def require_finite(what, *arrays):
        if not all(np.all(np.isfinite(a)) for a in arrays):
            sys.exit(f"{what}: nonfinite values")


    def compare(label, observed, expected, tol):
        """Print one case; return True when observed and expected differ by more than tol."""
        bad = bool(np.any(np.abs(np.asarray(observed) - np.asarray(expected)) > tol))
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    exact_x, exact_xp, exact_ct, exact_angle = exact()
    wrong_limit = canonical_limit()
    require_finite("oracle", exact_x, exact_xp, exact_ct, exact_angle, wrong_limit)
    defect = False
    print("columns: spin angle atan2(sx, sz) in rad, expected from the exact circle and Thomas-BMT")
    for mode in modes():
        angles = {}
        for slices in SLICES:
            out = track(mode, slices)
            spin = np.column_stack([out["spx"], out["spy"], out["spz"]])
            require_finite(f"{mode} N={slices}", *(out[k] for k in ("x", "xp", "t", "p")), spin)
            if np.max(np.abs(np.linalg.norm(spin, axis=1) - 1)) > 1e-12:
                sys.exit(f"{mode} N={slices}: the spin norm is not 1")
            angle = np.arctan2(out["spx"], out["spz"])
            require_finite(f"{mode} N={slices}: spin angle", angle)
            angles[slices] = angle
            # The orbit must match the same circle, or the spin comparison would be invalid.
            orbit = max(
                np.max(np.abs(out["x"] - exact_x)),
                np.max(np.abs(out["xp"] - exact_xp)),
                np.max(np.abs(C * out["t"] - exact_ct)),
            )
            if orbit > 2e-10:
                sys.exit(f"{mode} N={slices}: orbit differs from the exact circle by {orbit:g}")
            for row in (CONTROL, SMALL, LARGE):
                label = f"{mode} N={slices} delta={DELTA[row]:+.1f} xp={XP[row]}"
                print(f"{label}: observed={fmt(angle[row])} expected={fmt(exact_angle[row])}")
        # The spin is sampled after each kick, which leaves an O(1/N) error; remove it by
        # extrapolating the two finest results (4 f(4096) - f(1024))/3.
        limit = (4 * angles[4096] - angles[1024]) / 3
        if compare(
            f"{mode} control delta=+0.0 xp=0.0, N=4096",
            angles[4096][CONTROL],
            exact_angle[CONTROL],
            TOL,
        ):
            sys.exit("The on-axis control disagrees with the exact angle: the comparison is not valid.")
        require_finite(f"{mode}: extrapolated angle", limit)
        for row in (SMALL, LARGE):
            label = f"{mode} extrapolated delta={DELTA[row]:+.1f} xp={XP[row]}"
            compare(label, limit[row], exact_angle[row], TOL)
        label = (
            f"{mode} extrapolated delta={DELTA[LARGE]:+.1f} xp={XP[LARGE]} vs canonical-momentum limit"
        )
        compare(label, limit[LARGE], wrong_limit[LARGE], TOL)
        # A particle that matches the exact angle is correct. One that does not must follow the limit
        # with canonical momenta in place of slopes; any other result is a failed run, not a verdict.
        wrong = np.abs(limit - exact_angle) > TOL
        spread = np.max(np.abs(limit - wrong_limit)[wrong], initial=0.0)
        print(
            f"{mode} max |extrapolated - canonical-momentum limit| over {wrong.sum()} of {N} "
            f"particles that differ from the exact angle: {spread:.3g}"
        )
        if spread > TOL:
            sys.exit("The extrapolated angles match neither the exact angle nor the canonical limit.")
        defect |= bool(wrong.any())
    print(f"comparison: abs tolerance {TOL:g} rad on extrapolated and control angles")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source reading, not executed for this claim |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

