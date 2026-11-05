---
title: "DOC-05: Automatic CSBEND field expansion order is n+2, not the n+3 the manual states"
hide:
  - toc
---

# DOC-05 — Automatic CSBEND field expansion order is n+2, not the n+3 the manual states

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`computeCSBENDFieldCoefficients`, csbend.c:313–330](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L313-L330)<br>[evaluator, csbend.c:127–170](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L127-L170)<br>[`Kn` to `b[n]`, csbend.c:1029–1048](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1029-L1048)<br>[coefficient setup, csbend.c:1198](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1198)<br>[CSRCSBEND setup, csbend.c:2889](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2889)<br>[manual, csbend.tex:42–51](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L42-L51) |

For a CSBEND with EXPANSION_ORDER=0, the manual promises field terms through total degree n+3 in transverse coordinates x and y, with a minimum degree of 4, where n is the highest nonzero multipole order. For normal sextupole (n = 2) and octupole (n = 3) components, tracking instead matches explicit order n+2. Independent vacuum-field integration validates the documented-order trajectories. In a 0.25 m bend of curvature 1 m⁻¹ with K₃ = 3000 m⁻⁴, particles starting at x = 15–30 mm and y = 20–40 mm with zero slopes acquire slope differences up to 2.823×10⁻⁷. The tested quadrupole-only and midplane controls are unaffected.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)

??? example "Reproducer: DOC-05.py"

    [Download DOC-05.py](DOC-05.py)

    ```python
    """DOC-05: Automatic CSBEND field expansion order is n+2, not the n+3 the manual states

    For a CSBEND with EXPANSION_ORDER=0, the manual promises field terms through total degree n+3 in
    transverse coordinates x and y, with a minimum degree of 4, where n is the highest nonzero multipole
    order. For normal sextupole (n = 2) and octupole (n = 3) components, tracking instead matches
    explicit order n+2. Independent vacuum-field integration validates the documented-order
    trajectories. In a 0.25 m bend of curvature 1 m⁻¹ with K₃ = 3000 m⁻⁴, particles starting at x =
    15–30 mm and y = 20–40 mm with zero slopes acquire slope differences up to 2.823×10⁻⁷. The tested
    quadrupole-only and midplane controls are unaffected.

    The script tracks particles through a CSBEND with a K2 or K3 term at the automatic expansion order
    and at explicit orders, and compares the automatic result with the documented order n+3 and with
    order n+2. An independent integration of the vacuum field of the documented order, derived from the
    curved-coordinate Laplace equation, validates the documented-order tracking. The controls are a
    quadrupole-only bend and midplane particles, where the omitted terms do not act.
    """

    import sys
    from math import factorial

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from numpy.polynomial.polynomial import polyder, polymul, polyval
    from scipy.integrate import solve_ivp

    LENGTH, ANGLE, SLICES = 0.25, 0.25, 512
    H = ANGLE / LENGTH  # curvature, 1/m
    P0 = 30.0  # m_e c; particles start on momentum
    N = 12
    X = np.linspace(0.015, 0.03, N)  # m
    Y = np.linspace(0.02, 0.04, N)  # m
    ATOL = 2e-14  # automatic and explicit order are the same computation when the degrees agree
    ODE_TOL = 1e-12  # native explicit-order tracking against the independent integration


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def documented_order(n):
        # doc/csbend.tex: n + 3 for the highest nonzero multipole order n, but never less than 4.
        return max(4, n + 3)


    def field(n, kn, degree):
        """Vacuum field (Fx, Fy)/B0 of a curved dipole with one K_n term, as polynomials of degree D.

        In curvilinear coordinates a longitudinally invariant scalar potential obeys
        Psi_xx + h/(1 + h x) Psi_x + Psi_yy = 0. Midplane symmetry gives Psi = sum_j f_j(x) y^(2j+1)
        with f_0 = 1 + K_n x^n/(h n!) (the dipole plus the multipole, in units of the dipole field),
        and the Laplace equation gives f_(j+1) = -(f_j'' + h/(1 + h x) f_j')/((2j + 3)(2j + 2)),
        expanded here as a power series in x (1/(1 + h x) = sum (-h x)^k). Truncating Psi at degree D+1
        makes the field Fx = dPsi/dx, Fy = dPsi/dy exact through degree D, which is what
        EXPANSION_ORDER=D keeps. Returns the polynomial coefficient arrays bx[i, j] x^i y^j of Fx
        and by[i, j] of Fy.
        """
        size = degree + 2
        f = np.zeros(size)
        f[0] = 1.0
        f[n] = kn / H / factorial(n)
        bx = np.zeros((size, size))
        by = np.zeros((size, size))
        for j in range((degree + 1) // 2 + 1):
            power = 2 * j + 1  # power of y multiplying f
            if power > degree + 1:
                break
            f[degree + 2 - power :] = 0.0  # terms that would exceed degree D+1 in Psi
            for i, a in enumerate(f):
                if i and i - 1 + power <= degree:
                    bx[i - 1, power] = i * a
                if i + power - 1 <= degree:
                    by[i, power - 1] = power * a
            d1, d2 = polyder(f), polyder(f, 2)
            operator = polymul(H * (-H) ** np.arange(size), d1)[:size]
            operator = np.pad(operator, (0, size - len(operator)))
            operator[: len(d2)] += d2
            f = -operator / ((power + 2) * (power + 1))
        return bx, by


    def trajectory(n, kn, degree):
        """Exit (x, xp, y, yp) of each particle from integrating the Lorentz force in the field."""
        bx, by = field(n, kn, degree)
        result = []
        for x0, y0 in zip(X, Y, strict=True):
            # State (x, qx, y, qy): qx, qy are the transverse canonical momenta in units of P0 (for
            # p = P0, so w = 1); qs = sqrt(w^2 - qx^2 - qy^2); dx/ds = (1 + h x) qx/qs;
            # dqx/ds = h qs - h (1 + h x) Fy; dqy/ds = h (1 + h x) Fx, with fields in units of B0.
            def rhs(_s, z):
                x, qx, y, qy = z
                qs = np.sqrt(1 - qx * qx - qy * qy)
                fx = polyval(y, polyval(x, bx))
                fy = polyval(y, polyval(x, by))
                m = 1 + H * x
                return [m * qx / qs, H * qs - H * m * fy, m * qy / qs, H * m * fx]

            solution = solve_ivp(
                rhs, (0, LENGTH), [x0, 0, y0, 0], method="DOP853", rtol=3e-13, atol=2e-15
            )
            # An unfinished or non-physical integration is an invalid run, not an exit trajectory.
            if not solution.success or solution.t[-1] != LENGTH or not np.all(np.isfinite(solution.y)):
                sys.exit(f"integration for K{n}={kn:g}, degree {degree} failed: {solution.message}")
            x, qx, y, qy = solution.y[:, -1]
            qs2 = 1 - qx * qx - qy * qy
            if not (np.isfinite(qs2) and qs2 > 0):
                sys.exit(f"integration for K{n}={kn:g}, degree {degree} ends with 1 - qx^2 - qy^2 <= 0")
            qs = np.sqrt(qs2)
            result.append([x, qx / qs, y, qy / qs])
        return np.array(result)


    def track(mode, n, kn, order, y_scale=1.0):
        """Exit (x, xp, y, yp) per particle for EXPANSION_ORDER=order (0 selects automatically)."""
        lattice = (
            f"B: CSBEND,L={LENGTH!r},ANGLE={ANGLE!r},K{n}={kn!r},EXPANSION_ORDER={order},"
            f"N_SLICES={SLICES},INTEGRATION_ORDER=4,EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,SYNCH_RAD=0,"
            "ISR=0,REFERENCE_CORRECTION=0\nBL: LINE=(B)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n&track &end\n"
        )
        beam = {"p": np.full(N, P0), "x": X, "y": y_scale * Y, "particleID": np.arange(1, N + 1)}
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})
        out = particles(result["out.sdds"])
        if len(out["p"]) != N:
            sys.exit(f"{mode}: {N - len(out['p'])} particles lost")
        return np.column_stack([out[c] for c in ("x", "xp", "y", "yp")])


    def compare(label, observed, expected, tol):
        """Print the last particle's (x, xp, y, yp); return True when any particle differs by > tol."""
        if not (np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))):
            sys.exit(f"{label}: nonfinite comparison operand: invalid run")
        worst = np.max(np.abs(observed - expected))
        print(
            f"{label}: observed={fmt(observed[-1])} expected={fmt(expected[-1])} "
            f"max|difference|={worst:.6g}"
        )
        return bool(worst > tol)


    defect = []
    print("columns: x (m), xp, y (m), yp of the last particle; expected at the documented order")
    for mode in modes():
        for n, kn in ((2, 30.0), (3, 3000.0), (3, 1500.0)):
            label = f"{mode} K{n}={kn:g}"
            documented = documented_order(n)
            automatic = track(mode, n, kn, 0)
            # The independent oracle first validates the documented-order native run, so that the
            # documented-order tracking can serve as the expectation for the automatic one.
            explicit = track(mode, n, kn, documented)
            if compare(
                f"{label}, explicit order {documented} vs integration (control)",
                explicit,
                trajectory(n, kn, documented),
                ODE_TOL,
            ):
                sys.exit("The explicit-order run disagrees with the integration: invalid comparison.")
            defect.append(
                compare(f"{label}, automatic vs explicit order {documented}", automatic, explicit, ATOL)
            )
            # The automatic order is the same computation as explicit order n+2.
            lower = track(mode, n, kn, max(4, n + 2))
            if compare(f"{label}, automatic vs explicit order {max(4, n + 2)}", automatic, lower, ATOL):
                sys.exit("The automatic order is not n+2: the diagnosis does not hold.")
        # Controls: a quadrupole-only bend selects 4 under both rules, and on the midplane the omitted
        # terms (odd in y for Fx, y-dependent for Fy) vanish.
        if compare(
            f"{mode} K1=0.7, automatic vs explicit order 4 (control)",
            track(mode, 1, 0.7, 0),
            track(mode, 1, 0.7, 4),
            ATOL,
        ):
            sys.exit("The quadrupole control differs: invalid comparison.")
        midplane = track(mode, 3, 3000.0, 0, y_scale=0.0)
        if compare(
            f"{mode} K3=3000, y=0, automatic vs explicit order 6 (control)",
            midplane,
            track(mode, 3, 3000.0, 6, y_scale=0.0),
            ATOL,
        ):
            sys.exit("The midplane control differs: invalid comparison.")
    print(f"comparison: automatic against explicit order to {ATOL:g}; integration to {ODE_TOL:g}")
    verdict(all(defect))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

