---
title: "TRK-40: KQUAD with RADIAL=1 radiates from the unscaled field shape instead of the applied kick"
hide:
  - toc
---

# TRK-40 — KQUAD with RADIAL=1 radiates from the unscaled field shape instead of the applied kick

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`, head build) |
| Source | [`applyRadialCanonicalMultipoleKicks`, multipole.c:1584–1616](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1584-L1616)<br>[`integrate_kick_multipole_ordn`, multipole.c:1437–1440](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1437-L1440)<br>[multipole.c:1488–1510](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1488-L1510)<br>[multipole.h:205–208](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.h#L205-L208)<br>[track_data.c:1305](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1305)<br>[kquad.tex:119–123](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/kquad.tex#L119-L123) |

A KQUAD with RADIAL=1, a radially focusing lens, computes its classical radiation from field shapes that are not scaled by the strength K₁, so it radiates at K₁ = 0 and the loss grows with the square of N_SLICES. For a 3.066 GeV electron at (x, y) = (2, 1) mm through a 0.7 m lens, integrating the orbit and its radiation gives fractional energy losses of 0 at K₁ = 0 and 1.952×10⁻⁹ at K₁ = 1.3 m⁻²; with 32 slices Elegant gives 2.650×10⁻⁶ and 2.155×10⁻⁶. On the y = 0 plane the normal quadrupole, which exerts the same force, agrees with the integration.

Run `python TRK-40.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-40.py"

    [Download TRK-40.py](TRK-40.py)

    ```python
    """TRK-40: KQUAD with RADIAL=1 radiates from the unscaled field shape instead of the applied kick

    A KQUAD with RADIAL=1, a radially focusing lens, computes its classical radiation from field shapes
    that are not scaled by the strength K₁, so it radiates at K₁ = 0 and the loss grows with the square
    of N_SLICES. For a 3.066 GeV electron at (x, y) = (2, 1) mm through a 0.7 m lens, integrating the
    orbit and its radiation gives fractional energy losses of 0 at K₁ = 0 and 1.952×10⁻⁹ at K₁ = 1.3
    m⁻²; with 32 slices Elegant gives 2.650×10⁻⁶ and 2.155×10⁻⁶. On the y = 0 plane the normal
    quadrupole, which exerts the same force, agrees with the integration.

    The script tracks an electron at (x, y) = (2, 1) mm through a 0.7 m KQUAD with RADIAL=1 and
    SYNCH_RAD=1 for K₁ = 0 and 1.3 m⁻² at 8, 16 and 32 slices, and compares the momentum loss with an
    independent integration of the radial-lens orbit and its radiation. Controls: on the y = 0 plane
    the radial and normal lenses exert the same force, so with radiation off they give the same exit
    coordinates, and the normal KQUAD's loss there agrees with the integration.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from scipy.integrate import solve_ivp

    P0 = 6000.0  # reference momentum in m_e c
    LENGTH, X0, Y0 = 0.7, 2e-3, 1e-3
    RE = 2.8179403205e-15  # classical electron radius (m), CODATA 2022
    RTOL = 1e-4  # the normal-lens control agrees with the integration to about 1.4e-7 relative
    N = 12  # identical particles, so that every MPI worker tracks some


    def exit_state(mode, k1, radial, slices, synch_rad, y0, order=4):
        lattice = (
            f"Q: KQUAD,L={LENGTH!r},K1={k1!r},RADIAL={radial},N_SLICES={slices},INTEGRATION_ORDER={order},"
            f"SYNCH_RAD={synch_rad},ISR=0\nBL: LINE=(Q)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        beam = {"x": np.full(N, X0), "y": np.full(N, y0), "p": np.full(N, P0)}
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if len(out["p"]) != N or np.ptp(out["p"]) > 0:
            sys.exit(f"{mode}: particles lost or not identical: invalid")
        return np.array([out[c][0] for c in ("x", "xp", "y", "yp")]), 1 - out["p"][0] / P0


    def lens_loss(k1, y0):
        """Classical loss through a radial lens, by direct integration of the orbit.

        The radial lens kicks P_x/P0 by -K1 x and P_y/P0 by -K1 y per unit length, i.e. the field in
        curvature units has magnitude K1 r. Cartesian equations with canonical slopes (qx, qy):
        x' = qx/pz, y' = qy/pz with pz = sqrt(1 - qx^2 - qy^2), qx' = -K1 x, qy' = -K1 y, and the loss
        integrand (2/3) r_e gamma^3 K1^2 (x^2 + y^2) sqrt(1 + x'^2 + y'^2). On y = 0 this is the
        normal quadrupole as well.
        """

        def rhs(_, z):
            x, y, qx, qy, _ = z
            pz = np.sqrt(1 - qx * qx - qy * qy)
            xp, yp = qx / pz, qy / pz
            return [
                xp,
                yp,
                -k1 * x,
                -k1 * y,
                k1 * k1 * (x * x + y * y) * np.sqrt(1 + xp * xp + yp * yp),
            ]

        sol = solve_ivp(rhs, (0, LENGTH), [X0, y0, 0, 0, 0], method="DOP853", rtol=1e-12, atol=1e-18)
        return 2 / 3 * RE * np.hypot(1.0, P0) ** 3 * sol.y[4, -1]


    defect = []
    for mode in modes():
        # Control: on y = 0 the radial and normal lenses exert the same force.
        normal_orbit, _ = exit_state(mode, 1.3, 0, 128, 0, 0.0, order=6)
        radial_orbit, _ = exit_state(mode, 1.3, 1, 128, 0, 0.0, order=6)
        if np.max(np.abs(normal_orbit - radial_orbit)) > 0:
            sys.exit("The radial and normal lenses differ on y = 0 without radiation: invalid.")
        print(
            f"{mode} y=0 K1=1.3 128 slices, radiation off: radial and normal exit (x, xp, y, yp) identical"
        )
        expected = lens_loss(1.3, 0.0)
        normal = exit_state(mode, 1.3, 0, 128, 1, 0.0, order=6)[1]
        print(
            f"{mode} y=0 K1=1.3 normal KQUAD loss (control): observed={normal:.10g} expected={expected:.10g}"
        )
        if abs(normal / expected - 1) > RTOL:
            sys.exit("The normal-lens radiation control disagrees with the integration: invalid.")
        radial = exit_state(mode, 1.3, 1, 128, 1, 0.0, order=6)[1]
        print(
            f"{mode} y=0 K1=1.3 RADIAL=1 loss: observed={radial:.10g} expected={expected:.10g}"
            f" observed/expected={radial / expected:.6g}"
        )
        defect.append(abs(radial / expected - 1) > RTOL)
        for k1 in (0.0, 1.3):
            expected = lens_loss(k1, Y0)
            for slices in (8, 16, 32):
                observed = exit_state(mode, k1, 1, slices, 1, Y0)[1]
                print(
                    f"{mode} K1={k1} RADIAL=1 N_SLICES={slices} loss: observed={observed:.10g} expected={expected:.10g}"
                )
                defect.append(abs(observed - expected) > RTOL * max(expected, 1e-12))
    print(f"comparison: relative tolerance {RTOL:g} (absolute 1e-16 for zero expected loss)")
    verdict(all(defect))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Found<br><small>radial lens loss versus normal quadrupole and integration</small> | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

