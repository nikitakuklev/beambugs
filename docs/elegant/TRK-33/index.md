---
title: "TRK-33: KQUAD and KSEXT steering kicks radiate nothing horizontally and four times too much vertically"
hide:
  - toc
---

# TRK-33 — KQUAD and KSEXT steering kicks radiate nothing horizontally and four times too much vertically

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`integrate_kick_multipole_ordn`, multipole.c:1430–1444](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1430-L1444)<br>[multipole.h:186–209](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.h#L186-L209)<br>[multipole.c:1488–1510](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1488-L1510)<br>[manual, kquad.tex:64–67](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/kquad.tex#L64-L67)<br>[kquad.tex:119–123](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/kquad.tex#L119-L123)<br>[SYNCH_RAD, track_data.c:1171](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1171) |

A KQUAD or KSEXT with SYNCH_RAD=1 and a steering kick HKICK or VKICK computes the radiating field after subtracting the horizontal kick and doubling the vertical one. For a 0.8 m magnet with zero main strength and a 1 mrad kick at 3.066 GeV, the classical radiation integral along the uniform steering field gives a fractional energy loss of 5.072×10⁻¹⁰ in either plane; HKICK gives zero and VKICK 2.029×10⁻⁹, four times too much. The moments_output quantum excitation of the KQUAD is zero for HKICK and eight times the Sands diffusion integral for VKICK. This steering-field error is absent when HKICK and VKICK are both zero.

Run `python TRK-33.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-33.py"

    [Download TRK-33.py](TRK-33.py)

    ```python
    """TRK-33: KQUAD and KSEXT steering kicks radiate nothing horizontally and four times too much vertically

    A KQUAD or KSEXT with SYNCH_RAD=1 and a steering kick HKICK or VKICK computes the radiating
    field after subtracting the horizontal kick and doubling the vertical one. For a 0.8 m magnet
    with zero main strength and a 1 mrad kick at 3.066 GeV, the classical radiation integral along
    the uniform steering field gives a fractional energy loss of 5.072×10⁻¹⁰ in either plane; HKICK
    gives zero and VKICK 2.029×10⁻⁹, four times too much. The moments_output quantum excitation of
    the KQUAD is zero for HKICK and eight times the Sands diffusion integral for VKICK. This
    steering-field error is absent when HKICK and VKICK are both zero.

    The script tracks an on-axis electron through a KQUAD and a KSEXT of zero main strength with
    SYNCH_RAD=1 and a horizontal (HKICK) or vertical (VKICK) steering kick, and compares the momentum
    loss with the classical radiation integral along the uniform steering field. It also compares the
    moments_output energy-spread growth of the KQUAD with ISR=1 against the quantum-diffusion integral.
    The control is a KQUAD with K₁ ≠ 0 and no steering, tracked off axis, whose loss is checked against an
    independent integration of the orbit and its radiation.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from scipy.integrate import solve_ivp

    P0 = 6000.0  # reference momentum in m_e c (3.066 GeV electrons)
    LENGTH, KICK, SLICES = 0.8, 0.001, 256
    RE = 2.8179403205e-15  # classical electron radius (m), CODATA 2022
    LAMBDA_C = 3.8615926744e-13  # reduced Compton wavelength of the electron (m), CODATA 2022
    # The slice quadrature of the radiation integral and the energy loss fed back into the orbit
    # (about 1e-9 relative) agree with the continuous integral to about 1e-6 relative at 256 slices.
    RTOL = 1e-4
    N = 12  # identical particles, so that every MPI worker tracks some


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.atleast_1d(values)) + "]"


    def lattice(element, params):
        extra = "".join(f",{k}={v!r}" for k, v in params.items())
        # The marker keeps moments_output off its one-element-beamline crash.
        return (
            f"M: {element},L={LENGTH!r},N_SLICES={SLICES},INTEGRATION_ORDER=4{extra}\n"
            "END: MARK\nBL: LINE=(M,END)\n"
        )


    def loss(mode, element, params, x0=0.0):
        """Classical momentum loss -delta of electrons entering at x = x0 with SYNCH_RAD=1, ISR=0."""
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        files = {
            "lattice.lte": lattice(element, {**params, "SYNCH_RAD": 1, "ISR": 0}),
            "run.ele": commands,
        }
        beam = {"x": np.full(N, x0), "p": np.full(N, P0)}
        out = particles(run(files, mode, {"beam.sdds": beam})["out.sdds"])
        if len(out["p"]) != N or np.ptp(out["p"]) > 0:
            sys.exit(f"{mode} {element} {params}: particles lost or not identical: invalid")
        return 1 - out["p"][0] / P0


    def diffusion(mode, element, params):
        """Growth of the rms energy spread squared through the element, from moments_output."""
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r} &end\n"
            "&moments_output filename=out.mom, matched=0, equilibrium=0, radiation=1,"
            " beta_x=2, beta_y=3, beta_z=1 &end\n"
            "&run_control &end\n&bunched_beam &end\n&track &end\n"
        )
        files = {
            "lattice.lte": lattice(element, {**params, "SYNCH_RAD": 1, "ISR": 1}),
            "run.ele": commands,
        }
        moments = run(files, mode, outputs=("out.mom",))["out.mom"]
        s6 = np.asarray(moments[0, "s6"])
        return float(s6[-1] ** 2 - s6[0] ** 2)


    def compare(label, observed, expected):
        bad = abs(observed - expected) > RTOL * abs(expected)
        ratio = observed / expected
        print(
            f"{label}: observed={observed:.10g} expected={expected:.10g} observed/expected={ratio:.7g}"
        )
        return bad


    # A steering kick k over length L is a uniform field of curvature h = k/L in the Cartesian frame; on
    # the unperturbed orbit the normalized transverse momentum grows linearly to k, so the path is L*asin(k)/k. The classical loss
    # is (2/3) r_e gamma^3 h^2 * path, and the quantum diffusion of delta is
    # (55/(24 sqrt 3)) r_e lambda_c gamma^5 h^3 * path (Sands); gamma = P0 to 1.4e-8 here.
    path = LENGTH * np.arcsin(KICK) / KICK
    gamma = np.hypot(1.0, P0)
    expected_loss = 2 / 3 * RE * gamma**3 * (KICK / LENGTH) ** 2 * path
    expected_diffusion = 55 / (24 * np.sqrt(3)) * RE * LAMBDA_C * gamma**5 * (KICK / LENGTH) ** 3 * path


    def quad_loss(k1, x0):
        """Classical loss of an electron entering a quadrupole at x = x0, by direct integration.

        Cartesian equations with canonical slope q = P_x/P0: x' = q/sqrt(1 - q^2), q' = -K1*x; the
        field magnitude in curvature units is |K1*x|, and the loss integrand is
        (2/3) r_e gamma^3 (K1 x)^2 sqrt(1 + x'^2).
        """

        def rhs(_, z):
            x, q, _ = z
            xp = q / np.sqrt(1 - q * q)
            return [xp, -k1 * x, (k1 * x) ** 2 * np.sqrt(1 + xp * xp)]

        sol = solve_ivp(rhs, (0, LENGTH), [x0, 0.0, 0.0], method="DOP853", rtol=1e-12, atol=1e-18)
        return 2 / 3 * RE * gamma**3 * sol.y[2, -1]


    defect = []
    print(f"p0={P0:g} m_e c, L={LENGTH} m, kick={KICK} rad, {SLICES} slices, INTEGRATION_ORDER=4")
    for mode in modes():
        # Control: radiation from the main quadrupole field, no steering.
        control = loss(mode, "KQUAD", {"K1": 0.5}, x0=0.001)
        if compare(
            f"{mode} KQUAD K1=0.5 x0=1 mm, no steering (control)", control, quad_loss(0.5, 0.001)
        ):
            sys.exit("The quadrupole radiation control disagrees with the integration: invalid.")
        for element in ("KQUAD", "KSEXT"):
            for plane in ("HKICK", "VKICK"):
                observed = loss(mode, element, {plane: KICK})
                defect.append(compare(f"{mode} {element} {plane}={KICK} loss", observed, expected_loss))
        for plane in ("HKICK", "VKICK"):
            observed = diffusion(mode, "KQUAD", {plane: KICK})
            defect.append(
                compare(
                    f"{mode} KQUAD {plane}={KICK} moments sigma_delta^2 growth",
                    observed,
                    expected_diffusion,
                )
            )
    print(f"comparison: relative tolerance {RTOL:g}")
    verdict(all(defect))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gemini-4-argon-medium`</span> |  | <span class="bb-nowrap">2026-11-04</span> | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Found | <span class="bb-nowrap">`gemini-4-argon-medium`</span> |  | <span class="bb-nowrap">2026-11-04</span> | master (`7514db36f764`), source reading, not executed for this claim |
| Found<br><small class="bb-scope">comparison with the analytic loss and a CSBEND reference</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-04</span> | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-04</span> | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

