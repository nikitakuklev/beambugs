---
title: "TRK-23: CSBEND dipole-only strength error rescales its octupole and higher multipoles"
hide:
  - toc
---

# TRK-23 — CSBEND dipole-only strength error rescales its octupole and higher multipoles

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`track_through_csbend`, csbend.c:1029–1081](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1029-L1081)<br>[curvature radius, csbend.c:1111–1116](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1111-L1116)<br>[body kicks, csbend.c:1921–1927](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1921-L1927)<br>[`computeCSBENDFieldCoefficients`, csbend.c:355–359](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L355-L359)<br>[parameter metadata, track_data.c:1961–1962](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1961-L1962)<br>[manual, csbend.tex:26–32](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L26-L32) |

A CSBEND using FSE_DIPOLE to change only the dipole field should leave higher multipole strengths scaled solely by the common error FSE. Orders 3–8 instead receive an extra factor 1 + FSE_DIPOLE/(1 + FSE), while quadrupole and sextupole terms are compensated. For a 0.6 m, 0.09 rad bend with octupole strength K₃ = −3000 m⁻⁴, FSE = 0.07 and FSE_DIPOLE = 0.04, an electron of momentum 49 mₑc exits with horizontal slope x′ = −3.610×10⁻³ instead of −3.700×10⁻³, an error of 8.984×10⁻⁵. Independent Lorentz integration and an equivalent common-error configuration establish the expected trajectory.

Run `python TRK-23.py` with [_elegant.py](../helpers/_elegant.py) and [_multipole.py](../helpers/_multipole.py) in the same directory.

??? example "Reproducer: TRK-23.py"

    [Download TRK-23.py](TRK-23.py)

    ```python
    """TRK-23: CSBEND dipole-only strength error rescales its octupole and higher multipoles

    A CSBEND using FSE_DIPOLE to change only the dipole field should leave higher multipole strengths
    scaled solely by the common error FSE. Orders 3–8 instead receive an extra factor 1 + FSE_DIPOLE/(1
    + FSE), while quadrupole and sextupole terms are compensated. For a 0.6 m, 0.09 rad bend with
    octupole strength K₃ = −3000 m⁻⁴, FSE = 0.07 and FSE_DIPOLE = 0.04, an electron of momentum 49 mₑc
    exits with horizontal slope x′ = −3.610×10⁻³ instead of −3.700×10⁻³, an error of 8.984×10⁻⁵.
    Independent Lorentz integration and an equivalent common-error configuration establish the expected
    trajectory.

    The script tracks 12 electrons through the bend, once with the dipole error as FSE_DIPOLE and once
    with the same physical field written as FSE alone and a rescaled multipole strength, and compares
    both with an integration of the Lorentz equations in the laboratory frame. Controls are the
    equivalent FSE-only run, a zero dipole error, and K1 and K2, which the code compensates. A further
    check shows that the result equals the oracle with the multipole strength raised by the excess
    factor.
    """

    import sys
    from math import factorial

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from _multipole import P0, beam
    from scipy.integrate import solve_ivp

    LENGTH, ANGLE = 0.6, 0.09
    ATOL = 2e-12  # fourth-order integration with 384 slices; the correct field agrees to ~3e-13
    SPLIT = 1e-8  # the smallest tested defect produces a slope error of 2.555e-6, well above SPLIT=1e-8
    P_ATOL = 2e-13  # no radiation, so the momentum is conserved
    # (order n, K_n, FSE, FSE_DIPOLE); n = 1, 2 and zero dipole error are the controls.
    CASES = [
        (3, -3000.0, 0.07, 0.04),
        (3, -3000.0, 0.07, -0.04),
        (5, 1e8, 0.07, 0.04),
        (3, -3000.0, 0.0, 0.001),
        (3, -3000.0, 0.07, 0.0),
        (1, -0.8, 0.07, 0.04),
        (2, 30.0, 0.07, -0.04),
    ]


    class ExitPlane:
        """Event: the particle reaches the plane through the end of the reference arc."""

        terminal = True
        direction = 1

        def __init__(self, reference, tangent):
            self.reference, self.tangent = reference, tangent

        def __call__(self, _, state):
            return (state[:2] - self.reference) @ self.tangent


    def oracle(initial, n, k, fse, dipole):
        """Exit (x, xp) of each particle in the specified field, integrated in the laboratory frame.

        The orbit (X, Z, alpha) is followed against path length s in the plane of a circle of radius
        rho = L/angle centred at (-rho, 0): dX/ds = sin(alpha), dZ/ds = cos(alpha),
        dalpha/ds = -B_y p0/p, where B_y in units of the reference rigidity is
        B_y = (1 + FSE + FSE_DIPOLE)/rho + (1 + FSE) K_n r^n/n!, with r the radial offset from the
        reference arc. FSE scales every component; FSE_DIPOLE only the dipole. The orbit stops at the
        radial plane through the end of the reference arc, and x is the offset along that plane's
        normal, xp = tan(alpha + angle).
        """
        rho = LENGTH / ANGLE
        normal = np.array([np.cos(ANGLE), np.sin(ANGLE)])
        tangent = np.array([-np.sin(ANGLE), np.cos(ANGLE)])
        reference = rho * np.array([np.cos(ANGLE) - 1, np.sin(ANGLE)])
        result = []
        for x, xp, p in zip(initial["x"], initial["xp"], initial["p"], strict=True):

            def rhs(_, state, ratio=p / P0):
                xx, zz, alpha = state
                radial = np.hypot(xx + rho, zz) - rho
                by = (1 + fse + dipole) / rho + (1 + fse) * k * radial**n / factorial(n)
                return [np.sin(alpha), np.cos(alpha), -by / ratio]

            sol = solve_ivp(
                rhs,
                (0, 1.5 * LENGTH),
                [x, 0.0, np.arctan(xp)],
                events=ExitPlane(reference, tangent),
                method="DOP853",
                rtol=3e-13,
                atol=2e-15,
                max_step=0.02,  # resolve the exit-plane crossing
            )
            if not (sol.success and len(sol.t_events[0]) == 1):
                raise RuntimeError("laboratory-frame integration failed")
            xx, zz, alpha = sol.y_events[0][0]
            result.append([(np.array([xx, zz]) - reference) @ normal, np.tan(alpha + ANGLE)])
        return np.array(result)


    def track(mode, initial, n, k, fse, dipole):
        lattice = (
            f"B: CSBEND,L={LENGTH},ANGLE={ANGLE},K{n}={k!r},FSE={fse!r},FSE_DIPOLE={dipole!r},"
            "N_SLICES=384,INTEGRATION_ORDER=4,EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,SYNCH_RAD=0,ISR=0,"
            "REFERENCE_CORRECTION=0,EXPANSION_ORDER=10\n"
            "BL: LINE=(B)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0}, output=out.sdds, "
            "random_number_seed=753219 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        files = {"lattice.lte": lattice, "run.ele": commands}
        out = particles(run(files, mode, {"beam.sdds": initial})["out.sdds"])
        if len(out["p"]) != len(initial["p"]):
            sys.exit(f"{mode}: particles were lost")
        if np.max(np.abs(out["y"])) > 0 or np.max(np.abs(out["yp"])) > 0:
            sys.exit(f"{mode}: the midplane beam left the midplane")
        if not np.allclose(out["p"], initial["p"], rtol=0, atol=P_ATOL, equal_nan=False):
            sys.exit(f"{mode}: the momentum is not conserved")
        return np.column_stack((out["x"], out["xp"]))


    def compare(label, observed, expected):
        """Print the last particle's x, xp and the largest difference over the beam.

        Nonfinite values make the comparison invalid: the run terminates instead of returning a NaN
        that every later `> ATOL` test would accept.
        """
        if not (np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))):
            sys.exit(f"{label}: a nonfinite value makes the comparison invalid")
        error = float(np.max(np.abs(observed - expected)))
        print(
            f"{label}: observed={[float(v) for v in observed[-1]]} "
            f"expected={[float(v) for v in expected[-1]]} max|diff|={error:.4g}"
        )
        return error


    defect = False
    initial = beam() | {"y": np.zeros(12), "yp": np.zeros(12)}  # midplane beam
    for mode in modes():
        for n, k, fse, dipole in CASES:
            label = f"{mode} K{n}={k:g} FSE={fse:g} FSE_DIPOLE={dipole:g}"
            expected = oracle(initial, n, k, fse, dipole)
            component = track(mode, initial, n, k, fse, dipole)
            # The same field with the dipole error folded into FSE: B_y/rigidity has the same dipole
            # term (1 + g + d)/rho, and the multipole (1 + g) K_n becomes (1 + g + d) K_n'.
            k_equivalent = k * (1 + fse) / (1 + fse + dipole)
            equivalent = track(mode, initial, n, k_equivalent, fse + dipole, 0.0)
            if compare(f"{label} equivalent FSE-only", equivalent, expected) > ATOL:
                sys.exit(f"{label}: the equivalent-field control disagrees with the oracle")
            error = compare(label, component, expected)
            if n <= 2 or dipole == 0:
                if error > ATOL:
                    sys.exit(f"{label}: the control disagrees with the oracle")
                continue
            # What the code does: the multipole is multiplied by (1 + g + d)/(1 + g).
            excess = oracle(initial, n, k * (1 + fse + dipole) / (1 + fse), fse, dipole)
            if not np.all(np.isfinite(excess)):
                sys.exit(f"{label}: the excess-field oracle is not finite")
            if not np.allclose(component, excess, rtol=0, atol=ATOL, equal_nan=False):
                sys.exit(f"{label}: the result is not the excess-field prediction")
            defect |= error > SPLIT
    print(f"comparison: atol={ATOL:g} for agreement, {SPLIT:g} for disagreement")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

