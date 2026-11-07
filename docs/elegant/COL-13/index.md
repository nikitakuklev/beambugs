---
title: "COL-13: Round Gaussian BEAMBEAM produces NaNs at its center"
hide:
  - toc
---

# COL-13 — Round Gaussian BEAMBEAM produces NaNs at its center

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [radial kick, ionEffects.cc:1485](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ionEffects.cc#L1485)<br>[helper selection, ionEffects.cc:1411–1412](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ionEffects.cc#L1411-L1412)<br>[Gaussian branch, do_tracking.c:6316–6322](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L6316-L6322)<br>[BEAMBEAM manual, beambeam.tex:6–8](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/beambeam.tex#L6-L8) |

A Gaussian BEAMBEAM opposing bunch with transverse rms widths σₓ and σᵧ satisfying |σₓ − σᵧ|/σₓ < 10⁻⁹ turns a centered particle's slopes and momentum into NaN, even at zero opposing charge. The immediate output retains the invalid particles without reporting losses, and Elegant exits successfully. Symmetry and the continuous radial-force limit require zero kick at the center. For σₓ = σᵧ = 1.300 mm, off-axis particles at horizontal position x = 0.1σₓ have finite kicks agreeing with the analytic force, while a centered particle with σᵧ/σₓ = 1.02 retains zero slopes and momentum 1300 mₑc.

Run `python COL-13.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: COL-13.py"

    [Download COL-13.py](COL-13.py)

    ```python
    """COL-13: Round Gaussian BEAMBEAM produces NaNs at its center

    A Gaussian BEAMBEAM opposing bunch with transverse rms widths σₓ and σᵧ satisfying
    |σₓ − σᵧ|/σₓ < 10⁻⁹ turns a centered particle's slopes and momentum into NaN, even at zero opposing
    charge. The immediate output retains the invalid particles without reporting losses, and Elegant
    exits successfully. Symmetry and the continuous radial-force limit require zero kick at the center.
    For σₓ = σᵧ = 1.300 mm, off-axis particles at horizontal position x = 0.1σₓ have finite kicks
    agreeing with the analytic force, while a centered particle with σᵧ/σₓ = 1.02 retains zero slopes
    and momentum 1300 mₑc.

    The script tracks 12 particles at x = y = 0 through one Gaussian BEAMBEAM element for opposing
    charges 0 and ±0.9 nC, once with a round bunch and once with σᵧ/σₓ = 1.02, and compares the
    resulting slopes and momentum with the zero-force limit. Controls are the elliptical bunch and
    12 identical off-axis particles at each of x/σ = 0.1 and 0.01, whose kicks are compared with a
    closed form that evaluates 1 − exp(−r²/2σ²) stably.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    # SDDS constants.h, as Elegant uses them.
    QE, RE = 1.60217733e-19, 2.81794092e-15
    P0 = 1300.0  # momentum, units of m_e c
    SIGMA = 1.3e-3  # m
    N = 12
    EPS = np.finfo(float).eps


    def track(mode, charge, ratio, x):
        """(xp, yp, p) arrays after one BEAMBEAM with sigma_y = ratio * sigma_x, particles at (x, 0)."""
        lattice = (
            f"K: BEAMBEAM,DISTRIBUTION=gaussian,CHARGE={charge!r},XSIZE={SIGMA!r},YSIZE={ratio * SIGMA!r}\n"
            "BL: LINE=(K)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0!r},output=out.sdds,"
            "losses=lost.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {"x": np.full(N, x), "p": np.full(N, P0)}
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": beam},
            outputs=("out.sdds", "lost.sdds"),
        )
        out, lost = result["out.sdds"], result["lost.sdds"]
        if len(out[0, "xp"]) != N or len(lost[0, "particleID"]) != 0:
            sys.exit(f"{mode}: particles were lost, which is not the case under test")
        return tuple(np.asarray(out[0, name]) for name in ("xp", "yp", "p"))


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in values) + "]"


    defect = False
    for mode in modes():
        # At the center the unique zero-radius limit of the kick is zero, for any charge:
        # dp ~ C1 [1 - exp(-r^2/2s^2)]/r -> C1 r/(2 s^2) -> 0.
        expected = (np.zeros(N), np.zeros(N), np.full(N, P0))
        for charge in (0.0, 0.9e-9, -0.9e-9):
            for ratio in (1.0, 1.02):
                observed = track(mode, charge, ratio, 0.0)
                ok = all(
                    np.array_equal(o, e) for o, e in zip(observed, expected, strict=True)
                )  # NaN compares unequal
                name = f"{mode} center charge={charge * 1e9:+.1f} nC sigma_y/sigma_x={ratio}"
                print(
                    f"{name}: observed xp,yp,p={fmt([o[0] for o in observed])} "
                    f"expected xp,yp,p={fmt([e[0] for e in expected])}"
                )
                if ratio == 1.0:
                    defect |= not ok
                elif not ok:
                    sys.exit("The elliptical-bunch control does not give the zero center kick.")
        # Finite-radius control: the same routine at x = f sigma. Kick of a particle at (x, 0):
        # xp = -(2 r_e Q/e)/p0 [1 - exp(-x^2/2s^2)]/x, with 1 - exp(-u) = -expm1(-u).
        charge = 0.9e-9
        for fraction in (0.1, 0.01):
            x = fraction * SIGMA
            prefactor = 2 * RE * charge / QE / P0 / x
            expected_xp = -prefactor * -np.expm1(-(fraction**2) / 2)
            # The routine's 1 - exp(...) loses digits near zero: allow one half-ulp of exp(...)
            # times the prefactor, plus 8 ulp of rounding in the remaining arithmetic.
            tolerance = abs(prefactor) * EPS / 2 + abs(expected_xp) * 8 * EPS
            xp = track(mode, charge, 1.0, x)[0]
            print(
                f"{mode} off axis x/sigma={fraction}: observed xp={fmt(xp[:1])} "
                f"expected xp={fmt([expected_xp])}"
            )
            if not np.all(np.abs(xp - expected_xp) <= tolerance):
                sys.exit("The off-axis control disagrees with the closed form.")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |

