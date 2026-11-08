---
title: "OPT-01: Unmatched alpha derivatives report phase derivatives"
hide:
  - toc
---

# OPT-01 — Unmatched alpha derivatives report phase derivatives

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [`computeChromaticAlphaValue`, nonperiodic branch, src/chrom.c:1192–1200](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/chrom.c#L1192-L1200)<br>[`computeChromaticityValue`, src/chrom.c:1166–1167](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/chrom.c#L1166-L1167)<br>[`computeChromaticities`, assigning `dalphax` and `dalphay`, src/chrom.c:867–895](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/chrom.c#L867-L895)<br>[`dalphax/dp` and `dalphay/dp` are the chromatic derivatives of the alpha functions, dα/dδ, manual, doc/elegant.tex:8234–8235](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8234-L8235) |

Elegant's `twiss_output` with `matched=0` propagates user-given entrance optics through a non-periodic beamline and writes the momentum derivatives of the Twiss alpha functions, `dalphax/dp` and `dalphay/dp`. These columns instead hold the derivatives of the betatron phase, 2π times the chromaticities. For a 0.5 m quadrupole with K1 = 0.6 m⁻² followed by a 1 m drift, entering with βx = 3 m, αx = 0.4, βy = 2 m and αy = −0.3, the exact hard-edge map gives dαx/dδ = 0.004523 and dαy/dδ = 1.549, where δ is the relative momentum offset; Elegant writes −0.3591 and 0.07649. The tune and beta derivatives are correct.

Run `python OPT-01.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: OPT-01.py"

    [Download OPT-01.py](OPT-01.py)

    ```python
    """OPT-01: Unmatched alpha derivatives report phase derivatives

    Elegant's `twiss_output` with `matched=0` propagates user-given entrance optics through a
    non-periodic beamline and writes the momentum derivatives of the Twiss alpha functions, `dalphax/dp`
    and `dalphay/dp`. These columns instead hold the derivatives of the betatron phase, 2π times the
    chromaticities. For a 0.5 m quadrupole with K1 = 0.6 m⁻² followed by a 1 m drift, entering with βx =
    3 m, αx = 0.4, βy = 2 m and αy = −0.3, the exact hard-edge map gives dαx/dδ = 0.004523 and dαy/dδ =
    1.549, where δ is the relative momentum offset; Elegant writes −0.3591 and 0.07649. The tune and
    beta derivatives are correct.

    The script transports fixed entrance optics through the quadrupole and drift in serial Elegant (and
    with `--mpi` in 3-rank Pelegant), and compares the written alpha derivatives with a complex-step
    derivative of the exact hard-edge map with respect to the momentum offset. The written tune and beta
    derivatives are the control: they must agree with the same oracle, and the alpha derivatives are
    shown to equal 2π times the tune derivatives.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    K1, QUAD_L, DRIFT_L = 0.6, 0.5, 1.0  # m^-2, m, m
    ENTRANCE = {"x": (3.0, 0.4), "y": (2.0, -0.3)}  # (beta, alpha) at the entrance
    RTOL, ATOL = 1e-12, 1e-14
    STEP = 1e-30  # complex step: derivative = Im f(delta + i*STEP)/STEP, exact to rounding

    LATTICE = f"Q: QUAD,L={QUAD_L!r},K1={K1!r}\nD: DRIF,L={DRIFT_L!r}\nBL: LINE=(Q,D)\n"
    COMMANDS = (
        "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&twiss_output filename=twiss.sdds, matched=0, "
        "beta_x=3, alpha_x=0.4, beta_y=2, alpha_y=-0.3 &end\n"
        "&bunched_beam n_particles_per_bunch=12 &end\n"
        "&track &end\n"
    )


    def quad(k, length):
        """2x2 transfer matrix of a hard-edge quadrupole, analytic in k (k < 0 is defocusing)."""
        z = np.sqrt(k + 0j) * length  # cos z and sin(z)/z are even in z, so the branch is irrelevant
        return np.array([[np.cos(z), np.sin(z) / z * length], [-z * np.sin(z) / length, np.cos(z)]])


    def twiss_derivatives(plane, delta):
        """(d nu/d delta, d beta/d delta, d alpha/d delta) at the exit for fixed entrance optics.

        The gradient scales as K1/(1 + delta) (rigidity), the drift does not depend on delta. With the
        ellipse matrix B = [[beta, -alpha], [-alpha, gamma]] at the entrance, B1 = M B0 M^T gives
        beta1 = B1[0,0], alpha1 = -B1[0,1]. The phase advance is atan(M12/(M11 beta0 - M12 alpha0)).
        All three are analytic in delta, so a complex step differentiates them exactly.
        """
        beta0, alpha0 = ENTRANCE[plane]
        k = K1 if plane == "x" else -K1  # the quadrupole defocuses in y
        d = delta + 1j * STEP
        m = np.array([[1, DRIFT_L], [0, 1]]) @ quad(k / (1 + d), QUAD_L)
        b0 = np.array([[beta0, -alpha0], [-alpha0, (1 + alpha0**2) / beta0]])
        b1 = m @ b0 @ m.T
        phase = np.arctan(m[0, 1] / (m[0, 0] * beta0 - m[0, 1] * alpha0))
        return np.array([phase / (2 * np.pi), b1[0, 0], -b1[0, 1]]).imag / STEP


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected disagree."""
        bad = not np.allclose(observed, expected, rtol=RTOL, atol=ATOL)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    defect = False
    print(
        "columns: d nu/dp (1/(2 pi)), d beta/dp (m), d alpha/dp; expected from the exact hard-edge map"
    )
    for mode in modes():
        twiss = run({"lattice.lte": LATTICE, "run.ele": COMMANDS}, mode, outputs=("twiss.sdds",))[
            "twiss.sdds"
        ]
        for plane in "xy":
            observed = np.array([twiss[f"d{name}{plane}/dp"][0] for name in ("nu", "beta", "alpha")])
            expected = twiss_derivatives(plane, 0.0)
            # Control: the tune and beta derivatives agree with the oracle, so the entrance optics,
            # the K1/(1+delta) scaling and the sign conventions are right.
            if compare(f"{mode} {plane} control, nu and beta", observed[:2], expected[:2]):
                sys.exit(
                    "The tune/beta control disagrees with the oracle: the comparison is not valid."
                )
            defect |= compare(f"{mode} {plane} d alpha/dp", observed[2], expected[2])
            # The written alpha derivative equals 2 pi times the tune derivative, i.e. the phase one.
            print(f"{mode} {plane} 2*pi*d nu/dp={2 * np.pi * observed[0]:.16g}")
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-15</span> | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-15</span> | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-15</span> | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | high | <span class="bb-nowrap">2026-10-25</span> | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | high | <span class="bb-nowrap">2026-10-25</span> | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

