---
title: "COL-03: Round nonlinear space charge omits the integrating length and optics weighting"
hide:
  - toc
---

# COL-03 — Round nonlinear space charge omits the integrating length and optics weighting

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [round-beam branch, insertSCeffects.cc:552–563](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L552-L563)<br>[elliptical weights, insertSCeffects.cc:592–599](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L592-L599)<br>[weight accumulation, insertSCeffects.cc:837–856](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L837-L856)<br>[prefactor, insertSCeffects.cc:736–737](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L736-L737)<br>[nonlinear force, scmult.tex:24–39](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/scmult.tex#L24-L39)<br>[round limit, scmult.tex:44–47](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/scmult.tex#L44-L47) |

With `insert_sceffects` and `nonlinear=1`, SCMULT omits the integrating length and optical weights from the kick of a round Gaussian beam. The branch is selected when the rms sizes σₓ and σᵧ satisfy |σₓ − σᵧ|/σₓ < 10⁻⁶; an unsliced Gaussian control with σᵧ/σₓ = 1 + 10⁻⁵ agrees with an independent integrated-field calculation. In a symmetric ring with 1 nC of electrons at 10 mₑc and σₓ = σᵧ = 1 mm, the round kick is 4.000 times the correct value after a 0.25 m drift and 1.333 times it after a 0.75 m drift.

Run `python COL-03.py` with [_elegant.py](../helpers/_elegant.py) and [_scring.py](../helpers/_scring.py) in the same directory.

??? example "Reproducer: COL-03.py"

    [Download COL-03.py](COL-03.py)

    ```python
    """COL-03: Round nonlinear space charge omits the integrating length and optics weighting

    With `insert_sceffects` and `nonlinear=1`, SCMULT omits the integrating length and optical
    weights from the kick of a round Gaussian beam. The branch is selected when the rms sizes σₓ and
    σᵧ satisfy |σₓ − σᵧ|/σₓ < 10⁻⁶; an unsliced Gaussian control with σᵧ/σₓ = 1 + 10⁻⁵ agrees with
    an independent integrated-field calculation. In a symmetric ring with 1 nC of electrons at 10
    mₑc and σₓ = σᵧ = 1 mm, the round kick is 4.000 times the correct value after a 0.25 m drift and
    1.333 times it after a 0.75 m drift.

    The script tracks 512 electrons through one drift of a closed four-bend ring, with an SCMULT after
    the drift, and reads each particle's slope kick. It compares the kicks with the Gaussian field
    obtained from the Poisson integral, integrated over the drift with the effective length from the
    matched beta functions, for drifts of 0.25 m and 0.75 m. The control is a beam with a size ratio
    of 1.00001, just outside the round-beam branch.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _scring import beam, compare, kick, levels, track

    SIGMA = 1e-3  # m, rms horizontal size
    TIMES = 1e-10 * levels(8)  # s, arrival times of unit-rms Gaussian quantile levels, sigma_t = 0.1 ns
    RTOL = 1e-10  # the near-round control agrees to 2.3e-12

    defect = False
    for mode in modes():
        for length in (0.25, 0.75):
            name = f"{mode} L={length}"
            near = beam(SIGMA, 1.00001 * SIGMA, TIMES)
            observed, leff = track(mode, length, near)
            if compare(
                f"{name} near-round, sy/sx=1.00001", observed, kick(near, leff, nonlinear=True), RTOL
            ):
                sys.exit("The near-round control disagrees with the Gaussian field.")
            round_beam = beam(SIGMA, SIGMA, TIMES)
            observed, leff = track(mode, length, round_beam)
            expected = kick(round_beam, leff, nonlinear=True)
            wrong = compare(f"{name} round, sy/sx=1", observed, expected, RTOL)
            # The kick for a 1 m effective length: a slope change in rad, with no L and no optics.
            per_metre = kick(round_beam, np.ones(2), nonlinear=True)
            omits_length = not compare(
                f"{name} round, kick for 1 m effective length", observed, per_metre, RTOL
            )
            ratio = observed / expected
            print(
                f"{name} round: observed/expected={ratio.min():.10f}..{ratio.max():.10f}, "
                f"1/L_eff={1 / leff[0]:.10f}, {1 / leff[1]:.10f}"
            )
            defect |= wrong and omits_length
    print(f"comparison: rtol={RTOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Found<br><small>integrating length</small> | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

