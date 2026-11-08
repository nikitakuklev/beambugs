---
title: "COL-04: Nonlinear space charge ignores the horizontal and vertical plane switches"
hide:
  - toc
---

# COL-04 — Nonlinear space charge ignores the horizontal and vertical plane switches

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [applySCKick, insertSCeffects.cc:53–67](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L53-L67)<br>[unconditional kicks, insertSCeffects.cc:64–65](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L64-L65)<br>[nonlinearSCKick, insertSCeffects.cc:531–605](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L531-L605)<br>[switch storage, insertSCeffects.cc:124–125](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L124-L125)<br>[linear gating, insertSCeffects.cc:521–527](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L521-L527)<br>[switch description, elegant.tex:4485–4488](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4485-L4488)<br>[namelist defaults, insertSCeffects.nl:16–19](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.nl#L16-L19) |

With `insert_sceffects` set to `nonlinear=1`, SCMULT ignores the `horizontal` and `vertical` switches that enable space charge in each plane. For unsliced Gaussian bunches, successful nonlinear evaluations give the same kicks with either or both planes disabled; linear controls obey the switches. In the round-beam reproduction with 1 nC of electrons at 10 mₑc and 1 mm rms transverse sizes, a sampled particle receives −3.309×10⁻⁵ rad in each plane after a 0.5 m drift even with both switches off, instead of zero.

Run `python COL-04.py` with [_elegant.py](../helpers/_elegant.py) and [_scring.py](../helpers/_scring.py) in the same directory.

??? example "Reproducer: COL-04.py"

    [Download COL-04.py](COL-04.py)

    ```python
    """COL-04: Nonlinear space charge ignores the horizontal and vertical plane switches

    With `insert_sceffects` set to `nonlinear=1`, SCMULT ignores the `horizontal` and `vertical` switches that enable space charge in each plane. For unsliced Gaussian bunches, successful nonlinear evaluations give the same kicks with either or both planes disabled; linear controls obey the switches. In the round-beam reproduction with 1 nC of electrons at 10 mₑc and 1 mm rms transverse sizes, a sampled particle receives −3.309×10⁻⁵ rad in each plane after a 0.5 m drift even with both switches off, instead of zero.

    The script tracks 512 electrons through one drift of a closed four-bend ring with an SCMULT after
    it, with round and 2:1 elliptical beams and each plane selection, and reads each particle's slope
    kick. For every nonlinear run with a plane disabled it requires a finite, nonzero kick in each
    disabled component that equals the kick measured with both planes enabled for the same beam. The
    controls are the physical Gaussian field from the Poisson integral for the enabled components of
    the elliptical beam, the linear force with the same plane selections, and the all-enabled
    nonlinear run.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _scring import beam, compare, kick, levels, track

    SIGMA = 1e-3  # m, rms horizontal size
    TIMES = 1e-10 * levels(8)  # s, arrival times of unit-rms Gaussian quantile levels, sigma_t = 0.1 ns
    LENGTH = 0.5  # m, drift with the SCMULT
    RTOL = 1e-6  # the elliptical nonlinear field is accurate to 8e-8, see the Gaussian field entry

    ATOL = 1e-18  # rad

    # One entry per run: True when every disabled component of a nonlinear run keeps the full kick.
    leaks = []
    for mode in modes():
        reference = {}  # sy/sx -> kick measured with both planes enabled, nonlinear
        # The all-enabled nonlinear run of each beam comes first: it is the reference for the others.
        for ratio, nonlinear, planes in (
            (2.0, 0, (0, 0)),
            (2.0, 0, (1, 0)),
            (2.0, 1, (1, 1)),
            (2.0, 1, (0, 0)),
            (2.0, 1, (1, 0)),
            (2.0, 1, (0, 1)),
            (1.0, 1, (1, 1)),
            (1.0, 1, (0, 0)),
        ):
            particles_in = beam(SIGMA, ratio * SIGMA, TIMES)
            observed, leff = track(
                mode,
                LENGTH,
                particles_in,
                horizontal=planes[0],
                vertical=planes[1],
                nonlinear=nonlinear,
            )
            # The physical kick of an enabled plane, zero for a disabled plane. The enabled kick of a
            # round beam is not compared, since it has its own length defect: the round runs are
            # judged against the round kick measured with both planes enabled instead.
            enabled = np.array(planes, dtype=bool)
            expected = np.where(enabled, kick(particles_in, leff, nonlinear=bool(nonlinear)), 0.0)
            name = f"{mode} {'nonlinear' if nonlinear else 'linear'} sy/sx={ratio:g} h,v={planes[0]}{planes[1]}"
            bad = compare(name, observed, expected, RTOL)
            if nonlinear and planes == (1, 1):
                reference[ratio] = observed
            if not nonlinear or (ratio == 2.0 and planes == (1, 1)):
                if bad:
                    sys.exit("A control disagrees with the physical kick.")
            elif ratio == 2.0 and not np.allclose(
                observed[:, enabled], expected[:, enabled], rtol=RTOL, atol=ATOL
            ):
                sys.exit("An enabled component disagrees with the physical kick.")
            if nonlinear and planes != (1, 1):
                disabled = observed[:, ~enabled]
                leaks.append(
                    bool(
                        np.all(disabled != 0)
                        and np.allclose(disabled, reference[ratio][:, ~enabled], rtol=RTOL, atol=ATOL)
                    )
                )
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
    verdict(bool(leaks) and all(leaks))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-15</span> | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-04</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-15</span> | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-15</span> | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | high | <span class="bb-nowrap">2026-10-25</span> | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | high | <span class="bb-nowrap">2026-10-25</span> | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

