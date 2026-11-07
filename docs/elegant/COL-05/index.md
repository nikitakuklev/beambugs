---
title: "COL-05: Uniform nonlinear space charge is too weak by √12"
hide:
  - toc
---

# COL-05 — Uniform nonlinear space charge is too weak by √12

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [uniform prefactor, insertSCeffects.cc:545](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L545)<br>[linear uniform prefactor, insertSCeffects.cc:516](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L516)<br>[Gaussian prefactor, insertSCeffects.cc:547](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L547)<br>[switch description, elegant.tex:4489–4490](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4489-L4490)<br>[scmult.tex:24–39](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/scmult.tex#L24-L39) |

With `insert_sceffects`, `nonlinear=1` and `uniform_distribution=1`, SCMULT's unsliced coasting-beam prefactor is 1/√12 = 0.2887 of its correct value for both round and elliptical beams. An independent Gaussian transverse-field integral with uniform number density N/(√12 σₛ), where N is the particle population and σₛ the rms longitudinal length, establishes the missing normalization. In an 8 m ring with 1 nC of electrons at 10 mₑc and 1 mm rms transverse sizes, a sampled round-beam particle after a 1 m drift receives −3.558×10⁻⁷ rad instead of −1.233×10⁻⁶ rad, a 71.13% suppression. Linear controls obey their own force law, and the 1 m round-beam case removes the separate integration-weight error.

Run `python COL-05.py` with [_elegant.py](../helpers/_elegant.py) and [_scring.py](../helpers/_scring.py) in the same directory.

??? example "Reproducer: COL-05.py"

    [Download COL-05.py](COL-05.py)

    ```python
    """COL-05: Uniform nonlinear space charge is too weak by √12

    With `insert_sceffects`, `nonlinear=1` and `uniform_distribution=1`, SCMULT's unsliced coasting-beam prefactor is 1/√12 = 0.2887 of its correct value for both round and elliptical beams. An independent Gaussian transverse-field integral with uniform number density N/(√12 σₛ), where N is the particle population and σₛ the rms longitudinal length, establishes the missing normalization. In an 8 m ring with 1 nC of electrons at 10 mₑc and 1 mm rms transverse sizes, a sampled round-beam particle after a 1 m drift receives −3.558×10⁻⁷ rad instead of −1.233×10⁻⁶ rad, a 71.13% suppression. Linear controls obey their own force law, and the 1 m round-beam case removes the separate integration-weight error.

    The script tracks 512 electrons through one drift of a closed four-bend ring, with an SCMULT after
    the drift, and reads each particle's slope kick. The electrons are spread uniformly in time over
    one revolution. It compares the kicks of round and 2:1 elliptical beams with the Gaussian field
    from the Poisson integral and the uniform line density, and requires the nonlinear kicks to equal
    that expectation divided by √12. The controls are the same beams with the linear force, and a 1 m
    drift, whose effective length of 1 m is checked and gives a round beam its correct integration
    weight.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _scring import BETA, C, beam, circumference, compare, kick, levels, track

    SIGMA = 1e-3  # m, rms horizontal size
    LENGTH = 1.0  # m, drift with the SCMULT; its effective length of 1 m is what a round beam is given
    # A coasting beam fills the ring: its arrival times are uniform over one revolution, whose
    # duration is circumference/(beta c), so that sqrt(12) sigma_z equals the circumference.
    TIMES = circumference(LENGTH) / (BETA * C) * levels(8, uniform=True) / np.sqrt(12)
    RTOL = 1e-6  # the elliptical nonlinear field is accurate to 8e-8, see the Gaussian field entry

    defect_cases = []
    for mode in modes():
        for ratio in (1.0, 2.0):
            particles_in = beam(SIGMA, ratio * SIGMA, TIMES)
            for nonlinear in (0, 1):
                observed, leff = track(mode, LENGTH, particles_in, nonlinear=nonlinear, uniform=1)
                if not np.allclose(leff, LENGTH, rtol=1e-9, atol=0):
                    sys.exit(f"The effective lengths {leff} differ from {LENGTH} m.")
                expected = kick(particles_in, leff, nonlinear=bool(nonlinear), uniform=True)
                name = f"{mode} {'nonlinear' if nonlinear else 'linear'} uniform sy/sx={ratio:g}"
                bad = compare(name, observed, expected, RTOL)
                quotient = observed / expected
                print(
                    f"{name}: observed/expected={quotient.min():.10f}..{quotient.max():.10f}, "
                    f"1/sqrt(12)={1 / np.sqrt(12):.10f}"
                )
                if nonlinear:
                    # The defect is the extra factor 1/sqrt(12): the kicks must match the expectation
                    # divided by sqrt(12) and differ from the expectation itself.
                    scaled = np.allclose(observed, expected / np.sqrt(12), rtol=RTOL, atol=1e-18)
                    defect_cases.append(bad and scaled)
                elif bad:
                    sys.exit("The linear control disagrees with the physical kick.")
    print(f"comparison: rtol={RTOL:g}")
    verdict(bool(defect_cases) and all(defect_cases))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

