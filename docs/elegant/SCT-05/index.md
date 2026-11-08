---
title: "SCT-05: Standalone electron recoil omits energy loss"
hide:
  - toc
---

# SCT-05 — Standalone electron recoil omits energy loss

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [event probability, matter.c:127–128](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L127-L128)<br>[matter.c:187](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L187)<br>[matter.c:246–251](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L246-L251)<br>[matter.c:258–283](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L258-L283)<br>[`ELECTRON_RECOIL`, track_data.c:2143](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2143)<br>[matter.c:84–85](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L84-L85) |

In MATTER, enabling ELECTRON_RECOIL while ENERGY_DECAY and NUCLEAR_BREMSSTRAHLUNG are disabled omits the documented recoil energy loss. The energy-processing gates ignore standalone recoil, although collisions with target electrons must transfer positive energy. For 20000 electrons of momentum 1000 mₑc crossing 0.01 m of dilute helium, a Poisson approximation to the option's event rate predicts 6894 ± 67.21 electrons losing energy, but recoil-only output is bit-identical to all-off output. Enabling bremsstrahlung activates thousands of recoil losses, isolating the activation failure.

Run `python SCT-05.py` with [_elegant.py](../helpers/_elegant.py) and [_matter.py](../helpers/_matter.py) in the same directory.

??? example "Reproducer: SCT-05.py"

    [Download SCT-05.py](SCT-05.py)

    ```python
    """SCT-05: Standalone electron recoil omits energy loss

    In MATTER, enabling ELECTRON_RECOIL while ENERGY_DECAY and NUCLEAR_BREMSSTRAHLUNG are disabled omits
    the documented recoil energy loss. The energy-processing gates ignore standalone recoil, although
    collisions with target electrons must transfer positive energy. For 20000 electrons of momentum 1000
    mₑc crossing 0.01 m of dilute helium, a Poisson approximation to the option's event rate predicts
    6894 ± 67.21 electrons losing energy, but recoil-only output is bit-identical to all-off output.
    Enabling bremsstrahlung activates thousands of recoil losses, isolating the activation failure.

    The script tracks 20000 electrons through the target with every combination of ELECTRON_RECOIL and
    NUCLEAR_BREMSSTRAHLUNG and counts the electrons whose momentum drops. It compares each count with
    the number of recoil events the option's own rate predicts, and checks that the recoil-only output
    equals the all-off output coordinate by coordinate. The run with both options enabled shows that the
    recoil sampler works and confirms the expected count; the bremsstrahlung-only run is the control for
    the losses.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _matter import AMU, BREMS_PER_ELECTRON, BS_Y0, LENGTH, P0, RE, RHO, A, Z, beam, track

    COUNT, SEED = 20000, 90131
    THRESHOLD = 1e-9  # m_e c; a momentum decrease above this is a real energy loss, not rounding
    # PLIMIT 0.005 keeps every section's event probability far below one.
    ELEMENT = (
        f"L={LENGTH!r},PLIMIT=0.005,ENERGY_DECAY=0,NUCLEAR_BREMSSTRAHLUNG={{nb}},ELECTRON_RECOIL={{er}}"
    )
    COLUMNS = ("x", "xp", "y", "yp", "t", "p")

    # Mean recoil events per electron in the model's own process (matter.c:128): the collision rate
    # n_atom 2 pi Z r_e^2 L / P (1/y0 - 1), with y0 the sampler's lower cutoff. It only sizes the
    # expected number of electrons that suffer a recoil, in a Poisson approximation (at least one
    # event); that a recoil must lower the momentum follows from the option's documented contract and
    # the positive Moller cross section. The element actually has 61 independent Bernoulli sections of
    # probability MEAN_RECOIL/61 each, whose exact at-least-one expectation is 6913.18 with binomial
    # sigma 67.26; the recorded serial and MPI counts, 6910 and 6956, pass either 5-sigma comparison.
    MEAN_RECOIL = LENGTH * RHO / (AMU * A) * 2 * np.pi * Z * RE**2 / P0 * (1 / BS_Y0 - 1)
    RECOIL_EXPECTED = COUNT * -np.expm1(-MEAN_RECOIL)
    RECOIL_SIGMA = np.sqrt(RECOIL_EXPECTED * np.exp(-MEAN_RECOIL))  # Poisson-approximation sigma
    BREMS_EXPECTED = COUNT * BREMS_PER_ELECTRON
    # Bremsstrahlung alone may produce a few losses (mean 0.14 events); more than 10 has probability
    # below 1e-16, so it would mean something other than recoil is acting.
    BREMS_MAX = 10

    CASES = (  # label, (nb, er), expected electrons that lose momentum
        ("all off", (0, 0), 0.0),
        ("bremsstrahlung only", (1, 0), BREMS_EXPECTED),
        ("recoil + bremsstrahlung", (1, 1), RECOIL_EXPECTED),
        ("recoil only", (0, 1), RECOIL_EXPECTED),
    )

    defect = False
    print(f"electrons with a momentum decrease above {THRESHOLD:g} m_e c, of {COUNT}")
    for mode in modes():
        outs, counts = {}, {}
        for label, (nb, er), expected in CASES:
            outs[nb, er] = track(mode, ELEMENT.format(nb=nb, er=er), SEED, beam(COUNT))
            counts[nb, er] = int(np.sum(outs[nb, er]["p"] < P0 - THRESHOLD))
            spread = f"±{RECOIL_SIGMA:.2f}" if er else ""
            print(f"{mode} {label}: observed={counts[nb, er]} expected={expected:.2f}{spread}")
        # The controls must be clean before the combined-option losses can be attributed to recoil.
        if counts[0, 0] != 0:
            sys.exit(f"{mode}: the all-off run loses momentum in {counts[0, 0]} electrons")
        if counts[1, 0] > BREMS_MAX:
            sys.exit(f"{mode}: bremsstrahlung alone loses momentum in {counts[1, 0]} electrons")
        # With bremsstrahlung enabled, recoil reaches the sampler: its count must match the model rate.
        # Otherwise the expectation for the standalone case is not valid.
        if abs(counts[1, 1] - RECOIL_EXPECTED) > 5 * RECOIL_SIGMA:
            sys.exit(f"{mode}: recoil with bremsstrahlung gives {counts[1, 1]}, not the model rate")
        same = all(np.array_equal(outs[0, 1][k], outs[0, 0][k]) for k in COLUMNS)
        print(
            f"{mode} recoil only vs all off, six coordinates bit-identical: observed={same} expected=False"
        )
        defect |= same or counts[0, 1] < RECOIL_EXPECTED - 5 * RECOIL_SIGMA
    print(f"comparison: 5 sigma on the recoil count, loss threshold {THRESHOLD:g} m_e c")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-15</span> | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-04</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-15</span> | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-15</span> | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | high | <span class="bb-nowrap">2026-10-25</span> | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | high | <span class="bb-nowrap">2026-10-25</span> | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

