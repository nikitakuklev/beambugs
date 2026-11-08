---
title: "OPT-14: Periodic vertical and b-mode dispersion slopes have a denominator typo"
hide:
  - toc
---

# OPT-14 — Periodic vertical and b-mode dispersion slopes have a denominator typo

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`twiss_from_mat6`, bmad/code/twiss_from_mat6.f90:110–139](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/twiss_from_mat6.f90#L110-L139)<br>[Bmad manual, “Dispersion Calculation”, `s:dispersion`, `dexds`, bmad/doc/optics.tex:381–417](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/optics.tex#L381-L417) |

Calling the periodic Twiss initializer with vertical reference-orbit momentum gives incorrect vertical and b-mode dispersion slopes. The documented paraxial derivative is η′/q − pᵧ/q², where q = 1 + δ, δ is fractional momentum offset, pᵧ is vertical momentum divided by reference momentum, and η′ = ∂pᵧ/∂δ. At δ = 0.2, pᵧ = −0.02 and η′ = −0.05, both slopes are −0.008333 instead of −0.02778; horizontal and a-mode slopes agree with the derivative.

Run `python OPT-14.py` with [_bmad.py](../helpers/_bmad.py) and [_dispersion_and_moments.py](../helpers/_dispersion_and_moments.py) in the same directory.

??? example "Reproducer: OPT-14.py"

    [Download OPT-14.py](OPT-14.py)

    ```python
    """OPT-14: Periodic vertical and b-mode dispersion slopes have a denominator typo

    Calling the periodic Twiss initializer with vertical reference-orbit momentum gives incorrect
    vertical and b-mode dispersion slopes. The documented paraxial derivative is η′/q − pᵧ/q²,
    where q = 1 + δ, δ is fractional momentum offset, pᵧ is vertical momentum divided by reference
    momentum, and η′ = ∂pᵧ/∂δ. At δ = 0.2, pᵧ = −0.02 and η′ = −0.05, both slopes are −0.008333
    instead of −0.02778; horizontal and a-mode slopes agree with the derivative.

    The script constructs a stable periodic map with a prescribed fixed-point derivative and calls the
    compiled Twiss initializer. It checks the ordinary dispersion and horizontal slopes before comparing
    the vertical and b-mode slopes with the paraxial derivative.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _dispersion_and_moments import periodic, require

    values, orbit, eta, _ = periodic(False)
    require(values[:4], eta)
    # Differentiate the paraxial slope p_y/(1+delta) with respect to delta.
    expected = eta[[1, 3]] / (1 + orbit[5]) - orbit[[1, 3]] / (1 + orbit[5]) ** 2
    require(values[[4, 6]], np.repeat(expected[0], 2))
    verdict(
        compare(
            "vertical and b-mode deta/ds, from d[p_y/(1+delta)]/d(delta)",
            values[[5, 7]],
            np.repeat(expected[1], 2),
            atol=1e-15,
            rtol=3e-14,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

