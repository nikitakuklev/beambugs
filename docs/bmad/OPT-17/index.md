---
title: "OPT-17: Projected dispersion slopes use a momentum ratio as momentum"
hide:
  - toc
---

# OPT-17 — Projected dispersion slopes use a momentum ratio as momentum

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [`twiss_propagate1`, overwritten `rel_p2`, bmad/code/twiss_propagate1.f90:133–145](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/twiss_propagate1.f90#L133-L145)<br>[projected slope conversion, bmad/code/twiss_propagate1.f90:283–289](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/twiss_propagate1.f90#L283-L289)<br>[Bmad manual, “Dispersion Calculation”, `s:dispersion`, `dexds`, bmad/doc/optics.tex:381–417](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/optics.tex#L381-L417) |

Propagating an off-momentum lattice can return projected dispersion slopes that disagree with the modal slopes even without transverse coupling. The projected conversion uses the exit-to-entrance momentum ratio where it requires the actual exit relative momentum q = 1 + δ, with δ the fractional momentum offset. For δ = 0.2, normalized transverse momenta (0.03, −0.02), and zero momentum dispersion, the projected slopes are (−0.03, 0.02) instead of the documented paraxial derivatives (−0.02083, 0.01389).

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _dispersion_and_moments.py](../helpers/_dispersion_and_moments.py) · [View source](../helpers/index.md#_dispersion_and_moments)

??? example "Reproducer: OPT-17.py"

    [Download OPT-17.py](OPT-17.py)

    ```python
    """OPT-17: Projected dispersion slopes use a momentum ratio as momentum

    Propagating an off-momentum lattice can return projected dispersion slopes that disagree with
    the modal slopes even without transverse coupling. The projected conversion uses the
    exit-to-entrance momentum ratio where it requires the actual exit relative momentum q = 1 + δ,
    with δ the fractional momentum offset. For δ = 0.2, normalized transverse momenta (0.03,
    −0.02), and zero momentum dispersion, the projected slopes are (−0.03, 0.02) instead of the
    documented paraxial derivatives (−0.02083, 0.01389).

    The script tracks an off-momentum drift and checks unchanged momenta and zero momentum
    dispersion. It compares projected slopes with the paraxial quotient-rule derivative and
    validates modal slopes as a control. BMAD_DIST selects the build.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _dispersion_and_moments import OPEN_LINE, execute, lattice, require

    initial = [0, 0.03, 0, -0.02, 0, 0.2]
    values = execute(OPEN_LINE, {"lat.bmad": lattice("e: drift, l=1.7", start=initial)})
    require(values["orbit"][[1, 3, 5]], np.array(initial)[[1, 3, 5]])
    require(values["exit"][[1, 3]], [0, 0])
    # The paraxial slope px/q has derivative etap/q - px/q^2, q=1+delta.
    expected = -np.array([0.03, -0.02]) / 1.2**2
    require(values["slopes"][2:], expected)
    verdict(
        compare(
            "projected deta/ds, derivative of px/(1+delta) and py/(1+delta)",
            values["slopes"][:2],
            expected,
            atol=1e-15,
            rtol=3e-14,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

