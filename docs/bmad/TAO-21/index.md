---
title: "TAO-21: show universe labels momentum chromaticity as an energy derivative"
hide:
  - toc
---

# TAO-21 — show universe labels momentum chromaticity as an energy derivative

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P5 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [show universe chromaticity calculation and label, tao/code/tao_show_this.f90:5943–5956](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_show_this.f90#L5943-L5956)<br>[chrom_calc low-momentum orbit, bmad/code/chrom_calc.f90:120–134](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/chrom_calc.f90#L120-L134) |

Tao’s show universe output labels chromaticity as dQ/(dE/E) while displaying the derivative with respect to relative momentum. Here Q is tune, E is total energy, and δ is relative momentum deviation; the energy derivative is (dQ/dδ)/β₀², where β₀ is the reference speed divided by the speed of light. For a 1 MeV/c electron FODO ring, the displayed horizontal value is −0.02977 while the energy derivative is −0.03754. Independent matrix differentiation matches the displayed momentum derivative; at 1 GeV/c the conversion is below the display precision.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _tao_contracts_normal_form.py](../helpers/_tao_contracts_normal_form.py) · [View source](../helpers/index.md#_tao_contracts_normal_form)

??? example "Reproducer: TAO-21.py"

    [Download TAO-21.py](TAO-21.py)

    ```python
    """TAO-21: show universe labels momentum chromaticity as an energy derivative

    Tao’s show universe output labels chromaticity as dQ/(dE/E) while displaying the derivative with respect to relative momentum. Here Q is tune, E is total energy, and δ is relative momentum deviation; the energy derivative is (dQ/dδ)/β₀², where β₀ is the reference speed divided by the speed of light. For a 1 MeV/c electron FODO ring, the displayed horizontal value is −0.02977 while the energy derivative is −0.03754. Independent matrix differentiation matches the displayed momentum derivative; at 1 GeV/c the conversion is below the display precision.

    The script runs show universe for the same FODO optics at two reference momenta and reads both the chromaticity values and their label. It differentiates independent thick-quadrupole/drift matrices and applies the exact relativistic energy-to-momentum conversion.
    """

    import re

    import numpy as np
    from _bmad import compare, verdict
    from _tao_contracts_normal_form import FODO, block, fodo_matrix, tao


    def tunes(delta):
        matrix = fodo_matrix(delta)
        return np.array(
            [np.arccos(np.trace(matrix[i : i + 2, i : i + 2]) / 2) / (2 * np.pi) for i in (0, 2)]
        )


    # Bmad's chrom_calc uses the central momentum secant at delta=+-1e-4.
    h = 1e-4
    momentum_derivative = (tunes(h) - tunes(-h)) / (2 * h)
    derivative = (8 * (tunes(h) - tunes(-h)) - (tunes(2 * h) - tunes(-2 * h))) / (12 * h)
    assert np.allclose(momentum_derivative, derivative, rtol=4e-8, atol=0)
    bad = False
    for p0c in (1e6, 1e9):
        result = tao(FODO.replace("1e9", str(p0c)), ["show universe"])
        matches = re.findall(
            r"^\s*Chrom\s+(.+?)\s*!\s*(.+)$", block(result.log, "show universe"), re.MULTILINE
        )
        assert len(matches) == 1
        numbers, label = matches[0]
        displayed = np.array(numbers.split(), dtype=float)[[0, 2]]
        assert not compare(
            f"p0c={p0c:g} eV, displayed momentum secant control",
            displayed,
            momentum_derivative,
            atol=5e-7,
        )
        # E^2=p^2*c^2+m^2*c^4 gives dE/E=beta0^2*dp/p at the reference.
        beta2 = p0c**2 / (p0c**2 + 510998.95069**2)
        energy_derivative = derivative / beta2
        print(f"p0c={p0c:g} eV: label={label!r}, beta0^2={beta2:.15g}")
        discrepancy = compare(
            f"p0c={p0c:g} eV, energy derivative required by label",
            displayed,
            energy_derivative,
            atol=5e-7,
        )
        bad |= label == "dQ/(dE/E)" and discrepancy
    verdict(bad)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found (source/contract discrepancy) | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed (source/contract discrepancy) | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

