---
title: "TAO-26: Ad hoc pipe srdt expressions return uncomputed values as valid"
hide:
  - toc
---

# TAO-26 — Ad hoc pipe srdt expressions return uncomputed values as valid

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [srdt calculation gate, tao/code/tao_lattice_calc.f90:257–270](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_lattice_calc.f90#L257-L270)<br>[initial data setup, tao/code/tao_init_data_mod.f90:582–585](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_init_data_mod.f90#L582-L585)<br>[run-time data scan, tao/code/tao_run_cmd.f90:85–94](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_run_cmd.f90#L85-L94)<br>[srdt expression evaluation, tao/code/tao_evaluate_a_datum.f90:3113–3155](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_evaluate_a_datum.f90#L3113-L3155)<br>[summation_rdt_struct, bmad/modules/srdt_mod.f90:11–34](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/srdt_mod.f90#L11-L34) |

Evaluating an ad hoc srdt expression through Tao’s pipe interface returns a numerical value before the resonance-driving terms have been calculated. In a ring with either zero sextupole strength or K₂L = 10 m⁻², the ad hoc h₂₁₀₀₀ magnitude is the same 0.01000 m⁻¹ᐟ², while initialized data and an analytic sextupole formula give 0 and 3.990 m⁻¹ᐟ² respectively. The ad hoc h₂₂₀₀₀ real part is −1.000 m⁻¹ in both cases. Calculation is enabled by initialized data, but the expression path reads the stored fields and marks them valid without enabling it.

Run `python TAO-26.py` with [_bmad.py](../helpers/_bmad.py) and [_tao_contracts_normal_form.py](../helpers/_tao_contracts_normal_form.py) in the same directory.

??? example "Reproducer: TAO-26.py"

    [Download TAO-26.py](TAO-26.py)

    ```python
    """TAO-26: Ad hoc pipe srdt expressions return uncomputed values as valid

    Evaluating an ad hoc srdt expression through Tao’s pipe interface returns a numerical value before the resonance-driving terms have been calculated. In a ring with either zero sextupole strength or K₂L = 10 m⁻², the ad hoc h₂₁₀₀₀ magnitude is the same 0.01000 m⁻¹ᐟ², while initialized data and an analytic sextupole formula give 0 and 3.990 m⁻¹ᐟ² respectively. The ad hoc h₂₂₀₀₀ real part is −1.000 m⁻¹ in both cases. Calculation is enabled by initialized data, but the expression path reads the stored fields and marks them valid without enabling it.

    The script compares ad hoc queries with initialized srdt data in separate fresh processes for zero and nonzero sextupole strength. It checks the initialized values against an analytic thin-sextupole term using independent periodic optics, and prints the second uncomputed term as well.
    """

    from _bmad import compare, verdict
    from _tao_contracts_normal_form import THIN_RING, evaluated, tao, thin_beta

    expression = "lat::srdt.h21000.a"
    init = """&tao_d2_data
     d2_data%name = 'rdt'
     n_d1_data = 1
    /
    &tao_d1_data
     ix_d1_data = 1
     d1_data%name = 'h'
     datum(1)%data_type = 'srdt.h21000.a'
    /
    """
    bad = False
    for strength in (0, 10):
        # One sextupole at phase zero: |h21000|=K2L*beta**1.5/16.
        expected = strength * thin_beta() ** 1.5 / 16
        lat = THIN_RING.format(strength=strength)
        control = tao(
            lat,
            ["pipe evaluate " + expression],
            files={"rdt.init": init},
            options=["-lat", "lat.bmad", "-init", "rdt.init"],
        )
        assert not compare(
            f"K2L={strength}, initialized srdt control",
            evaluated(control.log, expression),
            expected,
            atol=2e-13,
        )
        result = tao(lat, ["pipe evaluate " + expression, "pipe evaluate lat::srdt.h22000.r"])
        bad |= compare(
            f"K2L={strength}, ad hoc srdt marked valid",
            evaluated(result.log, expression),
            expected,
            atol=2e-13,
        )
        print(
            f"K2L={strength}, ad hoc h22000.r marked valid: {evaluated(result.log, 'lat::srdt.h22000.r'):.16g}"
        )
    verdict(bad)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-10-24 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

