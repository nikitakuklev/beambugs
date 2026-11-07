---
title: "TAO-25: New normal.h monomials return zero after the first cached calculation"
hide:
  - toc
---

# TAO-25 — New normal.h monomials return zero after the first cached calculation

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [normal.h append and cache gate, tao/code/tao_evaluate_a_datum.f90:1812–1818](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_evaluate_a_datum.f90#L1812-L1818)<br>[tao_add_to_normal_mode_h_array, tao/code/tao_init_data_mod.f90:918–940](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_init_data_mod.f90#L918-L940)<br>[cache reset, tao/code/tao_ptc_normal_form.f90:49–50](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_ptc_normal_form.f90#L49-L50)<br>[normal.h data contract, tao/doc/data.tex:1394–1403](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/data.tex#L1394-L1403) |

Requesting a new normal.h monomial after the first cached calculation in a Tao session returns zero instead of its coefficient. In a thin-sextupole ring, h₃₀₀₀₀₀ is −1.330 m⁻¹ᐟ² when requested first, but a later h₂₁₀₀₀₀ query returns zero instead of −3.990 m⁻¹ᐟ². Reversing the request order reverses which term is lost; registering both before calculation gives both nonzero values. Independent thin-lens optics and the sextupole generator predict the coefficients, while the source shows that adding a monomial leaves the cache marked complete.

Run `python TAO-25.py` with [_bmad.py](../helpers/_bmad.py) and [_tao_contracts_normal_form.py](../helpers/_tao_contracts_normal_form.py) in the same directory.

??? example "Reproducer: TAO-25.py"

    [Download TAO-25.py](TAO-25.py)

    ```python
    """TAO-25: New normal.h monomials return zero after the first cached calculation

    Requesting a new normal.h monomial after the first cached calculation in a Tao session returns zero instead of its coefficient. In a thin-sextupole ring, h₃₀₀₀₀₀ is −1.330 m⁻¹ᐟ² when requested first, but a later h₂₁₀₀₀₀ query returns zero instead of −3.990 m⁻¹ᐟ². Reversing the request order reverses which term is lost; registering both before calculation gives both nonzero values. Independent thin-lens optics and the sextupole generator predict the coefficients, while the source shows that adding a monomial leaves the cache marked complete.

    The script runs both query orders in fresh processes on the same ring and checks a control that registers both terms before calculation. It computes the periodic beta from independent thin-lens matrices and compares the coefficients against the analytic sextupole generator.
    """

    from _bmad import compare, verdict
    from _tao_contracts_normal_form import THIN_RING, evaluated, tao, thin_beta

    terms = ["lat::normal.h.300000.r", "lat::normal.h.210000.r"]
    # At SX the phase is zero. K2L=10 means b3L=5: h30000=-K2L*beta**1.5/48;
    # the combinatorial coefficient of h21000 is three times larger.
    beta = thin_beta()
    expected = [-10 * beta**1.5 / 48, -10 * beta**1.5 / 16]
    print(f"independent thin-lens periodic beta at SX: {beta:.15g} m")
    init = """&tao_design_lattice
     design_lattice(1)%file = 'lat.bmad'
     design_lattice(1)%one_turn_map_calc = T
    /
    &tao_d2_data
     d2_data%name = 'rdt'
     n_d1_data = 1
    /
    &tao_d1_data
     ix_d1_data = 1
     d1_data%name = 'h'
     datum(1)%data_type = 'normal.h.300000.r'
     datum(2)%data_type = 'normal.h.210000.r'
    /
    """
    control = tao(
        THIN_RING.format(strength=10),
        ["pipe evaluate " + term for term in terms],
        options=["-init", "rdt.init"],
        files={"rdt.init": init},
    )
    for term, target in zip(terms, expected, strict=True):
        assert not compare(
            f"registered before calculation {term} (m^(-1/2))",
            evaluated(control.log, term),
            target,
            atol=2e-11,
        )
    bad = False
    for order in ([0, 1], [1, 0]):
        result = tao(
            THIN_RING.format(strength=10),
            ["set universe 1 one_turn_map_calc = T", *["pipe evaluate " + terms[i] for i in order]],
        )
        first, second = order
        assert not compare(
            f"first request {terms[first]}, analytic sextupole coefficient",
            evaluated(result.log, terms[first]),
            expected[first],
            atol=2e-11,
        )
        bad |= compare(
            f"second request {terms[second]}, same analytic coefficient",
            evaluated(result.log, terms[second]),
            expected[second],
            atol=2e-11,
        )
    verdict(bad)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-10-24 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

