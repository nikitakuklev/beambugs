---
title: "TAO-24: normal.h driving terms are the complex conjugates of Bengtsson’s"
hide:
  - toc
---

# TAO-24 — normal.h driving terms are the complex conjugates of Bengtsson’s

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [normal_form_rd_terms contract, bmad/ptc/ptc_layout_mod.f90:1075–1080](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/ptc_layout_mod.f90#L1075-L1080)<br>[normal_form_rd_terms phasor extraction, bmad/ptc/ptc_layout_mod.f90:1146–1152](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/ptc_layout_mod.f90#L1146-L1152)<br>[from_phasor, forest/code/Ci_tpsa.f90:3653–3656](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/forest/code/Ci_tpsa.f90#L3653-L3656)<br>[Tao Data Types, normal.h, tao/doc/data.tex:1394–1397](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/data.tex#L1394-L1397)<br>[normalform2 basis, tao/doc/data.tex:1464–1470](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/data.tex#L1464-L1470) |

Tao’s normal.h resonance-driving terms use the complex conjugate of the Bengtsson convention promised by its documentation. For one thin sextupole at nonzero betatron phase, normal.h gives h₂₁₀₀₀ = (0.5273 + 1.258i) m⁻¹ᐟ² while Bengtsson’s formula and srdt give (0.5273 − 1.258i) m⁻¹ᐟ². Eight cubic terms show the same conjugation: their real parts and magnitudes agree, while their imaginary parts reverse sign. The analytic sextupole generator and a strength-scaling control establish the convention mismatch.

Run `python TAO-24.py` with [_bmad.py](../helpers/_bmad.py) and [_tao_contracts_normal_form.py](../helpers/_tao_contracts_normal_form.py) in the same directory.

??? example "Reproducer: TAO-24.py"

    [Download TAO-24.py](TAO-24.py)

    ```python
    """TAO-24: normal.h driving terms are the complex conjugates of Bengtsson’s

    Tao’s normal.h resonance-driving terms use the complex conjugate of the Bengtsson convention promised by its documentation. For one thin sextupole at nonzero betatron phase, normal.h gives h₂₁₀₀₀ = (0.5273 + 1.258i) m⁻¹ᐟ² while Bengtsson’s formula and srdt give (0.5273 − 1.258i) m⁻¹ᐟ². Eight cubic terms show the same conjugation: their real parts and magnitudes agree, while their imaginary parts reverse sign. The analytic sextupole generator and a strength-scaling control establish the convention mismatch.

    The script defines all requested terms at initialization in a 16-bend FODO ring and subtracts the zero-sextupole background. It compares normal.h and srdt with Bengtsson’s analytic coefficients at the sextupole’s Twiss values and repeats at one hundredth strength.
    """

    import cmath

    import numpy as np
    from _bmad import compare, verdict
    from _tao_contracts_normal_form import block, tao

    RING = """parameter[geometry] = closed
    parameter[taylor_order] = 3
    beginning[e_tot] = 1e9
    qf: quadrupole, l = 0.5, k1 = 0.8
    qd: quadrupole, l = 0.5, k1 = -0.8
    b: sbend, l = 2, angle = 2*pi/16
    d: drift, l = 0.5
    cell: line = (qf, d, b, d, qd, d, b, d)
    s: multipole, k2l = {k2l}
    ring: line = (cell, qf, d, s, b, d, qd, d, b, d, 6*cell)
    use, ring
    """
    TERMS = ["h21000", "h30000", "h10110", "h10020", "h10200", "h20001", "h00201", "h10002"]
    PTC = {t: "normal.h." + t[1:5] + "0" + t[5] for t in TERMS}


    def bengtsson(b3l, ba, bb, eta, pa, pb):
        """Sextupole parts of Bengtsson's eqs. (96)-(97) for one thin sextupole."""
        e = cmath.exp
        return {
            "h21000": -b3l / 8 * ba**1.5 * e(1j * pa),
            "h30000": -b3l / 24 * ba**1.5 * e(3j * pa),
            "h10110": b3l / 4 * ba**0.5 * bb * e(1j * pa),
            "h10020": b3l / 8 * ba**0.5 * bb * e(1j * (pa - 2 * pb)),
            "h10200": b3l / 8 * ba**0.5 * bb * e(1j * (pa + 2 * pb)),
            "h20001": -2 * b3l * eta / 8 * ba * e(2j * pa),
            "h00201": 2 * b3l * eta / 8 * bb * e(2j * pb),
            "h10002": -b3l * eta / 2 * eta * ba**0.5 * e(1j * pa),
        }


    def values(strength):
        names = [f"srdt.{t}.{p}" for t in TERMS for p in "ri"]
        names += [f"{PTC[t]}.{p}" for t in TERMS for p in "ri"]
        datums = "\n".join(f' datum({i})%data_type = "{name}"' for i, name in enumerate(names, 1))
        init = (
            """&tao_design_lattice
     design_lattice(1)%file = 'lat.bmad'
     design_lattice(1)%one_turn_map_calc = T
    /
    &tao_params
     global%srdt_use_cache = F
    /
    &tao_d2_data
     d2_data%name = 'x'
     n_d1_data = 1
    /
    &tao_d1_data
     ix_d1_data = 1
     d1_data%name = 'd'
    """
            + datums
            + "\n/\n"
        )
        twiss_command = "pipe lat_list -no_slaves s ele.a.beta,ele.b.beta,ele.x.eta,ele.a.phi,ele.b.phi"
        result = tao(
            RING.format(k2l=strength),
            ["pipe data_d_array x.d", twiss_command],
            options=["-init", "rdt.init"],
            files={"rdt.init": init},
        )
        rows = [line.split(";") for line in block(result.log, "pipe data_d_array x.d").splitlines()]
        assert len(rows) == len(names)
        vals = {}
        for row in rows:
            assert len(row) == 14 and row[2] == "target" and row[13] == "T", row
            vals[row[1]] = float(row[7])
        twiss = [float(x) for x in block(result.log, twiss_command).split(";")]
        assert len(twiss) == 5 and np.isfinite(twiss).all()
        return vals, twiss


    def coefficient(vals, name):
        return complex(vals[name + ".r"], vals[name + ".i"])


    v1, twiss = values(1)
    v0, twiss0 = values(0)
    vs, twiss_small = values(0.01)
    assert not compare(
        "linear optics independent of thin sextupole strength", twiss, twiss0, atol=2e-13
    )
    assert not compare("linear optics at K2L=0.01", twiss_small, twiss0, atol=2e-13)
    print(f"SX beta_a,beta_b,eta_x (m), phi_a,phi_b (rad): {twiss}")
    # Bengtsson b3L=K2L/2; h+ = x-i*p and the transported factor is exp(+i*phi).
    hand = bengtsson(0.5, *twiss)
    bad = False
    errors = np.zeros(4)
    for term in TERMS:
        expected = hand[term]
        srdt = coefficient(v1, "srdt." + term) - coefficient(v0, "srdt." + term)
        ptc = coefficient(v1, PTC[term]) - coefficient(v0, PTC[term])
        small = 100 * (coefficient(vs, PTC[term]) - coefficient(v0, PTC[term]))
        if not np.isfinite([expected, srdt, ptc, small]).all():
            raise ValueError(f"Nonfinite coefficient: {term}")
        assert abs(srdt - expected) < 3e-13, (term, srdt, expected)
        # The degree-three generator terms are linear in K2L. The scaled subtraction
        # amplifies roundoff from the nonzero bend background by a factor of 100.
        assert abs(ptc - small) < 2e-9, (term, ptc, small)
        unit = "m^(-1/2)" if term in TERMS[:5] else "1" if term != "h10002" else "m^(1/2)"
        print(
            f"{term} [{unit}]: normal.h={ptc:.12g}, expected Bengtsson={expected:.12g}, srdt={srdt:.12g}, scaled weak={small:.12g}"
        )
        bad |= abs(ptc - expected) > 3e-11
        if abs(ptc - expected) > 3e-11:
            assert abs(ptc - expected.conjugate()) < 3e-11, (term, ptc, expected)
        errors = np.maximum(
            errors,
            [
                abs(srdt - expected),
                abs(ptc - expected.conjugate()),
                abs(ptc - expected),
                abs(ptc - small),
            ],
        )
    print(
        f"numeric residual maxima in each term's stated SI units [srdt-B, normal-conj(B), normal-B, strength scaling]: {errors.tolist()}"
    )
    verdict(bad)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>normal-form conjugation convention (source check)</small> | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`) |
| Confirmed<br><small>geometric h300000 and h210000 conjugation</small> | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |
| Confirmed | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |

