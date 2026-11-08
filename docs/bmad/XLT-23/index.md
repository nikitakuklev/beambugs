---
title: "XLT-23: MAD-X sbend k1 export ignores the changed reference momentum"
hide:
  - toc
---

# XLT-23 — MAD-X sbend k1 export ignores the changed reference momentum

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [quadrupole and sbend k1 serialization, bmad/output/write_lattice_mad_format.f90:740–756](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L740-L756)<br>[lcavity reference-momentum update, bmad/output/write_lattice_mad_format.f90:927–934](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L927-L934) |

Exporting a combined-function bend after an accelerating cavity leaves its normalized quadrupole gradient k₁ at the local momentum normalization. The MAD-X writer converts the cavity to an RF cavity and renormalizes downstream strengths to the starting momentum, including an ordinary quadrupole with the same k₁. After acceleration from 1 GeV to 2 GeV, k₁ = 0.2 m⁻² exports as 0.4000 m⁻² for the quadrupole but remains 0.2 m⁻² for the bend. Magnetic-rigidity scaling requires 0.4000 m⁻² for both, so the bend’s exported gradient is half the required value.

Run `python XLT-23.py` with [_bmad.py](../helpers/_bmad.py) and [_lattice_import_export.py](../helpers/_lattice_import_export.py) in the same directory.

??? example "Reproducer: XLT-23.py"

    [Download XLT-23.py](XLT-23.py)

    ```python
    """XLT-23: MAD-X sbend k1 export ignores the changed reference momentum

    Exporting a combined-function bend after an accelerating cavity leaves its normalized quadrupole gradient k₁ at the local momentum normalization. The MAD-X writer converts the cavity to an RF cavity and renormalizes downstream strengths to the starting momentum, including an ordinary quadrupole with the same k₁. After acceleration from 1 GeV to 2 GeV, k₁ = 0.2 m⁻² exports as 0.4000 m⁻² for the quadrupole but remains 0.2 m⁻² for the bend. Magnetic-rigidity scaling requires 0.4000 m⁻² for both, so the bend’s exported gradient is half the required value.

    The script exports matching quadrupole and bend strengths after an lcavity and after a drift control. It derives the required strength ratio from the reference momentum and checks the quadrupole export before comparing the bend.
    """

    import math

    from _bmad import compare, verdict
    from _lattice_import_export import HEADER, definition, inspect, number, records, valid

    defect = False
    for cavity in [False, True]:
        first = "lcavity,l=1,voltage=1e9,rf_frequency=1e9" if cavity else "drift,l=1"
        lattice = (
            HEADER.replace("parameter[p0c] = 3e9", "parameter[E_tot] = 1e9")
            + f"""c:{first}
    q:quadrupole,l=.5,k1=.2
    b:sbend,l=1,angle=.01,k1=.2
    lat:line=(c,q,b)
    use,lat
    """
        )
        result = inspect(lattice, "MAD-X")
        valid(result)
        text = result.outputs["out"].decode()
        p0c = records(result, "P0C")
        ratio = p0c[1][0] / p0c[0][0]
        # Fixed gradient G has k1=qG/p0; normalizing to the initial p0 multiplies by p_local/p_initial.
        mass = 510998.95069
        expected_ratio = math.sqrt(((2 if cavity else 1) * 1e9) ** 2 - mass**2) / math.sqrt(
            1e18 - mass**2
        )
        if compare("reference momentum ratio control", ratio, expected_ratio, atol=0, rtol=3e-14):
            raise RuntimeError("Reference momentum control failed")
        expected = 0.2 * expected_ratio
        if compare(
            f"acceleration={cavity}: quadrupole k1 control",
            number(definition(text, "Q"), "k1"),
            expected,
            atol=0,
            rtol=3e-14,
        ):
            raise RuntimeError("Quadrupole normalization control failed")
        bad = compare(
            f"acceleration={cavity}: sbend k1 (m^-2)",
            number(definition(text, "B"), "k1"),
            expected,
            atol=0,
            rtol=3e-14,
        )
        if not cavity and bad:
            raise RuntimeError("Nonaccelerating bend control failed")
        defect |= bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-10-22</span> | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

