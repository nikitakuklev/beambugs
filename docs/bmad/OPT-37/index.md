---
title: "OPT-37: Averaged-wiggler I0 omits length and misnormalizes the planar average"
hide:
  - toc
---

# OPT-37 — Averaged-wiggler I0 omits length and misnormalizes the planar average

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [radiation_integrals averaged branch, bmad/code/radiation_integrals.f90:518–534](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/radiation_integrals.f90#L518-L534)<br>[Bmad manual, Synchrotron Radiation Integrals, bmad/doc/synchrotron-radiation.tex:283–295](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/synchrotron-radiation.tex#L283-L295)<br>[Bmad manual, photon-number relation, bmad/doc/synchrotron-radiation.tex:332–338](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/synchrotron-radiation.tex#L332-L338)<br>[Bmad manual, Periodic Type Wiggler fields, bmad/doc/elements.tex:5664–5695](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L5664-L5695) |

For averaged planar and helical wigglers, the photon-count radiation integral I₀ omits the element length; the planar formula also uses 2/3 instead of the absolute-cosine average 2/π. The integral I₀ = ∫γ|g|ds must scale with length and be additive under subdivision, where γ is the relativistic energy factor and g is curvature. At 650 MeV/c and 0.3 T, a 0.48 m planar wiggler returns 117.3 instead of 53.78, and splitting it into four pieces raises the result to 469.3. The corresponding helical values are 176.0 and 704.0 instead of 84.48; the squared- and cubed-curvature integrals remain correct.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _srdt_wiggler_maps.py](../helpers/_srdt_wiggler_maps.py) · [View source](../helpers/index.md#_srdt_wiggler_maps)

??? example "Reproducer: OPT-37.py"

    [Download OPT-37.py](OPT-37.py)

    ```python
    """OPT-37: Averaged-wiggler I0 omits length and misnormalizes the planar average

    For averaged planar and helical wigglers, the photon-count radiation integral I₀ omits the element length; the planar formula also uses 2/3 instead of the absolute-cosine average 2/π. The integral I₀ = ∫γ|g|ds must scale with length and be additive under subdivision, where γ is the relativistic energy factor and g is curvature. At 650 MeV/c and 0.3 T, a 0.48 m planar wiggler returns 117.3 instead of 53.78, and splitting it into four pieces raises the result to 469.3. The corresponding helical values are 176.0 and 704.0 instead of 84.48; the squared- and cubed-curvature integrals remain correct.

    The script compares complete-period planar and helical wigglers at two lengths and with four-piece subdivision. It derives I0, I2 and I3 from the absolute field moments, checks three cache modes and validates the I2/I3 and zero-field controls. BMAD_DIST selects the production build.
    """

    import math

    import numpy as np
    from _bmad import compare, verdict
    from _srdt_wiggler_maps import radiation

    LATTICE = """parameter[geometry]=open
    parameter[particle]=electron
    parameter[p0c]=650e6
    beginning[beta_a]=7
    beginning[alpha_a]=.3
    beginning[beta_b]=11
    beginning[alpha_b]=-.2
    particle_start[x]=XSTART
    m: SPEC
    line1: line=(PIECES*m)
    use,line1
    """
    defect = False
    for model in ("planar_model", "helical_model"):
        for length, pieces, field in ((0.24, 1, 0.3), (0.48, 1, 0.3), (0.48, 4, 0.3), (0.48, 1, 0)):
            spec = f"wiggler, l={length / pieces}, l_period=.06, b_max={field}, field_calc={model}, tracking_method=bmad_standard"
            lat = LATTICE.replace("XSTART", "0").replace("SPEC", spec).replace("PIECES", str(pieces))
            meta, total, _ = radiation(lat)
            # Derive energy and magnetic rigidity from the input momentum, the
            # electron rest energy and c; validate Bmad's reference metadata separately.
            mass, p0c, c = 510998.95069, 650e6, 299792458.0
            energy = math.hypot(p0c, mass)
            np.testing.assert_allclose(meta[:4], [p0c, mass, energy, c], rtol=2e-15, atol=0)
            gamma = energy / mass
            curvature = c * abs(field) / p0c
            # Absolute cosine moments over a complete period are 2/pi, 1/2, 4/(3*pi).
            # The helical model has constant |curvature|, so all three moments are one.
            moments = (
                np.array([2 / math.pi, 0.5, 4 / (3 * math.pi)])
                if model == "planar_model"
                else np.ones(3)
            )
            expected = length * np.array([gamma * curvature, curvature**2, curvature**3]) * moments
            np.testing.assert_allclose(total[2:4], expected[1:], rtol=4e-15, atol=1e-30)
            bad = compare(
                f"{model}, L={length} m, pieces={pieces}, B={field} T: dimensionless I0",
                total[0],
                expected[0],
                atol=2e-12,
                rtol=4e-15,
            )
            if field == 0 and bad:
                raise AssertionError("Zero-field control failed")
            defect |= bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed (averaged planar/helical I0; undulator/shared dispatch extensions have source support) | `gpt-6.1-sol` | xhigh | 2026-10-29 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

