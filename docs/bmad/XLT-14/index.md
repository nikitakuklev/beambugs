---
title: "XLT-14: MAD export uses incorrect finite-beta starting T and PT"
hide:
  - toc
---

# XLT-14 — MAD export uses incorrect finite-beta starting T and PT

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [write_lattice_mad_format starting coordinates, bmad/output/write_lattice_mad_format.f90:1126–1139](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L1126-L1139)<br>[bmad/doc/phase-space.tex:227–240](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/phase-space.tex#L227-L240) |

Exporting a finite-beta starting particle to MAD-X writes incorrect longitudinal and energy coordinates. The writer uses T = zβ₀ and PT = δ/β₀, where z is the Bmad longitudinal coordinate, δ is relative momentum offset, and β₀ is reference speed divided by light speed. The required coordinates are T = z/β and PT = (E − E₀)/(p₀c), with particle speed βc, particle energy E, reference energy E₀ and reference momentum p₀. A 1 MeV/c electron with z = 0.03 m and δ = 0.2 gets T = 0.02671 m instead of 0.03261 m and PT = 0.2246 instead of 0.1813; the transverse coordinates remain correct.

Run `python XLT-14.py` with [_bmad.py](../helpers/_bmad.py) and [_lattice_import_export.py](../helpers/_lattice_import_export.py) in the same directory.

??? example "Reproducer: XLT-14.py"

    [Download XLT-14.py](XLT-14.py)

    ```python
    """XLT-14: MAD export uses incorrect finite-beta starting T and PT

    Exporting a finite-beta starting particle to MAD-X writes incorrect longitudinal and energy coordinates. The writer uses T = zβ₀ and PT = δ/β₀, where z is the Bmad longitudinal coordinate, δ is relative momentum offset, and β₀ is reference speed divided by light speed. The required coordinates are T = z/β and PT = (E − E₀)/(p₀c), with particle speed βc, particle energy E, reference energy E₀ and reference momentum p₀. A 1 MeV/c electron with z = 0.03 m and δ = 0.2 gets T = 0.02671 m instead of 0.03261 m and PT = 0.2246 instead of 0.1813; the transverse coordinates remain correct.

    The script exports the finite-beta drift fixture and compares its starting record with relativistic energy and speed. It checks all transverse coordinates and initial Twiss fields before evaluating T and PT.
    """

    import math

    from _bmad import compare, verdict
    from _lattice_import_export import HEADER, definition, export, number

    text = export(
        HEADER.replace("3e9", "1e6")
        .replace("beta_a] = 10", "beta_a] = 3")
        .replace("beta_b] = 10", "beta_b] = 4")
        + """beginning[alpha_a]=.4
    particle_start[x]=.001
    particle_start[y]=-.002
    particle_start[z]=.03
    particle_start[pz]=.2
    e:drift,l=.1
    lat:line=(e)
    use,lat
    """
    )
    initial = definition(text, "initial")
    control = [number(initial, k) for k in ["x", "px", "y", "py", "betx", "bety", "alfx", "alfy"]]
    if compare(
        "transverse coordinates and Twiss control",
        control,
        [0.001, 0, -0.002, 0, 3, 4, 0.4, 0],
        atol=2e-15,
    ):
        raise RuntimeError("Transverse/Twiss control failed")
    p0, mass, delta, z = 1e6, 510998.95069, 0.2, 0.03
    # Bmad z=-beta*c*dt; MAD-X T=-c*dt. PT is energy offset divided by p0*c.
    p = p0 * (1 + delta)
    beta = p / math.hypot(p, mass)
    expected = [z / beta, (math.hypot(p, mass) - math.hypot(p0, mass)) / p0]
    verdict(
        compare(
            "MAD-X [T (m),PT] from exact kinematics",
            [number(initial, k) for k in ["t", "pt"]],
            expected,
            atol=2e-16,
            rtol=3e-14,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

