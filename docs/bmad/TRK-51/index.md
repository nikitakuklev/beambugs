---
title: "TRK-51: A straight_reference multipole gives no on-momentum native dipole kick"
hide:
  - toc
---

# TRK-51 — A straight_reference multipole gives no on-momentum native dipole kick

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [track1_bmad multipole dispatch](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track1_bmad.f90#L142-L161)<br>[reference-bending kick](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/multipole_mod.f90#L248-L254)<br>[Bmad Manual, “Multipole” (s:mult)](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L3860-L3878) |

A multipole with nonzero integrated dipole strength K₀L and straight_reference should apply a simple dipole kick, but native tracking uses the reference-bending map instead. For an on-axis, on-momentum particle, K₀L = 0.0007300 produces zero normalized horizontal momentum pₓ instead of −0.0007300. PTC and an equivalent ab_multipole give the expected kick and displacement after a 1 m drift. The native dispatch ignores K0L_status when selecting the reference-orbit correction.

Run `python TRK-51.py` with [_bmad.py](../helpers/_bmad.py) and [_tracking_contracts_materials.py](../helpers/_tracking_contracts_materials.py) in the same directory.

??? example "Reproducer: TRK-51.py"

    [Download TRK-51.py](TRK-51.py)

    ```python
    """TRK-51: A straight_reference multipole gives no on-momentum native dipole kick

    A multipole with nonzero integrated dipole strength K₀L and straight_reference should apply a simple dipole kick, but native tracking uses the reference-bending map instead. For an on-axis, on-momentum particle, K₀L = 0.0007300 produces zero normalized horizontal momentum pₓ instead of −0.0007300. PTC and an equivalent ab_multipole give the expected kick and displacement after a 1 m drift. The native dispatch ignores K0L_status when selecting the reference-orbit correction.

    The script tracks straight-reference, reference-bending and equivalent ab_multipole cases with
    native and PTC methods. It compares the dipole kick and exact downstream drift with an analytic
    expectation and checks the reference-bending control.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _tracking_contracts_materials import lattice, orbit, tao

    k = 0.00073
    # A thin straight dipole changes canonical px by -K0L; exact drift uses px/p_s.
    expected = np.array([-k / np.sqrt(1 - k * k), -k])
    defects = []
    for name, element in [
        ("straight", "multipole, k0l_status=straight_reference, k0l=.00073"),
        ("bend", "multipole, k0l_status=bends_reference, k0l=.00073"),
        ("ab", "ab_multipole, b0=.00073"),
    ]:
        for method in ("bmad_standard", "symp_lie_ptc"):
            lat = lattice(f"e: {element}, tracking_method={method}\nd: drift,l=1", line="e,d")
            cmd = "pipe ele:orbit 2"
            actual = orbit(tao(lat, [cmd])[cmd])[:2]
            target = np.zeros(2) if name == "bend" else expected
            bad = compare(
                f"{name} {method} (x m, px), dipole plus exact drift",
                actual,
                target,
                atol=3e-15,
                rtol=3e-13,
            )
            if name == "straight" and method == "bmad_standard":
                defects.append(bad)
            elif bad:
                raise AssertionError("Dipole control failed")
    verdict(any(defects))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-10-22</span> | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

