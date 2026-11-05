---
title: "OPT-27: srdt_lsq_solution gives every variable the same strength"
hide:
  - toc
---

# OPT-27 — srdt_lsq_solution gives every variable the same strength

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [srdt_lsq_solution, bmad/modules/srdt_mod.f90:729–800](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/srdt_mod.f90#L729-L800)<br>[multipole slice weight, bmad/modules/srdt_mod.f90:558–565](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/srdt_mod.f90#L558-L565)<br>[Bmad manual, Transfer Maps: Driving Terms, bmad/doc/subroutines.tex:3476–3513](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/subroutines.tex#L3476-L3513) |

When srdt_lsq_solution is asked to choose sextupole strengths that cancel first-order driving terms, it replaces every response column by the sum of all columns. The least-squares solver therefore returns the same strength for every variable and leaves large residuals. An eight-cell ring with 24 thick sextupoles returns the sextupole strength K₂ = 0.007165 m⁻³ everywhere and a largest first-order residual of 7.170, while an independently formed response matrix reduces all residuals below 2.139×10⁻¹¹. Source inspection also finds a negative slice weight for thin-multipole variables and incorrect signs and normalization for nonzero chromaticity targets.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _srdt_wiggler_maps.py](../helpers/_srdt_wiggler_maps.py) · [View source](../helpers/index.md#_srdt_wiggler_maps)

??? example "Reproducer: OPT-27.py"

    [Download OPT-27.py](OPT-27.py)

    ```python
    """OPT-27: srdt_lsq_solution gives every variable the same strength

    When srdt_lsq_solution is asked to choose sextupole strengths that cancel first-order driving terms, it replaces every response column by the sum of all columns. The least-squares solver therefore returns the same strength for every variable and leaves large residuals. An eight-cell ring with 24 thick sextupoles returns the sextupole strength K₂ = 0.007165 m⁻³ everywhere and a largest first-order residual of 7.170, while an independently formed response matrix reduces all residuals below 2.139×10⁻¹¹. Source inspection also finds a negative slice weight for thin-multipole variables and incorrect signs and normalization for nonzero chromaticity targets.

    The script reconstructs the 18 by 24 response matrix by varying one sextupole at a time in thick and thin FODO rings. NumPy computes the minimum-norm solution and the script reapplies it in Bmad as a control, then compares the library solver result. BMAD_DIST selects the production build.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _srdt_wiggler_maps import LSQ_PROGRAM, THICK_RING, THIN_RING, records

    defect = False
    for name, lattice in (("thick", THICK_RING), ("thin", THIN_RING)):
        files = {"lat.bmad": lattice}
        columns = fortran(LSQ_PROGRAM, ["lat.bmad", "columns"], files=files).log
        b = records(columns, "B")[0]
        a = records(columns, "A")[:, 1:].T
        if a.shape != (18, 24):
            raise RuntimeError("Expected 24 response columns")
        # First-order resonance terms are affine in sextupole strength. SVD gives
        # the minimum Euclidean-norm solution of A*x = -b independently of the solver.
        expected = -np.linalg.pinv(a) @ b
        files["strengths.txt"] = " ".join(map(str, expected)) + "\n"
        applied = fortran(LSQ_PROGRAM, ["lat.bmad", "apply", "strengths.txt"], files=files).log
        residual = records(applied, "Q")[0]
        if compare(f"{name}: independent solution residual", residual, np.zeros(18), atol=5e-10):
            raise AssertionError("Independent minimum-norm control failed")
        log = fortran(LSQ_PROGRAM, ["lat.bmad", "lsq", "0", "0"], files=files).log
        actual = records(log, "SOLN")[0]
        defect |= compare(
            f"{name}: 24 strengths (K2 m^-3 or K2L m^-2)", actual, expected, atol=5e-10, rtol=2e-10
        )
        print(
            f"{name}: max first-order residual={np.max(np.abs(records(log, 'Q'))):.16g}; expected <5e-10"
        )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed<br><small>rank-one solver failure; thin weights/targets source-only</small> | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

