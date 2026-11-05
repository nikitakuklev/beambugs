---
title: "OPT-30: Tilted Taylor-map Jacobians lose the coordinate-frame transformations"
hide:
  - toc
---

# OPT-30 — Tilted Taylor-map Jacobians lose the coordinate-frame transformations

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [track1_taylor, bmad/low_level/track1_taylor.f90:83–116](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track1_taylor.f90#L83-L116)<br>[Bmad manual, taylor_map_include_offsets Attribute, bmad/doc/methods.tex:919–938](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L919-L938) |

With taylor_map_includes_offsets set to false, the Taylor tracker returns a body-frame Jacobian for a tilted element even though its particle trajectories are transformed correctly. The laboratory Jacobian must include the entrance and exit rotations. For a 0.8 m quadrupole with normalized strength 0.7 m⁻² and tilt 0.31 rad, M₂₃ is zero instead of −0.3259 m⁻¹, where M₂₃ is the derivative of horizontal normalized momentum with respect to vertical position. Including offsets in the map, or setting the tilt to zero, gives the correct matrix.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _srdt_wiggler_maps.py](../helpers/_srdt_wiggler_maps.py) · [View source](../helpers/index.md#_srdt_wiggler_maps)

??? example "Reproducer: OPT-30.py"

    [Download OPT-30.py](OPT-30.py)

    ```python
    """OPT-30: Tilted Taylor-map Jacobians lose the coordinate-frame transformations

    With taylor_map_includes_offsets set to false, the Taylor tracker returns a body-frame Jacobian for a tilted element even though its particle trajectories are transformed correctly. The laboratory Jacobian must include the entrance and exit rotations. For a 0.8 m quadrupole with normalized strength 0.7 m⁻² and tilt 0.31 rad, M₂₃ is zero instead of −0.3259 m⁻¹, where M₂₃ is the derivative of horizontal normalized momentum with respect to vertical position. Including offsets in the map, or setting the tilt to zero, gives the correct matrix.

    The script builds first-order Taylor maps of a tilted quadrupole with offsets excluded, offsets included, and zero tilt. It compares the returned transverse matrix and same-map trajectory differences with the analytically rotated focusing and defocusing blocks. BMAD_DIST selects the production build.
    """

    from _bmad import compare, fortran, verdict
    from _srdt_wiggler_maps import MAP_LATTICE, MAP_PROGRAM, records

    log = fortran(MAP_PROGRAM, files={"lat.bmad": MAP_LATTICE}).log
    rows = records(log, "TAYLOR")
    if rows.shape != (3, 6):
        raise RuntimeError("Incomplete Taylor-map result")
    defect = False
    for row in rows:
        i, actual, fd, analytic, fd_error, matrix_error = row
        if compare(f"case {int(i)}: finite-difference M23 (1/m)", fd, analytic, atol=2e-13):
            raise AssertionError("Rotated-quadrupole oracle failed")
        bad = compare(f"case {int(i)}: reported M23 (1/m)", actual, analytic, atol=2e-13)
        if i > 1 and bad:
            raise AssertionError("Offsets-included or untilted control failed")
        print(
            f"case {int(i)}: maximum transverse matrix error={matrix_error:.16g}; finite-difference error={fd_error:.16g}"
        )
        defect |= bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

