---
title: "TRK-08: Segmented surfaces reverse their height slopes"
hide:
  - toc
---

# TRK-08 — Segmented surfaces reverse their height slopes

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [z_at_surface, segmented plane, bmad/photon/photon_utils_mod.f90:118–122](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/photon/photon_utils_mod.f90#L118-L122)<br>[init_surface_segment, bmad/parsing/bmad_parser_mod.f90:8222–8299](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/bmad_parser_mod.f90#L8222-L8299)<br>[Bmad manual, Displacement, H_Misalign, and Segmented Surface Grids, bmad/doc/attributes.tex:1469–1481](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L1469-L1481) |

Segmented photon surfaces reverse the signed height slopes of their flat facets. For a facet of the surface z=−1.7x²−0.6y², the height at (93, −101) mm is −23.60 mm instead of the supporting plane’s −25.20 mm. Returned slope values describe the correct plane, while finite differences of the returned heights have the opposite signs. A photon intersection follows the incorrect height, producing a 1.600 mm displacement error.

Run `python TRK-08.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_19_fields_photons_ptc.py](../helpers/_s3_19_fields_photons_ptc.py) in the same directory.

??? example "Reproducer: TRK-08.py"

    [Download TRK-08.py](TRK-08.py)

    ```python
    """TRK-08: Segmented surfaces reverse their height slopes

    Segmented photon surfaces reverse the signed height slopes of their flat facets. For a facet of the
    surface z=−1.7x²−0.6y², the height at (93, −101) mm is −23.60 mm instead of the supporting plane’s
    −25.20 mm. Returned slope values describe the correct plane, while finite differences of the
    returned heights have the opposite signs. A photon intersection follows the incorrect height,
    producing a 1.600 mm displacement error.

    The script constructs a segmented quadratic surface, derives the plane through its four support
    vertices, and compares heights and slopes with z_at_surface. It also checks track_to_surface and
    samples just inside each corner to keep facet selection unambiguous;
    """

    import os

    # Set numerical pool sizes before NumPy/SciPy are imported.
    for _thread_variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_thread_variable] = "1"


    from _bmad import compare, verdict
    from _s3_19_fields_photons_ptc import control, photon_data, rows

    output = photon_data(3)
    x, y, actual, reported_x, reported_y = map(float, rows(output, "SEG")[0])
    x0, y0, z0, sx, sy = map(float, rows(output, "PLANE")[0])
    # The four supporting vertices of this separable quadratic lie on one plane.
    expected_z0 = -1.7 * (x0 * x0 + 0.04**2) - 0.6 * (y0 * y0 + 0.06**2)
    control([z0, sx, sy], [expected_z0, -3.4 * x0, -1.2 * y0], 1e-16)
    expected = expected_z0 + sx * (x - x0) + sy * (y - y0)
    fd_x, fd_y = map(float, rows(output, "SEG_FD")[0])
    control([reported_x, reported_y], [sx, sy], 1e-16)
    defect = compare("Facet height (m), supporting plane", actual, expected, atol=1e-13)
    defect |= compare("Finite-difference slopes, supporting plane", [fd_x, fd_y], [sx, sy], atol=1e-12)
    hit = rows(output, "SEG_HIT")[0]
    assert int(hit[0]) == 1
    control(float(hit[5]), actual, 1e-13)
    defect |= compare("Ray intersection z (m), supporting plane", float(hit[5]), expected, atol=1e-13)
    corners = rows(output, "CORNER")
    assert len(corners) == 4
    for row in corners:
        xx, yy, zz = map(float, row)
        defect |= compare(
            "Support corner height (m), quadratic surface", zz, -1.7 * xx**2 - 0.6 * yy**2, atol=2e-12
        )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

