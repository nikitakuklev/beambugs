---
title: "TRK-09: Polynomial surface derivatives corrupt curved-detector coordinates"
hide:
  - toc
---

# TRK-09 — Polynomial surface derivatives corrupt curved-detector coordinates

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [z_at_surface, polynomial and ellipse, bmad/photon/photon_utils_mod.f90:132–148](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/photon/photon_utils_mod.f90#L132-L148)<br>[qfunc, arc-length integrand, bmad/photon/to_surface_coords.f90:34–39](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/photon/to_surface_coords.f90#L34-L39)<br>[Bmad manual, Surface Properties for X-Ray elements, xs2ij4, bmad/doc/attributes.tex:1262–1279](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L1262-L1279)<br>[Detector, bmad/doc/elements.tex:1592–1594](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L1592-L1594) |

Curved photon detectors combining polynomial and elliptical curvature report the wrong surface coordinate because polynomial height derivatives have reversed signs. For z=−0.7x²+(√(1−0.64x²)−1)/1.3 at x=0.17 m, the reported slope is 0.1535 instead of −0.3225. The detector’s arc coordinate is 0.1707 m instead of 0.1729 m, an error of −2.228 mm. Analytic derivatives, finite differences and independent quadrature agree; pure polynomial or pure elliptical curvature gives the correct arc length.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _s3_19_fields_photons_ptc.py](../helpers/_s3_19_fields_photons_ptc.py) · [View source](../helpers/index.md#_s3_19_fields_photons_ptc)

??? example "Reproducer: TRK-09.py"

    [Download TRK-09.py](TRK-09.py)

    ```python
    """TRK-09: Polynomial surface derivatives corrupt curved-detector coordinates

    Curved photon detectors combining polynomial and elliptical curvature report the wrong surface
    coordinate because polynomial height derivatives have reversed signs. For
    z=−0.7x²+(√(1−0.64x²)−1)/1.3 at x=0.17 m, the reported slope is 0.1535 instead of −0.3225. The
    detector’s arc coordinate is 0.1707 m instead of 0.1729 m, an error of −2.228 mm. Analytic
    derivatives, finite differences and independent quadrature agree; pure polynomial or pure elliptical
    curvature gives the correct arc length.

    The script evaluates polynomial, elliptical and mixed one-dimensional detector surfaces at x=0.17 m.
    It checks derivatives against analytic formulas and a fourth-order stencil, then compares surface
    coordinates with independent arc-length quadrature; BMAD_DIST selects the build.
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


    import math

    from _bmad import compare, verdict
    from _s3_19_fields_photons_ptc import control, photon_data, rows
    from scipy.integrate import quad

    output = photon_data(4)
    data = rows(output, "DETECTOR")
    assert len(data) == 3 and {int(r[0]) for r in data} == {1, 2, 3}
    defect = False
    for row in data:
        index = int(row[0])
        x, z, reported_slope, fd_slope, actual_arc = map(float, row[1:])
        polynomial = 0.7 if index != 2 else 0
        ellipse = 0.64 / 1.3 if index != 1 else 0
        control(z, -polynomial * x * x + (math.sqrt(1 - 0.64 * x * x) - 1) / 1.3 * (index != 1), 2e-16)

        # Derivative of negative polynomial plus ellipse; surface distance is its arc integral.
        def slope(xx, polynomial=polynomial, ellipse=ellipse):
            return -2 * polynomial * xx - ellipse * xx / math.sqrt(1 - 0.64 * xx * xx)

        expected_slope = slope(x)
        arc, error = quad(lambda xx: math.sqrt(1 + slope(xx) ** 2), 0, x, epsabs=2e-15, epsrel=2e-14)
        assert error < 3e-15
        control(fd_slope, expected_slope, 3e-13)
        if index != 3:
            control(actual_arc, arc, 5e-13)
        defect |= compare(
            f"detector={index} slope, analytic derivative", reported_slope, expected_slope, atol=3e-13
        )
        defect |= compare(
            f"detector={index} arc (m), integral sqrt(1+z'²)", actual_arc, arc, atol=5e-13
        )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

