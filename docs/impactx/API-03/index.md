---
title: "API-03: Marker insertion multiplies an exact bend's angle"
hide:
  - toc
---

# API-03 — Marker insertion multiplies an exact bend's angle

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P1 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX development (`fa0a43b`) |
| Issue | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1723]], filed from this evidence |
| Pull request | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1724]], filed from this evidence |
| Source | [`Insert.cpp#L64-L74`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/transformation/Insert.cpp#L64-L74) |

`insert_element_every_ds()`, which places a thin element such as a `Marker` at regular spacing by splitting the thick elements it falls in, gives each piece of a split `ExactSbend` the full original angle `phi`. A uniform sector bend of length L and angle θ must keep its total deflection and its arc of radius L/θ when split, so a piece of length Lᵢ needs the angle θ·Lᵢ/L; the analytic exit coordinates of that arc fix the correct result. A marker at the midpoint of a 1 m, 20° `ExactSbend` makes it bend by 40° in total, and markers every 0.3 m make 80°, while the path length stays 1 m. Exact bends given by angle alone (`B=0`) or with an explicit field are affected; `Sbend`, `CFbend`, `Quad` and `ChrAcc` split exactly.

??? example "Reproducer: API-03.py"

    [Download API-03.py](API-03.py)

    ```python
    """API-03: Marker insertion multiplies an exact bend's angle

    `insert_element_every_ds()`, which places a thin element such as a `Marker` at regular spacing by
    splitting the thick elements it falls in, gives each piece of a split `ExactSbend` the full original
    angle `phi`. A uniform sector bend of length L and angle θ must keep its total deflection and its
    arc of radius L/θ when split, so a piece of length Lᵢ needs the angle θ·Lᵢ/L; the analytic exit
    coordinates of that arc fix the correct result. A marker at the midpoint of a 1 m, 20° `ExactSbend`
    makes it bend by 40° in total, and markers every 0.3 m make 80°, while the path length stays 1 m.
    Exact bends given by angle alone (`B=0`) or with an explicit field are affected; `Sbend`, `CFbend`,
    `Quad` and `ChrAcc` split exactly.

    The script pushes a 1000 MeV proton reference particle through a 1 m, 20° `ExactSbend` given by
    angle alone and through copies split by inert markers every 0.5 m and 0.3 m, and compares the exit
    position, direction and path length with the analytic arc x = (L/θ)(cos θ − 1), z = (L/θ) sin θ. The
    unsplit bend must match the arc within 2×10⁻¹⁵ (m or rad); as a control, setting each piece's angle
    to θ·Lᵢ/L restores the arc to the same tolerance.
    """

    import os

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[name] = "1"

    from math import atan2, cos, degrees, radians, sin

    import numpy as np
    from impactx import RefPart, elements, push

    LENGTH = 1.0
    ANGLE = 20.0
    theta = radians(ANGLE)
    radius = LENGTH / theta
    expected = np.array([radius * (cos(theta) - 1), radius * sin(theta), theta, LENGTH])


    def track(lattice):
        ref = RefPart().set_species("proton")
        ref.set_kin_energy_MeV(1000)
        for el in lattice:
            push(ref, el)
        return np.array([ref.x, ref.z, atan2(-ref.px, ref.pz), ref.s])


    original = elements.KnownElementsList([elements.ExactSbend(ds=LENGTH, phi=ANGLE)])
    baseline = track(original)
    np.testing.assert_allclose(baseline, expected, rtol=0, atol=2e-15)
    print(f"original: angle_deg={degrees(baseline[2]):.15g} x={baseline[0]:.15g} z={baseline[1]:.15g}")
    for spacing in (0.5, 0.3):
        split = elements.transformation.insert_element_every_ds(
            original, ds=spacing, element=elements.Marker("inert")
        )
        pieces = [el for el in split if isinstance(el, elements.ExactSbend)]
        assert original[0].ds == LENGTH and abs(original[0].phi - theta) < 1e-14
        assert abs(sum(el.ds for el in pieces) - LENGTH) < 1e-15
        observed = track(split)
        np.testing.assert_allclose(observed[2], len(pieces) * theta, rtol=0, atol=2e-15)
        print(
            f"spacing={spacing}: piece_lengths={[el.ds for el in pieces]} piece_angles_deg={[degrees(el.phi) for el in pieces]}"
        )
        print(
            f"  observed_angle_deg={degrees(observed[2]):.15g} expected_angle_deg={ANGLE:g} s={observed[3]:.15g}"
        )
        assert np.max(np.abs(observed - expected)) > 0.1
        # A sector with the same field has the same curvature in every segment.
        for el in pieces:
            # Constructor phi is degrees; this property's documented units are radians.
            el.phi = theta * el.ds / LENGTH
        repaired = track(split)
        np.testing.assert_allclose(repaired, expected, rtol=0, atol=2e-15)
        print(
            f"  angle-scaled control: angle_deg={degrees(repaired[2]):.15g} max_arc_error={np.max(np.abs(repaired - expected)):.3g}"
        )
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

