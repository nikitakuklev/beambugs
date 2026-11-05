---
title: "TRK-12: Laue displacement includes an absent polarization component"
hide:
  - toc
---

# TRK-12 — Laue displacement includes an absent polarization component

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [track1_crystal, trajectory weighting, bmad/photon/track1_photon_mod.f90:961–972](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/photon/track1_photon_mod.f90#L961-L972)<br>[Bmad manual, Incoherent Photon Tracking, panda1, bmad/doc/xray-tracking.tex:27–48](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/xray-tracking.tex#L27-L48) |

Laue crystal tracking includes an absent polarization when calculating the displacement of a purely polarized photon. A 20 µm silicon (111) crystal with asymmetry parameter 0.55 gives the same surface-frame displacement to pure π, pure σ and mixed inputs. For an incoming angle of −20 µrad, pure σ public tracking returns exit x=0.3413 µm instead of 0.3786 µm, an error of −37.27 nm. Independent two-beam dispersion and energy-flow calculations recover the individual polarization paths within the implemented branch-averaging model.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _s3_19_fields_photons_ptc.py](../helpers/_s3_19_fields_photons_ptc.py) · [View source](../helpers/index.md#_s3_19_fields_photons_ptc)

??? example "Reproducer: TRK-12.py"

    [Download TRK-12.py](TRK-12.py)

    ```python
    """TRK-12: Laue displacement includes an absent polarization component

    Laue crystal tracking includes an absent polarization when calculating the displacement of a purely
    polarized photon. A 20 µm silicon (111) crystal with asymmetry parameter 0.55 gives the same
    surface-frame displacement to pure π, pure σ and mixed inputs. For an incoming angle of −20 µrad,
    pure σ public tracking returns exit x=0.3413 µm instead of 0.3786 µm, an error of −37.27 nm.
    Independent two-beam dispersion and energy-flow calculations recover the individual polarization
    paths within the implemented branch-averaging model.

    The script tracks pure π, pure σ and mixed polarization through an asymmetric Si(111) Laue crystal
    at three incoming angles. A two-by-two dispersion eigensystem and Poynting energy flow independently
    calculate each individual polarization path, which is then checked against pure-input displacement
    and public track1 exit coordinates. BMAD_DIST selects the build.
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


    import numpy as np
    from _bmad import compare, verdict
    from _s3_19_fields_photons_ptc import photon_data, polarization_paths, rows

    output = photon_data(7)
    displacements = {int(row[0]): list(map(float, row[1:])) for row in rows(output, "DR")}
    crystal_parameters = {int(row[0]): list(map(float, row[1:])) for row in rows(output, "CP")}
    factors = {int(row[0]): list(map(float, row[1:])) for row in rows(output, "FACTORS")}
    vectors = {int(row[0]): list(map(float, row[1:])) for row in rows(output, "VECTORS")}
    actuals = {(int(row[0]), int(row[1])): list(map(float, row[2:])) for row in rows(output, "POL")}
    assert set(displacements) == set(crystal_parameters) == set(factors) == set(vectors) == {1, 2, 3}
    assert set(actuals) == {(k, p) for k in (1, 2, 3) for p in (1, 2, 3)}
    assert len(rows(output, "POL")) == 9
    assert all(np.isfinite(values).all() for values in actuals.values())
    defect = False
    for k, displacement in displacements.items():
        expected = polarization_paths(k, crystal_parameters, factors, vectors, displacements)
        residual = float(np.max(np.abs(expected - np.asarray(displacement).reshape(2, 3))))
        print(
            f"Angle case {k}: eigensystem/Poynting oracle versus individual paths residual={residual:.3g} m"
        )
        for pol in (1, 2):
            assert actuals[k, pol][4 if pol == 1 else 3] == 0
            defect |= compare(
                f"Angle case {k}, pure polarization {pol}, displacement (m), single present component",
                actuals[k, pol][:3],
                expected[pol - 1],
                atol=8e-20,
            )
    public = rows(output, "PUBLIC")
    assert len(public) == 6
    assert {(int(r[0]), int(r[1])) for r in public} == {(k, p) for k in (1, 2, 3) for p in (1, 2)}
    for row in public:
        actual, expected_x = map(float, row[2:])
        defect |= compare(
            f"Public track1 angle={row[0]} polarization={row[1]}, x (m), single-component exit transform",
            actual,
            expected_x,
            atol=2e-19,
        )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed (pure-polarization Laue displacement) | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

