---
title: "TRK-10: Coherent forward-Laue transport gives the β branch an extra π phase"
hide:
  - toc
---

# TRK-10 — Coherent forward-Laue transport gives the β branch an extra π phase

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [crystal_diffraction_field_calc, coherent branches, bmad/photon/photon_utils_mod.f90:384–410](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/photon/photon_utils_mod.f90#L384-L410)<br>[Bmad manual, crystal entrance boundary equations, bmad/doc/xray-tracking.tex:1009–1022](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/xray-tracking.tex#L1009-L1022)<br>[Coherent Laue Crystal Tracking, bmad/doc/xray-tracking.tex:1032–1041](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/xray-tracking.tex#L1032-L1041) |

Coherent forward-Laue crystal tracking gives one internal branch an extra π phase. At the zero-thickness limit of symmetric silicon (111), sampled fields split between +1 and −1 although the entrance boundary conditions require every normalized sample to equal +1. Public tracking through a positive 1e-13 m thickness also produces both signs. Incoherent transport gives the correct positive field at zero thickness.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _s3_19_fields_photons_ptc.py](../helpers/_s3_19_fields_photons_ptc.py) · [View source](../helpers/index.md#_s3_19_fields_photons_ptc)

??? example "Reproducer: TRK-10.py"

    [Download TRK-10.py](TRK-10.py)

    ```python
    """TRK-10: Coherent forward-Laue transport gives the β branch an extra π phase

    Coherent forward-Laue crystal tracking gives one internal branch an extra π phase. At the
    zero-thickness limit of symmetric silicon (111), sampled fields split between +1 and −1 although the
    entrance boundary conditions require every normalized sample to equal +1. Public tracking through a
    positive 1e-13 m thickness also produces both signs. Incoherent transport gives the correct positive
    field at zero thickness.

    The script checks 128 coherent zero-thickness complex fields against the entrance boundary condition
    and compares 12 public track1 branch fields at positive 1e-13 m thickness. It uses symmetric Laue
    Si(111), an incoherent control, fixed seed 71439 and BMAD_DIST.
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


    import cmath

    import numpy as np
    from _bmad import compare, verdict
    from _s3_19_fields_photons_ptc import control, photon_data, rows

    output = photon_data(5)
    zero = np.array([list(map(float, r)) for r in rows(output, "ZERO_SAMPLE")])
    assert zero.shape == (128, 3) and np.isfinite(zero).all()
    control(zero[:, 0], np.arange(1, 129), 0)
    control(zero[:, 1], np.ones(128), 2e-15)
    zero_fields = zero[:, 1] * np.exp(1j * zero[:, 2])
    npos, nneg = int(np.sum(zero_fields.real > 0)), int(np.sum(zero_fields.real < 0))
    # Entrance matching gives +1 for each probability-normalized branch, including its phase.
    print(f"Zero thickness: positive={npos} negative={nneg}; expected positive=128 negative=0")
    defect = compare(
        "Zero thickness max |E/E0-1|, entrance boundary",
        float(max(abs(zero_fields - 1))),
        0.0,
        atol=2e-15,
    )
    finite = np.array([list(map(float, row)) for row in rows(output, "FINITE")])
    assert finite.shape == (12, 4) and np.isfinite(finite).all()
    control(finite[:, 0], np.ones(12), 0)
    control(finite[:, 1], np.ones(12), 2e-7)
    fields = finite[:, 1] * np.exp(1j * finite[:, 2])
    control(fields.real, finite[:, 3], 2e-15)
    # A common propagation phase cancels from the distance between sampled branch fields.
    # At 1e-13 m, the two-beam dispersion gives a branch phase difference of 3.033e-8 rad.
    # The 2e-7 bound also covers the 7.368e-11 difference in branch amplitudes.
    separation = float(np.max(abs(fields[:, None] - fields[None, :])))
    defect |= compare(
        "Positive thickness max pairwise |E/E0 difference|, common entrance branch field",
        separation,
        0.0,
        atol=2e-7,
    )
    print(f"Positive thickness Re(E/E0) range: {min(fields.real):.16g} to {max(fields.real):.16g}")
    positive = [row[2] for row in finite if row[3] > 0]
    negative = [row[2] for row in finite if row[3] < 0]
    if positive and negative:
        phase_split = abs(cmath.phase(cmath.exp(1j * (negative[0] - positive[0]))))
        print(f"Phase split={phase_split:.16g} rad; expected common branch phase within 2e-7 rad")
    amp, phase = map(float, rows(output, "INCOHERENT_ZERO")[0])
    control([(amp * cmath.exp(1j * phase)).real, (amp * cmath.exp(1j * phase)).imag], [1, 0], 2e-15)
    print("Incoherent zero-thickness control: E/E0=+1 within 2e-15; seed=71439")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

