---
title: "TRK-06: Multilayer mirrors use the wrong π vacuum-interface coefficient"
hide:
  - toc
---

# TRK-06 — Multilayer mirrors use the wrong π vacuum-interface coefficient

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [multilayer_track, interior interface, bmad/photon/track1_photon_mod.f90:549–555](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/photon/track1_photon_mod.f90#L549-L555)<br>[vacuum interface, bmad/photon/track1_photon_mod.f90:583–595](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/photon/track1_photon_mod.f90#L583-L595)<br>[Bmad manual, Multilayer_mirror, bmad/doc/elements.tex:3934–3945](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L3934-L3945) |

Multilayer mirrors use an incorrectly normalized vacuum-interface coefficient for π polarization, whose electric field lies in the incidence plane. A one-cell tungsten/boron-carbide mirror at 9 keV and 0.02 rad grazing angle returns reflectivity 0.001190 instead of 0.001193, an error of −0.2443%. Matching tangential electric and magnetic fields at each layer independently establishes the expected reflection. At π/4 the result is 1.793e-9 instead of 1.546e-18, a large relative error in a tiny reflected intensity; perpendicular σ polarization passes.

Run `python TRK-06.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_19_fields_photons_ptc.py](../helpers/_s3_19_fields_photons_ptc.py) in the same directory.

??? example "Reproducer: TRK-06.py"

    [Download TRK-06.py](TRK-06.py)

    ```python
    """TRK-06: Multilayer mirrors use the wrong π vacuum-interface coefficient

    Multilayer mirrors use an incorrectly normalized vacuum-interface coefficient for π polarization,
    whose electric field lies in the incidence plane. A one-cell tungsten/boron-carbide mirror at 9 keV
    and 0.02 rad grazing angle returns reflectivity 0.001190 instead of 0.001193, an error of −0.2443%.
    Matching tangential electric and magnetic fields at each layer independently establishes the
    expected reflection. At π/4 the result is 1.793e-9 instead of 1.546e-18, a large relative error in a
    tiny reflected intensity; perpendicular σ polarization passes.

    The script tracks 9 keV photons through W/boron-carbide mirrors with 1, 3 and 7 cells at grazing
    angles 0.02, 0.25 and π/4 rad. It matches each interface independently with the same semi-infinite
    bottom-material termination and checks σ polarization as a control;
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

    from _bmad import compare, verdict
    from _s3_19_fields_photons_ptc import control, fresnel_stack, photon_data, rows

    output = photon_data(1)
    wavelength, r1, i1, r2, i2, d1, d2 = map(float, rows(output, "ML_CONFIG")[0])
    data = rows(output, "ML")
    assert len(data) == 9
    assert {(int(r[0]), int(r[1])) for r in data} == {(n, j) for n in (1, 3, 7) for j in (1, 2, 3)}
    defect = False
    for row in data:
        count, _angle_id = map(int, row[:2])
        angle, api, asi, ppi, psi = map(float, row[2:7])
        assert int(row[7]) == 1
        for pol, amplitude, phase in [("pi", api, ppi), ("sigma", asi, psi)]:
            expected = fresnel_stack(
                wavelength, complex(r1, i1), complex(r2, i2), d1, d2, count, angle, pol
            )
            actual = amplitude * cmath.exp(1j * phase)
            if pol == "sigma":
                control([actual.real, actual.imag], [expected.real, expected.imag], 3e-13)
            defect |= compare(
                f"N={count} angle={angle:.12g} {pol}, complex E/E0, interface recursion",
                [actual.real, actual.imag],
                [expected.real, expected.imag],
                atol=3e-13,
            )
            print(f"Reflectivity: observed={amplitude**2:.16g} expected={abs(expected) ** 2:.16g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

