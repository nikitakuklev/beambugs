---
title: "TRK-11: Oblique mask transmission omits the longer material path"
hide:
  - toc
---

# TRK-11 — Oblique mask transmission omits the longer material path

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [track1_diffraction_plate_or_mask, attenuation and phase, bmad/photon/track1_photon_mod.f90:199–209](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/photon/track1_photon_mod.f90#L199-L209)<br>[path-length coefficient contract, bmad/interface/xraylib_interface.f90:28–31](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/interface/xraylib_interface.f90#L28-L31)<br>[Bmad manual, Mask Wall For Diffraction Plate and Mask Elements, bmad/doc/attributes.tex:1913–1916](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L1913-L1916)<br>[Mask, bmad/doc/elements.tex:3525–3531](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L3525-L3531) |

Oblique photons transmitted through a material-filled mask receive the normal-incidence attenuation and phase. For a 20 µm titanium section at 9 keV and local longitudinal direction cosine 0.8, the field amplitude is 0.5154 instead of 0.4367. Straight-ray geometry requires a 25 µm material path, and Beer attenuation and optical phase both use that length. An oblique ray and a laboratory-axis ray through a pitched mask give the same error; normal incidence passes.

Run `python TRK-11.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_19_fields_photons_ptc.py](../helpers/_s3_19_fields_photons_ptc.py) in the same directory.

??? example "Reproducer: TRK-11.py"

    [Download TRK-11.py](TRK-11.py)

    ```python
    """TRK-11: Oblique mask transmission omits the longer material path

    Oblique photons transmitted through a material-filled mask receive the normal-incidence attenuation
    and phase. For a 20 µm titanium section at 9 keV and local longitudinal direction cosine 0.8, the
    field amplitude is 0.5154 instead of 0.4367. Straight-ray geometry requires a 25 µm material path,
    and Beer attenuation and optical phase both use that length. An oblique ray and a laboratory-axis
    ray through a pitched mask give the same error; normal incidence passes.

    The script tracks photons through a titanium mask at normal and oblique incidence and through the
    same mask pitched by acos(0.8). It compares attenuation and phase with the local material path
    d/abs(vz), using the normal-incidence case as a control and BMAD_DIST.
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
    import math

    from _bmad import compare, verdict
    from _s3_19_fields_photons_ptc import photon_data, rows

    output = photon_data(6)
    coeffs = {int(row[0]): (row[1], *map(float, row[2:])) for row in rows(output, "ABS")}
    data = rows(output, "MASK")
    assert len(data) == 6 and set(coeffs) == {5, 6, 7}
    assert {(int(r[0]), int(r[1])) for r in data} == {(i, j) for i in (5, 6, 7) for j in (1, 2)}
    defect = False
    for row in data:
        index, direction, state = map(int, row[:3])
        if index == 5:
            continue
        vz, ex, ey, phase_x, phase_y = map(float, row[3:])
        assert all(math.isfinite(v) for v in (vz, ex, ey, phase_x, phase_y))
        assert 0 < abs(vz) <= 1 and phase_x == phase_y
        flagged, alpha, phase_per_m = coeffs[index]
        assert math.isfinite(alpha) and alpha > 0 and math.isfinite(phase_per_m)
        actual = math.hypot(ex, ey)
        # Absorption coefficient is for field amplitude per metre along the ray.
        path = 2e-5 / abs(vz)
        expected = math.exp(-alpha * path)
        phase_expected = -phase_per_m * path
        if index != 5:
            assert flagged == "F" and state == 1
        status_bad = flagged != "F" or state != 1
        amplitude_bad = compare(
            f"mask={index} direction={direction}, E/E0, Beer attenuation", actual, expected, atol=3e-15
        )
        phase_bad = compare(
            f"mask={index} direction={direction}, exp(i phase), optical path",
            [cmath.exp(1j * phase_x).real, cmath.exp(1j * phase_x).imag],
            [cmath.exp(1j * phase_expected).real, cmath.exp(1j * phase_expected).imag],
            atol=3e-13,
        )
        print(
            f"vz_local={vz:.12g}: error_flag={flagged} expected=F; state={state} expected=1; phase={phase_x:.15g} expected={phase_expected:.15g} modulo 2pi"
        )
        if index == 6 and direction == 1:
            assert not (status_bad or amplitude_bad or phase_bad)
        defect |= status_bad or amplitude_bad or phase_bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

