---
title: "TRK-07: Recognized crystal materials falsely lose photons in transmitting masks"
hide:
  - toc
---

# TRK-07 — Recognized crystal materials falsely lose photons in transmitting masks

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [material-helper output contract, bmad/interface/xraylib_interface.f90:28–31](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/interface/xraylib_interface.f90#L28-L31)<br>[successful crystal lookup, bmad/interface/xraylib_interface.f90:59–73](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/interface/xraylib_interface.f90#L59-L73)<br>[mask transmission, bmad/photon/track1_photon_mod.f90:199–209](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/photon/track1_photon_mod.f90#L199-L209)<br>[Bmad manual, Mask Wall For Diffraction Plate and Mask Elements, bmad/doc/attributes.tex:1913–1916](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L1913-L1916) |

A transmitting mask filled with a recognized crystal material marks valid photons as lost. A 20 µm silicon section at 9 keV returns a lost photon with unchanged unit field instead of a live photon with field amplitude 0.9001. The material lookup computes finite absorption and phase coefficients but leaves its error flag set. A titanium section at normal incidence follows the successful elemental lookup and transmits with the expected amplitude 0.5154.

Run `python TRK-07.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_19_fields_photons_ptc.py](../helpers/_s3_19_fields_photons_ptc.py) in the same directory.

??? example "Reproducer: TRK-07.py"

    [Download TRK-07.py](TRK-07.py)

    ```python
    """TRK-07: Recognized crystal materials falsely lose photons in transmitting masks

    A transmitting mask filled with a recognized crystal material marks valid photons as lost. A 20 µm
    silicon section at 9 keV returns a lost photon with unchanged unit field instead of a live photon
    with field amplitude 0.9001. The material lookup computes finite absorption and phase coefficients
    but leaves its error flag set. A titanium section at normal incidence follows the successful
    elemental lookup and transmits with the expected amplitude 0.5154.

    The script calls the material helper and public track1 for silicon and titanium clear mask sections
    at normal incidence. It checks successful coefficient calculation, live state, Beer attenuation and
    optical phase modulo 2π;
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

    output = photon_data(2)
    assert "SILICON_RECOGNIZED" in output.splitlines()
    print("Crystal_GetCrystal('Si'): recognized material control passed.")
    coeffs = {int(row[0]): (row[1], *map(float, row[2:])) for row in rows(output, "ABS")}
    data = rows(output, "MASK")
    assert len(data) == 6 and set(coeffs) == {5, 6, 7}
    assert {(int(r[0]), int(r[1])) for r in data} == {(i, j) for i in (5, 6, 7) for j in (1, 2)}
    defect = False
    for row in data:
        index, direction, state = map(int, row[:3])
        if direction != 1 or index == 7:
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

