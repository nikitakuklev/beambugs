---
title: "TRK-05: AC cylindrical magnetic modes contain an extra radial azimuthal factor"
hide:
  - toc
---

# TRK-05 — AC cylindrical magnetic modes contain an extra radial azimuthal factor

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [em_field_calc, AC cylindrical branch, bmad/code/em_field_calc.f90:1231–1249](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/em_field_calc.f90#L1231-L1249)<br>[frequency and Cartesian conversion, bmad/code/em_field_calc.f90:1256–1279](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/em_field_calc.f90#L1256-L1279)<br>[Bmad manual, AC Cylindrical Map Decomposition, aiew, bmad/doc/electromagnetic-fields.tex:733–780](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/electromagnetic-fields.tex#L733-L780)<br>[Cylindrical_Map Field Map, bmad/doc/attributes.tex:2427–2441](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L2427-L2441) |

AC cylindrical magnetic modes with azimuthal order above one return an extra radial magnetic field. For an order-two, 80 MHz mode at (11, −7, 190) mm, the horizontal field is −1.491 µT instead of 372.4 µT from the curl of the documented electric field. Cartesian finite differences give a magnetic divergence of −0.06803 T/m where Maxwell’s equations require zero. Electric-coefficient modes and order-one magnetic modes pass the same checks.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _s3_19_cylindrical.py](../helpers/_s3_19_cylindrical.py) · [View source](../helpers/index.md#_s3_19_cylindrical)
- [Download _s3_19_fields_photons_ptc.py](../helpers/_s3_19_fields_photons_ptc.py) · [View source](../helpers/index.md#_s3_19_fields_photons_ptc)

??? example "Reproducer: TRK-05.py"

    [Download TRK-05.py](TRK-05.py)

    ```python
    """TRK-05: AC cylindrical magnetic modes contain an extra radial azimuthal factor

    AC cylindrical magnetic modes with azimuthal order above one return an extra radial magnetic field.
    For an order-two, 80 MHz mode at (11, −7, 190) mm, the horizontal field is −1.491 µT instead of
    372.4 µT from the curl of the documented electric field. Cartesian finite differences give a
    magnetic divergence of −0.06803 T/m where Maxwell’s equations require zero. Electric-coefficient
    modes and order-one magnetic modes pass the same checks.

    The script constructs 28 parsed maps and evaluates 4,116 samples around 84 centers for orders 1, 2
    and 4, both Fourier signs, two frequencies and both coefficient families. An entire Bessel series
    supplies an independent analytic curl, while Cartesian stencils check divergence and Faraday’s law.
    BMAD_DIST selects the build.
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
    from _s3_19_cylindrical import PROGRAM, modes_and_samples, oracle
    from _s3_19_fields_photons_ptc import control, execute

    modes, samples, groups, files = modes_and_samples()
    output = execute(PROGRAM, [], files)
    fields = {}
    for line in output.splitlines():
        if not line.startswith("SAMPLE "):
            continue
        tokens = line.split()
        ident, ie = map(int, tokens[1:3])
        values = np.array(list(map(float, tokens[3:])))
        assert ident not in fields and samples[ident][0] == ie
        control(values[:4], samples[ident][1], 0)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Nonfinite sample {ident}")
        fields[ident] = values[4:]
    assert len(fields) == len(samples)
    defect = False
    max_control = max_div = max_faraday = 0.0
    for group in groups:
        mode = modes[group["ie"] - 1]
        e, b, extra = oracle(mode, group["pos"])
        actual = fields[group["center"]]
        control(actual[:3], e, 2e-10 + 3e-13 * max(abs(e)))
        is_control = mode["family"] == "e" or mode["m"] == 1
        error = float(max(abs(actual[3:] - b)))
        if is_control:
            max_control = max(max_control, control(actual[3:], b, 3e-15))
        else:
            defect |= error > 3e-15
        compare(
            f"mode={group['ie']} family={mode['family']} m={mode['m']} "
            f"f={mode['frequency']:g} Hz Fourier-index={mode['index']} point={group['point']}, "
            "B (T), curl(E)/(i omega)",
            actual[3:],
            b,
            atol=3e-15,
        )
        # The 4-point stencils differentiate the library E and B in Cartesian coordinates.
        for stencils in group["stencils"]:
            d = np.array(
                [
                    (np.array([1, -8, 8, -1]) / (12 * h)) @ np.array([fields[i] for i in ids])
                    for h, ids in stencils
                ]
            )
            curl_e = np.array([d[1, 2] - d[2, 1], d[2, 0] - d[0, 2], d[0, 1] - d[1, 0]])
            divergence = float(np.trace(d[:3, 3:]))
            scale = max(float(max(abs(curl_e))), float(max(abs(d[3, 3:]))))
            faraday = float(max(abs(curl_e + d[3, 3:]))) / scale
            if is_control:
                assert abs(divergence) < 2e-9 and faraday < 2e-9
            max_div = max(max_div, abs(divergence))
            max_faraday = max(max_faraday, faraday)
            if (
                mode["family"] == "b"
                and mode["m"] == 2
                and mode["index"] == 1
                and mode["frequency"] == 8e7
                and mode["coefficient"].real > 0
                and group["point"] == 0
            ):
                print(
                    f"h={stencils[0][0]:g} m: div B={divergence:.16g} T/m expected=0; Faraday x={(curl_e + d[3, 3:])[0]:.16g} V/m² expected=0"
                )
    print(
        f"{len(modes)} maps, {len(groups)} centers, {len(fields)} samples; electric/m=1 control max B error={max_control:.3g} T"
    )
    print(
        f"Max |div B|={max_div:.16g} T/m; max relative Faraday residual={max_faraday:.16g}; expected=0"
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed (executed azimuthal orders 2 and 4; other orders have source support) | `gpt-6.1-sol` | xhigh | 2026-10-29 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

