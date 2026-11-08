---
title: "TAO-06: RMS reduction integrates squared endpoints instead of the linear interpolant"
hide:
  - toc
---

# TAO-06 — RMS reduction integrates squared endpoints instead of the linear interpolant

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`tao_datum_integrate` second moments, tao/code/tao_data_and_eval_mod.f90:769–775](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_data_and_eval_mod.f90#L769-L775)<br>[Tao manual, “Calculation of δD for a Datum”, RMS definition (`s:del.d`), tao/doc/optimization.tex:168–171](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/optimization.tex#L168-L171) |

Tao’s RMS range reduction squares the endpoint samples and applies trapezoidal quadrature, instead of integrating the square of the documented linear interpolation. An exactly linear, centered orbit over a one-metre drift returns 0.5000 mm instead of 0.2887 mm, making the variance three times too large. Its mean and integral are zero. Integrating a linear segment with endpoint values a and b requires a second moment proportional to (a² + ab + b²)/3.

Run `python TAO-06.py` with [_bmad.py](../helpers/_bmad.py) and [_tao_optimization_io.py](../helpers/_tao_optimization_io.py) in the same directory.

??? example "Reproducer: TAO-06.py"

    [Download TAO-06.py](TAO-06.py)

    ```python
    """TAO-06: RMS reduction integrates squared endpoints instead of the linear interpolant

    Tao’s RMS range reduction squares the endpoint samples and applies trapezoidal quadrature,
    instead of integrating the square of the documented linear interpolation. An exactly linear,
    centered orbit over a one-metre drift returns 0.5000 mm instead of 0.2887 mm, making the
    variance three times too large. Its mean and integral are zero. Integrating a linear segment
    with endpoint values a and b requires a second moment proportional to (a² + ab + b²)/3.

    The script centers an orbit in a drift and requests RMS, average, and integral constraints. It
    checks the endpoints and zero mean, then compares RMS with direct integration of the linear orbit.
    """

    import math

    from _bmad import compare, verdict
    from _tao_optimization_io import LINE, RMS_INIT, block, control, data_values, numbers, tao

    # Over a one-metre drift x(s)=dx*(s-.5), so integral(x^2 ds)=dx^2/12.
    dx = 0.001 / math.sqrt(1 - 0.001**2)
    commands = [
        f"set particle_start x = {-dx / 2:.17g}",
        "pipe lat_list beginning,d1 orbit.vec.1",
        "pipe data_parameter range.x model_value",
    ]
    result = tao(commands, lattice=LINE, init=RMS_INIT)
    control(
        "linear orbit endpoints (m)",
        numbers(block(result.log, commands[1])).ravel(),
        [-dx / 2, dx / 2],
        atol=1e-16,
    )
    values = data_values(result.log, commands[2])
    control("centered mean (m) and integral (m^2)", values[1:], [0, 0], atol=1e-16)
    verdict(
        compare("centered linear-orbit RMS (m), dx/sqrt(12)", values[0], dx / math.sqrt(12), atol=1e-15)
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

