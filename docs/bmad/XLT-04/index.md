---
title: "XLT-04: Converter polarization interpolation leaves its output uninitialized"
hide:
  - toc
---

# XLT-04 — Converter polarization interpolation leaves its output uninitialized

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`calc_out_coords`, track_a_converter.f90:268–300](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_converter.f90#L268-L300)<br>[`calc_out_coords2`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_converter.f90#L332-L335)<br>[the spin assignment](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_converter.f90#L251-L252)<br>[`converter_distribution_parser`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/converter_distribution_parser.f90#L172-L201)<br>[converter-model paper, “Spin Tracking”](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/converter_element_modeling/paper/converter.tex#L242-L247) |

Interpolating a converter distribution between incoming momenta leaves the outgoing longitudinal polarization unassigned. Two tables with polarization transfers 0.25 and 0.75 require a transfer of 0.5 at their momentum midpoint. Diagnostic initialization exposes an undefined value in all 128 samples; adding the missing weighted interpolation gives exactly 0.5 in every sample. The ordinary production library returns zero for this fixture.

Run `python XLT-04.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_24_converter.py](../helpers/_s3_24_converter.py) in the same directory.

??? example "Reproducer: XLT-04.py"

    [Download XLT-04.py](XLT-04.py)

    ```python
    """XLT-04: Converter polarization interpolation leaves its output uninitialized

    Interpolating a converter distribution between incoming momenta leaves the outgoing longitudinal polarization unassigned. Two tables with polarization transfers 0.25 and 0.75 require a transfer of 0.5 at their momentum midpoint. Diagnostic initialization exposes an undefined value in all 128 samples; adding the missing weighted interpolation gives exactly 0.5 in every sample. The ordinary production library returns zero for this fixture.

    The script runs the production converter and compiles a diagnostic copy of the source selected by BMAD_SOURCE with quiet-NaN initialization. A second copy adds only the missing interpolation assignment. Both diagnostic copies link against BMAD_DIST and run wrapper and direct-call controls.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _s3_24_converter import samples

    production, _, _ = samples()
    print(
        f"production polarization range: {[float(production[:, 2].min()), float(production[:, 2].max())]} expected=0.5"
    )
    bad = False
    for mode in ("converter", "direct"):
        observed, _, _ = samples(mode=mode, diagnostic=True)
        assigned, _, _ = samples(mode=mode, diagnostic=True, assigned=True)
        # Equal incoming-momentum distances give equal weights: (0.25+0.75)/2=0.5.
        if compare(
            f"{mode} assigned interpolation maximum residual",
            np.max(np.abs(assigned[:, 2] - 0.5)),
            0,
            atol=2e-15,
        ):
            raise RuntimeError("Assigned interpolation control failed")
        count = int(np.count_nonzero(np.isnan(observed[:, 2])))
        print(f"{mode} diagnostic undefined polarization: observed_nan_count={count}/128 expected=0")
        if count not in (0, 128):
            raise RuntimeError("Diagnostic spin samples are inconsistent")
        if not count:
            bad |= compare(
                f"{mode} diagnostic polarization", observed[:, 2], np.full(128, 0.5), atol=2e-15
            )
        bad |= count > 0
    verdict(bad)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed<br><small>undefined polarization-table interpolation output</small> | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

