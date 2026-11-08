---
title: "TAO-10: Wrapped average, integral, and RMS pass the unreordered value buffer"
hide:
  - toc
---

# TAO-10 — Wrapped average, integral, and RMS pass the unreordered value buffer

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`tao_load_this_datum` wrapped reductions, tao/code/tao_data_and_eval_mod.f90:595–612](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_data_and_eval_mod.f90#L595-L612)<br>[`tao_datum_integrate` length check, tao/code/tao_data_and_eval_mod.f90:739–757](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_data_and_eval_mod.f90#L739-L757)<br>[“Calculation of δD for a Datum” (`s:del.d`), tao/doc/optimization.tex:145–171](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/optimization.tex#L145-L171) |

Average, integral, and RMS constraints over a range that wraps around the start of a ring are rejected when Tao passes the whole-branch value buffer alongside the shorter wrapped position array. A valid one-metre range in a two-metre ring receives an internal-error diagnostic and invalid model, base, and design values. Its default invalid-value setting leaves the residual and merit at zero, while the clock average should be 4.336 ns and its integral 4.336 ns·m. The constructed wrapped positions and reordered values must be passed together.

Run `python TAO-10.py` with [_bmad.py](../helpers/_bmad.py) and [_tao_optimization_io.py](../helpers/_tao_optimization_io.py) in the same directory.

??? example "Reproducer: TAO-10.py"

    [Download TAO-10.py](TAO-10.py)

    ```python
    """TAO-10: Wrapped average, integral, and RMS pass the unreordered value buffer

    Average, integral, and RMS constraints over a range that wraps around the start of a ring are
    rejected when Tao passes the whole-branch value buffer alongside the shorter wrapped position
    array. A valid one-metre range in a two-metre ring receives an internal-error diagnostic and
    invalid model, base, and design values. Its default invalid-value setting leaves the residual
    and merit at zero, while the clock average should be 4.336 ns and its integral 4.336 ns·m. The
    constructed wrapped positions and reordered values must be passed together.

    The script requests all three reductions over a wrapped range and a non-wrapped control in separate
    Tao processes. It verifies every clock sample analytically, identifies the exact invalid-data
    diagnostic, and checks the resulting flags and merit.
    """

    import math

    import numpy as np
    from _bmad import compare, verdict
    from _tao_optimization_io import RING_LATTICE, block, control, diagnostics, numbers, parameters, tao

    # ref_time(s)=s/(beta*c); p0c=1 GeV and Bmad electron rest energy in eV.
    scale = math.hypot(1e9, 510998.95069) / (1e9 * 299792458.0)
    positions = np.array([0, 0.2, 1, 1.2, 2, 2])
    defect = False
    for reduction in ("average", "integral", "rms"):
        init = """&tao_design_lattice design_lattice(1)%file = 'lat.bmad' /
    &tao_params global%opt_with_ref = F, global%opt_with_base = F /
    &tao_d2_data d2_data%name = 'range', n_d1_data = 2 /
    """
        for index, name, start, end, weight in [
            (1, "clock", "QD", "QF", 1),
            (2, "control", "QF", "QD", 0),
        ]:
            init += f"""&tao_d1_data
    ix_d1_data = {index}
    d1_data%name = '{name}'
    default_data_type = 'ref_time'
    default_merit_type = '{reduction}'
    datum(1)%ele_start_name = '{start}'
    datum(1)%ele_name = '{end}'
    datum(1)%meas = 0
    datum(1)%weight = {weight}
    /
    """
        commands = [
            "pipe lat_list -index_order 1@0>>*|model ele.ix_ele,ele.s,ele.ref_time",
            "pipe data range.control[1]",
            "pipe data range.clock[1]",
            "pipe merit",
        ]
        result = tao(commands, lattice=RING_LATTICE, init=init, allow_errors=True)
        samples = numbers(block(result.log, commands[0]))
        control(
            f"{reduction} branch positions (m)",
            samples[:, :2],
            np.column_stack((np.arange(6), positions)),
            atol=3e-16,
        )
        control(
            f"{reduction} clock samples (s), s/(beta*c)", samples[:, 2], positions * scale, atol=1e-23
        )
        nonwrapped = parameters(block(result.log, commands[1]))
        if nonwrapped["good_model"] != "T" or float(nonwrapped["weight"]) != 0:
            raise RuntimeError(f"Non-wrapped control invalid: {nonwrapped}")
        if reduction != "rms":
            control(
                f"non-wrapped {reduction}, linear clock integration",
                float(nonwrapped["model_value"]),
                0.7 * scale,
                atol=1e-23,
            )
        else:
            # The independent RMS quadrature discrepancy is separate from buffer validity.
            if not math.isfinite(float(nonwrapped["model_value"])):
                raise RuntimeError(f"Nonfinite non-wrapped RMS: {nonwrapped}")
        datum = parameters(block(result.log, commands[2]))
        if any(
            datum[key] != value
            for key, value in {
                "exists": "T",
                "useit_opt": "T",
                "ix_ele_start": "3",
                "ix_ele": "1",
                "data_type": "ref_time",
                "data^merit_type": reduction,
            }.items()
        ):
            raise RuntimeError(f"Wrong wrapped datum: {datum}")
        errors = diagnostics(result.log)
        if errors:
            expected_error = [
                (
                    "tao_set_invalid",
                    (
                        "INTERNAL ERROR. PLEASE REPORT!",
                        "FOR DATUM: range.clock[1] with data_type: ref_time",
                    ),
                )
            ]
            print(f"wrapped {reduction}: observed=INTERNAL ERROR. PLEASE REPORT!; expected=valid datum")
            if errors != expected_error:
                raise RuntimeError(f"Unexpected errors: {errors}")
            if any(datum[key] != "F" for key in ("good_model", "good_base", "good_design")):
                raise RuntimeError(f"Inconsistent invalid flags: {datum}")
            control(
                f"{reduction} invalid value, residual, datum merit and total merit",
                [
                    float(datum["invalid_value"]),
                    float(datum["delta_merit"]),
                    float(datum["merit"]),
                    float(block(result.log, commands[3])),
                ],
                [0, 0, 0, 0],
                atol=0,
            )
        flags = [datum[key] == "T" for key in ("good_model", "good_base", "good_design")]
        defect |= compare(f"wrapped {reduction} valid flags", flags, [1, 1, 1], atol=0)
        if reduction != "rms":
            # Piecewise linear ref_time samples on s=(1.2,2,2,2,2.2) integrate to 1.3*scale.
            print(
                f"wrapped {reduction}: expected={1.3 * scale:.17g} from paired wrapped samples; good_model={datum['good_model']}"
            )
            if datum["good_model"] == "T":
                defect |= compare(
                    f"wrapped {reduction} value", float(datum["model_value"]), 1.3 * scale, atol=1e-23
                )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

