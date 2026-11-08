---
title: "TAO-15: A later lattice-file option cancels explicit -noinit"
hide:
  - toc
---

# TAO-15 — A later lattice-file option cancels explicit -noinit

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [tao_parse_command_args, tao/code/tao_parse_command_args.f90:145–157](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_parse_command_args.f90#L145-L157)<br>[tao_init, tao/code/tao_init.f90:87–94](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_init.f90#L87-L94)<br>[Command Line Arguments, -noinit, tao/doc/initialization.tex:138–140](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/initialization.tex#L138-L140) |

Starting Tao with -noinit before -lattice_file allows an ambient initialization file to change the run. Reversing these options suppresses that file as documented. An ambient chromaticity step of 0.007000 is loaded in the first order, while the reversed order retains the compiled default of 0.0001000. The parser explicitly clears the suppression flag when it encounters the lattice filename.

Run `python TAO-15.py` with [_bmad.py](../helpers/_bmad.py) and [_tao_contracts_normal_form.py](../helpers/_tao_contracts_normal_form.py) in the same directory.

??? example "Reproducer: TAO-15.py"

    [Download TAO-15.py](TAO-15.py)

    ```python
    """TAO-15: A later lattice-file option cancels explicit -noinit

    Starting Tao with -noinit before -lattice_file allows an ambient initialization file to change the run. Reversing these options suppresses that file as documented. An ambient chromaticity step of 0.007000 is loaded in the first order, while the reversed order retains the compiled default of 0.0001000. The parser explicitly clears the suppression flag when it encounters the lattice filename.

    The script supplies an ambient tao.init with a distinctive chromaticity step and runs both option orders in fresh processes. It checks the reversed-order default as a control and compares the resulting global parameter.
    """

    from _bmad import compare, verdict
    from _tao_contracts_normal_form import LATTICE, block, scalar, tao

    values = []
    for options in (["-noinit", "-lattice_file", "lat.bmad"], ["-lattice_file", "lat.bmad", "-noinit"]):
        result = tao(
            LATTICE,
            ["pipe global"],
            options=options,
            files={"tao.init": "&tao_params\n global%delta_e_chrom = 0.007\n/\n"},
        )
        values.append(scalar(block(result.log, "pipe global"), "delta_e_chrom"))
    assert not compare("control, -noinit last: delta_e_chrom", values[1], 0.0001, atol=0)
    verdict(
        compare(
            "-noinit first: delta_e_chrom; explicit suppression in either order",
            values[0],
            values[1],
            atol=0,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

