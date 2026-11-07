---
title: "TAO-12: Aliases with placeholders reject documented surplus arguments"
hide:
  - toc
---

# TAO-12 — Aliases with placeholders reject documented surplus arguments

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`tao_alias_translate`, tao/code/tao_get_user_input_mod.f90:534–568](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_get_user_input_mod.f90#L534-L568)<br>[“alias” (`s:alias`), tao/doc/command-list.tex:99–124](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/command-list.tex#L99-L124) |

Tao rejects surplus arguments to an alias that uses a placeholder, although the manual says to append arguments without matching placeholders. Defining inspect as pipe ele:mat6 [[1]] and invoking inspect E mat6 reports an extra-arguments error. The equivalent direct matrix query succeeds and gives the analytic drift matrix. The alias should expand to that same query under the documented rule.

Run `python TAO-12.py` with [_bmad.py](../helpers/_bmad.py) and [_tao_optimization_io.py](../helpers/_tao_optimization_io.py) in the same directory.

??? example "Reproducer: TAO-12.py"

    [Download TAO-12.py](TAO-12.py)

    ```python
    """TAO-12: Aliases with placeholders reject documented surplus arguments

    Tao rejects surplus arguments to an alias that uses a placeholder, although the manual says to
    append arguments without matching placeholders. Defining inspect as pipe ele:mat6 [[1]] and
    invoking inspect E mat6 reports an extra-arguments error. The equivalent direct matrix query
    succeeds and gives the analytic drift matrix. The alias should expand to that same query under
    the documented rule.

    The script checks a direct drift-matrix query analytically, then invokes the same query through a
    one-placeholder alias with a surplus argument. It accepts only the specific extra-arguments
    diagnostic as the failure.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _tao_optimization_io import LATTICE, block, control, diagnostics, records, tao

    commands = ["pipe ele:mat6 E mat6", "alias inspect pipe ele:mat6 [[1]]", "inspect E mat6"]
    result = tao(commands, lattice=LATTICE, allow_errors=True)
    # Linear drift: R12=R34=L, R56=L/gamma^2 for Bmad z and relative momentum.
    expected = np.eye(6)
    expected[0, 1] = expected[2, 3] = 1.7
    expected[4, 5] = 1.7 * 510998.95069**2 / (1e12 + 510998.95069**2)


    def matrix(command):
        rows = records(block(result.log, command))
        if len(rows) != 6 or any(len(row) != 9 or row[1:3] != ["REAL_ARR", "F"] for row in rows):
            raise RuntimeError(f"Malformed matrix response: {rows}")
        return np.array([[row[0], *row[3:]] for row in rows], dtype=float)


    direct = matrix(commands[0])
    control("direct matrix row indices", direct[:, 0], np.arange(1, 7), atol=0)
    control("direct drift matrix", direct[:, 1:], expected, atol=2e-15)
    errors = diagnostics(result.log)
    if errors:
        wanted = [("tao_alias_translate", ("EXTRA ARGUMENTS PRESENT IN COMMAND! inspect E mat6",))]
        if errors != wanted:
            raise RuntimeError(f"Unexpected diagnostic: {errors}")
        print(
            "alias expansion: observed=EXTRA ARGUMENTS PRESENT IN COMMAND! inspect E mat6; expected=pipe ele:mat6 E mat6"
        )
        defect = True
    else:
        via_alias = matrix(commands[2])
        control("alias matrix row indices", via_alias[:, 0], np.arange(1, 7), atol=0)
        defect = compare("alias drift matrix", via_alias[:, 1:], expected, atol=2e-15)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

