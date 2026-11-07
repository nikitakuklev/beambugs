---
title: "TAO-13: Command-file nesting contradicts the documented unlimited depth"
hide:
  - toc
---

# TAO-13 — Command-file nesting contradicts the documented unlimited depth

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [`tao_call_cmd` nesting check, tao/code/tao_call_cmd.f90:59–64](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_call_cmd.f90#L59-L64)<br>[“call” (`s:call`), tao/doc/command-list.tex:139–144](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/command-list.tex#L139-L144) |

Tao aborts a finite, acyclic command-file chain when it reaches active nesting level 31, despite the manual’s unlimited-nesting promise. The startup file consumes one of the thirty allowed levels. A chain of thirty further files never executes its deepest calculation or the startup file’s final quit command. Tao labels the finite chain an infinite loop and closes the entire command stack.

Run `python TAO-13.py` with [_bmad.py](../helpers/_bmad.py) and [_tao_optimization_io.py](../helpers/_tao_optimization_io.py) in the same directory.

??? example "Reproducer: TAO-13.py"

    [Download TAO-13.py](TAO-13.py)

    ```python
    """TAO-13: Command-file nesting contradicts the documented unlimited depth

    Tao aborts a finite, acyclic command-file chain when it reaches active nesting level 31, despite
    the manual’s unlimited-nesting promise. The startup file consumes one of the thirty allowed
    levels. A chain of thirty further files never executes its deepest calculation or the startup
    file’s final quit command. Tao labels the finite chain an infinite loop and closes the entire
    command stack.

    The script builds acyclic chains just below and at the failing boundary. It verifies that the
    shorter chain reaches its calculation, then checks every call and the exact stack-abort diagnostics
    for the longer chain.
    """

    from _bmad import verdict
    from _tao_optimization_io import LATTICE, block, control, diagnostics, numbers, tao


    def chain(depth):
        files = {f"depth{i}.tao": f"call depth{i + 1}.tao\n" for i in range(1, depth)}
        files[f"depth{depth}.tao"] = "pipe evaluate 271828\n"
        return files


    control_result = tao(["call depth1.tao"], lattice=LATTICE, files=chain(29))
    control(
        "30 active levels, deepest calculation",
        numbers(block(control_result.log, "pipe evaluate 271828"))[:, 1],
        [271828],
        atol=0,
    )
    result = tao(
        ["call depth1.tao"],
        lattice=LATTICE,
        files=chain(30),
        allow_errors=True,
        allow_aborted_script=True,
    )
    for i in range(1, 31):
        if result.log.count(f"Tao: call depth{i}.tao\n") != 1:
            raise RuntimeError(f"Call {i} did not reach the depth boundary:\n{result.log}")
    errors = diagnostics(result.log)
    if errors:
        expected = [
            (
                "tao_call_cmd",
                ("COMMAND FILE LEVEL > 30 INDICATES INFINITE LOOP. CLOSING ALL COMMAND FILES.",),
            )
        ]
        expected += [
            ("tao_abort_command_file", (f"ABORTING COMMAND FILE DUE TO DETECTED ERROR: {name}",))
            for name in [*(f"depth{i}.tao" for i in range(29, 0, -1)), "run.tao"]
        ]
        if errors != expected:
            raise RuntimeError(f"Unexpected diagnostics: {errors}")
        if "Tao: pipe evaluate 271828" in result.log or "Tao: quit" in result.log:
            raise RuntimeError(f"Inconsistent stack abort:\n{result.log}")
        print(
            "31 active levels: observed=finite chain aborted as infinite loop; expected=deepest calculation 271828 and scripted quit"
        )
        defect = True
    else:
        control(
            "31 active levels, deepest calculation",
            numbers(block(result.log, "pipe evaluate 271828"))[:, 1],
            [271828],
            atol=0,
        )
        if result.log.count("Tao: quit\n") != 1:
            raise RuntimeError(result.log)
        defect = False
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

