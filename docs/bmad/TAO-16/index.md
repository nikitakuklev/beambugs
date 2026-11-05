---
title: "TAO-16: Startup and batch-command failures exit with process status zero"
hide:
  - toc
---

# TAO-16 — Startup and batch-command failures exit with process status zero

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [tao_top_level initialization status, tao/code/tao_top_level.f90:42–59](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_top_level.f90#L42-L59)<br>[tao_program, tao/program/tao_program.f90:7–17](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/program/tao_program.f90#L7-L17)<br>[command loop, tao/code/tao_top_level.f90:74–111](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_top_level.f90#L74-L111) |

Starting Tao with an invalid command-line option or a missing explicit initialization file produces error diagnostics, executes none of the requested commands, and returns status zero. Invalid pipe and set commands also return zero after reporting errors. An unknown top-level command aborts its command file and returns the same success status. Automation needs an unsuccessful exit status to distinguish these failures from a completed run.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _tao_contracts_normal_form.py](../helpers/_tao_contracts_normal_form.py) · [View source](../helpers/index.md#_tao_contracts_normal_form)

??? example "Reproducer: TAO-16.py"

    [Download TAO-16.py](TAO-16.py)

    ```python
    """TAO-16: Startup and batch-command failures exit with process status zero

    Starting Tao with an invalid command-line option or a missing explicit initialization file produces error diagnostics, executes none of the requested commands, and returns status zero. Invalid pipe and set commands also return zero after reporting errors. An unknown top-level command aborts its command file and returns the same success status. Automation needs an unsuccessful exit status to distinguish these failures from a completed run.

    The script runs a successful completion-marker control, two startup failures, and four command failures in separate processes. It checks each specific diagnostic and whether the command file reaches its completion marker, then compares the exit status with the required batch failure status.
    """

    from _bmad import RunFailure, verdict
    from _tao_contracts_normal_form import ERROR, LATTICE, evaluated, tao

    control = tao(LATTICE, ["pipe evaluate 314159"])
    assert evaluated(control.log, "314159") == 314159
    print("successful command file: status=0, completion=True; expected success")
    cases = [
        (["-not_a_tao_option"], [], "BAD COMMAND LINE ARGUMENT", False),
        (["-init_file", "missing.init"], [], "TAO INITIALIZATION FILE NOT FOUND", False),
        (None, ["pipe no_such_subcommand"], "is not recognized", True),
        (None, ["pipe ele:mat6 1"], "INVALID", True),
        (None, ["set bmad_com no_such_parameter = T"], "BAD COMPONENT OR NUMBER", True),
        (None, ["this_is_not_a_command"], "ABORTING COMMAND FILE", False),
    ]
    bad = False
    for options, commands, diagnostic, continues in cases:
        try:
            result = tao(LATTICE, [*commands, "pipe evaluate 314159"], options=options, check=False)
            status, log = 0, result.log
        except RunFailure as exc:
            status, log = exc.status, exc.log
        assert diagnostic in log and ERROR.search(log), log
        completed = "Tao: pipe evaluate 314159\n" in log
        if status == 0:
            assert completed == continues, log
            if completed:
                assert evaluated(log, "314159") == 314159
        print(
            f"{diagnostic}: status={status}, completion={completed}; expected nonzero batch error status"
        )
        bad |= status == 0
    verdict(bad)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

