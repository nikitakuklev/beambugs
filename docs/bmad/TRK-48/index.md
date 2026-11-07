---
title: "TRK-48: Legacy regression runner passes a list object to open"
hide:
  - toc
---

# TRK-48 — Legacy regression runner passes a list object to open

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [legacy regression runner help, parsing and open](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/regression_tests/scripts/run_tests.py#L42-L87) |

Running the legacy regression script with its documented -list filename option raises TypeError before any test starts. Argument parsing wraps the filename in a Python list and passes that list to open. An empty alternate test list reproduces the failure, while the same empty list under the default filename runs successfully. The option should read the named file just as the default path does.

Run `python TRK-48.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: TRK-48.py"

    [Download TRK-48.py](TRK-48.py)

    ```python
    """TRK-48: Legacy regression runner passes a list object to open

    Running the legacy regression script with its documented -list filename option raises TypeError
    before any test starts. Argument parsing wraps the filename in a Python list and passes that
    list to open. An empty alternate test list reproduces the failure, while the same empty list
    under the default filename runs successfully. The option should read the named file just as the
    default path does.

    The script executes the actual legacy runner under BMAD_SOURCE in a temporary directory. It
    checks the default empty-list control before matching the specific TypeError at the
    alternate-list open call.
    """

    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from _bmad import source, verdict

    script = source() / "regression_tests/scripts/run_tests.py"
    with tempfile.TemporaryDirectory(prefix="bmad-list-") as tmp:
        work = Path(tmp)
        (work / "TESTS.LIST").write_text("")
        (work / "empty.list").write_text("")
        control = subprocess.run(
            [sys.executable, "-B", str(script)],
            cwd=work,
            text=True,
            capture_output=True,
            check=True,
            timeout=290,
        )
        if "Number of programs" not in control.stdout and "Number of Program" not in control.stdout:
            raise RuntimeError(control.stdout)
        print("default empty-list control: exit=0 expected=0")
        result = subprocess.run(
            [sys.executable, "-B", str(script), "-list", "empty.list"],
            cwd=work,
            text=True,
            capture_output=True,
            check=False,
            timeout=290,
        )
        defect = (
            result.returncode == 1
            and "dir_file = open (test_list_file, 'r')" in result.stderr
            and result.stderr.rstrip().endswith(
                "TypeError: expected str, bytes or os.PathLike object, not list"
            )
        )
        if not defect:
            result.check_returncode()
        print(f"-list empty.list: exit={result.returncode} expected=0; list-to-open TypeError={defect}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

