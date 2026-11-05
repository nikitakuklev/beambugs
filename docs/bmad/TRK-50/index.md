---
title: "TRK-50: Legacy regression documentation names the wrong script and list file"
hide:
  - toc
---

# TRK-50 — Legacy regression documentation names the wrong script and list file

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [legacy regression README](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/regression_tests/README.md#L171-L182)<br>[help and actual default](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/regression_tests/scripts/run_tests.py#L42-L61) |

Copying the legacy regression command from the README names a script that is absent from the distribution. The README and executable help spell it run_test.py, while the shipped program is run_tests.py. Both also advertise test.list as the default list, but execution opens TESTS.LIST. The documented command and default therefore disagree with the shipped interface.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)

??? example "Reproducer: TRK-50.py"

    [Download TRK-50.py](TRK-50.py)

    ```python
    """TRK-50: Legacy regression documentation names the wrong script and list file

    Copying the legacy regression command from the README names a script that is absent from the
    distribution. The README and executable help spell it run_test.py, while the shipped program is
    run_tests.py. Both also advertise test.list as the default list, but execution opens TESTS.LIST.
    The documented command and default therefore disagree with the shipped interface.

    The script reads the README and runs the shipped help from BMAD_SOURCE in scratch. It then
    distinguishes the two default list filenames by executing the runner with only test.list and
    with TESTS.LIST.
    """

    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from _bmad import source, verdict

    root = source() / "regression_tests"
    script = root / "scripts/run_tests.py"
    readme = (root / "README.md").read_text()
    with tempfile.TemporaryDirectory(prefix="bmad-help-") as tmp:
        work = Path(tmp)
        help_result = subprocess.run(
            [sys.executable, "-B", str(script), "-help"],
            cwd=work,
            text=True,
            capture_output=True,
            check=True,
            timeout=290,
        )
        help_text = help_result.stdout
        (work / "test.list").write_text("")
        result = subprocess.run(
            [sys.executable, "-B", str(script)],
            cwd=work,
            text=True,
            capture_output=True,
            check=False,
            timeout=290,
        )
        missing_default = (
            result.returncode == 1
            and "FileNotFoundError" in result.stderr
            and "'TESTS.LIST'" in result.stderr
        )
        if not missing_default:
            result.check_returncode()
        (work / "TESTS.LIST").write_text("")
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
    wrong_script = (
        "scripts/run_test.py" in readme
        and "run_test.py" in help_text
        and not (root / "scripts/run_test.py").exists()
    )
    wrong_list = "test.list" in readme and '"test.list"' in help_text and missing_default
    print(f"README/help script absent={wrong_script}; expected shipped run_tests.py")
    print(
        f"advertised test.list ignored={wrong_list}; actual TESTS.LIST control exit={control.returncode} expected=0"
    )
    verdict(wrong_script or wrong_list)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found (source/contract discrepancy) | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed (source/contract discrepancy) | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

