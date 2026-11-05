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
