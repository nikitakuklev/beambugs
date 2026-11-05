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
