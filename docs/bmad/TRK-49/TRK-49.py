"""TRK-49: Upstream pytest comparator accepts missing trailing records

An upstream regression test can pass after writing only a matching prefix of its expected
output. The pytest comparator pairs available records and repeats the check for excess current
records, so it never rejects a shorter current file. A current file with one matching record
passes against two expected records. Equal-length output and an extra-current-record control
establish that the missing-tail check is the omitted branch.

The script imports the comparator from BMAD_SOURCE in an isolated child process and stubs only
executable launch. It supplies complete, truncated and overlong files and checks each acceptance
decision.
"""

import subprocess
import sys
import tempfile

from _bmad import source, verdict

CHILD = r"""
import pathlib
import sys
sys.path.insert(0,sys.argv[1])
import test_fortran as runner
root=pathlib.Path.cwd()
runner.TESTS_ROOT=root
runner.run_fortran_code=lambda *args: None
first='"first" ABS 1e-14 1\n'
second='"second" ABS 1e-14 2\n'
for name,now,correct in [('complete',first,first),('short',first,first+second),('long',first+second,first)]:
    case=root/name
    case.mkdir()
    (case/'output.now').write_text(now)
    (case/'output.correct').write_text(correct)
    loaded=runner.load_output_file(case/'output.now')
    expected=runner.load_output_file(case/'output.correct')
    assert len(loaded)==len(now.splitlines())
    assert len(expected)==len(correct.splitlines())
    assert loaded[:min(len(loaded),len(expected))]==expected[:min(len(loaded),len(expected))]
    try:
        runner.test_fortran(root,name)
    except ValueError as exc:
        if str(exc) not in ('More lines in output.now than in output.correct','More lines in output.correct than in output.now'):
            raise
        accepted=0
    else:
        accepted=1
    print('RESULT',name,accepted,flush=True)
"""
with tempfile.TemporaryDirectory(prefix="bmad-comparator-") as tmp:
    result = subprocess.run(
        [sys.executable, "-B", "-c", CHILD, str(source() / "regression_tests")],
        cwd=tmp,
        text=True,
        capture_output=True,
        check=True,
        timeout=290,
    )
records = {}
for line in result.stdout.splitlines():
    if line.startswith("RESULT "):
        _, name, accepted = line.split()
        records[name] = int(accepted)
if (
    set(records) != {"complete", "short", "long"}
    or records["complete"] != 1
    or records["long"] != 0
):
    raise AssertionError(result.stdout)
for name, expected in [("complete", 1), ("long", 0), ("short", 0)]:
    print(f"{name}: accepted={records[name]} expected={expected}")
verdict(records["short"] == 1)
