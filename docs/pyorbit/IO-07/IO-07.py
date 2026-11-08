"""IO-07: `Bunch.readBunch` never returns for a missing file

`Bunch.readBunch(path)` checks the opened stream with `ifstream::bad()`, which stays false when a
file is missing or unreadable, so the open failure is not reported. The coordinate reader then loops
on a stream that never reaches end-of-file: the call spins at full CPU and never returns, with no
message, and a batch job with a misspelled bunch path hangs until its wall-clock limit. The expected
behavior is an immediate error naming the file. The bunch-attribute, synchronous-particle and
particle-attribute readers use the same test and return silently without reading.

The script runs each read in a child interpreter with a 10 s limit: `readBunch` on a missing path,
`readBunch` on a file without read permission (skipped when running as root, which ignores file
modes), and `initBunchAttr` and `readPartAttr` on a missing path. It reports whether each child
finished and what it printed. The control child reads a valid one-particle file written by
`dumpBunch` and returns at once.
"""

import os
import subprocess
import sys
import tempfile

LIMIT_S = 10.0

tmp = tempfile.mkdtemp()
missing = os.path.join(tmp, "no_such_bunch.txt")
unreadable = os.path.join(tmp, "unreadable.txt")
valid = os.path.join(tmp, "bunch.txt")
with open(unreadable, "w") as f:
    f.write("1 0 0 0 0 0\n")
os.chmod(unreadable, 0)

CHILD = (
    "import sys\n"
    "from orbit.core.bunch import Bunch\n"
    "b = Bunch()\n"
    "if sys.argv[1] == 'write':\n"
    "    b.addParticle(1e-3, 0, 0, 0, 0, 0)\n"
    "    b.dumpBunch(sys.argv[2])\n"
    "else:\n"
    "    getattr(b, sys.argv[1])(sys.argv[2])\n"
    "    print('returned, size', b.getSize())\n"
)


def run(method, path):
    """Run Bunch().<method>(path) in a child; return (finished, exit status, stdout + stderr)."""
    try:
        r = subprocess.run(
            [sys.executable, "-I", "-c", CHILD, method, path],
            capture_output=True,
            text=True,
            timeout=LIMIT_S,
            check=False,
        )
        return True, r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, None, ""


def report(label, finished, rc, out, expected):
    state = (
        f"finished (exit {rc}, output {out!r})"
        if finished
        else f"still running after {LIMIT_S:.0f} s"
    )
    print(f"{label}: observed {state}; expected {expected}")


if not run("write", valid)[0]:
    sys.exit("could not write the control file")

# Expected for an unopenable file: the call stops promptly with an error naming the file, as the
# reader's own branch "Can not open file:<name>" intends.
hung = []
for label, method, path in (
    ("readBunch(missing file)", "readBunch", missing),
    ("readBunch(file without read permission)", "readBunch", unreadable),
):
    if path == unreadable and os.geteuid() == 0:
        print(f"{label}: skipped, running as root")
        continue
    finished, rc, out = run(method, path)
    report(label, finished, rc, out, "prompt error naming the file")
    hung.append(not finished)

silent = []
for label, method in (
    ("initBunchAttr(missing file)", "initBunchAttr"),
    ("readPartAttr(missing file)", "readPartAttr"),
):
    finished, rc, out = run(method, missing)
    report(label, finished, rc, out, "prompt error naming the file")
    silent.append(finished and "Can not open" not in out)

finished, rc, out = run("readBunch", valid)
report(
    "readBunch(valid one-particle dump) (control)",
    finished,
    rc,
    out,
    "finished, 'returned, size 1'",
)
control = finished and rc == 0 and out.endswith("size 1")

if not control:
    sys.exit("control read failed")
print("DEFECT REPRODUCED" if all(hung) and all(silent) else "NOT REPRODUCED")
