"""API-20: Serial `MPI_Recv` reads an uninitialized message size and returns garbage, crashes or stops at random

In builds without MPI, `orbit_mpi.MPI_Recv(type, source, tag, comm)` sizes its result from the serial `MPI_Get_count` stub, which sets the status count but never writes the count it returns, so the wrapper branches on an uninitialized `int`. A serial receive has no sender, and the wrapper's check for a size below 1 makes the intended result a fatal-error stop. Instead the call stops, returns a scalar, returns a tuple of up to hundreds of millions of elements that no one sent, or ends with a segmentation fault, varying from run to run; the tuple holds leftover memory for small sizes and zeros for large ones. Valgrind reports the uninitialized read on every run.

Setup: the script calls `valgrind` (memcheck), and it caps each child interpreter's address space at 4 GiB so that a garbage size cannot exhaust the machine's memory.

The script runs `MPI_Recv(MPI_INT, 0, 0, MPI_COMM_WORLD)` once under valgrind memcheck, which stops at the first error and names the function and the origin of the uninitialized value, and then `MPI_Recv` of `MPI_INT` and `MPI_DOUBLE` in five fresh child interpreters each, printing each child's result or exit status. The expected result of every run is the wrapper's "MPI_Recv(...) - fatal error. STOP." with exit status 1. It applies to a build without MPI (`MPI_Initialized() == 0`) and reports NOT REPRODUCED on an MPI build.
"""

import os
import resource
import shutil
import subprocess
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from orbit.core import orbit_mpi as mpi

serial = mpi.MPI_Initialized() == 0


def code(datatype):
    return (
        "from orbit.core import orbit_mpi as m\n"
        f"r = m.MPI_Recv(m.mpi_datatype.{datatype}, 0, 0, m.mpi_comm.MPI_COMM_WORLD)\n"
        "print('returned %s of %d elements' % (type(r).__name__, len(r)) if isinstance(r, tuple) else "
        "'returned %r' % (r,))\n"
    )


# The serial Probe stub sets status.count = 0; a count of 0 takes the wrapper's size < 1 branch, which prints
# "MPI_Recv(...) - fatal error. STOP." and exits through ORBIT_MPI_Finalize with status 1.
EXPECTED = "exit 1, MPI_Recv(...) - fatal error. STOP."


def cap_memory():
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))


if shutil.which("valgrind") is None:
    print("valgrind not found")
    sys.exit(2)

env = dict(os.environ, PYTHONMALLOC="malloc")  # let memcheck see Python's allocations
vg = subprocess.run(
    [
        "valgrind",
        "--exit-on-first-error=yes",
        "--error-exitcode=99",
        "--track-origins=yes",
        sys.executable,
        "-I",
        "-c",
        code("MPI_INT"),
    ],
    capture_output=True,
    text=True,
    timeout=600,
    env=env,
    preexec_fn=cap_memory,
    check=False,
)
lines = [ln.split("== ", 1)[-1] for ln in vg.stderr.splitlines() if ln.startswith("==")]
first = next((i for i, ln in enumerate(lines) if "uninitialised" in ln), None)
in_recv = False
if first is not None:
    frame = lines[first + 1] if first + 1 < len(lines) else ""
    origin = next(
        (lines[j + 1] for j in range(first, len(lines) - 1) if "was created by" in lines[j]), ""
    )
    in_recv = "mpi_recv" in frame and "mpi_recv" in origin
    where = "stack allocation" if "stack" in " ".join(lines[first : first + 20]) else ""
    # library paths are shortened to their file name
    frame_s = (
        frame.strip().replace(os.path.dirname(frame.split("(in ")[-1]) + "/", "")
        if "(in " in frame
        else frame
    )
    origin_s = (
        origin.strip().replace(os.path.dirname(origin.split("(in ")[-1]) + "/", "")
        if "(in " in origin
        else origin
    )
    print(
        f"valgrind: first error {lines[first]!r} {frame_s.strip()}; origin {where!r} {origin_s.strip()}"
    )
else:
    print(f"valgrind: no uninitialised-value error (exit {vg.returncode})")
print(
    "valgrind verdict: observed %s, expected no uninitialised value (the serial count should be written as 0)"
    % ("uninitialised size used in mpi_recv" if in_recv else "none in mpi_recv")
)

for datatype in ("MPI_INT", "MPI_DOUBLE"):
    for i in range(5):
        try:
            p = subprocess.run(
                [sys.executable, "-I", "-c", code(datatype)],
                capture_output=True,
                text=True,
                timeout=60,
                preexec_fn=cap_memory,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # A garbage size of ~1e8 elements can keep the child building its tuple for minutes.
            print(
                f"MPI_Recv({datatype}) run {i}: observed no return within 60 s; expected {EXPECTED}"
            )
            continue
        err = p.stderr.strip().splitlines()
        if p.returncode < 0:
            obs = f"killed by signal {-p.returncode}"
        elif p.stdout.strip():
            obs = f"exit {p.returncode}, {p.stdout.strip()}"
        else:
            obs = f"exit {p.returncode}, {err[-1] if err else ''}"
        print(f"MPI_Recv({datatype}) run {i}: observed {obs}; expected {EXPECTED}")

print(f"serial build: {serial}")
# The verdict rests on the deterministic valgrind finding; the child outcomes vary from run to run.
print("DEFECT REPRODUCED" if serial and in_recv else "NOT REPRODUCED")
