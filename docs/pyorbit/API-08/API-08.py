"""API-08: MPI graph-topology wrappers end the process or crash instead of returning results

Three graph-topology wrappers in `orbit_mpi` cannot return a result. `MPI_Graph_get` and `MPI_Graph_neighbors` test the return value of `PyTuple_SetItem` inverted, so the first successfully stored element takes the fatal-error path and ends the process with status 1; `MPI_Graph_get` also sizes its edge tuple by the node count. `MPI_Graph_map` parses five arguments into three output pointers, so calls with three or four arguments end the process and a call with five crashes it with a segmentation fault. In a serial build `MPI_Graph_get(MPI_COMM_WORLD)` exits with status 1 where a pair of tuples is expected; on an MPI build, `MPI_Graph_neighbors` and `MPI_Graph_get` of a two-rank graph communicator exit with status 1 the same way.

The script calls `MPI_Graph_get` and `MPI_Graph_map` with three, four and five arguments, each in a child interpreter, and reports the exit status and the last line of error output. The control `MPI_Graphdims_get` returns the dimensions (1, 1) of the serial stub, so `MPI_Graph_get` must return two one-element tuples.
"""

import os
import subprocess
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

PRE = "from orbit.core import orbit_mpi as m\nW = m.mpi_comm.MPI_COMM_WORLD\n"


def run(code):
    """Exit status, stdout and last stderr line of `code` in a child interpreter."""
    p = subprocess.run(
        [sys.executable, "-I", "-c", PRE + code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    err = [ln.strip() for ln in p.stderr.splitlines() if ln.strip()]
    return p.returncode, p.stdout.strip() or "<none>", err[-1] if err else "<none>"


cases = [
    (
        "Graph_get(W)",
        "r = m.MPI_Graph_get(W)\nprint(type(r).__name__, len(r))",
        "exit 0, output 'tuple 2'",
    ),
    # The wrapper's own messages name MPI_Graph_map(MPI_Comm old, [...indexes], [...edges]).
    ("Graph_map(W, [1], [0])", "print(m.MPI_Graph_map(W, [1], [0]))", "exit 0, a rank"),
    (
        "Graph_map(W, [1], [0], 0)",
        "print(m.MPI_Graph_map(W, [1], [0], 0))",
        "exit 0 or a TypeError",
    ),
    (
        "Graph_map(W, [1], [0], 0, W)",
        "print(m.MPI_Graph_map(W, [1], [0], 0, W))",
        "exit 0 or a TypeError",
    ),
]
reproduced = False
for label, code, exp in cases:
    rc, out, err = run(code)
    print(f"{label}: observed exit {rc}, output {out}, stderr tail {err!r}; expected {exp}")
    reproduced |= rc != 0
rc, out, _ = run("print(m.MPI_Graphdims_get(W))")
print(f"control Graphdims_get(W): observed exit {rc}, output {out}; expected exit 0, output (1, 1)")

print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")
