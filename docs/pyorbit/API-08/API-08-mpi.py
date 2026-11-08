"""API-08: `MPI_Graph_neighbors` and `MPI_Graph_get` on a real graph communicator of an MPI build

Setup: needs the MPI build of PyORBIT and its launcher; set `PYORBIT_MPI_PYTHON` to that build's interpreter and `PYORBIT_MPIEXEC` to its `mpiexec`. The script launches itself under `mpiexec -n 2`.

The script creates a two-node graph communicator in which each rank is the other's only neighbour (index [1, 2], edges [1, 0]) with `MPI_Graph_create`, and then calls `MPI_Graph_neighbors(G, rank)`, which must return the other rank, and `MPI_Graph_get(G)`, which must return ((1, 2), (1, 0)). The controls are `MPI_Graph_create`, `MPI_Graphdims_get` and `MPI_Graph_neighbors_count` on the same communicator, which the MPI library answers through the wrappers without the inverted tuple check.
"""

import os
import subprocess
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def worker(call):
    from orbit.core import orbit_mpi as mpi

    W = mpi.mpi_comm.MPI_COMM_WORLD
    rank = mpi.MPI_Comm_rank(W)
    G = mpi.mpi_comm.MPI_Comm()
    rc = mpi.MPI_Graph_create(W, [1, 2], [1, 0], 0, G)
    if call == "control":
        print(
            f"rank {rank} create {rc} dims {mpi.MPI_Graphdims_get(G)} count {mpi.MPI_Graph_neighbors_count(G, rank)}",
            flush=True,
        )
    elif call == "neighbors":
        print(f"rank {rank} result {mpi.MPI_Graph_neighbors(G, rank)}", flush=True)
    else:
        print(f"rank {rank} result {mpi.MPI_Graph_get(G)}", flush=True)


def mpirun(n, *args, timeout=60):
    py, mx = os.environ.get("PYORBIT_MPI_PYTHON"), os.environ.get("PYORBIT_MPIEXEC")
    if not py or not mx:
        sys.exit("set PYORBIT_MPI_PYTHON and PYORBIT_MPIEXEC to the MPI build's python and mpiexec")
    p = subprocess.run(
        [mx, "-n", str(n), py, "-I", os.path.abspath(__file__), "--worker", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    return p.returncode, p.stdout


def main():
    status, out = mpirun(2, "control")
    lines = sorted(ln for ln in out.splitlines() if ln.startswith("rank "))
    print(
        f"control create/dims/count: observed {lines}, status {status}; expected create 0, dims (2, 2), count 1 on each rank"
    )
    if status != 0 or lines != [
        "rank 0 create 0 dims (2, 2) count 1",
        "rank 1 create 0 dims (2, 2) count 1",
    ]:
        raise RuntimeError("control failed")
    bad = False
    for call, expected in (
        ("neighbors", "rank 0 -> (1,), rank 1 -> (0,)"),
        ("get", "((1, 2), (1, 0)) on each rank"),
    ):
        status, out = mpirun(2, call)
        results = sorted(
            ln for ln in out.splitlines() if ln.startswith("rank ") and " result " in ln
        )
        err = next((ln.strip() for ln in out.splitlines() if "cannot create" in ln), "<none>")
        print(
            f"MPI_Graph_{call}: observed results {results or '<none>'}, status {status}, message {err!r}; expected {expected}, status 0"
        )
        bad |= status != 0 or len(results) != 2
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
    else:
        main()
