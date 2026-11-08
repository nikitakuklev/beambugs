---
title: "API-08: MPI graph-topology wrappers end the process or crash instead of returning results"
hide:
  - toc
---

# API-08 — MPI graph-topology wrappers end the process or crash instead of returning results

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Bindings and build |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [wrap_orbit_mpi_graph_functions.hh#L65-L76](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_graph_functions.hh#L65-L76)<br>[#L83-L87](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_graph_functions.hh#L83-L87)<br>[#L134-L153](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_graph_functions.hh#L134-L153) |

Three graph-topology wrappers in `orbit_mpi` cannot return a result. `MPI_Graph_get` and `MPI_Graph_neighbors` test the return value of `PyTuple_SetItem` inverted, so the first successfully stored element takes the fatal-error path and ends the process with status 1; `MPI_Graph_get` also sizes its edge tuple by the node count. `MPI_Graph_map` parses five arguments into three output pointers, so calls with three or four arguments end the process and a call with five crashes it with a segmentation fault. In a serial build `MPI_Graph_get(MPI_COMM_WORLD)` exits with status 1 where a pair of tuples is expected; on an MPI build, `MPI_Graph_neighbors` and `MPI_Graph_get` of a two-rank graph communicator exit with status 1 the same way.

Run `python API-08.py`.

??? example "Reproducer: API-08.py"

    [Download API-08.py](API-08.py)

    ```python
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

    ```

??? example "Reproducer: API-08-mpi.py"

    [Download API-08-mpi.py](API-08-mpi.py)

    ```python
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

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Found<br><small class="bb-scope">MPI-build run of the graph wrappers</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, MPICH 5.0.2, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

