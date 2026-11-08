---
title: "API-09: Communicators returned by MPI_Comm_create, MPI_Comm_dup and MPI_Comm_split are never released"
hide:
  - toc
---

# API-09 — Communicators returned by `MPI_Comm_create`, `MPI_Comm_dup` and `MPI_Comm_split` are never released

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Bindings and build |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [wrap_mpi_comm.cc#L162-L167](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_mpi_comm.cc#L162-L167)<br>[#L61-L68](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_mpi_comm.cc#L61-L68)<br>[#L47-L55](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_mpi_comm.cc#L47-L55)<br>[wrap_orbit_mpi_comm_functions.hh#L8-L57](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_comm_functions.hh#L8-L57)<br>[Bunch.cc#L59](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Bunch.cc#L59)<br>[#L107](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Bunch.cc#L107)<br>[wrap_mpi_group.cc#L155-L159](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_mpi_group.cc#L155-L159) |

The helper that creates the communicator objects returned by `orbit_mpi.MPI_Comm_create`, `MPI_Comm_dup` and `MPI_Comm_split` takes one reference more than it hands out, so these objects are never deallocated: `sys.getrefcount` reports 3 for a communicator held in one variable, where 2 is expected, and each call keeps 24 bytes. Their destructor, which frees the MPI communicator, therefore never runs; only an explicit `comm.free()` releases it. On an MPICH build, a loop that drops `MPI_Comm_dup` results without `free()` aborts the job after 2046 duplicates with "Too many communicators". Each `Bunch` also leaks its 24-byte communicator wrapper; groups from `MPI_Comm_group` are released correctly.

Run `python API-09.py`.

??? example "Reproducer: API-09.py"

    [Download API-09.py](API-09.py)

    ```python
    """API-09: Communicators returned by `MPI_Comm_create`, `MPI_Comm_dup` and `MPI_Comm_split` are never released

    The helper that creates the communicator objects returned by `orbit_mpi.MPI_Comm_create`, `MPI_Comm_dup` and `MPI_Comm_split` takes one reference more than it hands out, so these objects are never deallocated: `sys.getrefcount` reports 3 for a communicator held in one variable, where 2 is expected, and each call keeps 24 bytes. Their destructor, which frees the MPI communicator, therefore never runs; only an explicit `comm.free()` releases it. On an MPICH build, a loop that drops `MPI_Comm_dup` results without `free()` aborts the job after 2046 duplicates with "Too many communicators". Each `Bunch` also leaks its 24-byte communicator wrapper; groups from `MPI_Comm_group` are released correctly.

    The script prints the reference counts of the three communicators and of a group (control), each held in one variable. It then counts the bytes that 20000 discarded results of `MPI_Comm_dup`, `MPI_Comm_group` (control) and `Bunch()` leave allocated, with `tracemalloc`.
    """

    import gc
    import os
    import sys
    import tracemalloc

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core import orbit_mpi as mpi
    from orbit.core.bunch import Bunch

    W = mpi.mpi_comm.MPI_COMM_WORLD
    G = mpi.MPI_Comm_group(W)

    # sys.getrefcount counts the variable plus its own argument: 2 for an object held by one name.
    dup = mpi.MPI_Comm_dup(W)
    split = mpi.MPI_Comm_split(W, 0, 0)
    create = mpi.MPI_Comm_create(W, G)
    group = mpi.MPI_Comm_group(W)
    # Direct calls, so that no container or loop variable adds a reference.
    counts = {
        "MPI_Comm_dup": sys.getrefcount(dup),
        "MPI_Comm_split": sys.getrefcount(split),
        "MPI_Comm_create": sys.getrefcount(create),
    }
    n_group = sys.getrefcount(group)
    for label, n in counts.items():
        print(f"getrefcount({label} result): observed {n}, expected 2 (one variable + call argument)")
    print(f"control getrefcount(MPI_Comm_group result): observed {n_group}, expected 2")
    reproduced = any(n != 2 for n in counts.values())


    def retained(make, n=20000):
        """Bytes per call still allocated after n results of make() are discarded."""
        for _ in range(100):
            make()
        gc.collect()
        tracemalloc.start()
        a = tracemalloc.get_traced_memory()[0]
        for _ in range(n):
            make()
        gc.collect()
        b = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        return (b - a) / n


    # A discarded object is freed, so nothing should remain per call (allocator noise is < 1 byte).
    for label, make in (("MPI_Comm_dup(W)", lambda: mpi.MPI_Comm_dup(W)), ("Bunch()", Bunch)):
        print(f"bytes retained per discarded {label}: observed {retained(make):.2f}, expected 0")
    print(
        f"control bytes retained per discarded MPI_Comm_group(W): observed {retained(lambda: mpi.MPI_Comm_group(W)):.2f}, expected 0"
    )

    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

??? example "Reproducer: API-09-mpi.py"

    [Download API-09-mpi.py](API-09-mpi.py)

    ```python
    """API-09: leaked `MPI_Comm_dup` communicators exhaust the MPI library's communicator limit on an MPI build

    Setup: needs the MPI build of PyORBIT and its launcher; set `PYORBIT_MPI_PYTHON` to that build's interpreter and `PYORBIT_MPIEXEC` to its `mpiexec`. The script launches itself under `mpiexec -n 2`.

    The script duplicates `MPI_COMM_WORLD` 10000 times in a loop and drops each duplicate, which must release it, and reports how many duplicates were made before the run ended and with what error. The control frees each duplicate explicitly with `free()` and must complete the loop.
    """

    import os
    import subprocess
    import sys

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    LOOPS = 10000


    def worker(mode):
        from orbit.core import orbit_mpi as mpi

        W = mpi.mpi_comm.MPI_COMM_WORLD
        rank = mpi.MPI_Comm_rank(W)
        for i in range(LOOPS):
            c = mpi.MPI_Comm_dup(W)
            if mode == "free":
                c.free()
            del c  # the last reference held by the script
            if rank == 0:
                print(f"made {i + 1}", flush=True)
        print(f"rank {rank} completed {LOOPS}", flush=True)


    def mpirun(n, *args, timeout=120):
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
        bad = False
        for mode, label in (
            ("free", "dup + free() (control)"),
            ("drop", "dup, dropped without free()"),
        ):
            status, out = mpirun(2, mode)
            made = max([int(ln.split()[1]) for ln in out.splitlines() if ln.startswith("made ")] or [0])
            done = out.count(f"completed {LOOPS}")
            err = next(
                (ln.strip() for ln in out.splitlines() if "Too many communicators" in ln), "<none>"
            )
            print(
                f"{label}: ranks completed {done} of 2, {made} duplicates made on rank 0, status {status}, error {err!r}; "
                f"expected 2 of 2, status 0 (a dropped communicator is released)"
            )
            if mode == "free" and (done != 2 or status != 0):
                raise RuntimeError("control failed")
            bad |= mode == "drop" and (done != 2 or status != 0)
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
| Found<br><small class="bb-scope">communicator exhaustion on an MPI build</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, MPICH 5.0.2, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

