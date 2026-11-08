---
title: "API-13: MPI wrappers and Bunch.setMPIComm accept any Python object as a communicator"
hide:
  - toc
---

# API-13 — MPI wrappers and `Bunch.setMPIComm` accept any Python object as a communicator

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Bindings and build |
| Priority | P5 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [wrap_bunch.cc#L87-L102](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/wrap_bunch.cc#L87-L102)<br>[wrap_orbit_mpi_comm_functions.hh#L142-L150](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_comm_functions.hh#L142-L150)<br>[wrap_orbit_mpi_send_receive_functions.hh#L136-L145](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi_send_receive_functions.hh#L136-L145) |

`Bunch.setMPIComm` and the `orbit_mpi` functions cast their communicator, datatype and operation arguments to PyORBIT's wrapper structures without checking the Python type. A string is accepted as a bunch communicator and returned by `getMPIComm()`, and in a serial build `MPI_Comm_rank("x")` returns 0 and `MPI_Comm_size(None)` returns 1, with no `TypeError`. In an MPI build the handle is read from the foreign object's memory and passed to MPI: on MPICH, `MPI_Comm_rank("x")`, `MPI_Comm_rank(7)` and `getSizeGlobal()` of a bunch given a string abort the job with "Invalid communicator".

Run `python API-13.py`.

??? example "Reproducer: API-13.py"

    [Download API-13.py](API-13.py)

    ```python
    """API-13: MPI wrappers and `Bunch.setMPIComm` accept any Python object as a communicator

    `Bunch.setMPIComm` and the `orbit_mpi` functions cast their communicator, datatype and operation arguments to PyORBIT's wrapper structures without checking the Python type. A string is accepted as a bunch communicator and returned by `getMPIComm()`, and in a serial build `MPI_Comm_rank("x")` returns 0 and `MPI_Comm_size(None)` returns 1, with no `TypeError`. In an MPI build the handle is read from the foreign object's memory and passed to MPI: on MPICH, `MPI_Comm_rank("x")`, `MPI_Comm_rank(7)` and `getSizeGlobal()` of a bunch given a string abort the job with "Invalid communicator".

    The script passes a string, `None` and an integer where a communicator or operation is required and records whether each call raises `TypeError`. The control passes `MPI_COMM_WORLD`.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core import orbit_mpi as mpi
    from orbit.core.bunch import Bunch

    W = mpi.mpi_comm.MPI_COMM_WORLD
    INT = mpi.mpi_datatype.MPI_INT


    def attempt(f):
        try:
            return f"accepted, returned {f()!r}"
        except TypeError as e:
            return f"TypeError({e})"


    b = Bunch()
    cases = [
        ("Bunch.setMPIComm('not a comm')", lambda: (b.setMPIComm("not a comm"), b.getMPIComm())[1]),
        ("MPI_Comm_rank('x')", lambda: mpi.MPI_Comm_rank("x")),
        ("MPI_Comm_size(None)", lambda: mpi.MPI_Comm_size(None)),
        ("MPI_Barrier(3)", lambda: mpi.MPI_Barrier(3)),
        ("MPI_Allreduce(1, MPI_INT, None, W)", lambda: mpi.MPI_Allreduce(1, INT, None, W)),
    ]
    reproduced = False
    for label, f in cases:
        obs = attempt(f)
        print(
            f"{label}: observed {obs}; expected TypeError (the argument is not an orbit_mpi wrapper object)"
        )
        reproduced |= not obs.startswith("TypeError")
    print(f"control MPI_Comm_rank(MPI_COMM_WORLD): observed {mpi.MPI_Comm_rank(W)}, expected 0")

    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

??? example "Reproducer: API-13-mpi.py"

    [Download API-13-mpi.py](API-13-mpi.py)

    ```python
    """API-13: non-communicator objects passed as communicators on an MPI build

    Setup: needs the MPI build of PyORBIT and its launcher; set `PYORBIT_MPI_PYTHON` to that build's interpreter and `PYORBIT_MPIEXEC` to its `mpiexec`. The script launches itself under `mpiexec -n 2`.

    The script passes the string `"x"` and the integer 7 to `MPI_Comm_rank`, and sets the string `"text"` as a bunch's communicator before `getSizeGlobal`, each in its own two-rank run; each call must raise `TypeError`. It records the exit status and the MPI library's error, which shows the handle the wrapper read from the object's memory. The control passes `MPI_COMM_WORLD`.
    """

    import os
    import subprocess
    import sys

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")


    def worker(case):
        from orbit.core import orbit_mpi as mpi
        from orbit.core.bunch import Bunch

        try:
            if case == "control":
                v = mpi.MPI_Comm_rank(mpi.mpi_comm.MPI_COMM_WORLD)
            elif case == "rank_str":
                v = mpi.MPI_Comm_rank("x")
            elif case == "rank_int":
                v = mpi.MPI_Comm_rank(7)
            else:
                b = Bunch()
                b.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                b.setMPIComm("text")
                v = b.getSizeGlobal()
            print(f"RETURNED {v}", flush=True)
        except TypeError as e:
            print(f"TYPEERROR {e}", flush=True)


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
        bad = False
        for case, label in (
            ("control", "MPI_Comm_rank(MPI_COMM_WORLD) (control)"),
            ("rank_str", "MPI_Comm_rank('x')"),
            ("rank_int", "MPI_Comm_rank(7)"),
            ("bunch_str", "Bunch.setMPIComm('text'); getSizeGlobal()"),
        ):
            status, out = mpirun(2, case)
            lines = out.splitlines()
            res = sorted({ln for ln in lines if ln.startswith(("RETURNED", "TYPEERROR"))})
            err = next((ln.strip() for ln in lines if "failed" in ln and "comm=" in ln), "<none>")
            if case == "control":
                print(
                    f"{label}: observed {res}, status {status}; expected RETURNED 0 and RETURNED 1, status 0"
                )
                if status != 0 or res != ["RETURNED 0", "RETURNED 1"]:
                    raise RuntimeError("control failed")
                continue
            print(
                f"{label}: observed {res or '<no Python result>'}, status {status}, MPI error {err!r}; expected TypeError on both ranks, status 0"
            )
            bad |= not (res and all(r.startswith("TYPEERROR") for r in res))
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
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Found<br><small>MPI-build run of foreign communicator objects</small> | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, MPICH 5.0.2, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

