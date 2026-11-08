---
title: "API-19: Importing orbit.core after another component initialized MPI aborts the process, and orbit_mpi.finalize fails after another component finalized it"
hide:
  - toc
---

# API-19 — Importing `orbit.core` after another component initialized MPI aborts the process, and `orbit_mpi.finalize` fails after another component finalized it

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Bindings and build |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [orbit_mpi.cc#L49-L58](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/orbit_mpi.cc#L49-L58)<br>[wrap_orbit_mpi.cc#L181-L183](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/wrap_orbit_mpi.cc#L181-L183)<br>[orbit_mpi.cc#L84-L93](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/orbit_mpi.cc#L84-L93)<br>[#L33-L45](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/mpi/orbit_mpi.cc#L33-L45) |

Importing any `orbit.core` module of an MPI build calls `MPI_Init` without asking `MPI_Initialized`; `orbit_mpi.finalize` and the native fatal-error path query the world rank and call `MPI_Finalize` without asking `MPI_Finalized`. A program in which another component has already initialized MPI, such as one that imports `mpi4py` first, therefore aborts at the import with "Cannot call MPI_INIT or MPI_INIT_THREAD more than once", and a `try` block cannot catch it. After another component has finalized MPI, `orbit_mpi.finalize()` ends with status 1 instead of 0 and `finalize(message)` loses its message.

Run `python API-19.py`.

??? example "Reproducer: API-19.py"

    [Download API-19.py](API-19.py)

    ```python
    """API-19: Importing `orbit.core` after another component initialized MPI aborts the process, and `orbit_mpi.finalize` fails after another component finalized it

    Importing any `orbit.core` module of an MPI build calls `MPI_Init` without asking `MPI_Initialized`; `orbit_mpi.finalize` and the native fatal-error path query the world rank and call `MPI_Finalize` without asking `MPI_Finalized`. A program in which another component has already initialized MPI, such as one that imports `mpi4py` first, therefore aborts at the import with "Cannot call MPI_INIT or MPI_INIT_THREAD more than once", and a `try` block cannot catch it. After another component has finalized MPI, `orbit_mpi.finalize()` ends with status 1 instead of 0 and `finalize(message)` loses its message.

    Setup: needs the MPI build of PyORBIT and its launcher; set `PYORBIT_MPI_PYTHON` to that build's interpreter and `PYORBIT_MPIEXEC` to its `mpiexec`. The script launches itself under `mpiexec -n 2` and calls the linked MPICH library through `ctypes`, finding it in `/proc/self/maps` (Linux).

    The script stands in for the other MPI component by calling `MPI_Init`, `MPI_Init_thread` or `MPI_Finalize` of the MPI library that PyORBIT's extension links, through `ctypes`, before importing `orbit.core` inside a `try` block, or before calling `orbit_mpi.finalize()` and `orbit_mpi.finalize(message)`. It records the exit status, the first MPI error line and whether the import or the message got through. The controls run the same steps without the foreign call.
    """

    import ctypes
    import glob
    import importlib.util
    import os
    import subprocess
    import sys

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    MESSAGE = "API19 stop message"
    MPI_THREAD_MULTIPLE = 3  # MPICH's value; mpi4py initializes MPI with MPI_Init_thread at this level


    def libmpi():
        """Loads PyORBIT's MPI library without importing orbit.core, whose import calls MPI_Init."""
        spec = importlib.util.find_spec("orbit")
        if spec is None or spec.origin is None:
            raise RuntimeError("the orbit package is not importable")
        orbit_dir = os.path.dirname(spec.origin)
        ext = glob.glob(os.path.join(orbit_dir, "core", "orbit_mpi*.so"))[0]
        ctypes.CDLL(ext, mode=ctypes.RTLD_GLOBAL)  # dlopen only: the module initializer does not run
        with open("/proc/self/maps") as f:
            paths = {ln.split()[-1] for ln in f if "libmpi" in ln.split()[-1]}
        path = min(p for p in paths if os.path.basename(p).startswith("libmpi."))
        return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


    def worker(case):
        if case.startswith("init"):
            lib = libmpi()
            if case == "init_foreign":
                lib.MPI_Init(None, None)  # another component initializes MPI first
            elif case == "init_thread_foreign":
                provided = ctypes.c_int(0)
                lib.MPI_Init_thread(
                    None, None, ctypes.c_int(MPI_THREAD_MULTIPLE), ctypes.byref(provided)
                )
            try:
                from orbit.core import orbit_mpi as mpi
            except Exception as e:  # noqa: BLE001 - shows whether the failure can be caught
                print(f"import raised {type(e).__name__}", flush=True)
                return
            rank = mpi.MPI_Comm_rank(mpi.mpi_comm.MPI_COMM_WORLD)
            print(
                f"rank {rank} imported orbit.core, MPI_Initialized {mpi.MPI_Initialized()}", flush=True
            )
            return
        from orbit.core import orbit_mpi as mpi

        if case.startswith("fin_foreign"):
            libmpi().MPI_Finalize()  # another component finalizes MPI
        if case.endswith("_msg"):
            mpi.finalize(MESSAGE)
        else:
            mpi.finalize()


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


    def first_mpi_error(out):
        for line in out.splitlines():
            if "Cannot call MPI_INIT" in line or "after finalizing" in line:
                return line.strip()
        return "<none>"


    def main():
        bad = False
        for case, label in (
            ("init_control", "import orbit.core (control)"),
            ("init_foreign", "MPI_Init by another component, then import orbit.core"),
            (
                "init_thread_foreign",
                "MPI_Init_thread(MPI_THREAD_MULTIPLE) by another component, then import orbit.core",
            ),
        ):
            status, out = mpirun(2, case)
            imported = out.count("imported orbit.core, MPI_Initialized 1")
            caught = out.count("import raised")
            print(
                f"{label}: ranks imported {imported} of 2, exceptions caught {caught}, status {status}, "
                f"MPI error {first_mpi_error(out)!r}; expected 2 of 2, status 0 (MPI_Initialized tells the import "
                f"whether MPI is already up)"
            )
            if case == "init_control" and (imported != 2 or status != 0):
                raise RuntimeError("control failed")
            bad |= case != "init_control" and (imported != 2 or status != 0)
        for case, label, want in (
            ("fin_control", "orbit_mpi.finalize() (control)", 0),
            ("fin_foreign", "MPI_Finalize by another component, then orbit_mpi.finalize()", 0),
            ("fin_control_msg", f"orbit_mpi.finalize({MESSAGE!r}) (control)", 1),
            (
                "fin_foreign_msg",
                f"MPI_Finalize by another component, then orbit_mpi.finalize({MESSAGE!r})",
                1,
            ),
        ):
            status, out = mpirun(2, case)
            shown = MESSAGE in out
            msg = f", message printed {shown}" if case.endswith("_msg") else ""
            exp = ", message printed True" if case.endswith("_msg") else ""
            # finalize() is the documented normal stop (status 0); finalize(message) stops with status 1 and the message
            print(
                f"{label}: status {status}{msg}, MPI error {first_mpi_error(out)!r}; "
                f"expected status {want}{exp} (finalize's own exit, whoever finalized MPI)"
            )
            ok = status == want and (shown or not case.endswith("_msg"))
            if "control" in case and not ok:
                raise RuntimeError("control failed")
            bad |= not ok
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
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, MPICH 5.0.2, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, MPICH 5.0.2, CPU, double |

