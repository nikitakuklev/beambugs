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
