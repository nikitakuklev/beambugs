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
