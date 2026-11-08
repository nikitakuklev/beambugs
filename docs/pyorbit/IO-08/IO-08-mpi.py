"""IO-08: the empty-rank failure of `collect_bunch` under a real two-rank MPI run

Setup: needs the MPI build of PyORBIT and its launcher; set `PYORBIT_MPI_PYTHON` to that build's interpreter and `PYORBIT_MPIEXEC` to its `mpiexec`. The script launches itself under `mpiexec -n 2`.

The script distributes x = 1, 2, 3 on rank 0 and nothing on rank 1 over a two-rank `MPI_COMM_WORLD` bunch and calls `collect_bunch` into a fresh shared directory; the root must return the three rows. The controls give rank 1 one particle (x = 4) and leave the root empty with rank 1 holding x = 4, 5.
"""

import os
import subprocess
import sys
import tempfile

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

CASES = {
    "rank1_empty": ({0: [1, 2, 3], 1: []}, [1.0, 2.0, 3.0]),
    "rank1_one": ({0: [1, 2, 3], 1: [4]}, [1.0, 2.0, 3.0, 4.0]),
    "root_empty": ({0: [], 1: [4, 5]}, [4.0, 5.0]),
}


def worker(case, outdir):
    from orbit.bunch_utils.serialize import collect_bunch
    from orbit.core import orbit_mpi as mpi
    from orbit.core.bunch import Bunch

    rank = mpi.MPI_Comm_rank(mpi.mpi_comm.MPI_COMM_WORLD)
    b = Bunch()
    for v in CASES[case][0][rank]:
        b.addParticle(float(v), 0.0, 0.0, 0.0, 0.0, 0.0)
    try:
        d = collect_bunch(b, output_dir=outdir, return_memmap=False)
        if rank == 0 and d is not None:
            print(f"ROOT {d['coords'][:, 0].tolist()}", flush=True)
    except Exception as e:  # noqa: BLE001 - the root reports whatever collect_bunch raises
        print(
            f"ROOT {type(e).__name__}: {e}" if rank == 0 else f"RANK{rank} {type(e).__name__}",
            flush=True,
        )


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
        ("rank1_empty", "rank 0 holds [1, 2, 3], rank 1 empty"),
        ("rank1_one", "rank 0 holds [1, 2, 3], rank 1 holds [4] (control)"),
        ("root_empty", "rank 0 empty, rank 1 holds [4, 5] (control)"),
    ):
        with tempfile.TemporaryDirectory() as d:
            status, out = mpirun(2, case, d)
        root = next(
            (ln[5:] for ln in out.splitlines() if ln.startswith("ROOT ")), "<no root output>"
        )
        expected = CASES[case][1]  # root's rows, then each other rank's rows in rank order
        print(f"{label}: root observed {root}; expected {expected}; status {status}")
        if case != "rank1_empty" and root != str(expected):
            raise RuntimeError("control failed")
        bad |= case == "rank1_empty" and root != str(expected)
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
    else:
        main()
