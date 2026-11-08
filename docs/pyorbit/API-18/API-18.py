"""API-18: A native fatal error on one MPI rank hangs the job while the other ranks wait in a collective, and its message is lost unless the rank is 0

PyORBIT's native fatal errors call `MPI_Finalize` and exit on the failing rank alone; nothing calls `MPI_Abort`. When the other ranks have entered a collective operation, the failing rank blocks in `MPI_Finalize` and the job runs until it is killed. The messages print only on rank 0, and `orbit_mpi.finalize(message)` prints after `MPI_Finalize` returns, so such a job can end or hang with no message at all. In a two-rank job, a missing bunch attribute or `orbit_mpi.finalize(message)` on one rank leaves the job running after 20 s while the other rank waits in an all-reduce; `MPI_Abort` from the same rank ends it at once.

Setup: needs the MPI build of PyORBIT and its launcher; set `PYORBIT_MPI_PYTHON` to that build's interpreter and `PYORBIT_MPIEXEC` to its `mpiexec`. The script launches itself under `mpiexec -n 2`, kills a job after 20 s, takes about two minutes, and calls `MPI_Abort` of the linked MPICH library through `ctypes`, finding it in `/proc/self/maps` (Linux).

The script runs two-rank jobs in which one rank hits a native fatal error, `Bunch.bunchAttrDouble` of a missing attribute or `orbit_mpi.finalize` with a message, while the other rank continues into `MPI_Allreduce`, as a tracking loop would. It records whether the error message appears, the exit status and whether the job ends within 20 s, with the failing rank 1 and then rank 0. The controls make both ranks hit the same error, and call `MPI_Abort` on rank 1, which must end the job at once; a last line shows that when the other rank ends its script instead of waiting, the job ends but rank 1's message is still lost.
"""

import ctypes
import os
import signal
import subprocess
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

TIMEOUT = 20.0
MESSAGE = "API18 fatal message"
ABORT_CODE = 7


def mpi_abort():
    """MPI_Abort(MPI_COMM_WORLD, ABORT_CODE) of the MPI library that orbit.core has loaded."""
    with open("/proc/self/maps") as f:
        paths = {ln.split()[-1] for ln in f if "libmpi" in ln.split()[-1]}
    lib = ctypes.CDLL(min(p for p in paths if os.path.basename(p).startswith("libmpi.")))
    lib.MPI_Abort(
        ctypes.c_int(0x44000000), ctypes.c_int(ABORT_CODE)
    )  # MPICH's MPI_COMM_WORLD handle


def worker(error, who, other):
    from orbit.core import orbit_mpi as mpi
    from orbit.core.bunch import Bunch

    W = mpi.mpi_comm.MPI_COMM_WORLD
    rank = mpi.MPI_Comm_rank(W)
    if who == "all" or int(who) == rank:
        if error == "attribute":
            Bunch().bunchAttrDouble("absent")  # native fatal error: "No attribute with name:absent"
        elif error == "finalize":
            mpi.finalize(MESSAGE)
        else:
            print(MESSAGE, flush=True)
            mpi_abort()
    if other == "end":
        return  # the other rank reaches the end of its script
    mpi.MPI_Allreduce(
        1, mpi.mpi_datatype.MPI_INT, mpi.mpi_op.MPI_SUM, W
    )  # the next collective of a run
    print(f"rank {rank} passed the all-reduce", flush=True)


def mpirun(n, *args):
    py, mx = os.environ.get("PYORBIT_MPI_PYTHON"), os.environ.get("PYORBIT_MPIEXEC")
    if not py or not mx:
        sys.exit("set PYORBIT_MPI_PYTHON and PYORBIT_MPIEXEC to the MPI build's python and mpiexec")
    t0 = time.monotonic()
    p = subprocess.Popen(
        [mx, "-n", str(n), py, "-I", os.path.abspath(__file__), "--worker", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = p.communicate(timeout=TIMEOUT)
        return p.returncode, out, time.monotonic() - t0
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGTERM)  # mpiexec cleans up its ranks on SIGTERM
        try:
            out, _ = p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
        return None, out, None


def main():
    bad = False
    attr = ("attribute", "Bunch().bunchAttrDouble('absent')", "No attribute with name:absent")
    fin = ("finalize", f"orbit_mpi.finalize({MESSAGE!r})", MESSAGE)
    for (error, label, banner), who, other, role in (
        (attr, "1", "wait", "case"),
        (attr, "0", "wait", "case"),
        (fin, "1", "wait", "case"),
        (fin, "0", "wait", "case"),
        (attr, "all", "wait", "control"),
        (fin, "all", "wait", "control"),
        (("abort", f"MPI_Abort(MPI_COMM_WORLD, {ABORT_CODE})", MESSAGE), "1", "wait", "control"),
        (attr, "1", "end", "scope"),
    ):
        status, out, dt = mpirun(2, error, who, other)
        shown = banner in out
        ended = (
            f"ended in {dt:.1f} s with status {status}"
            if status is not None
            else f"still running after {TIMEOUT:.0f} s"
        )
        where = {"1": "rank 1", "0": "rank 0", "all": "both ranks"}[who]
        peer = (
            "the other rank in an all-reduce"
            if other == "wait"
            else "the other rank at the end of its script"
        )
        tag = {"case": "", "control": " (control)", "scope": ""}[role]
        print(
            f"{label} on {where}, {peer}{tag}: message printed {shown}, job {ended}; "
            f"expected message printed True and the job ended at once"
        )
        if role == "control" and not (shown and status is not None and dt < 5.0):
            raise RuntimeError("control failed: the job did not print its message and stop")
        if role == "case":
            bad |= not shown or status is None
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:])
    else:
        main()
