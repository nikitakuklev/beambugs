---
title: "DIA-14: profiles hangs or aborts an MPI run for a bunch with fewer than two particles or a zero-width coordinate range"
hide:
  - toc
---

# DIA-14 — `profiles` hangs or aborts an MPI run for a bunch with fewer than two particles or a zero-width coordinate range

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Diagnostics |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [profiles.py#L26-L32](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/profiles.py#L26-L32)<br>[#L71-L76](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/profiles.py#L71-L76)<br>[#L55-L69](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/profiles.py#L55-L69)<br>[#L78-L81](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/profiles.py#L78-L81)<br>[Grid1D.cc#L568-L580](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/Grid1D.cc#L568-L580) |

`orbit.diagnostics.profiles(bunch, coord, file)` writes a note and returns an empty histogram when the bunch has fewer than two particles or the coordinate range has zero width, but only on rank 0 of the bunch's communicator. The other ranks continue: with fewer than two particles they enter the collective extrema search, which meets a different collective on rank 0, and with a zero-width range they bin on a zero-width grid, where a native fatal error that prints nothing stops them. A `dE` profile of a monoenergetic bunch leaves a two-rank job running after 20 s, and a bunch with 0 or 1 particles aborts it at the next all-reduce with an MPI message-truncated error. On one rank both calls return the empty histogram.

Run `python DIA-14.py`.

??? example "Reproducer: DIA-14.py"

    [Download DIA-14.py](DIA-14.py)

    ```python
    """DIA-14: `profiles` hangs or aborts an MPI run for a bunch with fewer than two particles or a zero-width coordinate range

    `orbit.diagnostics.profiles(bunch, coord, file)` writes a note and returns an empty histogram when the bunch has fewer than two particles or the coordinate range has zero width, but only on rank 0 of the bunch's communicator. The other ranks continue: with fewer than two particles they enter the collective extrema search, which meets a different collective on rank 0, and with a zero-width range they bin on a zero-width grid, where a native fatal error that prints nothing stops them. A `dE` profile of a monoenergetic bunch leaves a two-rank job running after 20 s, and a bunch with 0 or 1 particles aborts it at the next all-reduce with an MPI message-truncated error. On one rank both calls return the empty histogram.

    Setup: needs the MPI build of PyORBIT and its launcher; set `PYORBIT_MPI_PYTHON` to that build's interpreter and `PYORBIT_MPIEXEC` to its `mpiexec`. The script launches itself under `mpiexec` with 1 and 2 ranks and kills a job after 20 s.

    The script calls `profiles` for `dE` on a bunch of six particles with dE = 0, three per rank, for `x` on a bunch whose only particle is on the last rank, and for `x` on an empty bunch, then lets every rank enter `MPI_Allreduce`, as a tracking loop would. It records how many ranks pass that all-reduce, the exit status and whether the job ends within 20 s. The controls are the first two cases on one rank and an ordinary `x` profile on two ranks.
    """

    import os
    import signal
    import subprocess
    import sys
    import tempfile
    import time

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    TIMEOUT = 20.0


    def worker(case, outdir):
        from orbit.core import orbit_mpi as mpi
        from orbit.core.bunch import Bunch
        from orbit.diagnostics.profiles import profiles

        W = mpi.mpi_comm.MPI_COMM_WORLD
        rank, size = mpi.MPI_Comm_rank(W), mpi.MPI_Comm_size(W)
        b = Bunch()
        b.macroSize(1.0)
        if case == "mono":  # three particles per rank, all with dE = 0
            for i in range(3):
                b.addParticle(1e-3 * (i + 3 * rank), 0.0, 0.0, 0.0, 0.1 * i, 0.0)
            coord = "dE"
        elif case == "single":  # one particle in the whole bunch, on the last rank
            if rank == size - 1:
                b.addParticle(1e-3, 0.0, 0.0, 0.0, 0.0, 0.0)
            coord = "x"
        elif case == "empty":  # no particles on any rank
            coord = "x"
        else:  # ordinary spread (control)
            for i in range(3):
                b.addParticle(1e-3 * (i + 3 * rank), 0.0, 0.0, 0.0, 0.1 * i, 1e-4 * i)
            coord = "x"
        profiles(b, coord, os.path.join(outdir, "profile.txt"))
        mpi.MPI_Allreduce(
            1, mpi.mpi_datatype.MPI_INT, mpi.mpi_op.MPI_SUM, W
        )  # next collective of a run
        print(f"rank {rank} passed", flush=True)


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
        for case, label in (
            ("mono", "dE profile of a monoenergetic bunch"),
            ("single", "x profile of a one-particle bunch"),
            ("empty", "x profile of an empty bunch"),
            ("spread", "x profile of an ordinary bunch (control)"),
        ):
            for n in (1, 2) if case in ("mono", "single") else (2,):
                with tempfile.TemporaryDirectory() as d:
                    status, out, dt = mpirun(n, case, d)
                    note = "<none>"
                    if os.path.exists(os.path.join(d, "profile.txt")):
                        with open(os.path.join(d, "profile.txt")) as f:
                            note = f.readline().strip()
                passed = sum(f"rank {r} passed" in out for r in range(n))
                ended = (
                    f"ended in {dt:.1f} s, status {status}"
                    if status is not None
                    else f"still running after {TIMEOUT:.0f} s"
                )
                print(
                    f"{label}, {n} rank(s): ranks past the next all-reduce {passed} of {n}, job {ended}, "
                    f"file note {note!r}; expected {n} of {n} and status 0 (profiles returns on every rank)"
                )
                ok = passed == n and status == 0
                if (n == 1 or case == "spread") and not ok:
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

