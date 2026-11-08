---
title: "IO-10: ParticleIdNumber.addParticleIdNumbers reduces over MPI_COMM_WORLD, so numbering a bunch on its own communicator hangs"
hide:
  - toc
---

# IO-10 — `ParticleIdNumber.addParticleIdNumbers` reduces over `MPI_COMM_WORLD`, so numbering a bunch on its own communicator hangs

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Particles and I/O |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">contract-unsettled</span> [?](../../methodology.md#tags) |
| Source | [particleidnumber.py#L19-L57](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/bunch_utils/particleidnumber.py#L19-L57) |

`ParticleIdNumber.addParticleIdNumbers(bunch)` takes the rank, the rank count and an all-reduce of the per-rank particle counts from `MPI_COMM_WORLD`, not from the bunch's communicator, and performs the all-reduce even when `fixedidnumber` makes it unnecessary. A bunch on a sub-communicator or on `MPI_COMM_SELF` therefore cannot be numbered unless every world rank calls the function: a call from the bunch's ranks alone hangs. When every group calls, each group's IDs start after the particle counts of the earlier world ranks, 6–11 for the second of two 6-particle bunches.

Run `python IO-10.py`.

??? example "Reproducer: IO-10.py"

    [Download IO-10.py](IO-10.py)

    ```python
    """IO-10: `ParticleIdNumber.addParticleIdNumbers` reduces over `MPI_COMM_WORLD`, so numbering a bunch on its own communicator hangs

    `ParticleIdNumber.addParticleIdNumbers(bunch)` takes the rank, the rank count and an all-reduce of the per-rank particle counts from `MPI_COMM_WORLD`, not from the bunch's communicator, and performs the all-reduce even when `fixedidnumber` makes it unnecessary. A bunch on a sub-communicator or on `MPI_COMM_SELF` therefore cannot be numbered unless every world rank calls the function: a call from the bunch's ranks alone hangs. When every group calls, each group's IDs start after the particle counts of the earlier world ranks, 6–11 for the second of two 6-particle bunches.

    Setup: needs the MPI build of PyORBIT and its launcher; set `PYORBIT_MPI_PYTHON` to that build's interpreter and `PYORBIT_MPIEXEC` to its `mpiexec`. The script launches itself under `mpiexec` with 2, 3 and 4 ranks, kills a job after 20 s and takes about a minute and a half.

    The script gives each rank three particles of a bunch whose communicator is set with `setMPIComm`, lets only the bunch's ranks call the helper while the other ranks do not, and records the identifiers and whether the job ends within 20 s: ranks 0 and 1 of three share a bunch on a split communicator, with and without `fixedidnumber`, and rank 0 of two numbers its own bunch on `MPI_COMM_SELF`. The control numbers a bunch on `MPI_COMM_WORLD` over two ranks, which must give 0–5. A last line shows the identifiers when both groups of a four-rank split number their own bunches.
    """

    import os
    import signal
    import subprocess
    import sys
    import time

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    TIMEOUT = 20.0


    def worker(layout):
        from orbit.bunch_utils import ParticleIdNumber
        from orbit.core import orbit_mpi as mpi
        from orbit.core.bunch import Bunch

        W = mpi.mpi_comm.MPI_COMM_WORLD
        rank = mpi.MPI_Comm_rank(W)
        if layout == "world":
            group, comm, numbers = 0, W, True
        elif layout == "self":  # every rank has its own bunch; only rank 0 numbers it
            group, comm, numbers = rank, mpi.mpi_comm.MPI_COMM_SELF, rank == 0
        else:
            group = rank // 2  # ranks 0, 1 -> bunch 0; ranks 2, 3 -> bunch 1
            comm = mpi.MPI_Comm_split(W, group, rank)
            numbers = group == 0 or layout == "groups"
        b = Bunch()
        b.setMPIComm(comm)
        for i in range(3):
            b.addParticle(1e-3 * i, 0.0, 0.0, 0.0, 0.0, 0.0)
        if not numbers:
            print(f"bunch {group} rank {rank} did not number", flush=True)
            return  # this rank does other work and does not number its bunch
        if layout == "subset_fixed":
            ParticleIdNumber.addParticleIdNumbers(b, fixedidnumber=7)  # needs no data from other ranks
        else:
            ParticleIdNumber.addParticleIdNumbers(b)
        ids = [int(b.partAttrValue("ParticleIdNumber", i, 0)) for i in range(b.getSize())]
        print(f"bunch {group} rank {rank} ids {' '.join(map(str, ids))}", flush=True)


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


    def bunch_ids(out):
        ids = {}
        for line in out.splitlines():
            if " ids " in line:
                head, tail = line.split(" ids ")
                ids.setdefault(int(head.split()[1]), []).extend(int(v) for v in tail.split())
        return {g: sorted(v) for g, v in ids.items()}


    def span(v):
        if not v:
            return "<none>"
        return f"{v[0]} for all {len(v)}" if len(v) > 1 and len(set(v)) == 1 else f"{v[0]}-{v[-1]}"


    def main():
        bad = False
        for n, layout, label, expected in (
            (2, "world", "2 ranks, bunch on MPI_COMM_WORLD (control)", list(range(6))),
            (3, "subset", "3 ranks, bunch on ranks 0-1 numbered, rank 2 not numbering", list(range(6))),
            (3, "subset_fixed", "3 ranks, same with fixedidnumber=7", [7] * 6),
            (
                2,
                "self",
                "2 ranks, rank 0 numbers its MPI_COMM_SELF bunch, rank 1 not numbering",
                list(range(3)),
            ),
        ):
            status, out, dt = mpirun(n, layout)
            ended = (
                f"ended in {dt:.1f} s, status {status}"
                if status is not None
                else f"still running after {TIMEOUT:.0f} s"
            )
            got = bunch_ids(out).get(0, [])
            # expected: the bunch's particles numbered 0..N-1 in rank order of its own communicator (3 per rank),
            # or every ID equal to fixedidnumber
            print(
                f"{label}: bunch 0 ids observed {span(got)}, job {ended}; expected {span(expected)} and the job ended"
            )
            ok = got == expected and status == 0
            if layout == "world" and not ok:
                raise RuntimeError("control failed")
            bad |= not ok
        status, out, dt = mpirun(4, "groups")
        ids = bunch_ids(out)
        ended = (
            f"ended in {dt:.1f} s, status {status}"
            if status is not None
            else f"still running after {TIMEOUT:.0f} s"
        )
        print(
            f"4 ranks, two communicators of 2, both groups numbering (information): bunch 0 ids {span(ids.get(0, []))}, "
            f"bunch 1 ids {span(ids.get(1, []))}, job {ended}; unique across bunches, offset by the counts of the "
            f"earlier world ranks"
        )
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
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, MPICH 5.0.2, CPU, double |
| Confirmed<br><small>the hang; the offset identifiers as a contract question</small> | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, MPICH 5.0.2, CPU, double |

