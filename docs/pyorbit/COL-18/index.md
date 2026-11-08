---
title: "COL-18: SpaceChargeCalcSliceBySlice2D leaves slices without space charge when the bunch communicator lacks some MPI_COMM_WORLD ranks"
hide:
  - toc
---

# COL-18 — `SpaceChargeCalcSliceBySlice2D` leaves slices without space charge when the bunch communicator lacks some `MPI_COMM_WORLD` ranks

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [SpaceChargeCalcSliceBySlice2D.cc#L86-L108](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalcSliceBySlice2D.cc#L86-L108)<br>[#L261](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/SpaceChargeCalcSliceBySlice2D.cc#L261-L261) |

`SpaceChargeCalcSliceBySlice2D.trackBunch` solves the 2-D potential of slice iz only on the rank whose `MPI_COMM_WORLD` rank equals iz modulo the world size, zeroes it on the other ranks, and then sums the potentials over the bunch's own communicator. For a bunch whose communicator lacks some world ranks, set with `setMPIComm`, the slices owned by the missing ranks keep zero potential and the transverse kicks are silently wrong. With two ranks each tracking its own uniform-disk bunch on `MPI_COMM_SELF`, every second of 16 slices is unsolved and the kicks differ from the one-rank kicks by up to 0.7886 of the peak kick. Bunches on `MPI_COMM_WORLD` or a duplicate of it match the one-rank run to 3×10⁻¹⁴.

Run `python COL-18.py`.

??? example "Reproducer: COL-18.py"

    [Download COL-18.py](COL-18.py)

    ```python
    """COL-18: `SpaceChargeCalcSliceBySlice2D` leaves slices without space charge when the bunch communicator lacks some `MPI_COMM_WORLD` ranks

    `SpaceChargeCalcSliceBySlice2D.trackBunch` solves the 2-D potential of slice iz only on the rank whose `MPI_COMM_WORLD` rank equals iz modulo the world size, zeroes it on the other ranks, and then sums the potentials over the bunch's own communicator. For a bunch whose communicator lacks some world ranks, set with `setMPIComm`, the slices owned by the missing ranks keep zero potential and the transverse kicks are silently wrong. With two ranks each tracking its own uniform-disk bunch on `MPI_COMM_SELF`, every second of 16 slices is unsolved and the kicks differ from the one-rank kicks by up to 0.7886 of the peak kick. Bunches on `MPI_COMM_WORLD` or a duplicate of it match the one-rank run to 3×10⁻¹⁴.

    Setup: needs the MPI build of PyORBIT and its launcher; set `PYORBIT_MPI_PYTHON` to that build's interpreter and `PYORBIT_MPIEXEC` to its `mpiexec`. The script launches itself under `mpiexec` with 1, 2 and 4 ranks.

    The script builds a uniform round beam of radius 5 mm, uniform over 40 m in z so that every slice carries the same transverse field, and tracks it once through `SpaceChargeCalcSliceBySlice2D` on a 64 × 64 × 16 grid. It compares every particle's horizontal kick away from the z-grid ends with the one-rank run, and marks each z plane whose on-axis potential is at least half the one-rank value as solved, for two ranks each with the full bunch on `MPI_COMM_SELF` and for four ranks split into two communicators of two ranks. The controls distribute the bunch over two ranks on `MPI_COMM_WORLD` and on a duplicate of it, which must reproduce the one-rank kicks.
    """

    import json
    import math
    import os
    import signal
    import subprocess
    import sys
    import tempfile

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    N, A, LZ = 12000, 5e-3, 40.0  # particles, beam radius (m), bunch length (m)
    NX, NY, NZ = 64, 64, 16  # grid


    def particles():
        # deterministic uniform disc of radius A (square-root quantile radius, golden-angle azimuth), z uniform on
        # [-LZ/2, LZ/2] from the base-3 radical inverse of i, which is uncorrelated with the azimuth
        out = []
        for i in range(N):
            r = A * math.sqrt((i + 0.5) / N)
            th = i * math.pi * (3.0 - math.sqrt(5.0))
            u, f, m = 0.0, 1.0 / 3.0, i
            while m:
                u += f * (m % 3)
                m //= 3
                f /= 3.0
            out.append((r * math.cos(th), r * math.sin(th), -LZ / 2 + LZ * u))
        return out


    def worker(layout, outdir):
        from orbit.core import orbit_mpi as mpi
        from orbit.core.bunch import Bunch
        from orbit.core.spacecharge import SpaceChargeCalcSliceBySlice2D

        W = mpi.mpi_comm.MPI_COMM_WORLD
        rank, size = mpi.MPI_Comm_rank(W), mpi.MPI_Comm_size(W)
        comm, group, sub = W, 0, (rank, size)
        if layout == "dup":  # same ranks as the world, another handle
            comm = mpi.MPI_Comm_dup(W)
        elif layout == "self":  # every rank tracks its own full bunch
            comm, group, sub = mpi.mpi_comm.MPI_COMM_SELF, rank, (0, 1)
        elif layout == "halves":  # ranks 0..size/2-1 and size/2..size-1 each share one bunch
            group = rank // (size // 2)
            comm = mpi.MPI_Comm_split(W, group, rank)
            sub = (mpi.MPI_Comm_rank(comm), mpi.MPI_Comm_size(comm))
        elif layout != "world":
            raise ValueError(layout)
        mine = [
            i for i in range(N) if i % sub[1] == sub[0]
        ]  # the bunch dealt round-robin over its communicator
        b = Bunch()
        b.mass(0.93827208816)
        b.macroSize(1e8)
        b.getSyncParticle().kinEnergy(1.0)
        b.setMPIComm(comm)
        P = particles()
        for i in mine:
            b.addParticle(P[i][0], 0.0, P[i][1], 0.0, P[i][2], 0.0)
        calc = SpaceChargeCalcSliceBySlice2D(NX, NY, NZ)
        calc.trackBunch(b, 1.0)
        phi = calc.getPhiGrid()  # potential after the solver's MPI sum
        axis = [phi.getValueOnGrid(NX // 2, NY // 2 + 1, iz) for iz in range(NZ)]
        with open(os.path.join(outdir, f"rank{rank}.json"), "w") as f:
            json.dump(
                {"group": group, "axis": axis, "kicks": [[i, b.xp(k)] for k, i in enumerate(mine)]}, f
            )


    def mpirun(n, *args, timeout=180):
        py, mx = os.environ.get("PYORBIT_MPI_PYTHON"), os.environ.get("PYORBIT_MPIEXEC")
        if not py or not mx:
            sys.exit("set PYORBIT_MPI_PYTHON and PYORBIT_MPIEXEC to the MPI build's python and mpiexec")
        p = subprocess.Popen(
            [mx, "-n", str(n), py, "-I", os.path.abspath(__file__), "--worker", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            _, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            raise RuntimeError(f"mpiexec -n {n} {args} did not finish in {timeout} s")
        if p.returncode != 0:
            raise RuntimeError(f"mpiexec -n {n} {args} failed with {p.returncode}: {err[-400:]}")


    def run(n, layout):
        """Returns {group: ({particle: kick}, [on-axis potential of each z plane])}."""
        with tempfile.TemporaryDirectory() as d:
            mpirun(n, layout, d)
            groups = {}
            for r in range(n):
                with open(os.path.join(d, f"rank{r}.json")) as f:
                    res = json.load(f)
                kicks, _ = groups.setdefault(res["group"], ({}, res["axis"]))
                kicks.update({i: k for i, k in res["kicks"]})
        return groups


    def main():
        P = particles()
        dz = LZ / (NZ - 1)
        interior = [i for i in range(N) if abs(P[i][2]) < LZ / 2 - 2 * dz]  # away from the z-grid ends
        ref, ref_axis = run(1, "world")[0]
        peak = max(abs(ref[i]) for i in interior)
        bad = False
        for n, layout, label in (
            (2, "world", "2 ranks, MPI_COMM_WORLD (control)"),
            (2, "dup", "2 ranks, MPI_Comm_dup(MPI_COMM_WORLD) (control)"),
            (2, "self", "2 ranks, each on MPI_COMM_SELF"),
            (4, "halves", "4 ranks, two communicators of 2"),
        ):
            for g, (kicks, axis) in sorted(run(n, layout).items()):
                dev = max(abs(kicks[i] - ref[i]) for i in interior) / peak
                # a plane is solved when its on-axis potential is at least half the one-rank value; zeroed planes give 0
                planes = "".join("1" if abs(a) > 0.5 * abs(b) else "0" for a, b in zip(axis, ref_axis))
                print(
                    f"{label}, bunch {g}: solved z planes [{planes}], expected all 1; "
                    f"max |kick - one-rank kick|/peak {dev:.4g}, expected 0 (same bunch, same solver)"
                )
                if layout in ("world", "dup") and dev > 1e-12:
                    raise RuntimeError("control failed: a world-communicator run differs from one rank")
                bad |= layout not in ("world", "dup") and dev > 1e-12
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

