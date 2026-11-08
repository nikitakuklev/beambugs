---
title: "COL-17: TImpedance kicks each MPI rank's particles with the wake of that rank's particles only"
hide:
  - toc
---

# COL-17 — `TImpedance` kicks each MPI rank's particles with the wake of that rank's particles only

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [TImpedance.cc#L270-L279](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Impedances/TImpedance.cc#L270-L279)<br>[Grid1D.cc#L404-L418](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/Grid1D.cc#L404-L418)<br>[Grid1D.cc#L762-L798](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/spacecharge/Grid1D.cc#L762-L798)<br>[TImpedance.cc#L285-L335](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Impedances/TImpedance.cc#L285-L335)<br>[#L253-L257](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Impedances/TImpedance.cc#L253-L257)<br>[test_fd_TImpedance.py#L58-L68](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/tests/examples/Impedances/test_fd_TImpedance.py#L58-L68) |

`TImpedance.trackBunch` bins the transverse dipole moments of the bunch (x, x′ and y, y′) along z into arrays that stay local to each MPI rank, and sums over the bunch's communicator only the particle counts, which the kick does not use. Each rank therefore kicks its particles with the wake of its own share of the charge, while the impedance acts on the whole bunch. A rigidly offset bunch spread evenly over 2, 3 and 4 ranks receives 0.5000, 0.3333 and 0.2500 times the kick of the same bunch on one rank, in x and in y. A bunch held entirely on one rank, with the other ranks empty, gets the correct kick.

Run `python COL-17.py`.

??? example "Reproducer: COL-17.py"

    [Download COL-17.py](COL-17.py)

    ```python
    """COL-17: `TImpedance` kicks each MPI rank's particles with the wake of that rank's particles only

    `TImpedance.trackBunch` bins the transverse dipole moments of the bunch (x, x′ and y, y′) along z into arrays that stay local to each MPI rank, and sums over the bunch's communicator only the particle counts, which the kick does not use. Each rank therefore kicks its particles with the wake of its own share of the charge, while the impedance acts on the whole bunch. A rigidly offset bunch spread evenly over 2, 3 and 4 ranks receives 0.5000, 0.3333 and 0.2500 times the kick of the same bunch on one rank, in x and in y. A bunch held entirely on one rank, with the other ranks empty, gets the correct kick.

    Setup: needs the MPI build of PyORBIT and its launcher; set `PYORBIT_MPI_PYTHON` to that build's interpreter and `PYORBIT_MPIEXEC` to its `mpiexec`. The script launches itself under `mpiexec` with 1 to 4 ranks.

    The script builds a mirror-symmetric Gaussian bunch with every particle at x = y = 1 mm and x′ = y′ = 0, repeats it twelve times and tracks it once through `TImpedance` with one resistive sideband in each plane. It deals the copies so that every rank holds an identical subset, so a correct reduction must reproduce the one-rank kick of every particle exactly, and prints the least-squares ratio of each run's kicks to the one-rank kicks and the largest deviation. The control runs two ranks with all particles on rank 0 and rank 1 empty, which must also reproduce the one-rank kicks.
    """

    import json
    import os
    import signal
    import subprocess
    import sys
    import tempfile

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    COPIES = 12  # each particle appears COPIES times, so the splits over 2, 3 and 4 ranks are exact


    def worker(layout, outdir):
        import numpy as np
        from orbit.core import orbit_mpi as mpi
        from orbit.core.bunch import Bunch
        from orbit.core.impedances import TImpedance

        W = mpi.mpi_comm.MPI_COMM_WORLD
        rank, size = mpi.MPI_Comm_rank(W), mpi.MPI_Comm_size(W)
        rng = np.random.default_rng(7)
        half = 10.0 * np.abs(rng.standard_normal(500))
        zs = np.concatenate([half, -half])  # rms length 10 m, symmetric about z = 0
        nz = len(zs)
        # global particle index g = copy * nz + j holds z = zs[j]
        if layout == "even":  # copy c lives on rank c % size: identical subsets on all ranks
            mine = [c * nz + j for c in range(COPIES) if c % size == rank for j in range(nz)]
        elif layout == "rank0":  # control: everything on rank 0, the other ranks empty
            mine = list(range(COPIES * nz)) if rank == 0 else []
        else:
            raise ValueError(layout)
        b = Bunch()
        b.mass(0.93827208816)
        b.macroSize(1e10)
        b.getSyncParticle().kinEnergy(1.0)
        for g in mine:
            b.addParticle(1e-3, 0.0, 1e-3, 0.0, float(zs[g % nz]), 0.0)
        C, nb = 248.0, 64  # ring length (m), bins
        T = TImpedance(C, 1, nb, 1, 1)
        T.assignLatFuncs(6.23, 0.0, 10.0, 5.81, 0.0, 12.0)
        Zp = [0j] * (nb // 2)
        Zm = [0j] * (nb // 2)
        Zp[0] = 1e6 + 0j  # one resistive sideband line
        Zm[0] = -np.conj(Zp[0])  # transverse symmetry Z(-f) = -conj(Z(f))
        T.assignImpedance("X", Zp, Zm)
        T.assignImpedance("Y", Zp, Zm)
        T.trackBunch(b)
        with open(os.path.join(outdir, f"rank{rank}.json"), "w") as f:
            json.dump([[g, b.xp(k), b.yp(k)] for k, g in enumerate(mine)], f)


    def mpirun(n, *args, timeout=120):
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
        """Returns ([x' kick], [y' kick]) of every particle in global order."""
        with tempfile.TemporaryDirectory() as d:
            mpirun(n, layout, d)
            kicks = {}
            for r in range(n):
                with open(os.path.join(d, f"rank{r}.json")) as f:
                    kicks.update({g: (kx, ky) for g, kx, ky in json.load(f)})
        order = sorted(kicks)
        return [kicks[g][0] for g in order], [kicks[g][1] for g in order]


    def compare(k, ref):
        # least-squares ratio of this run's kicks to the one-rank kicks, and the worst deviation over the peak
        ratio = sum(a * b for a, b in zip(k, ref)) / sum(b * b for b in ref)
        dev = max(abs(a - b) for a, b in zip(k, ref)) / max(abs(b) for b in ref)
        return ratio, dev


    def main():
        ref = run(1, "even")
        bad = False
        for n, layout in ((2, "even"), (3, "even"), (4, "even"), (2, "rank0")):
            k = run(n, layout)
            label = f"{n} ranks, {'identical subsets' if layout == 'even' else 'all on rank 0, rank 1 empty (control)'}"
            for plane, kp, rp in (("x'", k[0], ref[0]), ("y'", k[1], ref[1])):
                ratio, dev = compare(kp, rp)
                print(
                    f"{label}, {plane} kick: kick / one-rank kick observed {ratio:.4f}, max |diff|/peak {dev:.3e}; "
                    f"expected 1 and 0 (the impedance acts on the whole bunch, however it is distributed)"
                )
                if layout == "rank0" and dev > 1e-12:
                    raise RuntimeError(
                        "control failed: an undistributed bunch does not reproduce the one-rank kick"
                    )
                bad |= layout == "even" and dev > 1e-12
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

