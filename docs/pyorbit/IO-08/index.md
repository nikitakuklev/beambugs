---
title: "IO-08: collect_bunch fails on the root when another MPI rank holds no particles"
hide:
  - toc
---

# IO-08 — `collect_bunch` fails on the root when another MPI rank holds no particles

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Particles and I/O |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [serialize.py#L174-L190](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/bunch_utils/serialize.py#L174-L190)<br>[serialize.py#L221-L235](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/bunch_utils/serialize.py#L221-L235)<br>[serialize.py#L272-L274](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/bunch_utils/serialize.py#L272-L274) |

`collect_bunch(bunch, output_dir)` has every rank write its local coordinates to a NumPy
memory-map file, then the root reads the other ranks' files as float64. A rank with zero local
particles creates its file through a zero-row memory map, which NumPy stores as a one-byte file; the
root's reader rejects it with "Size of available data is not a multiple of the data-type size".
Collecting or saving a bunch therefore raises `ValueError` whenever any non-root rank is empty, for
example a few particles spread over many ranks or after losses, although the expected result is the
global coordinate array. An empty root rank is handled correctly.

Run `python IO-08.py`.

??? example "Reproducer: IO-08.py"

    [Download IO-08.py](IO-08.py)

    ```python
    """IO-08: `collect_bunch` fails on the root when another MPI rank holds no particles

    `collect_bunch(bunch, output_dir)` has every rank write its local coordinates to a NumPy memory-map
    file, then the root reads the other ranks' files as float64. A rank with zero local particles
    creates its file through a zero-row memory map, which NumPy stores as a one-byte file; the root's
    reader rejects it with "Size of available data is not a multiple of the data-type size". Collecting
    or saving a bunch therefore raises `ValueError` whenever any non-root rank is empty, for example a
    few particles spread over many ranks or after losses, although the expected result is the global
    coordinate array. An empty root rank is handled correctly.

    The script first shows the NumPy behavior directly: the size of the file a zero-row write-mode
    memory map leaves and the error raised when it is read as float64. It then replays the two-rank file
    protocol of `collect_bunch` in one process, calling it as rank 1 and then as rank 0 with the
    module's `orbit_mpi` rank, size and barrier calls replaced by a stub and each rank's bunch reporting
    the two-rank total as its global size. Controls give rank 1 one particle, and leave the root empty
    while rank 1 holds two.
    """

    import os
    import tempfile
    import types

    import numpy as np
    from orbit.bunch_utils import serialize
    from orbit.core.bunch import Bunch

    f = os.path.join(tempfile.mkdtemp(), "zero_rows.dat")
    np.memmap(f, dtype=np.float64, mode="w+", shape=(0, 6)).flush()
    try:
        np.memmap(f, dtype=np.float64, mode="r")
        read = "read without error"
    except ValueError as e:
        read = f"ValueError: {e}"
    print(
        f"NumPy {np.__version__} zero-row memory map: file size {os.path.getsize(f)} byte(s); read as float64: {read}"
    )


    class RankBunch:
        """A serial bunch seen as one rank of a two-rank bunch: getSizeGlobal() is the two-rank total."""

        def __init__(self, xs, global_size):
            self._b = Bunch()
            for x in xs:
                self._b.addParticle(x, 0.0, 0.0, 0.0, 0.0, 0.0)
            self._n = global_size

        def getSizeGlobal(self):
            return self._n

        def __getattr__(self, name):
            return getattr(self._b, name)


    def collect_two_ranks(xs0, xs1):
        """Rank 1 writes its file and returns None; then the root (rank 0) gathers both files."""
        d = tempfile.mkdtemp()
        n = len(xs0) + len(xs1)
        rank = {"value": 1}
        stub = types.SimpleNamespace(
            MPI_Comm_rank=lambda comm: rank["value"],
            MPI_Comm_size=lambda comm: 2,
            MPI_Barrier=lambda comm: None,
        )
        real = serialize.orbit_mpi
        serialize.orbit_mpi = stub
        try:
            if serialize.collect_bunch(RankBunch(xs1, n), d) is not None:  # type: ignore[arg-type]
                raise RuntimeError("rank 1 returned a result")
            rank["value"] = 0
            try:
                r = serialize.collect_bunch(RankBunch(xs0, n), d, return_memmap=False)  # type: ignore[arg-type]
            except ValueError as e:
                return f"ValueError: {e}"
            if r is None:
                raise RuntimeError("root returned no result")
            return np.asarray(r["coords"])[:, 0].tolist()
        finally:
            serialize.orbit_mpi = real


    # Expected: the root returns rank 0's rows followed by rank 1's rows (collect_bunch fills rows
    # 0..n0-1 from its own bunch and appends each other rank's file in rank order).
    obs = collect_two_ranks([1.0, 2.0, 3.0], [])
    print(f"rank 0 holds [1, 2, 3], rank 1 empty: observed {obs!r}; expected [1.0, 2.0, 3.0]")
    obs_c1 = collect_two_ranks([1.0, 2.0, 3.0], [4.0])
    print(
        f"rank 0 holds [1, 2, 3], rank 1 holds [4] (control): observed {obs_c1!r}; expected [1.0, 2.0, 3.0, 4.0]"
    )
    obs_c2 = collect_two_ranks([], [4.0, 5.0])
    print(f"rank 0 empty, rank 1 holds [4, 5] (control): observed {obs_c2!r}; expected [4.0, 5.0]")

    ok = isinstance(obs, str) and obs_c1 == [1.0, 2.0, 3.0, 4.0] and obs_c2 == [4.0, 5.0]
    print("DEFECT REPRODUCED" if ok else "NOT REPRODUCED")

    ```

??? example "Reproducer: IO-08-mpi.py"

    [Download IO-08-mpi.py](IO-08-mpi.py)

    ```python
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

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Found<br><small class="bb-scope">two-rank MPI run</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, MPICH 5.0.2, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

