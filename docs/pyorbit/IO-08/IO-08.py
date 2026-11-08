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
