"""API-03: `MPI_INT` all-reduce and broadcast wrap integers outside the 32-bit range

`orbit_mpi.MPI_Allreduce` and `MPI_Bcast` with `MPI_INT` read each Python integer as a C `long` and cast it to a 32-bit `int` without a range check. Values outside −2³¹ … 2³¹ − 1 wrap silently: an all-reduce of 2³¹ returns −2147483648, one of `[2**32 + 5]` returns `(5,)`, and a broadcast of 2³¹ returns −2147483648. Global counts above 2³¹ − 1, such as particle totals summed over ranks, are corrupted without an error, where an `OverflowError` is expected.

The script reduces and broadcasts integers just outside the 32-bit range on one rank, where SUM and a broadcast must return the input or raise `OverflowError`, and prints the two's-complement wrap beside each result. The control reduces 2³¹ − 1, the largest C `int`.
"""

import ctypes
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from orbit.core import orbit_mpi as mpi

W = mpi.mpi_comm.MPI_COMM_WORLD
INT = mpi.mpi_datatype.MPI_INT
SUM = mpi.mpi_op.MPI_SUM
assert mpi.MPI_Comm_size(W) == 1  # one rank: SUM and broadcast return the input


def call(f):
    try:
        return f()
    except OverflowError:
        return "OverflowError"


cases = [
    ("Allreduce(2**31, SUM)", lambda: mpi.MPI_Allreduce(2**31, INT, SUM, W), 2**31),
    (
        "Allreduce(-2**31 - 1, SUM)",
        lambda: mpi.MPI_Allreduce(-(2**31) - 1, INT, SUM, W),
        -(2**31) - 1,
    ),
    (
        "Allreduce([2**32 + 5], SUM)",
        lambda: mpi.MPI_Allreduce([2**32 + 5], INT, SUM, W),
        (2**32 + 5,),
    ),
    ("Bcast(2**31)", lambda: mpi.MPI_Bcast(2**31, INT, 0, W), 2**31),
]
reproduced = False
for label, f, exp in cases:
    obs = call(f)
    v = exp[0] if isinstance(exp, tuple) else exp
    # (int) of a long keeps the low 32 bits read as signed, which ctypes.c_int reproduces.
    print(
        f"{label}: observed {obs}, expected {exp} or OverflowError (32-bit wrap {ctypes.c_int(v).value})"
    )
    reproduced |= obs not in (exp, "OverflowError")
obs = call(lambda: mpi.MPI_Allreduce(2**31 - 1, INT, SUM, W))
print(f"control Allreduce(2**31 - 1, SUM): observed {obs}, expected {2**31 - 1}")

print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")
