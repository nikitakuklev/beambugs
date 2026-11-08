"""API-13: MPI wrappers and `Bunch.setMPIComm` accept any Python object as a communicator

`Bunch.setMPIComm` and the `orbit_mpi` functions cast their communicator, datatype and operation arguments to PyORBIT's wrapper structures without checking the Python type. A string is accepted as a bunch communicator and returned by `getMPIComm()`, and in a serial build `MPI_Comm_rank("x")` returns 0 and `MPI_Comm_size(None)` returns 1, with no `TypeError`. In an MPI build the handle is read from the foreign object's memory and passed to MPI: on MPICH, `MPI_Comm_rank("x")`, `MPI_Comm_rank(7)` and `getSizeGlobal()` of a bunch given a string abort the job with "Invalid communicator".

The script passes a string, `None` and an integer where a communicator or operation is required and records whether each call raises `TypeError`. The control passes `MPI_COMM_WORLD`.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from orbit.core import orbit_mpi as mpi
from orbit.core.bunch import Bunch

W = mpi.mpi_comm.MPI_COMM_WORLD
INT = mpi.mpi_datatype.MPI_INT


def attempt(f):
    try:
        return f"accepted, returned {f()!r}"
    except TypeError as e:
        return f"TypeError({e})"


b = Bunch()
cases = [
    ("Bunch.setMPIComm('not a comm')", lambda: (b.setMPIComm("not a comm"), b.getMPIComm())[1]),
    ("MPI_Comm_rank('x')", lambda: mpi.MPI_Comm_rank("x")),
    ("MPI_Comm_size(None)", lambda: mpi.MPI_Comm_size(None)),
    ("MPI_Barrier(3)", lambda: mpi.MPI_Barrier(3)),
    ("MPI_Allreduce(1, MPI_INT, None, W)", lambda: mpi.MPI_Allreduce(1, INT, None, W)),
]
reproduced = False
for label, f in cases:
    obs = attempt(f)
    print(
        f"{label}: observed {obs}; expected TypeError (the argument is not an orbit_mpi wrapper object)"
    )
    reproduced |= not obs.startswith("TypeError")
print(f"control MPI_Comm_rank(MPI_COMM_WORLD): observed {mpi.MPI_Comm_rank(W)}, expected 0")

print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")
