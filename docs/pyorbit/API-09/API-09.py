"""API-09: Communicators returned by `MPI_Comm_create`, `MPI_Comm_dup` and `MPI_Comm_split` are never released

The helper that creates the communicator objects returned by `orbit_mpi.MPI_Comm_create`, `MPI_Comm_dup` and `MPI_Comm_split` takes one reference more than it hands out, so these objects are never deallocated: `sys.getrefcount` reports 3 for a communicator held in one variable, where 2 is expected, and each call keeps 24 bytes. Their destructor, which frees the MPI communicator, therefore never runs; only an explicit `comm.free()` releases it. On an MPICH build, a loop that drops `MPI_Comm_dup` results without `free()` aborts the job after 2046 duplicates with "Too many communicators". Each `Bunch` also leaks its 24-byte communicator wrapper; groups from `MPI_Comm_group` are released correctly.

The script prints the reference counts of the three communicators and of a group (control), each held in one variable. It then counts the bytes that 20000 discarded results of `MPI_Comm_dup`, `MPI_Comm_group` (control) and `Bunch()` leave allocated, with `tracemalloc`.
"""

import gc
import os
import sys
import tracemalloc

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from orbit.core import orbit_mpi as mpi
from orbit.core.bunch import Bunch

W = mpi.mpi_comm.MPI_COMM_WORLD
G = mpi.MPI_Comm_group(W)

# sys.getrefcount counts the variable plus its own argument: 2 for an object held by one name.
dup = mpi.MPI_Comm_dup(W)
split = mpi.MPI_Comm_split(W, 0, 0)
create = mpi.MPI_Comm_create(W, G)
group = mpi.MPI_Comm_group(W)
# Direct calls, so that no container or loop variable adds a reference.
counts = {
    "MPI_Comm_dup": sys.getrefcount(dup),
    "MPI_Comm_split": sys.getrefcount(split),
    "MPI_Comm_create": sys.getrefcount(create),
}
n_group = sys.getrefcount(group)
for label, n in counts.items():
    print(f"getrefcount({label} result): observed {n}, expected 2 (one variable + call argument)")
print(f"control getrefcount(MPI_Comm_group result): observed {n_group}, expected 2")
reproduced = any(n != 2 for n in counts.values())


def retained(make, n=20000):
    """Bytes per call still allocated after n results of make() are discarded."""
    for _ in range(100):
        make()
    gc.collect()
    tracemalloc.start()
    a = tracemalloc.get_traced_memory()[0]
    for _ in range(n):
        make()
    gc.collect()
    b = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    return (b - a) / n


# A discarded object is freed, so nothing should remain per call (allocator noise is < 1 byte).
for label, make in (("MPI_Comm_dup(W)", lambda: mpi.MPI_Comm_dup(W)), ("Bunch()", Bunch)):
    print(f"bytes retained per discarded {label}: observed {retained(make):.2f}, expected 0")
print(
    f"control bytes retained per discarded MPI_Comm_group(W): observed {retained(lambda: mpi.MPI_Comm_group(W)):.2f}, expected 0"
)

print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")
