"""API-02: String `MPI_Bcast` and `MPI_Send` send as many UTF-8 bytes as the string has characters

For `MPI_CHAR` data, `orbit_mpi.MPI_Bcast` and `MPI_Send` take the message length from the Python character count but transmit the string's UTF-8 bytes. A string with non-ASCII characters is longer in bytes than in characters, so its tail is cut: broadcasting `"héllo"` returns `"héll"` on every rank, including the root, and broadcasting `"é"`, `"αβγ"` or `"中文"` raises `UnicodeDecodeError`, because the cut falls inside a character. ASCII strings are unaffected.

The script broadcasts non-ASCII strings from rank 0 to itself, where the result must equal the root's string, and prints the character and byte counts beside each result. The ASCII string `"hello"` is the control.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from orbit.core import orbit_mpi as mpi

W = mpi.mpi_comm.MPI_COMM_WORLD
CHAR = mpi.mpi_datatype.MPI_CHAR

reproduced = False
for s in ("héllo", "é", "αβγ", "中文", "hello"):
    try:
        obs = repr(mpi.MPI_Bcast(s, CHAR, 0, W))
    except UnicodeDecodeError as e:
        obs = f"UnicodeDecodeError({e.reason})"
    # A broadcast from the root returns the root's own string. The wrapper sends len(s) bytes of
    # s.encode(); that is the whole string only when len(s) == len(s.encode()), i.e. for ASCII.
    control = s.isascii()
    label = ("control " if control else "") + f"Bcast({s!r}, MPI_CHAR)"
    print(
        f"{label}: observed {obs}, expected {s!r} (characters {len(s)}, UTF-8 bytes {len(s.encode())})"
    )
    reproduced |= not control and obs != repr(s)

print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")
