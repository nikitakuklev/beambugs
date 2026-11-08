"""LAT-06: `RF_AxisFieldsStore.addAxisFieldsForAccSeq` raises `NameError` on every call

`RF_AxisFieldsStore.addAxisFieldsForAccSeq(accLattice, accSeqNamesList, dir_location)` loops over the undefined name `accSeqList` instead of its argument, so every call raises `NameError` before it reads any file, even for an empty list of sequence names. Its documented purpose, loading the axis fields of all RF gaps in the named sequences into the field store, is unavailable.

The script builds a linac lattice with one named sequence and no cavities and calls the method with that sequence name and with an empty list. The expected result is a normal return with nothing loaded. The control looks the sequence up by name, the step the method's loop performs first.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from orbit.py_linac.lattice import LinacAccLattice, RF_AxisFieldsStore, Sequence

lat = LinacAccLattice("lat")
lat.addSequence(Sequence("SEQ"))

seq = lat.getSequence("SEQ")
print(
    f"control, lat.getSequence('SEQ'): {seq.getName()} with {len(seq.getRF_Cavities())} cavities, expected SEQ with 0"
)
if seq.getName() != "SEQ":
    raise RuntimeError("control case failed: the sequence cannot be found by name")

reproduced = False
for names in (["SEQ"], []):
    try:
        RF_AxisFieldsStore.addAxisFieldsForAccSeq(lat, names, "")
        outcome = "returned"
    except NameError as err:
        outcome = f"NameError: {err}"
        reproduced = True
    print(
        f"addAxisFieldsForAccSeq(lat, {names}): {outcome}; expected a normal return (no cavities to load)"
    )
print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")
