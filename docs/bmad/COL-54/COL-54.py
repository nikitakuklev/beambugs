"""COL-54: BBU HOM-order filtering breaks mode-array allocation

Filtering higher-order modes in the beam-breakup simulation breaks the mode array when any modes are removed. Partial removal attempts to allocate an already allocated array, while complete removal leaves an unallocated array for downstream consumers. A cavity containing orders one and two fails with an allocation error at cutoff one, and a two-cavity case with one fully discarded mode crashes; retaining both orders completes.

The script checks matching one-cavity and two-cavity controls that retain all modes, then runs partial-removal and complete-removal cases in separate processes. It requires the specific allocation failure after successful parsing or segmentation fault after tracking begins. BMAD_DIST selects the build and all runs use seed 123.
"""

from _bmad import RunFailure, run, verdict
from _wake_tables_and_bbu import BBU_FILES, bbu_control

bbu_control()
bbu_control("control_all.init", "BBU two-cavity control HOM voltage gain")
defect = False
for name, status, diagnostic in [
    ("cutoff.init", 2, "Attempting to allocate already allocated variable"),
    ("cutoff_all.init", -11, "SIGSEGV"),
]:
    try:
        result = run("bbu", [name], files=BBU_FILES, outputs=["for_py.txt"])
    except RunFailure as exc:
        if (
            exc.status != status
            or diagnostic not in exc.log
            or "Lattice parse time" not in exc.log
            or (name == "cutoff_all.init" and "bbu_track_all loop begins" not in exc.log)
        ):
            raise
        print(f"{name}: observed=status {exc.status}, {diagnostic}; expected=normal completion")
        defect = True
    else:
        if "v_gain" not in result.outputs["for_py.txt"].decode():
            raise RuntimeError(result.log)
        print(f"{name}: observed=normal completion; expected=normal completion")
verdict(defect)
