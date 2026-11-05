"""COL-53: BBU ramping overwrites the circular bunch-queue index

Enabling the bunch-charge ramp in the beam-breakup simulation overwrites the circular queue slot with the chronological bunch number. The first insertion records slot zero even though the bunch occupies slot one. The next insertion reports a bunch-array overflow and aborts; the same lattice completes with ramping disabled.

The script runs the BBU executable with a fixed random seed on the same lattice with ramping disabled and enabled. It checks completion of the control and identifies the queue-overflow diagnostic and abort in the ramped case. BMAD_DIST selects the build.
"""

from _bmad import RunFailure, run, verdict
from _wake_tables_and_bbu import BBU_FILES, bbu_control

bbu_control()
try:
    result = run("bbu", ["ramp.init"], files=BBU_FILES, outputs=["for_py.txt"])
except RunFailure as exc:
    if exc.status != -4 or "BBU_BEAM%BUNCH ARRAY OVERFLOW" not in exc.log:
        raise
    print(f"ramping: observed=queue overflow, SIGILL ({exc.status}); expected=normal completion")
    defect = True
else:
    if "v_gain" not in result.outputs["for_py.txt"].decode():
        raise RuntimeError(result.log)
    print("ramping: observed=normal completion; expected=normal completion")
    defect = False
verdict(defect)
