"""TRK-42: Two compiled constants disagree with the claimed CODATA 2022 update

The physical-constants module claims CODATA 2022 values but supplies different central values for the classical electron radius and electron magnetic anomaly. The compiled radius exceeds the CODATA value by 0.7807 parts per billion, and the anomaly by 0.1121 parts per billion. Direct extraction of the compiled constants and comparison with the NIST CODATA table establish the mismatch.

The script compiles a small constants reader against BMAD_DIST and checks the CODATA-update
comment in BMAD_SOURCE. It compares both values with the fixed CODATA 2022 references at
near-roundoff tolerance.
"""

from _bmad import compare, source, verdict
from _tracking_contracts_materials import constants

text = (source() / "sim_utils/interfaces/physical_constants.f90").read_text()
if "Updated to 2022 CODATA" not in text:
    raise RuntimeError("The constants provenance comment changed; inspect the stated set.")
values = constants()["constants"]
# CODATA 2022 central values: https://physics.nist.gov/cuu/Constants/Table/allascii.txt
defects = [
    compare("electron radius (m), CODATA 2022", values[2], 2.8179403205e-15, atol=0, rtol=2e-16),
    compare("electron anomaly, CODATA 2022", values[3], 1.15965218046e-3, atol=0, rtol=2e-16),
]
verdict(any(defects))
