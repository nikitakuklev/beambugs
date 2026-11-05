"""OPT-28: srdt per-element terms have wrong factors and unset fields

Requesting per_ele_out from srdt_calc or srdt_calc_with_cache returns contributions that do not add up to the reported totals. The direct routine reverses three first-order signs and quarters the chromatic coefficient h₁₀₀₀₂, while the cached routine omits normalization factors altogether. In an eight-cell ring, the normal sextupole driving coefficient h₂₁₀₀₀ is −0.3827 − 0.3095i, but the direct contributions sum to its negative and cached contributions to −8 times the total. Source inspection finds that chromatic per-element fields are not assigned and second-order resonance fields are formed from unassigned slice fields.

The script assigns the repeating sextupole strengths (3, −4.5, 1.5) in an eight-cell ring. It sums the eight assigned first-order per-element complex fields from direct and cached calculations and compares them with the reported totals, requiring the correctly normalized direct fields to pass. BMAD_DIST selects the production build.
"""

import numpy as np
from _bmad import compare, fortran, verdict
from _srdt_wiggler_maps import LSQ_PROGRAM, THICK_RING

strengths = " ".join(map(str, [3, -4.5, 1.5] * 8)) + "\n"
log = fortran(
    LSQ_PROGRAM,
    ["lat.bmad", "perele", "strengths.txt"],
    files={"lat.bmad": THICK_RING, "strengths.txt": strengths},
).log
names = ["h20001", "h00201", "h10002", "h21000", "h30000", "h10110", "h10020", "h10200"]
rows = {
    fields[0]: np.array(list(map(float, fields[1:]))).reshape(3, 2)
    for line in log.splitlines()
    if (fields := line.split()) and fields[0] in names
}
if any(name not in rows for name in names):
    raise RuntimeError("Missing first-order terms")
defect = False
for name in names:
    total, direct, cached = rows[name]
    bad = compare(f"{name}: summed direct contributions", direct, total, atol=2e-12)
    if name in ("h20001", "h10110", "h10020", "h10200") and bad:
        raise AssertionError("Correctly normalized direct control failed")
    defect |= bad
    defect |= compare(f"{name}: summed cached contributions", cached, total, atol=2e-12)
verdict(defect)
