"""TAO-05: Single-element absolute constraints discard the reference value

A single-element abs_max constraint, or an abs_min constraint with identical start and end
elements, ignores its reference element when taking the absolute value. In a one-metre drift
with a 3 mm entrance offset, both return 4.000 mm instead of the referenced displacement of
1.000 mm. Tao’s documented order is to subtract the reference value before taking the absolute
value. Ordinary target constraints and a non-wrapped range retain the reference subtraction.

The script compares two single-element absolute constraints with the analytic drift displacement
and a target control. It then expands the absolute constraints to a genuine range and checks
their referenced extrema. BMAD_DIST supplies Tao.
"""

import math

from _bmad import compare, verdict
from _tao_optimization_io import ABS_INIT, LINE, control, data_values, tao

command = "pipe data_parameter dat.x model_value"
result = tao([command], lattice=LINE, init=ABS_INIT)
values = data_values(result.log, command)
# Bmad px = p_x/p0; with delta=0 and py=0, dx = L*px/sqrt(1-px^2).
dx = 0.001 / math.sqrt(1 - 0.001**2)
control("target displacement (m), exact drift", values[1], dx, atol=1e-14)
defect = compare("single-element abs_max and abs_min (m)", values[[0, 2]], [dx, dx], atol=1e-14)
result = tao(
    [
        "set data dat.x[1]|ele_start_name = beginning",
        "set data dat.x[3]|ele_start_name = beginning",
        command,
    ],
    lattice=LINE,
    init=ABS_INIT,
)
control(
    "range abs_max, target, abs_min (m), referenced drift extrema",
    data_values(result.log, command),
    [dx, dx, 0],
    atol=1e-14,
)
verdict(defect)
