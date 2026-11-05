"""TAO-11: Wrapped extrema compare the second candidate with itself

A maximum constraint over a range that crosses the start of a ring ignores the segment after the
starting element. In a two-metre ring, a wrapped reference-time maximum returns 0.6671 ns
instead of 6.671 ns and is marked valid. The comparison tests the second segment’s candidate
against itself, so it never replaces the first candidate. The maximum must include both segments
of the evaluation range.

The script samples the reference clock around a ring and requests its maximum over QD through
the ring boundary to QF. It verifies the clock samples from the speed and circumference, then
checks the valid datum against the maximum of both segments. BMAD_DIST supplies Tao.
"""

import math

import numpy as np
from _bmad import compare, verdict
from _tao_optimization_io import RING_LATTICE, block, control, numbers, parameters, tao

INIT = """&tao_design_lattice design_lattice(1)%file = 'lat.bmad' /
&tao_d2_data d2_data%name = 'range', n_d1_data = 1 /
&tao_d1_data
 ix_d1_data = 1
 d1_data%name = 'clock'
 default_data_type = 'ref_time'
 default_merit_type = 'max'
 datum(1)%ele_start_name = 'QD'
 datum(1)%ele_name = 'QF'
 datum(1)%meas = 0
 datum(1)%weight = 1
/
"""
commands = [
    "pipe lat_list -index_order 1@0>>*|model ele.s,ele.ref_time",
    "pipe data range.clock[1]",
]
result = tao(commands, lattice=RING_LATTICE, init=INIT)
samples = numbers(block(result.log, commands[0]))
scale = math.hypot(1e9, 510998.95069) / (1e9 * 299792458.0)
positions = np.array([0, 0.2, 1, 1.2, 2, 2])
control("ring sample positions (m)", samples[:, 0], positions, atol=3e-16)
control("reference clock (s), s/(beta*c)", samples[:, 1], positions * scale, atol=1e-23)
datum = parameters(block(result.log, commands[1]))
if any(
    datum[key] != value
    for key, value in {
        "good_model": "T",
        "useit_opt": "T",
        "ix_ele_start": "3",
        "ix_ele": "1",
    }.items()
):
    raise RuntimeError(f"Wrong datum: {datum}")
# QD..END plus BEGINNING..QF has clock positions 1.2,2,2,0,.2 m.
verdict(
    compare(
        "wrapped maximum clock (s), both segments",
        float(datum["model_value"]),
        2 * scale,
        atol=1e-23,
    )
)
