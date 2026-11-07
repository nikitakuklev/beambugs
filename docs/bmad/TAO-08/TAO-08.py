"""TAO-08: Pipe global duplicates two optimizer parameter records

A pipe global response contains two copies each of n_opti_cycles and n_opti_loops. Both copies
have identical values, matching the corresponding optimization-only response. Parameter names
must be unique within a response so clients can validate and consume its schema. Removing the
repeated records preserves all parameter values.

The script counts every parameter name in pipe global and checks the repeated values against pipe
global:optimization. It requires the expected parameter schema and successful command completion
before assigning a verdict.
"""

from collections import Counter

from _bmad import compare, verdict
from _tao_optimization_io import LATTICE, block, parameters, records, tao

commands = ["pipe global", "pipe global:optimization"]
result = tao(commands, lattice=LATTICE)
rows = records(block(result.log, commands[0]))
counts = Counter(row[0] for row in rows)
if not {"n_opti_loops", "n_opti_cycles", "plot_on", "track_type"} <= counts.keys():
    raise RuntimeError(f"Missing global parameters: {counts}")
opt = parameters(block(result.log, commands[1]))
defect = False
for name in ("n_opti_cycles", "n_opti_loops"):
    matching = [row for row in rows if row[0] == name]
    if any(row != matching[0] for row in matching) or matching[0][3] != opt[name]:
        raise RuntimeError(f"Inconsistent optimizer parameter: {matching}; subset={opt[name]}")
    defect |= compare(f"{name} record count, unique parameter contract", counts[name], 1, atol=0)
if any(
    count > 1 for name, count in counts.items() if name not in ("n_opti_cycles", "n_opti_loops")
):
    raise RuntimeError(f"Additional duplicate parameter: {counts}")
verdict(defect)
