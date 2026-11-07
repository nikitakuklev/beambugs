"""TAO-09: Track-only lattice listing uses branch zero’s tracking count

A track-only lattice listing truncates a secondary branch using the tracking-element count of
branch zero. A secondary branch with six tracking elements returns indices 0 through 3 instead
of 0 through 6 when the primary branch has three tracking elements. The unfiltered response
contains every element and its correct longitudinal position. The filter must use the selected
element’s own branch count.

The script creates a primary branch and a longer secondary branch and queries both branch counts. It
verifies the full secondary-branch table against its drift lengths, then compares the track-only
table with the same expected elements.
"""

import numpy as np
from _bmad import compare, verdict
from _tao_optimization_io import block, control, numbers, records, tao

LATTICE = """parameter[geometry] = open
parameter[particle] = electron
parameter[p0c] = 1e6
beginning[beta_a] = 3
beginning[beta_b] = 4
d: drift, l=0.1
f: fork, to_line=secondary
primary: line=(d,f)
secondary: line=(5*d)
secondary[geometry] = open
use, primary
"""
commands = [
    "pipe lat_branch_list 1",
    "pipe lat_list -index_order 1@1>>*|model ele.ix_ele,ele.s",
    "pipe lat_list -track_only -index_order 1@1>>*|model ele.ix_ele,ele.s",
]
result = tao(commands, lattice=LATTICE)
branches = records(block(result.log, commands[0]))
counts = [[int(row[i]) for i in (0, 2, 3)] for row in branches]
control("branch index, tracking and maximum element counts", counts, [[0, 3, 3], [1, 6, 6]], atol=0)
# Five 0.1-m drifts, with BEGINNING at 0 and END duplicating the exit at .5 m.
expected = np.column_stack((np.arange(7), [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5]))
complete = numbers(block(result.log, commands[1]))
control("unfiltered secondary branch indices and positions (m)", complete, expected, atol=2e-16)
filtered = numbers(block(result.log, commands[2]))
if filtered.ndim != 2 or filtered.shape[1] != 2:
    raise RuntimeError(f"Malformed filtered table: {filtered}")
indices = filtered[:, 0].astype(int)
if not np.array_equal(indices, filtered[:, 0]) or any(i < 0 or i > 6 for i in indices):
    raise RuntimeError(f"Invalid filtered indices: {filtered}")
control("filtered rows retain their correct positions (m)", filtered, expected[indices], atol=2e-16)
print(f"track-only indices: observed={indices.tolist()} expected={list(range(7))}")
defect = compare(
    "track-only record count, selected branch count plus BEGINNING", len(filtered), 7, atol=0
)
if len(filtered) == 7:
    control("complete track-only table", filtered, expected, atol=2e-16)
verdict(defect)
