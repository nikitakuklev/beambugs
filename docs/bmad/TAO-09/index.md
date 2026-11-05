---
title: "TAO-09: Track-only lattice listing uses branch zero’s tracking count"
hide:
  - toc
---

# TAO-09 — Track-only lattice listing uses branch zero’s tracking count

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [`tao_pipe_cmd`, `lat_list` filtering, tao/code/tao_pipe_cmd.f90:5162–5168](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_pipe_cmd.f90#L5162-L5168)<br>[Tao manual, “pipe lat_list” (`p:lat.list`), tao/doc/pipe-interface-commands.tex:1398–1416](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/pipe-interface-commands.tex#L1398-L1416) |

A track-only lattice listing truncates a secondary branch using the tracking-element count of branch zero. A secondary branch with six tracking elements returns indices 0 through 3 instead of 0 through 6 when the primary branch has three tracking elements. The unfiltered response contains every element and its correct longitudinal position. The filter must use the selected element’s own branch count.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _tao_optimization_io.py](../helpers/_tao_optimization_io.py) · [View source](../helpers/index.md#_tao_optimization_io)

??? example "Reproducer: TAO-09.py"

    [Download TAO-09.py](TAO-09.py)

    ```python
    """TAO-09: Track-only lattice listing uses branch zero’s tracking count

    A track-only lattice listing truncates a secondary branch using the tracking-element count of
    branch zero. A secondary branch with six tracking elements returns indices 0 through 3 instead
    of 0 through 6 when the primary branch has three tracking elements. The unfiltered response
    contains every element and its correct longitudinal position. The filter must use the selected
    element’s own branch count.

    The script creates a primary branch and a longer secondary branch and queries both branch
    counts. It verifies the full secondary-branch table against its drift lengths, then compares the
    track-only table with the same expected elements. BMAD_DIST supplies Tao.
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

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

