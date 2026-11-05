---
title: "TAO-08: Pipe global duplicates two optimizer parameter records"
hide:
  - toc
---

# TAO-08 — Pipe global duplicates two optimizer parameter records

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [`tao_pipe_cmd`, `global` response, tao/code/tao_pipe_cmd.f90:4604–4612](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_pipe_cmd.f90#L4604-L4612)<br>[Tao manual, “pipe global” and “pipe global:optimization”, tao/doc/pipe-interface-commands.tex:1251–1286](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/pipe-interface-commands.tex#L1251-L1286) |

A pipe global response contains two copies each of n_opti_cycles and n_opti_loops. Both copies have identical values, matching the corresponding optimization-only response. Parameter names must be unique within a response so clients can validate and consume its schema. Removing the repeated records preserves all parameter values.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _tao_optimization_io.py](../helpers/_tao_optimization_io.py) · [View source](../helpers/index.md#_tao_optimization_io)

??? example "Reproducer: TAO-08.py"

    [Download TAO-08.py](TAO-08.py)

    ```python
    """TAO-08: Pipe global duplicates two optimizer parameter records

    A pipe global response contains two copies each of n_opti_cycles and n_opti_loops. Both copies
    have identical values, matching the corresponding optimization-only response. Parameter names
    must be unique within a response so clients can validate and consume its schema. Removing the
    repeated records preserves all parameter values.

    The script counts every parameter name in pipe global and checks the repeated values against
    pipe global:optimization. It requires the expected parameter schema and successful command
    completion before assigning a verdict. BMAD_DIST supplies Tao.
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

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

