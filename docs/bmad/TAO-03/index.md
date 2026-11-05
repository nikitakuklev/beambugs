---
title: "TAO-03: Data namelist export omits every datum in its row-format path"
hide:
  - toc
---

# TAO-03 — Data namelist export omits every datum in its row-format path

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`tao_write_cmd` row-format dispatch, tao/code/tao_write_cmd.f90:820–827](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_write_cmd.f90#L820-L827)<br>[datum-row assembly, tao/code/tao_write_cmd.f90:896–914](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_write_cmd.f90#L896-L914)<br>[`namelist_item_out`, tao/code/tao_write_cmd.f90:1358–1390](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_write_cmd.f90#L1358-L1390)<br>[Tao manual, “write namelist” (`s:write.namelist`), tao/doc/command-list.tex:4318–4326](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/command-list.tex#L4318-L4326) |

Writing a Tao data namelist for an array of three constraints produces the array header but no datum definitions. Reloading the export with its universe selector quoted leaves all three datums nonexistent and reduces the merit from 2.475 × 10⁻⁵ to zero. A round trip must preserve the definitions and constraints. The row-format writer assembles each row in memory without writing it to the file.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _tao_optimization_io.py](../helpers/_tao_optimization_io.py) · [View source](../helpers/index.md#_tao_optimization_io)

??? example "Reproducer: TAO-03.py"

    [Download TAO-03.py](TAO-03.py)

    ```python
    """TAO-03: Data namelist export omits every datum in its row-format path

    Writing a Tao data namelist for an array of three constraints produces the array header but no
    datum definitions. Reloading the export with its universe selector quoted leaves all three
    datums nonexistent and reduces the merit from 2.475 × 10⁻⁵ to zero. A round trip must preserve
    the definitions and constraints. The row-format writer assembles each row in memory without
    writing it to the file.

    The script exports RMS, average, and integral constraints on a drift orbit and reloads the
    generated file in a fresh Tao process. It quotes the universe selector to isolate the missing
    rows, then compares datum existence and merit before and after the round trip. BMAD_DIST
    supplies Tao.
    """

    import re

    import numpy as np
    from _bmad import compare, verdict
    from _tao_optimization_io import LINE, RMS_INIT, block, control, data_values, records, tao

    commands = [
        "pipe data_parameter range.x exists",
        "pipe data_parameter range.x model_value",
        "pipe merit",
        "write namelist -data export.init",
    ]
    result = tao(commands, lattice=LINE, init=RMS_INIT, outputs=["export.init"])
    exists = records(block(result.log, commands[0]))
    if exists != [["1", "T", "T", "T"]]:
        raise RuntimeError(f"Input constraints missing: {exists}")
    values = data_values(result.log, commands[1])
    before = float(block(result.log, commands[2]))
    control(
        "initial merit, sum of three unit-weight squared values", before, values @ values, atol=1e-16
    )
    export = result.outputs["export.init"].decode()
    # Count datum identities, including a writer that emits fields on separate lines.
    row_count = len(set(re.findall(r"^\s*datum\(\s*(\d+)\s*\)", export, re.MULTILINE)))
    defect = compare("exported datum definitions", row_count, 3, atol=0)
    quoted = re.sub(r"(?m)^(\s*universe\s*=\s*)1\s*$", r"\1'1'", export)
    init = """&tao_start data_file = 'export.init' /
    &tao_design_lattice design_lattice(1)%file = 'line.bmad' /
    """
    commands = ["pipe data_parameter range.x exists", "pipe merit"]
    result = tao(commands, lattice=LINE, init=init, files={"export.init": quoted})
    rows = records(block(result.log, commands[0]))
    if len(rows) != 1 or rows[0][0] != "1" or any(v not in ("T", "F") for v in rows[0][1:]):
        raise RuntimeError(f"Malformed existence array: {rows}")
    observed = np.array([v == "T" for v in rows[0][1:]])
    defect |= compare("reloaded datum existence", observed, np.ones(3), atol=0)
    after = float(block(result.log, commands[1]))
    defect |= compare("reloaded merit, equal to pre-export merit", after, before, atol=1e-16)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

