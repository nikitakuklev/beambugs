---
title: "TAO-04: Data namelist export writes an unquoted character universe selector"
hide:
  - toc
---

# TAO-04 — Data namelist export writes an unquoted character universe selector

| | |
|---|---|
| Code | [Bmad](../index.md) · Tao |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [`tao_write_cmd`, tao/code/tao_write_cmd.f90:802–806](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_write_cmd.f90#L802-L806)<br>[`tao_init_data` character declaration, tao/code/tao_init_data_mod.f90:43–49](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_init_data_mod.f90#L43-L49) |

Tao writes the universe selector as an unquoted integer in an exported data namelist, although its reader declares that selector as character data. The build compiler’s namelist reader rejects the generated file with IOSTAT = 5010 and a missing-quote error. Quoting only the selector makes the same namelist readable and recovers universe 1. Generated data namelists must use the syntax accepted by the reader.

Run `python TAO-04.py` with [_bmad.py](../helpers/_bmad.py) and [_tao_optimization_io.py](../helpers/_tao_optimization_io.py) in the same directory.

??? example "Reproducer: TAO-04.py"

    [Download TAO-04.py](TAO-04.py)

    ```python
    """TAO-04: Data namelist export writes an unquoted character universe selector

    Tao writes the universe selector as an unquoted integer in an exported data namelist, although
    its reader declares that selector as character data. The build compiler’s namelist reader
    rejects the generated file with IOSTAT = 5010 and a missing-quote error. Quoting only the
    selector makes the same namelist readable and recovers universe 1. Generated data namelists must
    use the syntax accepted by the reader.

    The script generates a data export with Tao and compiles a small Fortran reader with the same
    character-valued selector declaration. It compares reads of the original export and a copy with only
    that selector quoted.
    """

    import re

    from _bmad import compare, fortran, verdict
    from _tao_optimization_io import LINE, RMS_INIT, control, tao

    result = tao(
        ["write namelist -data export.init"], lattice=LINE, init=RMS_INIT, outputs=["export.init"]
    )
    export = result.outputs["export.init"].decode()
    quoted = re.sub(r"(?m)^(\s*universe\s*=\s*)1\s*$", r"\1'1'", export)
    PROGRAM = """program ledger_namelist_probe
    implicit none
    type d2_type
      character(40) name
    end type
    character(40) universe
    character(200) message, filename
    type(d2_type) d2_data
    integer n_d1_data, status
    namelist /tao_d2_data/ d2_data, universe, n_d1_data
    call get_command_argument(1, filename)
    open(10, file=trim(filename), status='old')
    read(10, nml=tao_d2_data, iostat=status, iomsg=message)
    close(10)
    write(*,'(a,i0)') 'IOSTAT=', status
    if (status /= 0) then
      write(*,'(a)') trim(message)
    else
      write(*,'(a,a)') 'universe=', trim(universe)
    endif
    write(*,'(a)') 'PROBE COMPLETE'
    end program
    """
    statuses = []
    for name, content in (("original", export), ("quoted", quoted)):
        result = fortran(PROGRAM, ["export.init"], files={"export.init": content})
        match = re.search(r"^IOSTAT=(\d+)$", result.log, re.MULTILINE)
        if match is None or "PROBE COMPLETE" not in result.log:
            raise RuntimeError(result.log)
        status = int(match[1])
        if status and (status != 5010 or "Missing quote while reading item 2" not in result.log):
            raise RuntimeError(result.log)
        if not status and "universe=1" not in result.log:
            raise RuntimeError(result.log)
        statuses.append(status)
        print(f"{name} namelist read: observed IOSTAT={status}; expected=0, quoted character selector")
    control("quoted selector control IOSTAT", statuses[1], 0, atol=0)
    verdict(compare("original selector IOSTAT", statuses[0], 0, atol=0))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

