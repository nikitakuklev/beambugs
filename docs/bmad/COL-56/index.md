---
title: "COL-56: Sparse legacy wake rows take polarization from the wrong row"
hide:
  - toc
---

# COL-56 — Sparse legacy wake rows take polarization from the wrong row

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`parser_read_old_format_lr_wake`, bmad/parsing/bmad_parser_mod.f90:1877–1929](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/bmad_parser_mod.f90#L1877-L1929) |

A legacy long-range wake file with gaps between populated rows can silently change a retained mode’s polarization. The parser compacts the rows but reads the polarization angle from the output index instead of the input row. Modes in rows one and three with angles zero and 0.2500 turns parse successfully, but the second mode becomes unpolarized with angle zero; the same modes in consecutive rows preserve both angles.

Run `python COL-56.py` with [_bmad.py](../helpers/_bmad.py) and [_wake_tables_and_bbu.py](../helpers/_wake_tables_and_bbu.py) in the same directory.

??? example "Reproducer: COL-56.py"

    [Download COL-56.py](COL-56.py)

    ```python
    """COL-56: Sparse legacy wake rows take polarization from the wrong row

    A legacy long-range wake file with gaps between populated rows can silently change a retained mode’s polarization. The parser compacts the rows but reads the polarization angle from the output index instead of the input row. Modes in rows one and three with angles zero and 0.2500 turns parse successfully, but the second mode becomes unpolarized with angle zero; the same modes in consecutive rows preserve both angles.

    The script parses identical legacy long-range modes from consecutive rows and from rows one and three. It compares every mode’s frequency, polarization flag and angle with the input values.
    """

    from _bmad import compare, fortran, verdict
    from _wake_tables_and_bbu import require, rows

    PROGRAM = r"""
    program sparse_wake
    use bmad
    implicit none
    type(lat_struct) lat
    integer i
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) error stop 'Parser failed'
    if (size(lat%ele(1)%wake%lr%mode)/=2) error stop 'Wrong mode count'
    do i=1,2
     write(*,'(a,i0,3es25.16)') 'RESULT mode',i,lat%ele(1)%wake%lr%mode(i)%freq, &
     real(merge(1,0,lat%ele(1)%wake%lr%mode(i)%polarized),rp),lat%ele(1)%wake%lr%mode(i)%angle
    enddo
    end program
    """
    LATTICE = "no_digested\nparameter[particle]=electron\nparameter[p0c]=1e9\nparameter[geometry]=open\nw: pipe, l=1, lr_wake_file='modes.wake'\nln: line=(w)\nuse,ln\n"
    DENSE = "&long_range_modes\nlr(1)%freq=1e9, lr(1)%r_over_q=100, lr(1)%q=10000, lr(1)%m=1, lr(1)%angle='0'\nlr(2)%freq=2e9, lr(2)%r_over_q=200, lr(2)%q=20000, lr(2)%m=1, lr(2)%angle='0.25'\n/\n"
    SPARSE = "&long_range_modes\nlr(1)%freq=1e9, lr(1)%r_over_q=100, lr(1)%q=10000, lr(1)%m=1, lr(1)%angle='0'\nlr(3)%freq=2e9, lr(3)%r_over_q=200, lr(3)%q=20000, lr(3)%m=1, lr(3)%angle='0.25'\n/\n"
    control = rows(fortran(PROGRAM, files={"lat.bmad": LATTICE, "modes.wake": DENSE}).log)
    for i, expected in [(1, [1e9, 1, 0]), (2, [2e9, 1, 0.25])]:
        require(f"dense mode {i}: Hz, polarized, turns", control[f"mode{i}"], expected)
    actual = rows(fortran(PROGRAM, files={"lat.bmad": LATTICE, "modes.wake": SPARSE}).log)
    require("sparse first mode", actual["mode1"], control["mode1"])
    require("sparse second frequency (Hz)", actual["mode2"][0], 2e9)
    verdict(
        compare("sparse second polarization and angle (turns)", actual["mode2"][1:], [1, 0.25], atol=0)
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

