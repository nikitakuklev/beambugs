---
title: "COL-52: Replacing a table wake with modes leaves an active empty table"
hide:
  - toc
---

# COL-52 — Replacing a table wake with modes leaves an active empty table

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [`parser_read_sr_wake`, bmad/parsing/bmad_parser_mod.f90:1472–1484](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/bmad_parser_mod.f90#L1472-L1484)<br>[`parser_read_sr_wake`, bmad/parsing/bmad_parser_mod.f90:1663–1665](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/bmad_parser_mod.f90#L1663-L1665)<br>[`sr_z_long_wake`, bmad/multiparticle/wake_mod.f90:532–558](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L532-L558) |

Replacing a tabulated short-range wake with a mode-only definition leaves the old table spacing active after its arrays have been emptied. Tracking then reports that the bunch exceeds the table window and marks a particle lost. A single electron macroparticle receives the same normalized momentum kick, −0.0005000, as the mode-only control, but is lost instead of remaining alive.

Run `python COL-52.py` with [_bmad.py](../helpers/_bmad.py) and [_wake_tables_and_bbu.py](../helpers/_wake_tables_and_bbu.py) in the same directory.

??? example "Reproducer: COL-52.py"

    [Download COL-52.py](COL-52.py)

    ```python
    """COL-52: Replacing a table wake with modes leaves an active empty table

    Replacing a tabulated short-range wake with a mode-only definition leaves the old table spacing active after its arrays have been emptied. Tracking then reports that the bunch exceeds the table window and marks a particle lost. A single electron macroparticle receives the same normalized momentum kick, −0.0005000, as the mode-only control, but is lost instead of remaining alive.

    The script parses a table followed by a mode-only replacement, then parses the final definition alone as a control. Each fresh process tracks one charged particle and reports table size, spacing, state and kick.
    """

    import re

    from _bmad import compare, fortran, verdict
    from _wake_tables_and_bbu import require

    PROGRAM = r"""
    program replace_wake
    use bmad
    use wake_mod
    implicit none
    type(lat_struct) lat
    type(bunch_struct) bunch
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) error stop 'Parser failed'
    write(*,'(a,i0,a,es25.16)') 'TABLE ',size(lat%ele(1)%wake%sr%z_long%w),' ',lat%ele(1)%wake%sr%z_long%dz
    call reallocate_bunch(bunch,1)
    call init_coord(bunch%particle(1), [0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp], &
      lat%ele(0),element_end=downstream_end$)
    bunch%particle(1)%charge=1e-9_rp
    call order_particles_in_z(bunch)
    call track1_sr_wake(bunch,lat%ele(1))
    write(*,'(a,i0,a,es25.16)') 'PARTICLE ',bunch%particle(1)%state,' ',bunch%particle(1)%vec(6)
    end program
    """

    LATTICES = [
        "no_digested\nparameter[particle]=electron\nparameter[p0c]=1e9\nparameter[geometry]=open\nw: pipe, l=1, sr_wake={scale_with_length=F, longitudinal={1e15,0,0,0.25,none}}\nln: line=(w)\nuse,ln\n",
        "no_digested\nparameter[particle]=electron\nparameter[p0c]=1e9\nparameter[geometry]=open\nw: pipe, l=1, sr_wake={scale_with_length=F, z_long={w={-0.01 1e15, 0 1e15, 0.01 0,}}}\nw[sr_wake]={scale_with_length=F, longitudinal={1e15,0,0,0.25,none}}\nln: line=(w)\nuse,ln\n",
    ]


    def check(lattice):
        log = fortran(PROGRAM, files={"lat.bmad": lattice}).log
        table = re.search(r"TABLE\s+(\d+)\s+([\d.Ee+\-]+)", log)
        particle = re.search(r"PARTICLE\s+(\d+)\s+([\d.Ee+\-]+)", log)
        if table is None or particle is None:
            raise RuntimeError(log)
        return [float(table[1]), float(table[2]), float(particle[1]), float(particle[2])], log


    control, _ = check(LATTICES[0])
    # At phase pi/2 the modal self wake is q*A/2. Length scaling is disabled,
    # and the modal momentum law normalizes this voltage by p0c in energy units.
    self_kick = -1e-9 * 1e15 / (2 * 1e9)
    require("mode-only control: size, dz(m), state, delta", control, [0, 0, 1, self_kick], atol=1e-18)
    actual, log = check(LATTICES[1])
    require(
        "replacement table size and modal self kick", [actual[0], actual[3]], [0, self_kick], atol=1e-18
    )
    defect = compare(
        "replacement: dz(m), state; final definition equals control", actual[1:3], [0, 1], atol=0
    )
    if defect and "The bunch is longer than the sr-z wake can handle" not in log:
        raise RuntimeError(log)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

