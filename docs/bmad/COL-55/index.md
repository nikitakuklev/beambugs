---
title: "COL-55: BBU charge ramp patterns skip their final entry"
hide:
  - toc
---

# COL-55 — BBU charge ramp patterns skip their final entry

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [bbu_add_a_bunch ramp selection](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/code/bbu_track_mod.f90#L599-L608)<br>[ramp count construction](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/bbu/bbu_program.f90#L53-L61)<br>[ramp_pattern parameter](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/bbu/doc/bbu.tex#L237-L248) |

A BBU simulation with a repeating bunch-charge ramp omits the final supplied pattern entry. The
program counts all supplied values, but the insertion routine subtracts one again when wrapping
the pattern index. A three-entry pattern [1, 2, 3] produces charge factors [1, 2, 1] instead of
[1, 2, 3]. The documented pattern repeats all entries.

Run `python COL-55.py` with [_bmad.py](../helpers/_bmad.py) and [_ibs_charge_and_modes.py](../helpers/_ibs_charge_and_modes.py) in the same directory.

??? example "Reproducer: COL-55.py"

    [Download COL-55.py](COL-55.py)

    ```python
    """COL-55: BBU charge ramp patterns skip their final entry

    A BBU simulation with a repeating bunch-charge ramp omits the final supplied pattern entry. The
    program counts all supplied values, but the insertion routine subtracts one again when wrapping
    the pattern index. A three-entry pattern [1, 2, 3] produces charge factors [1, 2, 1] instead of
    [1, 2, 3]. The documented pattern repeats all entries.

    The script supplies a three-entry charge pattern and records the first three inserted bunch
    charges. It restores the queue end after each insertion to isolate pattern selection from the
    separate queue-index defect. The first two entries check initialization and charge
    normalization.
    """

    from _bmad import compare, verdict
    from _ibs_charge_and_modes import bbu_fortran

    PROGRAM = """program probe
    use bmad
    use beam_mod
    use bbu_track_mod
    implicit none
    type(lat_struct),target :: lat
    type(beam_init_struct) bi
    type(bbu_beam_struct) bb
    type(bbu_param_struct) bp
    integer i
    real(rp) factors(3)
    call ran_seed_put(12345)
    call bmad_parser('simple.bmad',lat)
    bi%n_particle=1
    bi%dt_bunch=1e-9_rp
    bi%bunch_charge=1e-9_rp
    bp%ramp_on=.true.
    bp%ramp_pattern(1:3)=[1.0_rp,2.0_rp,3.0_rp]
    bp%n_ramp_pattern=3
    call bbu_setup(lat,bi%dt_bunch,bp,bb)
    do i=1,3
      call bbu_add_a_bunch(lat,bb,bp,bi)
      factors(i)=bb%bunch(i)%charge_tot/bi%bunch_charge
      ! Restore the queue end after each call to isolate the pattern from its separate queue defect.
      bb%ix_bunch_end=i
    enddo
    write(*,'(a,3es25.16)') 'RESULT ',factors
    end program
    """
    log = bbu_fortran(
        PROGRAM,
        {
            "simple.bmad": """no_digested
    parameter[particle] = electron
    parameter[p0c] = 1e9
    parameter[geometry] = open
    beginning[beta_a] = 10
    beginning[beta_b] = 10
    w: pipe, l=1, lr_wake={mode={1e9,100,0,0,0,unpolarized}}
    ln: line=(w)
    use,ln
    """
        },
    ).log
    rows = [
        list(map(float, line.split()[1:])) for line in log.splitlines() if line.startswith("RESULT ")
    ]
    if len(rows) != 1:
        raise RuntimeError(log)
    if compare("first two pattern entries control", rows[0][:2], [1, 2], atol=2e-15):
        raise RuntimeError("Ramp initialization control failed")
    verdict(
        compare(
            "three-entry BBU charge pattern; repeat all supplied entries",
            rows[0],
            [1, 2, 3],
            atol=2e-15,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

