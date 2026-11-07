---
title: "COL-51: Whole-lattice wake reset leaves secondary-branch history intact"
hide:
  - toc
---

# COL-51 — Whole-lattice wake reset leaves secondary-branch history intact

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [zero_lr_wakes_in_lat contract and loop](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L58-L91)<br>[root element-array alias](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/allocate_lat_ele_array.f90#L36-L42)<br>[track_beam caller contract](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_mod.f90#L11-L16) |

Resetting long-range wakes through zero_lr_wakes_in_lat clears only the root branch.
Independent secondary branches retain their stored modal amplitudes and reference times, so a
subsequent run inherits the previous run’s wake history. In a two-branch lattice initialized
with amplitude 2 and reference time 3 s, the root branch resets both values to zero while the
secondary branch retains both. A whole-lattice reset must clear each branch.

Run `python COL-51.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-51.py"

    [Download COL-51.py](COL-51.py)

    ```python
    """COL-51: Whole-lattice wake reset leaves secondary-branch history intact

    Resetting long-range wakes through zero_lr_wakes_in_lat clears only the root branch.
    Independent secondary branches retain their stored modal amplitudes and reference times, so a
    subsequent run inherits the previous run’s wake history. In a two-branch lattice initialized
    with amplitude 2 and reference time 3 s, the root branch resets both values to zero while the
    secondary branch retains both. A whole-lattice reset must clear each branch.

    The script initializes all four modal amplitudes and the reference time in two independent
    branches, then calls the whole-lattice reset. It compares each component with zero and requires
    the root-branch control to pass. The branches have separate elements and wake allocations.
    """

    from _bmad import compare, fortran, verdict

    PROGRAM = """program probe
    use bmad
    use wake_mod
    implicit none
    type(lat_struct),target :: lat
    integer i
    call bmad_parser('branches.bmad',lat)
    if(ubound(lat%branch,1)/=1) error stop 'expected two branches'
    do i=0,1
      lat%branch(i)%ele(1)%wake%lr%mode(1)%b_sin=2
      lat%branch(i)%ele(1)%wake%lr%mode(1)%b_cos=4
      lat%branch(i)%ele(1)%wake%lr%mode(1)%a_sin=5
      lat%branch(i)%ele(1)%wake%lr%mode(1)%a_cos=6
      lat%branch(i)%ele(1)%wake%lr%t_ref=3
    enddo
    call zero_lr_wakes_in_lat(lat)
    do i=0,1
      write(*,'(a,i2,5es25.16)') 'RESULT ',i,lat%branch(i)%ele(1)%wake%lr%mode(1)%b_sin, &
        lat%branch(i)%ele(1)%wake%lr%mode(1)%b_cos,lat%branch(i)%ele(1)%wake%lr%mode(1)%a_sin, &
        lat%branch(i)%ele(1)%wake%lr%mode(1)%a_cos,lat%branch(i)%ele(1)%wake%lr%t_ref
    enddo
    end program
    """
    log = fortran(
        PROGRAM,
        files={
            "branches.bmad": """no_digested
    parameter[particle] = electron
    parameter[p0c] = 1e9
    parameter[geometry] = open
    w: pipe, l=1, lr_wake={mode={1e9,100,0,0,0,unpolarized}}
    v: pipe, l=1, lr_wake={mode={2e9,100,0,0,0,unpolarized}}
    ln: line=(w)
    ln2: line=(v)
    ln2[geometry]=open
    ln2[p0c]=1e9
    use,ln,ln2
    """
        },
    ).log
    rows = {
        int(p[1]): list(map(float, p[2:]))
        for line in log.splitlines()
        if (p := line.split()) and p[0] == "RESULT"
    }
    if set(rows) != {0, 1}:
        raise RuntimeError(log)
    if compare("root branch reset control [b_sin,b_cos,a_sin,a_cos,t_ref]", rows[0], [0] * 5, atol=0):
        raise RuntimeError("Root branch reset control failed")
    verdict(compare("secondary branch reset [b_sin,b_cos,a_sin,a_cos,t_ref]", rows[1], [0] * 5, atol=0))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

