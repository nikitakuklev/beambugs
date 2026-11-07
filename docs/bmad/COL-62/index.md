---
title: "COL-62: Short-range wake equations interchange source and witness offsets"
hide:
  - toc
---

# COL-62 — Short-range wake equations interchange source and witness offsets

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [`Short--Range Wakes (s:sr.wake.eq), delvp2 and delvp3`, bmad/doc/wakefields.tex:26–53](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/wakefields.tex#L26-L53)<br>[`Short-Range Wakes (s:sr.wake.syntax)`, bmad/doc/attributes.tex:3036–3049](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L3036-L3049)<br>[`sr_longitudinal_wake_particle`, bmad/multiparticle/wake_mod.f90:343–380](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L343-L380) |

The manual’s short-range longitudinal wake equations interchange the source and witness offsets for two position-dependent modes. The equation described as linear in the leading particle’s horizontal position uses the witness position, while the trailing-particle vertical equation weights the source. Tracking follows the attribute definitions: its kick ratios are the source horizontal offset, 0.002000 m, and the witness vertical offset, 0.009000 m, instead of 0.007000 m and 0.004000 m from the displayed equations.

Run `python COL-62.py` with [_bmad.py](../helpers/_bmad.py) and [_wake_tables_and_bbu.py](../helpers/_wake_tables_and_bbu.py) in the same directory.

??? example "Reproducer: COL-62.py"

    [Download COL-62.py](COL-62.py)

    ```python
    """COL-62: Short-range wake equations interchange source and witness offsets

    The manual’s short-range longitudinal wake equations interchange the source and witness offsets for two position-dependent modes. The equation described as linear in the leading particle’s horizontal position uses the witness position, while the trailing-particle vertical equation weights the source. Tracking follows the attribute definitions: its kick ratios are the source horizontal offset, 0.002000 m, and the witness vertical offset, 0.009000 m, instead of 0.007000 m and 0.004000 m from the displayed equations.

    The script reads the two equations from BMAD_SOURCE and measures the three modal kick cases on BMAD_DIST. Source self-kicks and the offset-independent witness kick are controls. It compares the displayed offset choices with the attribute contract and the measured ratios.
    """

    import re

    import numpy as np
    from _bmad import compare, fortran, source, verdict
    from _wake_tables_and_bbu import require, rows

    PROGRAM = r"""program modal_offsets
    use bmad
    use wake_mod
    implicit none
    type(ele_struct) ele
    type(coord_struct) orb
    integer i,dependence(3)
    character(12) wake_names(3)
    real(rp) kicks(2)
    call init_ele(ele,drift$)
      call init_wake(ele%wake, 1, 0, 0, 0)
      ele%value(l$) = 0.6_rp
      ele%value(p0c$) = 1.0e12_rp
      ele%value(p0c_start$) = 1.0e12_rp
      ele%value(e_tot$) = hypot(ele%value(p0c$),m_electron)
      ele%value(e_tot_start$) = ele%value(e_tot$)
      dependence = [none$, x_leading$, y_trailing$]
      wake_names = [character(12):: 'wake_none', 'wake_xlead', 'wake_ytrail']
      do i = 1, 3
        ele%wake%sr%long(1) = wake_sr_mode_struct()
        ele%wake%sr%long(1)%amp = 1.0e18_rp
        ele%wake%sr%long(1)%damp = 3.0_rp
        ele%wake%sr%long(1)%k = 4.0_rp
        ele%wake%sr%long(1)%phi = 0.13_rp
        ele%wake%sr%long(1)%position_dependence = dependence(i)
        ele%wake%sr%z_ref_long = 0.01_rp
        call init_coord(orb, [0.002_rp, 0.0_rp, 0.004_rp, 0.0_rp, 0.01_rp, 0.0_rp], ele, upstream_end$, electron$)
        orb%charge = 2.0e-12_rp
        call sr_longitudinal_wake_particle(ele, orb)
        kicks(1) = orb%vec(6)
        call init_coord(orb, [0.007_rp, 0.0_rp, 0.009_rp, 0.0_rp, -0.02_rp, 0.0_rp], ele, upstream_end$, electron$)
        orb%charge = 0.0_rp
        call sr_longitudinal_wake_particle(ele, orb)
        kicks(2) = orb%vec(6)
        write(*, '(3a, 2es25.16e3)') 'RESULT ', trim(wake_names(i)), ' ', kicks
      enddo

    end program
    """
    manual = (source() / "bmad/doc/wakefields.tex").read_text()


    def equation(label):
        matches = re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}", manual, re.DOTALL)
        return next(s for s in matches if f"\\label{{{label}}}" in s)


    x_indices = set(re.findall(r"x_([ij])", equation("delvp2")))
    y_indices = set(re.findall(r"y_([ij])", equation("delvp3")))
    if x_indices not in ({"i"}, {"i", "j"}) or y_indices not in ({"i"}, {"i", "j"}):
        raise RuntimeError("Unrecognized equation offsets")
    # i is the witness and j the source. Self terms can carry i in either equation.
    manual_x = 0.002 if "j" in x_indices else 0.007
    manual_y = 0.004 if "j" in y_indices else 0.009
    actual = rows(fortran(PROGRAM).log)
    scale = -1e18 * 2e-12 * 0.6 / 1e12
    phi = 2 * np.pi * 0.13
    base = scale * np.exp(-0.03 * 3) * np.sin(-0.03 * 4 + phi)
    for name, self_factor, witness_factor in [
        ("wake_none", 1, 1),
        ("wake_xlead", 0.002, 0.002),
        ("wake_ytrail", 0.004, 0.009),
    ]:
        require(
            f"{name} delta; attribute contract and causal mode sum",
            actual[name],
            [scale * np.sin(phi) * self_factor / 2, base * witness_factor],
            rtol=3e-13,
            atol=1e-22,
        )
    observed = np.array([actual["wake_xlead"][1], actual["wake_ytrail"][1]]) / actual["wake_none"][1]
    verdict(
        compare(
            "offset ratios (m); displayed delvp2, delvp3", observed, [manual_x, manual_y], atol=1e-17
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

