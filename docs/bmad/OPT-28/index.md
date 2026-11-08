---
title: "OPT-28: srdt per-element terms have wrong factors and unset fields"
hide:
  - toc
---

# OPT-28 — srdt per-element terms have wrong factors and unset fields

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [srdt_calc per_ele_out, bmad/modules/srdt_mod.f90:211–235](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/srdt_mod.f90#L211-L235)<br>[srdt_calc_with_cache per_ele_out, bmad/modules/srdt_mod.f90:362–386](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/srdt_mod.f90#L362-L386)<br>[total normalization, bmad/modules/srdt_mod.f90:133–142](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/srdt_mod.f90#L133-L142)<br>[Bmad manual, Transfer Maps: Driving Terms, bmad/doc/subroutines.tex:3476–3513](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/subroutines.tex#L3476-L3513) |

Requesting per_ele_out from srdt_calc or srdt_calc_with_cache returns contributions that do not add up to the reported totals. The direct routine reverses three first-order signs and quarters the chromatic coefficient h₁₀₀₀₂, while the cached routine omits normalization factors altogether. In an eight-cell ring, the normal sextupole driving coefficient h₂₁₀₀₀ is −0.3827 − 0.3095i, but the direct contributions sum to its negative and cached contributions to −8 times the total. Source inspection finds that chromatic per-element fields are not assigned and second-order resonance fields are formed from unassigned slice fields.

Run `python OPT-28.py` with [_bmad.py](../helpers/_bmad.py) and [_srdt_wiggler_maps.py](../helpers/_srdt_wiggler_maps.py) in the same directory.

??? example "Reproducer: OPT-28.py"

    [Download OPT-28.py](OPT-28.py)

    ```python
    """OPT-28: srdt per-element terms have wrong factors and unset fields

    Requesting per_ele_out from srdt_calc or srdt_calc_with_cache returns contributions that do not add up to the reported totals. The direct routine reverses three first-order signs and quarters the chromatic coefficient h₁₀₀₀₂, while the cached routine omits normalization factors altogether. In an eight-cell ring, the normal sextupole driving coefficient h₂₁₀₀₀ is −0.3827 − 0.3095i, but the direct contributions sum to its negative and cached contributions to −8 times the total. Source inspection finds that chromatic per-element fields are not assigned and second-order resonance fields are formed from unassigned slice fields.

    The script assigns the repeating sextupole strengths (3, −4.5, 1.5) in an eight-cell ring. It sums the eight assigned first-order per-element complex fields from direct and cached calculations and compares them with the reported totals, requiring the correctly normalized direct fields to pass.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _srdt_wiggler_maps import LSQ_PROGRAM, THICK_RING

    strengths = " ".join(map(str, [3, -4.5, 1.5] * 8)) + "\n"
    log = fortran(
        LSQ_PROGRAM,
        ["lat.bmad", "perele", "strengths.txt"],
        files={"lat.bmad": THICK_RING, "strengths.txt": strengths},
    ).log
    names = ["h20001", "h00201", "h10002", "h21000", "h30000", "h10110", "h10020", "h10200"]
    rows = {
        fields[0]: np.array(list(map(float, fields[1:]))).reshape(3, 2)
        for line in log.splitlines()
        if (fields := line.split()) and fields[0] in names
    }
    if any(name not in rows for name in names):
        raise RuntimeError("Missing first-order terms")
    defect = False
    for name in names:
        total, direct, cached = rows[name]
        bad = compare(f"{name}: summed direct contributions", direct, total, atol=2e-12)
        if name in ("h20001", "h10110", "h10020", "h10200") and bad:
            raise AssertionError("Correctly normalized direct control failed")
        defect |= bad
        defect |= compare(f"{name}: summed cached contributions", cached, total, atol=2e-12)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | xhigh | <span class="bb-nowrap">2026-10-24</span> | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed<br><small class="bb-scope">per-element direct/cache factors; unset fields source-only</small> | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

