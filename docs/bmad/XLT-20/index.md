---
title: "XLT-20: MAD-X bend k0s import applies the wrong strength normalization"
hide:
  - toc
---

# XLT-20 — MAD-X bend k0s import applies the wrong strength normalization

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [parse_and_write_element bend strength, util_programs/mad_to_bmad/madx_to_bmad.py:549–555](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/mad_to_bmad/madx_to_bmad.py#L549-L555)<br>[bend multipole scale, bmad/code/multipole_ele_to_ab.f90:535–539](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/multipole_ele_to_ab.f90#L535-L539)<br>[relative multipole application, bmad/code/multipole_ele_to_ab.f90:584–589](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/multipole_ele_to_ab.f90#L584-L589)<br>[Bmad manual, “Multipole Attributes: Magnetic and Electric” (`s:multip`), bmad/doc/attributes.tex:2076–2093](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L2076-L2093) |

Importing a MAD-X bend with skew dipole strength k₀s multiplies it by the bend angle and then applies Bmad’s default relative-multipole scaling. The physical integrated skew field should be k₀sL, where L is bend length. With L = 2 m, angle = 0.1 rad, and k₀s = 0.001 m⁻¹, the imported integrated strength is 0.00001000 instead of 0.002000, a factor of 200 too small. A zero-angle bend loses the skew field entirely; an explicit unscaled integrated coefficient restores the expected vertical momentum kick.

Run `python XLT-20.py` with [_bmad.py](../helpers/_bmad.py) and [_lattice_import_export.py](../helpers/_lattice_import_export.py) in the same directory.

??? example "Reproducer: XLT-20.py"

    [Download XLT-20.py](XLT-20.py)

    ```python
    """XLT-20: MAD-X bend k0s import applies the wrong strength normalization

    Importing a MAD-X bend with skew dipole strength k₀s multiplies it by the bend angle and then applies Bmad’s default relative-multipole scaling. The physical integrated skew field should be k₀sL, where L is bend length. With L = 2 m, angle = 0.1 rad, and k₀s = 0.001 m⁻¹, the imported integrated strength is 0.00001000 instead of 0.002000, a factor of 200 too small. A zero-angle bend loses the skew field entirely; an explicit unscaled integrated coefficient restores the expected vertical momentum kick.

    The script translates curved and zero-angle bends using BMAD_SOURCE and tracks the generated Bmad lattices. For the straight case it compares with the integrated-field result and MAD-X PTC, then changes only the generated multipole representation as a control.
    """

    import re

    from _bmad import compare, verdict
    from _lattice_import_export import inspect, madx, records, translate, valid

    defect = False
    for angle in [0.1, 0.0]:
        lattice = f"beam,particle=electron,pc=3; b:sbend,l=2,angle={angle},k0s=.001; lat:line=(b); use,period=lat;"
        status, log, output = translate(lattice)
        if status:
            raise RuntimeError(log)
        result = inspect(output)
        valid(result)
        a0 = records(result, "A0")[0][0]
        py = records(result, "ORBIT")[-1][3]
        # Query the integrated field used by tracking, including Bmad's multipole scaling.
        physical = records(result, "FIELD")[0][0]
        defect |= compare(f"angle={angle}: integrated skew strength k0s*L", physical, 0.002, atol=2e-16)
        print(f"angle={angle}: imported a0={a0:.17g}, tracked py={py:.17g}")
        if angle == 0:
            with madx() as mad:
                mad.input(lattice)
                mad.input(
                    "ptc_create_universe;ptc_create_layout,model=2,method=6,nst=10;ptc_start;ptc_track,icase=6,closed_orbit=false,turns=1,onetable;"
                )
                ptc_py = float(mad.table.trackone.py[-1])
                mad.input("ptc_track_end;ptc_end;")
            if compare("MAD-X PTC straight-field py control", ptc_py, 0.002, atol=0, rtol=1e-12):
                raise RuntimeError("PTC field control failed")
            fixed, count = re.subn(
                r"a0\s*=\s*\.001\s*\*\s*b\[angle\]", "scale_multipoles=F,a0=.001*b[l]", output
            )
            if count != 1:
                raise RuntimeError(output)
            control = inspect(fixed)
            valid(control)
            fixed_py = records(control, "ORBIT")[-1][3]
            if compare("corrected Bmad straight-field py control", fixed_py, 0.002, atol=0, rtol=2e-15):
                raise RuntimeError("Corrected skew-field control failed")
            defect |= compare("imported straight-field py", py, 0.002, atol=0, rtol=2e-15)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

