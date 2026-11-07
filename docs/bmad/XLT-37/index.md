---
title: "XLT-37: Sad_Mult documentation interchanges normal and skew coefficients"
hide:
  - toc
---

# XLT-37 — Sad_Mult documentation interchanges normal and skew coefficients

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P5 |
| Failure | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [manual passage](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L5036-L5038)<br>[magnetostatic field equation (`s:mag.field`, `qlpbb`)](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/electromagnetic-fields.tex#L92-L104)<br>[multipole attribute description](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L2061-L2074)<br>[Zero-length sad_mult tracking](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_sad_mult.f90#L60-L71) |

The Sad_Mult element documentation calls aₙ normal and bₙ skew, reversing Bmad’s magnetic-field convention. The correct labels are normal bₙ and skew aₙ. The defining field equation and direct tracking of separate quadrupole coefficients establish those labels; the two test kicks agree exactly with that convention.

Run `python XLT-37.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: XLT-37.py"

    [Download XLT-37.py](XLT-37.py)

    ```python
    """XLT-37: Sad_Mult documentation interchanges normal and skew coefficients

    The Sad_Mult element documentation calls aₙ normal and bₙ skew, reversing Bmad’s magnetic-field convention. The correct labels are normal bₙ and skew aₙ. The defining field equation and direct tracking of separate quadrupole coefficients establish those labels; the two test kicks agree exactly with that convention.

    The script reads the Sad_Mult description and field definition from BMAD_SOURCE, then tracks zero-length sad_mult elements with separate b1 and a1 coefficients against analytic quadrupole kicks. Both kick controls must pass before the documentation verdict.
    """

    import re

    import numpy as np
    from _bmad import compare, fortran, source, verdict

    root = source()
    elements = (root / "bmad/doc/elements.tex").read_text()
    fields = (root / "bmad/doc/electromagnetic-fields.tex").read_text()
    field_equation = r"\frac{q \, L}{P_0} \, (B_y + i B_x) = (b_n + i a_n) \, (x + i y)^n"
    if field_equation not in fields:
        raise RuntimeError("The defining magnetic-field equation changed")
    claim = re.search(
        r"\\vn\{sad_mult\} element (?:used|uses) normal \\vn\{(an|bn)\} and skew \\vn\{(an|bn)\}",
        elements,
    )
    if claim is None or claim.groups() not in [("an", "bn"), ("bn", "an")]:
        raise RuntimeError("The Sad_Mult coefficient description changed")
    program = """program coefficient_labels
    use bmad
    implicit none
    type(lat_struct) lat
    type(coord_struct) a,b
    integer i
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) error stop 'lattice parsing failed'
    do i=1,2
      call init_coord(a,[.001_rp,0.0_rp,.002_rp,0.0_rp,0.0_rp,0.0_rp],lat%ele(i),upstream_end$)
      call track1(a,lat%ele(i),lat%param,b)
      if(b%state/=alive$) error stop 'particle lost'
      print '(a,2es26.17)', 'KICK ',b%vec(2),b%vec(4)
    enddo
    print '(a)', 'COMPLETED'
    end program
    """
    lattice = """no_digested
    parameter[geometry]=open
    parameter[particle]=electron
    parameter[p0c]=1e9
    beginning[beta_a]=10
    beginning[beta_b]=10
    nb:sad_mult,l=0,b1=.5
    sa:sad_mult,l=0,a1=.5
    lat:line=(nb,sa)
    use,lat
    """
    result = fortran(program, files={"lat.bmad": lattice})
    if "COMPLETED" not in result.log:
        raise RuntimeError(result.log)
    observed = np.array(
        [
            [float(v) for v in line.split()[1:]]
            for line in result.log.splitlines()
            if line.startswith("KICK ")
        ]
    )
    # For n=1, -dpx+i*dpy=(b1+i*a1)*(x+i*y).
    # b1 gives normal focusing; a1 gives the skew cross-plane kick.
    expected = [[-0.5 * 0.001, 0.5 * 0.002], [0.5 * 0.002, 0.5 * 0.001]]
    if compare(
        "separate b1/a1 normalized kicks, analytic quadrupoles",
        observed,
        expected,
        atol=2e-19,
        rtol=3e-15,
    ):
        raise RuntimeError("Coefficient-convention tracking controls failed")
    print(f"Sad_Mult labels: normal={claim[1]}, skew={claim[2]}; expected normal=bn, skew=an")
    verdict(claim.groups() == ("an", "bn"))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

