---
title: "XLT-30: MAD-X export drops the whole-element tilt of a multipole"
hide:
  - toc
---

# XLT-30 — MAD-X export drops the whole-element tilt of a multipole

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`write_lattice_mad_format`, multipole branch](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L946-L972)<br>[`multipole_ele_to_ab`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/multipole_ele_to_ab.f90#L80-L104)<br>[the element-tilt rotation](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/multipole_ele_to_ab.f90#L232-L235)<br>[Bmad manual, element tilt](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L608-L616)<br>[Bmad manual, “Magnetostatic Multipole Fields”, `s:mag.field`, equation `qlpbb`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/electromagnetic-fields.tex#L92-L116)<br>[Bmad manual, “Multipole”, `s:mult`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L3860-L3878) |

Exporting a multipole with a whole-element tilt to MAD-X loses that rotation. For a thin quadrupole with integrated strength 0.5 m⁻¹ and tilt 0.17 rad, a particle at x = 1 mm and y = 2 mm receives normalized momenta (pₓ, pᵧ) = (−0.0005000, 0.001000) instead of (−0.0008049, 0.0007760), with momenta divided by reference momentum. A tilted order-zero reference bend also loses its vertical survey displacement. Per-component rotation and ab_multipole tilt are preserved; writing the whole-element TILT restores both the kick and the reference-bend plane.

Run `python XLT-30.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_24_translations.py](../helpers/_s3_24_translations.py) in the same directory.

??? example "Reproducer: XLT-30.py"

    [Download XLT-30.py](XLT-30.py)

    ```python
    """XLT-30: MAD-X export drops the whole-element tilt of a multipole

    Exporting a multipole with a whole-element tilt to MAD-X loses that rotation. For a thin quadrupole with integrated strength 0.5 m⁻¹ and tilt 0.17 rad, a particle at x = 1 mm and y = 2 mm receives normalized momenta (pₓ, pᵧ) = (−0.0005000, 0.001000) instead of (−0.0008049, 0.0007760), with momenta divided by reference momentum. A tilted order-zero reference bend also loses its vertical survey displacement. Per-component rotation and ab_multipole tilt are preserved; writing the whole-element TILT restores both the kick and the reference-bend plane.

    The script exports tilted quadrupole and sextupole multipoles, mixed component and element rotations, and passing rotation controls. It compares MAD-X TRACK and PTC with the complex multipole formula and Bmad tracking, then checks the reference-bend plane with MAD-X SURVEY. A handwritten MAD-X TILT supplies an independent receiving-code control.
    """

    import math
    import re

    import numpy as np
    from _bmad import compare, verdict
    from _s3_24_translations import bmad_kick, export, lattice, madx_track, track

    bad = False
    cases = [
        ("whole k1", "multipole,k1l=.5,tilt=.17", 1, 0.5, 0.17, 0.0, False),
        ("whole k2", "multipole,k2l=.8,tilt=.17", 2, 0.8, 0.17, 0.0, False),
        ("component plus whole", "multipole,k1l=.5,t1=.1,tilt=.07", 1, 0.5, 0.17, 0.0, False),
        ("component control", "multipole,k1l=.5,t1=.17", 1, 0.5, 0.17, 0.0, True),
        ("ab control", "ab_multipole,b1=.5,tilt=.17", 1, 0.5, 0.17, 0.0, True),
        (
            "normal and skew",
            "multipole,k1l=.5,k1sl=.2,tilt=.17,tracking_method=symp_lie_ptc",
            1,
            0.5,
            0.17,
            0.2,
            False,
        ),
    ]
    for label, ele, n, kn, tilt, skew, control in cases:
        lat = lattice(ele)
        expected = bmad_kick(n, kn, tilt, skew)
        native, _ = track(lat)
        if compare(f"{label} Bmad control (px,py)", native["ORBIT"][[1, 3]], expected, atol=8e-16):
            raise RuntimeError("Bmad multipole-formula control failed")
        text = export(lat, "madx")
        for ptc in (False, True):
            actual = madx_track(text, ptc=ptc)
            mismatch = compare(
                f"{label} MAD-X {'PTC' if ptc else 'TRACK'} (px,py)", actual, expected, atol=8e-16
            )
            if control and mismatch:
                raise RuntimeError("Per-component or ab rotation control failed")
            bad |= mismatch
        # Independent deck: component tilt and element tilt add for one order.
        knl = ",".join(["0"] * n + [str(kn)])
        ksl = ",".join(["0"] * n + [str(skew)])
        hand = f"beam,particle=electron,pc=1; m:multipole,knl={{{knl}}},ksl={{{ksl}}},tilt={tilt}; lat:line=(m); use,period=lat;"
        if compare(f"{label} handwritten MAD-X TILT control", madx_track(hand), expected, atol=8e-16):
            raise RuntimeError("Handwritten tilt control failed")
    lat = lattice("multipole,k0l=.001,k0l_status=bends_reference,tilt=.17", "d:drift,l=1", "m,d")
    native, _ = track(lat, start=(0, 0, 0, 0, 0, 0))
    expected = -math.sin(0.001) * np.array([math.cos(0.17), math.sin(0.17)])
    if compare("Bmad floor geometry control (m)", native["FLOOR"][:2], expected, atol=2e-17):
        raise RuntimeError("Native survey control failed")
    text = export(lat, "madx")
    bad |= compare("MAD-X exported floor (m)", madx_track(text, survey=True), expected, atol=2e-17)
    hand = re.sub(r"(M: multipole[^;]*);", r"\1, tilt=.17;", text)
    if hand == text or compare(
        "MAD-X added TILT floor control (m)", madx_track(hand, survey=True), expected, atol=2e-17
    ):
        raise RuntimeError("Reference-bend tilt control failed")
    verdict(bad)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-24 | madx-ksl-sign, source, production, CPU, double, serial |
| Confirmed<br><small>whole-element multipole tilt in MAD-X export</small> | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed<br><small>tilted reference-bend and thick-element error extensions</small> | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |

