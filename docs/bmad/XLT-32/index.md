---
title: "XLT-32: Elegant export drops order-zero multipole components"
hide:
  - toc
---

# XLT-32 — Elegant export drops order-zero multipole components

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">experimental</span> <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`write_lattice_elegant_format`, multipole branch](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_elegant_format.f90#L678-L698)<br>[Bmad manual, “Magnetostatic Multipole Fields”, `s:mag.field`, equation `qlpbb`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/electromagnetic-fields.tex#L92-L116)<br>[Bmad manual, “Multipole”, `s:mult`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L3860-L3878) |

Exporting order-zero multipole components to Elegant removes their dipole kicks. An ab_multipole with vertical dipole component a₀ = 0.0007300 should change normalized vertical momentum by 0.0007300, but its export gives zero; a straight-reference K0L dipole is likewise lost. When the omitted component belongs to a thick element’s generated end multipoles, an undefined element name also makes Elegant reject the lattice. Handwritten order-zero MULT elements restore the missing kicks in Elegant.

Run `python XLT-32.py` with [_bmad.py](../helpers/_bmad.py), [_s3_24_translations.py](../helpers/_s3_24_translations.py) and [_s4_17_translations.py](../helpers/_s4_17_translations.py) in the same directory.

??? example "Reproducer: XLT-32.py"

    [Download XLT-32.py](XLT-32.py)

    ```python
    """XLT-32: Elegant export drops order-zero multipole components

    Exporting order-zero multipole components to Elegant removes their dipole kicks. An ab_multipole with vertical dipole component a₀ = 0.0007300 should change normalized vertical momentum by 0.0007300, but its export gives zero; a straight-reference K0L dipole is likewise lost. When the omitted component belongs to a thick element’s generated end multipoles, an undefined element name also makes Elegant reject the lattice. Handwritten order-zero MULT elements restore the missing kicks in Elegant.

    Setup: `ELEGANT_BIN` set to a serial Elegant executable and `RPN_DEFNS` to the `defns.rpn` file of SDDS; the script runs the exported lattices in Elegant.

    The script exports normal and skew order-zero examples and runs Elegant. It compares the kicks with Bmad and handwritten order-zero MULT controls, and checks the thick-element rejection and missing reference geometry.
    """

    import math
    import re

    from _bmad import compare, verdict
    from _s3_24_translations import elegant_track, lattice, undefined_names
    from _s4_17_translations import export, track

    bad = False
    for label, ele, tilt, expected in [
        ("a0", "ab_multipole,a0=.00073", -math.pi / 2, [0, 0.00073]),
        (
            "k0 straight",
            "multipole,k0l=.00073,k0l_status=straight_reference,k1l=.5,tracking_method=symp_lie_ptc",
            0,
            [-0.00073, 0],
        ),
    ]:
        lat = lattice(ele)
        native, _ = track(lat, start=(0, 0, 0, 0, 0, 0))
        if compare(label + " Bmad kick control", native["ORBIT"][[1, 3]], expected, atol=3e-15):
            raise RuntimeError("Native kick control failed")
        text = export(lat, "elegant")
        actual = elegant_track(text, x=0, y=0)
        if actual is None:
            raise RuntimeError("Unexpected Elegant rejection")
        bad |= compare(label + " Elegant kick", actual[[1, 3]], expected, atol=3e-15)
        # Build the supported map independently of the export so that a corrected
        # writer does not receive a second copy of its restored dipole kick.
        hand = f"M0: mult,knl=.00073,tilt={tilt},order=0\nLAT: line=(M0)\n"
        control = elegant_track(hand, x=0, y=0)
        if control is None or compare(
            label + " handwritten ORDER=0 control", control[[1, 3]], expected, atol=3e-15
        ):
            raise RuntimeError("Order-zero receiving-code control failed")
    text = export(lattice("quadrupole,l=.5,a0=.00073,scale_multipoles=F"), "elegant")
    missing, line = undefined_names(text)
    print(f"thick a0 beamline: {line}; undefined={missing}")
    if missing:
        if missing != ["QMULTIPOLE_1"]:
            raise RuntimeError("Unexpected undefined names")
        bad |= elegant_track(text, x=0, y=0, rejected=missing[0]) is None
    lat = lattice("multipole,k0l=.00073,k0l_status=bends_reference,k1l=.5", "d:drift,l=1", "m,d")
    native, _ = track(lat, start=(0, 0, 0, 0, 0, 0))
    # floor_w_mat_to_angles wraps theta with modulo2(theta, pi). Adding/subtracting
    # pi shifts this angle by 1.026e-16 rad; allow one ulp at pi over the 1 m drift.
    if compare("native reference-bend floor x (m)", native["FLOOR"][0], -math.sin(0.00073), atol=5e-16):
        raise RuntimeError("Reference-bend floor control failed")
    text = export(lat, "elegant")
    orders = [int(v) for v in re.findall(r"order\s*=\s*(\d+)", text)]
    print(
        f"reference-bend export orders={orders}; expected explicit reference-bend support or diagnostic"
    )
    verdict(bad)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed<br><small>straight-reference dipole kick omission</small> | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

