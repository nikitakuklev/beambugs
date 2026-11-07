---
title: "XLT-13: MAD-X skew-multipole export reverses the kick"
hide:
  - toc
---

# XLT-13 — MAD-X skew-multipole export reverses the kick

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P1 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-20 on Bmad release [20261020-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261020-0) (`19b8afb`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Issue | [[gh:https://github.com/bmad-sim/bmad-ecosystem/issues/2228]], filed from this evidence |
| Pull request | [[gh:https://github.com/bmad-sim/bmad-ecosystem/pull/2229]], filed from this evidence |
| Source | [`write_lattice_mad_format`, bmad/output/write_lattice_mad_format.f90:960](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_mad_format.f90#L960)<br>[lines 267–286](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_mad_format.f90#L267-L286)<br>[line 960](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L960) |

Exporting a skew multipole of quadrupole order or higher to MAD-X negated its skew coefficients and reversed its transverse kick. The shared complex field expansion requires the same skew sign in both codes. For integrated skew-quadrupole strength 0.91 m⁻¹ at (x, y) = (1, 2) mm, the correct normalized transverse momentum changes are (0.001820, 0.0009100); the export gave (−0.001820, −0.0009100). The field expansion and Bmad tracking agree on the positive changes. The same writer handles skew field errors attached to thick magnets, while the main fields of thick skew magnets and the MAD8 export use separate, correct branches.

Run `python XLT-13.py` with [_bmad.py](../helpers/_bmad.py) and [_translators.py](../helpers/_translators.py) in the same directory.

??? example "Reproducer: XLT-13.py"

    [Download XLT-13.py](XLT-13.py)

    ```python
    """XLT-13: MAD-X skew-multipole export reverses the kick

    Exporting a skew multipole of quadrupole order or higher to MAD-X negated its skew coefficients
    and reversed its transverse kick. The shared complex field expansion requires the same skew sign
    in both codes. For integrated skew-quadrupole strength 0.91 m⁻¹ at (x, y) = (1, 2) mm, the
    correct normalized transverse momentum changes are (0.001820, 0.0009100); the export gave
    (−0.001820, −0.0009100). The field expansion and Bmad tracking agree on the positive changes.
    The same writer handles skew field errors attached to thick magnets, while the main fields of
    thick skew magnets and the MAD8 export use separate, correct branches.

    The script exports with bmad_to_mad_sad_elegant a thin skew quadrupole given as
    `ab_multipole, a1 = 0.91` and as `multipole, k1sl = 0.91`, reads the written `ksl`, and tracks
    a particle at (x, y) = (1, 2) mm through each export with MAD-X TRACK. The kick is compared
    with the field expansion (b₁ + i a₁)(x + i y) that Bmad and MAD-X share, and Bmad's own kick is
    the control. A copy of the export with `ksl` set to +0.91 is the receiving-code control.
    """

    import re

    import numpy as np
    from _bmad import compare, verdict
    from _translators import export, lattice, madx, madx_track, numbers, statements, tao

    X, Y, STRENGTH = 1e-3, 2e-3, 0.91
    # Thin skew quadrupole: F = (b1 + i a1)(x + i y) with b1 = 0, a1 = 0.91;
    # Delta px = -Re F, Delta py = +Im F.
    field = 1j * STRENGTH * complex(X, Y)
    EXPECTED = np.array([-field.real, field.imag])

    defect = False
    for element in ("ab_multipole, a1 = 0.91", "multipole, k1sl = 0.91"):
        lat = lattice(f"e: {element}", "e", start=[X, 0, Y, 0, 0, 0])
        kick = numbers(tao(lat, ["pipe ele:orbit 1"])["pipe ele:orbit 1"], ["px", "py"])
        if compare(f"{element}: Bmad (px, py)", kick, EXPECTED, atol=0, rtol=2e-15):
            raise SystemExit("Bmad disagrees with the field expansion: comparison not valid.")
        deck = export(lat, "madx")
        print(f"{element}: exported: E: {statements(deck, 'madx')['E']}")
        with madx(deck) as mad:
            end = madx_track(mad, [X, 0, Y, 0, 0, 0])
        defect |= compare(
            f"{element}: MAD-X TRACK (px, py) of the export", end[[1, 3]], EXPECTED, atol=0, rtol=2e-15
        )
        # Receiving-code controls: ksl = +0.91 must give the field-expansion kick, and the value
        # -0.91 that release 20261020-0 wrote must give the reversed kick.
        for sign, target in ((1, EXPECTED), (-1, -EXPECTED)):
            edited, count = re.subn(r"ksl = \{[^}]+\}", f"ksl = {{0, {sign * STRENGTH}}}", deck)
            if count != 1:
                raise SystemExit("Expected one ksl array in the export.")
            with madx(edited) as mad:
                end = madx_track(mad, [X, 0, Y, 0, 0, 0])
            if compare(
                f"{element}: MAD-X TRACK with ksl = {{0, {sign * STRENGTH}}}, expected {sign:+d} x field",
                end[[1, 3]],
                target,
                atol=0,
                rtol=2e-15,
            ):
                raise SystemExit("A ksl control disagrees: comparison not valid.")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

