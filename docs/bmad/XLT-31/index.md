---
title: "XLT-31: SAD export turns a K0L reference bend into a dipole kick"
hide:
  - toc
---

# XLT-31 — SAD export turns a K0L reference bend into a dipole kick

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`write_lattice_sad_format`, multipole branch](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_sad_format.f90#L517-L523)<br>[generic MULT coefficient block](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_sad_format.f90#L726-L738)<br>[ROTATE output](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_sad_format.f90#L809-L812)<br>[Bmad manual, “Multipole”, `s:mult`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L3860-L3878)<br>[Bmad manual, “Sad_Mult”, dipole components](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L5040-L5041) |

Exporting a multipole whose dipole parameter K0L bends the reference trajectory to SAD writes a dipole kick on a straight reference instead. With K0L = 1 mrad and tilt 0.17 rad, a 1 m downstream drift should end at floor coordinates (−0.9856, −0.1692) mm while an on-reference particle stays centered. Importing the exported MULT back into Bmad leaves the floor straight and drives the particle (−0.9856, −0.1692) mm off axis. A straight-reference K0L is correctly represented by that dipole kick; SAD’s source defines the reference bend through a zero-length BEND with ANGLE.

Run `python XLT-31.py` with [_bmad.py](../helpers/_bmad.py), [_s3_24_translations.py](../helpers/_s3_24_translations.py) and [_s4_17_translations.py](../helpers/_s4_17_translations.py) in the same directory.

??? example "Reproducer: XLT-31.py"

    [Download XLT-31.py](XLT-31.py)

    ```python
    """XLT-31: SAD export turns a K0L reference bend into a dipole kick

    Exporting a multipole whose dipole parameter K0L bends the reference trajectory to SAD writes a dipole kick on a straight reference instead. With K0L = 1 mrad and tilt 0.17 rad, a 1 m downstream drift should end at floor coordinates (−0.9856, −0.1692) mm while an on-reference particle stays centered. Importing the exported MULT back into Bmad leaves the floor straight and drives the particle (−0.9856, −0.1692) mm off axis. A straight-reference K0L is correctly represented by that dipole kick; SAD’s source defines the reference bend through a zero-length BEND with ANGLE.

    The script exports both reference-status choices, imports the SAD text through the upstream importer selected by BMAD_SOURCE, and tracks it as sad_mult. It checks the reference geometry and dipole kick analytically, with the straight-reference case as a control.
    """

    import math

    import numpy as np
    from _bmad import compare, verdict
    from _s3_24_translations import lattice, translate_sad
    from _s4_17_translations import export, track

    bad = False
    c, s = math.cos(0.17), math.sin(0.17)
    expected_kick = -0.001 * np.array([c, s])
    expected_orbit = np.array(
        [
            expected_kick[0] / math.sqrt(1 - 1e-6),
            expected_kick[0],
            expected_kick[1] / math.sqrt(1 - 1e-6),
            expected_kick[1],
        ]
    )
    for status in ("bends_reference", "straight_reference"):
        method = ",tracking_method=symp_lie_ptc" if status == "straight_reference" else ""
        lat = lattice(f"multipole,k0l=.001,k0l_status={status},tilt=.17" + method, "d:drift,l=1", "m,d")
        original, _ = track(lat, start=(0, 0, 0, 0, 0, 0))
        text = export(lat, "sad")
        print(status + ": " + next(line for line in text.splitlines() if line.startswith("MULT")))
        converted = translate_sad(text)
        for mode in ("bmad_standard", "symp_lie_ptc"):
            out, _ = track(converted + f"\nm[tracking_method]={mode}\n", start=(0, 0, 0, 0, 0, 0))
            if status == "bends_reference":
                expected_floor = -math.sin(0.001) * np.array([c, s])
                if compare(
                    "native centered orbit control (m, normalized momenta)",
                    original["ORBIT"][:4],
                    np.zeros(4),
                    atol=2e-16,
                ) or compare(
                    "native floor control (m)", original["FLOOR"][:2], expected_floor, atol=2e-17
                ):
                    raise RuntimeError("Reference-bend controls failed")
                bad |= compare(
                    f"{mode} SAD round-trip orbit", out["ORBIT"][:4], np.zeros(4), atol=2e-16
                )
                bad |= compare(
                    f"{mode} SAD round-trip floor (m)", out["FLOOR"][:2], expected_floor, atol=2e-17
                )
            else:
                for label, values in [("native PTC", original), ("SAD " + mode, out)]:
                    if compare(
                        label + " straight kick control",
                        values["ORBIT"][:4],
                        expected_orbit,
                        atol=8e-16,
                    ) or compare(
                        label + " straight floor control (m)", values["FLOOR"][:2], [0, 0], atol=2e-17
                    ):
                        raise RuntimeError("Straight-reference control failed")
    verdict(bad)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>SAD reference-bend translation (source check)</small> | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`) |
| Confirmed | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed<br><small>SAD reference-bend export and Bmad round trip</small> | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

