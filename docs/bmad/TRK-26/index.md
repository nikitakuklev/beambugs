---
title: "TRK-26: Spin Taylor evaluation uses the orbital map’s expansion origin"
hide:
  - toc
---

# TRK-26 — Spin Taylor evaluation uses the orbital map’s expansion origin

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [track1_spin_taylor, bmad/spin/track1_spin_taylor.f90:36–50](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/track1_spin_taylor.f90#L36-L50)<br>[stored spin expansion origin, bmad/spin/sprint_spin_taylor_map.f90:270–274](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/sprint_spin_taylor_map.f90#L270-L274)<br>[automatic SPRINT map, bmad/spin/ele_to_spin_taylor.f90:30–31](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/ele_to_spin_taylor.f90#L30-L31)<br>[track_taylor reference-orbit contract, bmad/modules/taylor_mod.f90:710–741](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/taylor_mod.f90#L710-L741) |

When ordinary tracking automatically builds a SPRINT spin map about a displaced incoming orbit, spin Taylor evaluation uses the orbital map’s expansion origin instead. This evaluates the stored spin polynomial at the wrong displacement. A 1 GeV/c electron entering a 1 m quadrupole with strength 0.5 m⁻² at horizontal position 1 mm leaves with horizontal spin −0.003004 instead of −0.001502. Evaluation about the stored spin origin and the normalized linear Thomas–BMT model agree; a map explicitly built at the zero origin passes.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _rf_and_spin.py](../helpers/_rf_and_spin.py) · [View source](../helpers/index.md#_rf_and_spin)

??? example "Reproducer: TRK-26.py"

    [Download TRK-26.py](TRK-26.py)

    ```python
    """TRK-26: Spin Taylor evaluation uses the orbital map’s expansion origin

    When ordinary tracking automatically builds a SPRINT spin map about a displaced incoming orbit, spin Taylor evaluation uses the orbital map’s expansion origin instead. This evaluates the stored spin polynomial at the wrong displacement. A 1 GeV/c electron entering a 1 m quadrupole with strength 0.5 m⁻² at horizontal position 1 mm leaves with horizontal spin −0.003004 instead of −0.001502. Evaluation about the stored spin origin and the normalized linear Thomas–BMT model agree; a map explicitly built at the zero origin passes.

    The script compares automatic and explicit zero-origin SPRINT maps at zero and signed horizontal offsets, using BMAD_DIST. It evaluates the stored polynomial about its spin origin and compares it with the normalized linear Thomas–BMT solution.
    """

    import math

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _rf_and_spin import TAYLOR_LATTICE, TAYLOR_PROGRAM, control, evaluate, spin, spin_data

    d, maps = spin_data(fortran(TAYLOR_PROGRAM, ["3"], files={"verify.bmad": TAYLOR_LATTICE}).log)
    defect = False
    for i in range(3):
        for j in range(2):
            tag = f"R_{i}_{j}"
            k, length, gamma, anomaly = d[tag + "_PARAM"]
            x = d[tag + "_IN"]
            # Integrate linear T-BMT along x(s)=x0*cos(sqrt(k)*s), then normalize SPRINT's quaternion.
            qy = -0.5 * (1 + anomaly * gamma) * math.sqrt(k) * math.sin(math.sqrt(k) * length) * x[0]
            initial = np.array([0.0, 0.0, 1.0])
            expected = spin(np.array([1.0, 0.0, qy, 0.0]), initial)
            represented = spin(evaluate(maps[tag], x, d[tag + "_SPINREF"]), initial)
            control(
                f"{tag} stored spin-origin polynomial, normalized T-BMT",
                represented,
                expected,
                atol=2e-14,
            )
            if j == 0 or i == 0:
                control(f"{tag} zero-origin control", d[tag + "_ACTUAL"], expected, atol=2e-14)
            else:
                print(
                    f"{tag} spin origin={d[tag + '_SPINREF'].tolist()} orbital origin={d[tag + '_ORBITREF'].tolist()}"
                )
                defect |= compare(
                    f"{tag} automatic spin map, normalized T-BMT",
                    d[tag + "_ACTUAL"],
                    expected,
                    atol=2e-14,
                )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

