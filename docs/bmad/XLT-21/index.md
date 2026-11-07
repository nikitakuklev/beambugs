---
title: "XLT-21: MAD-X and MAD8 export ignores is_on"
hide:
  - toc
---

# XLT-21 — MAD-X and MAD8 export ignores is_on

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P1 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`write_lattice_mad_format`, bmad/output/write_lattice_mad_format.f90:741–745](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L741-L745)<br>[lines 780–784](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L780-L784)<br>[908–913](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L908-L913)<br>[bmad/doc/attributes.tex:2010](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L2010)<br>[`multipole_ele_to_ab`, bmad/code/multipole_ele_to_ab.f90:62](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/multipole_ele_to_ab.f90#L62) |

MAD-X and MAD8 export reactivates quadrupoles, sextupoles, solenoids and RF cavities switched off with is_on = False. For a 0.5 m quadrupole with normalized gradient K₁ = 1.2 m⁻² between two 1 m drifts, the drift formula and Bmad give exit beta functions βₓ = βᵧ = 6.250 m from an initial waist with βₓ = βᵧ = 5 m. MAD-X TWISS of the export instead gives βₓ = 0.9211 m and βᵧ = 18.32 m, the switched-on optics. The exported sextupole, solenoid and cavity likewise retain their full strengths. Elegant correctly omits the switched-off quadrupole strength.

Run `python XLT-21.py` with [_bmad.py](../helpers/_bmad.py), [_s4_17_translations.py](../helpers/_s4_17_translations.py), [_translators.py](../helpers/_translators.py) and [_s3_24_translations.py](../helpers/_s3_24_translations.py) in the same directory.

??? example "Reproducer: XLT-21.py"

    [Download XLT-21.py](XLT-21.py)

    ```python
    """XLT-21: MAD-X and MAD8 export ignores is_on

    MAD-X and MAD8 export reactivates quadrupoles, sextupoles, solenoids and RF cavities switched off with is_on = False. For a 0.5 m quadrupole with normalized gradient K₁ = 1.2 m⁻² between two 1 m drifts, the drift formula and Bmad give exit beta functions βₓ = βᵧ = 6.250 m from an initial waist with βₓ = βᵧ = 5 m. MAD-X TWISS of the export instead gives βₓ = 0.9211 m and βᵧ = 18.32 m, the switched-on optics. The exported sextupole, solenoid and cavity likewise retain their full strengths. Elegant correctly omits the switched-off quadrupole strength.

    The script exports with bmad_to_mad_sad_elegant a 0.5 m quadrupole with K₁ = 1.2 m⁻² and
    is_on = False between two 1 m drifts, with initial β = 5 m in both planes. It runs MAD-X TWISS
    on the export and compares the exit β with Tao and with the 2.5 m drift that a switched-off
    magnet must be. It reads the strengths written to the MAD-X and MAD8 files for this quadrupole
    and for switched-off sextupole, solenoid and rfcavity elements; the Elegant export of the same
    quadrupole, which honours is_on, is the control.
    """

    import re

    from _bmad import compare, verdict
    from _s4_17_translations import export, tao
    from _translators import lattice, madx, numbers, statements, value

    BETA0, DRIFTS = 5.0, 2.5
    # A field-free 2.5 m line from a waist: beta = beta0 + L^2/beta0 = 6.25 m in both planes.
    EXPECTED = [BETA0 + DRIFTS**2 / BETA0] * 2

    quad = lattice(
        "d: drift, l = 1\nq: quadrupole, l = 0.5, k1 = 1.2, is_on = F",
        "d, q, d",
        beta=(BETA0, BETA0),
    )
    bmad = numbers(tao(quad, ["pipe ele:twiss 3"])["pipe ele:twiss 3"], ["beta_a", "beta_b"])
    if compare("Tao exit (beta_a, beta_b) m, expected 2.5 m drift", bmad, EXPECTED, atol=1e-12):
        raise SystemExit("Tao disagrees with the drift: comparison not valid.")

    defect = False
    deck = export(quad, "madx")
    with madx(deck) as mad:
        twiss = mad.twiss(beta0="initial")
        observed = [twiss.betx[-1], twiss.bety[-1]]
    defect |= compare("MAD-X TWISS exit (betx, bety) m of the export", observed, EXPECTED, atol=1e-12)

    # A switched-off element must be written without field strength (or as a drift).
    strengths = {"Q": "k1", "S": "k2", "SO": "ks", "RF": "volt"}
    off = lattice(
        "q: quadrupole, l = 0.5, k1 = 1.2, is_on = F\n"
        "s: sextupole, l = 0.2, k2 = 3, is_on = F\n"
        "so: solenoid, l = 0.3, ks = 0.4, is_on = F\n"
        "rf: rfcavity, l = 0.4, voltage = 1e6, rf_frequency = 5e8, is_on = F",
        "q, s, so, rf",
    )
    for fmt in ("madx", "mad8"):
        written = statements(export(off, fmt), fmt)
        for name, key in strengths.items():
            written_value = value(written[name], key, 0.0)
            print(f"{fmt}: off {name}: {written[name]}")
            defect |= compare(f"{fmt}: off {name}: written {key}, expected 0", written_value, 0, atol=0)

    lte = statements(export(quad, "elegant"), "elegant")["Q"]
    print(f"elegant control: Q: {lte}")
    if re.search(r"\bk1 =", lte):
        raise SystemExit("The Elegant export writes k1 for the off quadrupole: control not valid.")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | foreign-export-embedded-kicks (`5afc3c33a462`), source, production, CPU, double, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |

