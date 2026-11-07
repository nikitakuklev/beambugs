---
title: "XLT-16: MAD-X import drops pure-skew normal-magnet strengths"
hide:
  - toc
---

# XLT-16 — MAD-X import drops pure-skew normal-magnet strengths

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P1 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`parse_and_write_element`, util_programs/mad_to_bmad/madx_to_bmad.py:575–580](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/mad_to_bmad/madx_to_bmad.py#L575-L580)<br>[600–605](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/mad_to_bmad/madx_to_bmad.py#L600-L605)<br>[622–627](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/mad_to_bmad/madx_to_bmad.py#L622-L627) |

Importing a MAD-X quadrupole, sextupole or octupole whose only strength is k1s, k2s or k3s discards that strength after setting the skew roll. The imported magnet has zero field. A 0.7 m skew quadrupole with k1s = 1.3 m⁻² becomes a drift with no transverse coupling, while MAD-X and the analytic transfer matrix give R₁₃ = 0.3189 and R₄₁ = 0.9131 m⁻¹, using coordinates (x, pₓ, y, pᵧ) with momenta normalized to the reference momentum. Assigning k1 = 1.3 to the import restores that matrix. Specifying both normal and skew strengths selects a separate importer branch.

Run `python XLT-16.py` with [_bmad.py](../helpers/_bmad.py) and [_translators.py](../helpers/_translators.py) in the same directory.

??? example "Reproducer: XLT-16.py"

    [Download XLT-16.py](XLT-16.py)

    ```python
    """XLT-16: MAD-X import drops pure-skew normal-magnet strengths

    Importing a MAD-X quadrupole, sextupole or octupole whose only strength is k1s, k2s or k3s
    discards that strength after setting the skew roll. The imported magnet has zero field. A 0.7 m
    skew quadrupole with k1s = 1.3 m⁻² becomes a drift with no transverse coupling, while MAD-X and
    the analytic transfer matrix give R₁₃ = 0.3189 and R₄₁ = 0.9131 m⁻¹, using coordinates (x, pₓ,
    y, pᵧ) with momenta normalized to the reference momentum. Assigning k1 = 1.3 to the import
    restores that matrix. Specifying both normal and skew strengths selects a separate importer
    branch.

    The script translates with madx_to_bmad.py a 0.7 m MAD-X quadrupole, sextupole and octupole whose
    only strength is k1s, k2s or k3s = 1.3, and reads the imported length, tilt and normal strength with
    Tao. For the skew quadrupole it also compares the transverse transfer matrix of the imported element
    with MAD-X TWISS of the source and with the closed-form solution of the skew-quadrupole equations x″
    = K y, y″ = K x. A copy of the import with k1 = 1.3 added is the receiving-code control.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _translators import import_madx, madx, record, tao

    LENGTH, STRENGTH = 0.7, 1.3
    OPEN = "parameter[geometry] = open\nbeginning[beta_a] = 10\nbeginning[beta_b] = 10\n"


    def deck(kind, order):
        return (
            "beam, particle = electron, pc = 1;\n"
            f"q: {kind}, l = {LENGTH}, k{order}s = {STRENGTH};\n"
            "bl: line = (q);\nuse, period = bl;\n"
        )


    def skew_quad_matrix(k, length):
        """4x4 (x, px, y, py) matrix of x'' = k y, y'' = k x for one hard-edge length.

        u = (x + y)/sqrt(2) obeys u'' = k u (defocusing), v = (x - y)/sqrt(2) obeys v'' = -k v.
        """
        r = np.sqrt(k)
        defocus = np.array(
            [
                [np.cosh(r * length), np.sinh(r * length) / r],
                [r * np.sinh(r * length), np.cosh(r * length)],
            ]
        )
        focus = np.array(
            [
                [np.cos(r * length), np.sin(r * length) / r],
                [-r * np.sin(r * length), np.cos(r * length)],
            ]
        )
        plus, minus = (defocus + focus) / 2, (defocus - focus) / 2
        return np.block([[plus, minus], [minus, plus]])


    def bmad_matrix(text):
        lines = tao(text + OPEN, ["pipe ele:mat6 1 mat6"])["pipe ele:mat6 1 mat6"]
        rows = [[float(v) for v in s.split(";")[3:]] for s in lines if s.count(";") >= 8]
        return np.array(rows)[:4, :4]


    defect = False
    for order, kind in ((1, "quadrupole"), (2, "sextupole"), (3, "octupole")):
        text = import_madx(deck(kind, order))
        print(f"{kind}: imported: {next(s for s in text.splitlines() if s.startswith('q:'))}")
        attributes = record(tao(text + OPEN, ["pipe ele:gen_attribs 1"])["pipe ele:gen_attribs 1"])
        # The roll -pi/(2(n+1)) turns a normal 2(n+1)-pole into the MAD-X skew orientation.
        geometry = [float(attributes["L"]), float(attributes["TILT"])]
        if compare(
            f"{kind}: imported (L m, tilt rad)",
            geometry,
            [LENGTH, -np.pi / (2 * (order + 1))],
            atol=1e-15,
        ):
            raise SystemExit("The imported length or tilt is wrong: comparison not valid.")
        strength = float(attributes[f"K{order}"])
        defect |= compare(f"{kind}: imported K{order}, expected {STRENGTH}", strength, STRENGTH, atol=0)

    expected = skew_quad_matrix(STRENGTH, LENGTH)
    with madx(deck("quadrupole", 1)) as mad:
        twiss = mad.twiss(betx=10, bety=10, rmatrix=True)
        madx_matrix = np.array([[twiss[f"re{i}{j}"][-1] for j in range(1, 5)] for i in range(1, 5)])
    if compare("MAD-X skew quadrupole matrix, expected closed form", madx_matrix, expected, atol=1e-14):
        raise SystemExit("MAD-X disagrees with the closed form: comparison not valid.")
    text = import_madx(deck("quadrupole", 1))
    defect |= compare(
        "imported skew quadrupole matrix, expected closed form", bmad_matrix(text), expected, atol=1e-14
    )
    # Keep a valid strength-plus-roll control even after an upstream importer fix.
    definition = next(s for s in text.splitlines() if s.startswith("q:"))
    fixed = text.replace(definition, f"q: quadrupole, l = {LENGTH}, k1 = {STRENGTH}, tilt = -pi/4")
    if compare(
        "import with k1 = 1.3 added, expected closed form", bmad_matrix(fixed), expected, atol=1e-14
    ):
        raise SystemExit("The k1 = 1.3 control disagrees: comparison not valid.")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

