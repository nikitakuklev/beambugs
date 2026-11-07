---
title: "XLT-33: Elegant export leaves later copies of a split multipole undefined"
hide:
  - toc
---

# XLT-33 — Elegant export leaves later copies of a split multipole undefined

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">experimental</span> [?](../../methodology.md#tags) |
| Source | [`write_lattice_elegant_format`, duplicate-name skip](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_elegant_format.f90#L485-L488)<br>[multipole splitting](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_elegant_format.f90#L678-L698) |

An Elegant export becomes invalid when a multipole with multiple nonzero orders appears more than once. The first occurrence is split into one MULT per order, but later occurrences retain an original name that has no definition, so Elegant rejects the lattice. The same failure occurs for the paired end multipoles generated from a thick element’s field errors. Replacing the later reference with the split elements makes the repeated thin-multipole example agree with Bmad tracking.

Run `python XLT-33.py` with [_bmad.py](../helpers/_bmad.py), [_s3_24_translations.py](../helpers/_s3_24_translations.py) and [_s4_17_translations.py](../helpers/_s4_17_translations.py) in the same directory.

??? example "Reproducer: XLT-33.py"

    [Download XLT-33.py](XLT-33.py)

    ```python
    """XLT-33: Elegant export leaves later copies of a split multipole undefined

    An Elegant export becomes invalid when a multipole with multiple nonzero orders appears more than once. The first occurrence is split into one MULT per order, but later occurrences retain an original name that has no definition, so Elegant rejects the lattice. The same failure occurs for the paired end multipoles generated from a thick element’s field errors. Replacing the later reference with the split elements makes the repeated thin-multipole example agree with Bmad tracking.

    Setup: `ELEGANT_BIN` set to a serial Elegant executable and `RPN_DEFNS` to the `defns.rpn` file of SDDS; the script runs the exported lattices in Elegant.

    The script exports a repeated thin multipole and a quadrupole with two orders of field error. It verifies Elegant’s exact undefined-name error and checks a repaired thin-multipole beamline against Bmad.
    """

    import re

    from _bmad import compare, verdict
    from _s3_24_translations import elegant_track, lattice, undefined_names
    from _s4_17_translations import export, track

    bad = False
    for label, lat, name in [
        ("repeated thin", lattice("multipole,k1l=.1,k2l=.4", "d:drift,l=1", "m,d,m"), "M"),
        (
            "thick field errors",
            lattice("quadrupole,l=.5,k1=.3,b2=.4,b3=5,scale_multipoles=F"),
            "QMULTIPOLE_1",
        ),
    ]:
        text = export(lat, "elegant")
        missing, line = undefined_names(text)
        print(f"{label}: {line}; undefined={missing}; expected=[]")
        if missing:
            if missing != [name]:
                raise RuntimeError("Unexpected undefined names")
            actual = elegant_track(text, rejected=name)
            bad |= actual is None
        else:
            elegant_track(text)
        if label == "repeated thin":
            # Define the original name as a line of the already exported components.
            # Every later reference now uses exactly the first occurrence's definition.
            components = re.findall(r"(?m)^(M__\d+):", text)
            if components != ["M__1", "M__2"]:
                raise RuntimeError("Unexpected first-occurrence split")
            fixed = (
                text
                if not missing
                else text.replace("line_1:", f"M: line=({','.join(components)})\nline_1:")
            )
            result = elegant_track(fixed)
            native, _ = track(lat)
            if result is None or compare(
                "repaired repeated-multipole orbit (m,normalized momenta)",
                result,
                native["ORBIT"][:4],
                atol=3e-15,
            ):
                raise RuntimeError("Repeated-multipole receiving-code control failed")
    verdict(bad)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed<br><small>undefined names for repeated thin multipoles</small> | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

