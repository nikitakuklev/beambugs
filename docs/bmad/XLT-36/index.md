---
title: "XLT-36: Energy-time coordinate discussion contains three spelling errors"
hide:
  - toc
---

# XLT-36 — Energy-time coordinate discussion contains three spelling errors

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P5 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, committed 2026-10-31) |
| Source | [manual passage](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/phase-space.tex#L227-L243) |

The manual’s discussion of MAD and Bmad energy-time coordinates contains the words “keept”, “cocordinates” and “differnt”. Their correct spellings are “kept”, “coordinates” and “different”. A check of the three sentences against those spellings identifies the errors directly.

Run `python XLT-36.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: XLT-36.py"

    [Download XLT-36.py](XLT-36.py)

    ```python
    """XLT-36: Energy-time coordinate discussion contains three spelling errors

    The manual’s discussion of MAD and Bmad energy-time coordinates contains the words “keept”, “cocordinates” and “differnt”. Their correct spellings are “kept”, “coordinates” and “different”. A check of the three sentences against those spellings identifies the errors directly.

    The script reads the coordinate discussion from BMAD_SOURCE and compares each word in its sentence with the correct spelling. Each sentence must contain either the recorded error or its corrected form, so changed context fails explicitly.
    """

    from _bmad import source, verdict

    text = (source() / "bmad/doc/phase-space.tex").read_text()
    marker = r"\section{MAD and Bmad Energy-Time Phase Space Coordinates}"
    if text.count(marker) != 1:
        raise RuntimeError("Missing unique coordinate section")
    section = text.split(marker)[1].split(r"\section{")[0]
    errors = 0
    for before, after, typo, expected in [
        ("must be ", " in mind", "keept", "kept"),
        ("phase space ", " in cases", "cocordinates", "coordinates"),
        (r"\bmad uses a ", " energy coordinate", "differnt", "different"),
    ]:
        incorrect = before + typo + after
        correct = before + expected + after
        counts = (section.count(incorrect), section.count(correct))
        if counts not in [(1, 0), (0, 1)]:
            raise RuntimeError(f"Changed spelling context for {expected}: {counts}")
        observed = typo if counts[0] else expected
        print(f"coordinate prose: observed={observed}; expected={expected}")
        errors += counts[0]
    print(f"spelling errors: observed={errors}; expected=0")
    verdict(errors > 0)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`) |

