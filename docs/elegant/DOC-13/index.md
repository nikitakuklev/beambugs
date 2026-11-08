---
title: "DOC-13: SHRFDF phase parameters are labelled Hz but consumed as radians"
hide:
  - toc
---

# DOC-13 — SHRFDF phase parameters are labelled Hz but consumed as radians

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P5 |
| Failure | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [src/track_data.c:3954–3979](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3954-L3979)<br>[src/shrfdf.c:65–90](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/shrfdf.c#L65-L90)<br>[doc/shrfdf.tex:1–31](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/shrfdf.tex#L1-L31) |

SHRFDF labels PHASE0 through PHASE9 in Hz although they are angular offsets in radians. Each phase is added directly to the harmonic argument before taking its sine or cosine. Independent scans of all ten modes show 2π periodicity and a crest at π/2; an input of 90 gives 0.8940 of crest rather than the degree-based crest. The generated parameter dictionary repeats the incorrect unit, while PERIOD_PHASE is correctly labelled RAD. Tracking already uses radians consistently.

Run `python DOC-13.py` with [_harmonic_rf.py](../helpers/_harmonic_rf.py), [_slicing.py](../helpers/_slicing.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: DOC-13.py"

    [Download DOC-13.py](DOC-13.py)

    ```python
    """DOC-13: SHRFDF phase parameters are labelled Hz but consumed as radians

    SHRFDF labels PHASE0 through PHASE9 in Hz although they are angular offsets in radians. Each phase is added directly to the harmonic argument before taking its sine or cosine. Independent scans of all ten modes show 2π periodicity and a crest at π/2; an input of 90 gives 0.8940 of crest rather than the degree-based crest. The generated parameter dictionary repeats the incorrect unit, while PERIOD_PHASE is correctly labelled RAD. Tracking already uses radians consistently.

    Generate the native parameter dictionary and scan every harmonic at four input phases with on-axis zero-time rays. Compare crest-normalized canonical kicks with sin(phase), including a 2π shift and input 90, and check PERIOD_PHASE as a correctly labelled control.
    """

    import re
    import tempfile
    from pathlib import Path

    from _harmonic_rf import kick
    from _slicing import close

    # isort: split
    import _elegant as native
    import numpy as np


    def dictionary():
        # The generated ASCII dictionary has heterogeneous pages; inspect its SHRFDF block directly.
        with tempfile.TemporaryDirectory(prefix="harmonic-dictionary-") as tmp:
            path = Path(tmp) / "dictionary.sdds"
            native.run(
                {
                    "line.lte": "O:MARK\nBL:LINE=(O)\n",
                    "run.ele": (
                        "&run_setup lattice=line.lte,use_beamline=BL,p_central=3200 &end\n"
                        "&run_control n_steps=1 &end\n"
                        f'&print_dictionary filename="{path}",SDDS_form=1 &end\n'
                    ),
                },
                outputs=(),
            )
            block = (
                path.read_text()
                .split("!***** element type SHRFDF:", 1)[1]
                .split("!***** element type", 1)[0]
            )
            return dict(re.findall(r'^"([^"\n]+)" "([^"\n]*)"', block, flags=re.MULTILINE))


    def main():
        units = dictionary()
        if units["PERIOD_PHASE"] != "RAD":
            raise RuntimeError("PERIOD_PHASE control is not labelled RAD")
        faults = []
        phases = [np.pi / 2, -0.43, -0.43 + 2 * np.pi, 90.0]
        for mode in range(10):
            unit = units[f"PHASE{mode}"]
            if unit not in ("HZ", "RAD"):
                raise RuntimeError(f"Unexpected PHASE{mode} unit: {unit}")
            values = [float(kick(mode, p, np.zeros((1, 6)))[0]) for p in phases]
            ratios = np.array(values) / values[0]
            expected = np.sin(phases)
            close(ratios, expected, 2e-15, "radian sine law")
            faults.append(unit == "HZ")
            print(
                f"PHASE{mode}: dictionary={unit}, expected=RAD; normalized kicks={ratios.tolist()}, expected={expected.tolist()} (sin of input in radians)"
            )
        print("PERIOD_PHASE: dictionary=RAD, expected=RAD (angular-offset control)")
        native.verdict(any(faults))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

