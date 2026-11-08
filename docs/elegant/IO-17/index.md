---
title: "IO-17: XORBIT MULT export ignores FACTOR and the selected pole-tip strength"
hide:
  - toc
---

# IO-17 — XORBIT MULT export ignores FACTOR and the selected pole-tip strength

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/convert_to_xorbit.c:86–90](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_xorbit.c#L86-L90)<br>[src/convert_to_xorbit.c:156–165](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_xorbit.c#L156-L165)<br>[src/track_data.c:506–514](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L506-L514)<br>[src/multipole.c:532–537](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L532-L537)<br>[doc/mult.tex:5–21](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/mult.tex#L5-L21)<br>[doc/elegant.tex:9580–9615](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L9580-L9615)<br>[src/elegantto.c:88–93](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegantto.c#L88-L93)<br>[src/elegantto.c:265–270](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegantto.c#L265-L270) |

Exporting a MULT with elegantto -xorbit uses the raw KNL value instead of its effective strength. FACTOR is omitted, and nonzero BORE does not select the documented BTIPL pole-tip strength. A thin quadrupole with KNL=2 m⁻¹ and FACTOR=3 exports 2 m⁻¹ instead of 6 m⁻¹; FACTOR=0 still exports an active magnet. At the converter's 1 GeV/c reference, BORE=0.02000 m and BTIPL=0.01000 T m require 0.1499 m⁻¹, but the writer exports the inactive KNL or zero. Analytic thin kicks and independent pole-tip normalization match native tracking and establish the exported-strength error.

Run `python IO-17.py` with [_misalign.py](../helpers/_misalign.py), [_slicing.py](../helpers/_slicing.py), [_elegant.py](../helpers/_elegant.py) and [_diagnostics.py](../helpers/_diagnostics.py) in the same directory.

??? example "Reproducer: IO-17.py"

    [Download IO-17.py](IO-17.py)

    ```python
    """IO-17: XORBIT MULT export ignores FACTOR and the selected pole-tip strength

    Exporting a MULT with elegantto -xorbit uses the raw KNL value instead of its effective strength. FACTOR is omitted, and nonzero BORE does not select the documented BTIPL pole-tip strength. A thin quadrupole with KNL=2 m⁻¹ and FACTOR=3 exports 2 m⁻¹ instead of 6 m⁻¹; FACTOR=0 still exports an active magnet. At the converter's 1 GeV/c reference, BORE=0.02000 m and BTIPL=0.01000 T m require 0.1499 m⁻¹, but the writer exports the inactive KNL or zero. Analytic thin kicks and independent pole-tip normalization match native tracking and establish the exported-strength error.

    Setup: `elegantto` of the Elegant build on `PATH`; the script exports lattices with it.

    Export seven thin normal quadrupoles with scaled, folded and pole-tip strengths, each followed by an inert MARK. Recover the effective strength independently from native canonical kicks and compare it with the emitted normal/skew coefficients. The MARK avoids the separate terminal-MULT converter crash.
    """

    import math
    import re

    from _misalign import export
    from _slicing import close, discrepancy, track

    # isort: split
    import _elegant as native
    import numpy as np

    # SDDS pin constants, including elegantto's default p = 1 GeV/c in electron mc units.
    P0 = 1000 / 0.51099906
    RIGIDITY = P0 * 9.1093897e-31 * 299792458.0 / 1.60217733e-19
    POLE = 0.01 / (0.02 * RIGIDITY)


    def main():
        faults = []
        cases = (
            ("unit control", "KNL=2,FACTOR=1", 2.0, 2.0),
            ("factor three", "KNL=2,FACTOR=3", 6.0, 2.0),
            ("factor zero", "KNL=2,FACTOR=0", 0.0, 2.0),
            ("factor negative", "KNL=2,FACTOR=-1", -2.0, 2.0),
            ("folded control", "KNL=6,FACTOR=1", 6.0, 6.0),
            ("pole/stale KNL", "KNL=2,BORE=0.02,BTIPL=0.01,FACTOR=1", POLE, 2.0),
            ("pole/zero KNL", "KNL=0,BORE=0.02,BTIPL=0.01,FACTOR=1", POLE, 0.0),
        )
        print(
            f"pole normalization: p0={P0:.17g} mc, Brho={RIGIDITY:.17g} T m; expected K1L={POLE:.17g} m^-1 (BTIPL/(BORE*Brho))"
        )
        for label, params, effective, raw in cases:
            lattice = f"B:MULT,L=0,ORDER=1,{params}\nM:MARK\nBL:LINE=(B,M)\n"
            text = export(lattice, "xorbit")
            lines = [line for line in text.splitlines() if line.startswith("B ")]
            if len(lines) != 1:
                raise RuntimeError(f"Expected one MULT record: {text}")
            match = re.fullmatch(r"B\s+n\s*([+-]?\d+\.\d+)\s*([+-]?\d+\.\d+)", lines[0])
            if match is None:
                raise RuntimeError(f"Unexpected MULT serialization: {lines[0]}")
            emitted = np.array([float(v) for v in match.groups()])
            # At order 1 and zero roll the normal coefficient is K1L, with no factorial ambiguity.
            faults.append(discrepancy(emitted, [effective, 0], [raw, 0], 5.01e-7))
            ray = np.zeros((1, 6))
            ray[0, 0] = 0.001
            out = track(label, lattice, ray, p0=P0)["q"][0]
            # Thin canonical kick qx=-K1L*x; SDDS xp=qx/sqrt(1-qx^2).
            measured = -out[1] / (ray[0, 0] * math.sqrt(1 + out[1] ** 2))
            close(measured, effective, 3e-14, "native thin canonical kick")
            print(
                f"{label}: exported normal={emitted[0]:.6f}, skew={emitted[1]:.6f}; expected normal={effective:.17g}, skew=0; native K1L={measured:.17g} m^-1 (inverse thin canonical kick); diagnostics=none"
            )
        native.verdict(any(faults))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-06</span> | master (`0355dd6aca12`), source inspection |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

