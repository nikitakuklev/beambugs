---
title: "IO-09: MAD8 and SAD export ignore field-selected KQUAD and KSEXT strengths"
hide:
  - toc
---

# IO-09 — MAD8 and SAD export ignore field-selected KQUAD and KSEXT strengths

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/track_data.c:1253–1258](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1253-L1258)<br>[src/track_data.c:1151–1158](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1151-L1158)<br>[src/multipole.c:791–795](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L791-L795)<br>[src/multipole.c:852–863](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L852-L863)<br>[src/convert_to_mad8.c:64–68](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_mad8.c#L64-L68)<br>[src/convert_to_mad8.c:105–109](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_mad8.c#L105-L109)<br>[src/convert_to_mad8.c:253–257](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_mad8.c#L253-L257)<br>[src/convert_to_mad8.c:293–297](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/convert_to_mad8.c#L293-L297)<br>[src/compute_matrices.c:1647–1650](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1647-L1650)<br>[src/elegantto.c:284–294](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegantto.c#L284-L294)<br>[doc/elegant.tex:9587–9623](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L9587-L9623)<br>[src/elegantto.c:21–29](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegantto.c#L21-L29)<br>[src/elegantto.c:88–94](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegantto.c#L88-L94) |

Translating a KQUAD or KSEXT specified by pole-tip field B and bore radius BORE with elegantto's MAD8 or SAD output writes the inactive stored normalized strength instead of the selected field. With the default stored zero, the exported magnet has no quadrupole focusing or sextupole strength; a stale nonzero value can reverse the quadrupole's focusing sign. For electrons at momentum 1500 mc, B = 0.09000 T and BORE = 0.01200 m require normalized gradient K₁ = 2.933 m⁻², but MAD8 writes zero and changes R₂₁, the slope response to position, from −0.8418 m⁻¹ to zero. Native tracking and an independent rigidity calculation agree; equivalent explicit-strength controls export correctly.

Run `python IO-09.py` with [_diagnostics.py](../helpers/_diagnostics.py), [_elegant.py](../helpers/_elegant.py) and [_refmomentum.py](../helpers/_refmomentum.py) in the same directory.

??? example "Reproducer: IO-09.py"

    [Download IO-09.py](IO-09.py)

    ```python
    """IO-09: MAD8 and SAD export ignore field-selected KQUAD and KSEXT strengths

    Translating a KQUAD or KSEXT specified by pole-tip field B and bore radius BORE with elegantto's MAD8 or SAD output writes the inactive stored normalized strength instead of the selected field. With the default stored zero, the exported magnet has no quadrupole focusing or sextupole strength; a stale nonzero value can reverse the quadrupole's focusing sign. For electrons at momentum 1500 mc, B = 0.09000 T and BORE = 0.01200 m require normalized gradient K₁ = 2.933 m⁻², but MAD8 writes zero and changes R₂₁, the slope response to position, from −0.8418 m⁻¹ to zero. Native tracking and an independent rigidity calculation agree; equivalent explicit-strength controls export correctly.

    Setup: `elegantto` of the Elegant build on `PATH`; the script exports lattices with it.

    Derive quadrupole and sextupole strengths from electron magnetic rigidity at momentum 1500 mc, then compare native small-ray responses with MAD8 and SAD serialization. Track the MAD8 quadrupole representation and check equivalent explicit strengths, stale strengths and zero-field controls in fresh native processes.
    """

    import re
    import subprocess
    import tempfile
    from pathlib import Path

    from _diagnostics import conversion_defect
    from _elegant import verdict
    from _refmomentum import CHARGE, MASS, RAYS, C, close, quadrupole_slope, response, track


    def export(lattice, mode, order):
        with tempfile.TemporaryDirectory(prefix="elegant-export-repro-") as tmp:
            work = Path(tmp)
            (work / "lattice.lte").write_text(lattice + "\nUSE,BL\n")
            result = subprocess.run(
                ["elegantto", "lattice.lte", "output.txt", f"-{mode}"],
                cwd=work,
                capture_output=True,
                text=True,
                check=False,
                timeout=290,
            )
            log = result.stdout + result.stderr
            if result.returncode or re.search("warning|unsupported", log, re.IGNORECASE):
                raise RuntimeError(f"elegantto -{mode} failed or issued a diagnostic:\n{log}")
            text = (work / "output.txt").read_text()
            match = re.search(rf"\bK{order}=([^,\s)]+)", text)
            if match is None:
                raise RuntimeError(f"Missing K{order} in {mode} output")
            return float(match[1]), text


    def main():
        p0, length, bore, field = 1500.0, 0.3, 0.012, 0.09
        # B*rho = P/|q| = p0*m*c/|q|; K1 = (B/a)/(B*rho).
        rigidity = p0 * MASS * C / CHARGE
        k1 = field / (bore * rigidity)
        defect = False
        cases = (
            ("field/default", f"B={field},BORE={bore}", k1, 0.0),
            ("field/stale", f"B={field},BORE={bore},K1=-1.25", k1, -1.25),
            ("explicit control", f"K1={k1:.17g}", k1, k1),
            ("zero field control", f"B=0,BORE={bore}", 0.0, 0.0),
            ("zero field/stale", f"B=0,BORE={bore},K1=-1.25", 0.0, -1.25),
            ("zero bore control", f"B={field},BORE=0,K1=0.75", 0.75, 0.75),
        )
        for label, specification, physical, stored in cases:
            lattice = (
                f"Q: KQUAD,L={length},{specification},N_SLICES=96,INTEGRATION_ORDER=6,"
                "EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,SYNCH_RAD=0,ISR=0\nBL: LINE=(Q)\n"
            )
            native = response(track(lattice, p0, rays=RAYS)["particles"])
            expected = quadrupole_slope(physical, length)
            close(native, expected, rtol=2e-12, atol=1e-18)
            mad_k, text = export(lattice, "mad8", 1)
            sad_kl, _ = export(lattice, "sad", 1)
            defect |= conversion_defect(mad_k, physical, stored, rtol=4e-15)
            defect |= conversion_defect(sad_kl, physical * length, stored * length, rtol=4e-15)
            target = response(track(text, p0, rays=RAYS)["particles"])
            close(target, quadrupole_slope(mad_k, length), rtol=2e-12, atol=1e-18)
            print(
                f"{label}: native R21={native:.16g} expected={expected:.16g} m^-1; "
                f"MAD8 K1={mad_k:.16g} expected={physical:.16g} m^-2; "
                f"SAD K1L={sad_kl:.16g} expected={physical * length:.16g} m^-1; "
                f"target R21={target:.16g}; diagnostics=none "
                "(K1=(B/BORE)/(p0*m*c/|q|), R21=-sqrt(K1)*sin(sqrt(K1)*L))"
            )

        b2, a2, l2 = 0.003, 0.02, 0.2
        # B_y = (B/a^2)*x^2 and K2 = d^2(B_y)/dx^2 /(B*rho).
        k2 = 2 * b2 / (a2**2 * rigidity)
        for label, specification, stored in (
            ("sextupole/field", f"B={b2},BORE={a2}", 0.0),
            ("sextupole/explicit control", f"K2={k2:.17g}", k2),
        ):
            lattice = f"S: KSEXT,L={l2},{specification},N_SLICES=96,INTEGRATION_ORDER=6,SYNCH_RAD=0,ISR=0\nBL: LINE=(S)\n"
            native = response(track(lattice, p0, rays=RAYS)["particles"], order=2)
            close(native, -k2 * l2 / 2, rtol=2e-12)
            for mode, factor in (("mad8", 1), ("sad", l2)):
                observed, _ = export(lattice, mode, 2)
                defect |= conversion_defect(observed, k2 * factor, stored * factor, rtol=4e-15)
                print(
                    f"{label}/{mode}: K2{'L' if mode == 'sad' else ''}={observed:.16g} "
                    f"expected={k2 * factor:.16g}; native x^2 coefficient={native:.16g} "
                    f"expected={-k2 * l2 / 2:.16g} m^-2; diagnostics=none "
                    "(K2=2*B/(BORE^2*Brho), xprime coefficient=-K2*L/2)"
                )
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial elegant and elegantto, one thread |
| Confirmed<br><small class="bb-scope">MAD8/SAD and KSEXT, stale strengths and zero-field controls</small> | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial elegant and elegantto, one thread |

