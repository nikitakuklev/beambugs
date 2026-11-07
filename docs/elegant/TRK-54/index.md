---
title: "TRK-54: Finite exact correctors add false displacement at weak nonzero kick"
hide:
  - toc
---

# TRK-54 — Finite exact correctors add false displacement at weak nonzero kick

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant master (`0355dd6`, committed 2026-11-03) |
| Source | [src/exactCorrector.c:101–208](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/exactCorrector.c#L101-L208)<br>[src/exactCorrector.c:241–257](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/exactCorrector.c#L241-L257)<br>[element descriptions](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L333-L335)<br>[EHKICK manual](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ehkick.tex#L1-L3) |

Finite-length EHKICK, EVKICK and EKICKER can add false transverse displacement at small nonzero steering angles. For a 0.4 m element with 100 µrad incoming slope and momentum 1% above a 3 GeV/c reference, a 50 prad kick produces a 367.7 nm position error. Independent uniform-field quadrature gives the expected trajectory, and zero and 1 mrad kicks pass. Fifty picoradians is a published fast-corrector resolution; zero-length kicks avoid the displacement subtraction.

Run `python TRK-54.py` with [_precision.py](../helpers/_precision.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-54.py"

    [Download TRK-54.py](TRK-54.py)

    ```python
    """TRK-54: Finite exact correctors add false displacement at weak nonzero kick

    Finite-length EHKICK, EVKICK and EKICKER can add false transverse displacement at small nonzero steering angles. For a 0.4 m element with 100 µrad incoming slope and momentum 1% above a 3 GeV/c reference, a 50 prad kick produces a 367.7 nm position error. Independent uniform-field quadrature gives the expected trajectory, and zero and 1 mrad kicks pass. Fifty picoradians is a published fast-corrector resolution; zero-length kicks avoid the displacement subtraction.

    Track fixed planar rays through EHKICK, EVKICK and EKICKER with both signs of a 50 prad kick. Compare displacement with 70-digit uniform-field quadrature, checking the final slope, zero and 1 mrad strengths, an axial ray and a zero-length control.
    """

    from _precision import PIN, corrector, native, position_line, require_close, verdict


    def element(kind, kick, length):
        strength = f"HKICK={kick:.17g},VKICK=0" if kind == "EKICKER" else f"KICK={kick:.17g}"
        return f"{kind},L={length},{strength},ISR=0,SYNCH_RAD=0"


    def main():
        defect = False
        for kind in ("EHKICK", "EVKICK", "EKICKER"):
            pos, slope = ("y", "yp") if kind == "EVKICK" else ("x", "xp")
            for kick in (0.0, 1e-3, 5e-11, -5e-11):
                out, _ = native(element(kind, kick, 0.4), [{pos: 0.001, slope: 0.0001, "p": PIN}])
                expected, final_slope = corrector(kick)
                error = position_line(
                    f"{kind}/kick={kick:+.0e}",
                    out[pos][0],
                    expected,
                    "70-digit integral of u/sqrt(1-u^2), u linear in z",
                    coordinate=pos,
                )
                require_close(out[slope][0], final_slope, 1e-17)
                if kick == 0 or kick == 1e-3:
                    require_close(out[pos][0], expected, 5e-14)
                else:
                    defect |= error > 1e-12
        out, _ = native(element("EHKICK", 1e-8, 0.4), [{"p": PIN}])
        expected, final_slope = corrector(1e-8, x=0.0, xp=0.0)
        error = position_line(
            "EHKICK/axial/kick=+1e-08", out["x"][0], expected, "70-digit uniform-field integral"
        )
        require_close(out["xp"][0], final_slope, 1e-17)
        defect |= error > 1e-12
        for kind in ("EHKICK", "EVKICK", "EKICKER"):
            pos, slope = ("y", "yp") if kind == "EVKICK" else ("x", "xp")
            out, _ = native(element(kind, 5e-11, 0.0), [{pos: 0.001, slope: 0.0001, "p": PIN}])
            expected, final_slope = corrector(5e-11, length=0.0)
            position_line(
                f"{kind}/thin/kick=+5e-11", out[pos][0], expected, "zero-length position invariant"
            )
            require_close(out[pos][0], expected, 1e-18)
            require_close(out[slope][0], final_slope, 1e-17)
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial |

