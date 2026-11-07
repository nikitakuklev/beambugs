---
title: "TRK-38: CSBEND FSE_CORRECTION cancels a dipole-only strength error and a horizontal steering kick"
hide:
  - toc
---

# TRK-38 — CSBEND FSE_CORRECTION cancels a dipole-only strength error and a horizontal steering kick

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Source | [`csbend_update_fse_adjustment`, csbend.c:6678–6700](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L6678-L6700)<br>[`csbend_fse_adjustment_penalty`, csbend.c:6661–6674](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L6661-L6674)<br>[`track_through_csbend`, csbend.c:1111–1116](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1111-L1116)<br>[manual, csbend.tex:76–80](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L76-L80)<br>[parameter table, track_data.c:1961–1962](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1961-L1962)<br>[FSE_CORRECTION, track_data.c:2017](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2017) |

When a matrix calculation fits CSBEND's FSE_CORRECTION=1 or 2 for edge model 2 or 4, the calibration retains FSE_DIPOLE and XKICK and absorbs their horizontal exit-slope effects. For a 0.75 m, 0.09 rad bend, adding FSE_DIPOLE = 7×10⁻⁴ to the bend with its error-free correction held fixed gives exit slope x′ = −6.287×10⁻⁵, as the same error given as FSE does with refitting, but the refitted FSE_DIPOLE run gives |x′| < 10⁻¹⁵. XKICK = 8×10⁻⁵ should give x′ = 7.984×10⁻⁵ and also gives |x′| < 10⁻¹⁵. The vertical deflection of YKICK is kept.

Run `python TRK-38.py` with [_csbend_fse.py](../helpers/_csbend_fse.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-38.py"

    [Download TRK-38.py](TRK-38.py)

    ```python
    """TRK-38: CSBEND FSE_CORRECTION cancels a dipole-only strength error and a horizontal steering kick

    When a matrix calculation fits CSBEND's FSE_CORRECTION=1 or 2 for edge model 2 or 4, the
    calibration retains FSE_DIPOLE and XKICK and absorbs their horizontal exit-slope effects. For a
    0.75 m, 0.09 rad bend, adding FSE_DIPOLE = 7×10⁻⁴ to the bend with its error-free correction
    held fixed gives exit slope x′ = −6.287×10⁻⁵, as the same error given as FSE does with
    refitting, but the refitted FSE_DIPOLE run gives |x′| < 10⁻¹⁵. XKICK = 8×10⁻⁵ should give x′ =
    7.984×10⁻⁵ and also gives |x′| < 10⁻¹⁵. The vertical deflection of YKICK is kept.

    The script fits the correction once for the error-free bend (edge models 2 and 4, twiss_output
    before tracking) and reads the fitted FSE from the log. With refitting off and that FSE held fixed,
    it adds FSE, FSE_DIPOLE, XKICK or YKICK and tracks an on-axis electron: these are the expected exit
    slopes. It then tracks the same errors with FSE_CORRECTION=1, which refits with the error present.
    FSE and YKICK are the controls; FSE_DIPOLE and XKICK show the defect.
    """

    import sys

    from _csbend_fse import track
    from _elegant import modes, verdict

    # The fitted FSE is printed with 6 significant digits (about 5e-12 absolute here), which moves the
    # fixed-correction exit slope by below 1e-12 rad; the defects are near 6e-5 rad.
    TOL = 1e-10
    ERRORS = (
        ("FSE", 7e-4, "xp"),
        ("FSE_DIPOLE", 7e-4, "xp"),
        ("XKICK", 8e-5, "xp"),
        ("YKICK", 8e-5, "yp"),
    )

    defect = []
    for mode in modes():
        for model in (2, 4):
            calibration, fitted = track(mode, model, {"FSE_CORRECTION": 1}, twiss=True)
            if fitted is None or abs(calibration["xp"]) > 1e-14:
                sys.exit(f"{mode} model {model}: the error-free correction was not fitted: invalid")
            print(
                f"{mode} EDGE_EFFECTS={model}: error-free fitted FSE {fitted:.6g}, exit xp {calibration['xp']:.3g}"
            )
            for name, value, column in ERRORS:
                # Expected: the error added to the bend whose ideal edge correction is held fixed, so the
                # correction compensates only the edges, as documented.
                fixed = {"FSE_CORRECTION": 0, "FSE": fitted + (value if name == "FSE" else 0.0)}
                if name != "FSE":
                    fixed[name] = value
                expected = track(mode, model, fixed, twiss=True)[0][column]
                observed, refit = track(mode, model, {"FSE_CORRECTION": 1, name: value}, twiss=True)
                if refit is None:
                    sys.exit(f"{mode} model {model} {name}: the correction was not refitted: invalid")
                diff = observed[column] - expected
                print(
                    f"{mode} EDGE_EFFECTS={model} {name}={value:g} exit {column}: observed={observed[column]:.10g}"
                    f" expected={expected:.10g} diff={diff:.4g}"
                )
                if name in ("FSE", "YKICK"):
                    if abs(diff) > TOL:
                        sys.exit(f"{mode} model {model} {name}: the control disagrees: invalid")
                else:
                    defect.append(abs(diff) > TOL)
    print(f"comparison: abs tolerance {TOL:g} rad")
    verdict(bool(defect) and all(defect))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed<br><small>FSE_DIPOLE and XKICK; not YKICK</small> | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

