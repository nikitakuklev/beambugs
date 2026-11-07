---
title: "TRK-39: CSBEND FSE_CORRECTION is applied only after a command computes the bend's matrix"
hide:
  - toc
---

# TRK-39 — CSBEND FSE_CORRECTION is applied only after a command computes the bend's matrix

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Source | [`compute_matrix`, compute_matrices.c:1793–1798](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1793-L1798)<br>[`track_through_csbend`, csbend.c:1111](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1111)<br>[csbend.c:1504–1506](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1504-L1506)<br>[csbend.c:6678–6700](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L6678-L6700)<br>[manual, csbend.tex:76–80](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L76-L80) |

CSBEND fits the FSE_CORRECTION strength adjustment only when a command such as twiss_output computes the bend's transfer matrix; a run that only tracks applies no correction and prints no warning. For an error-free 0.75 m, 0.09 rad bend with edge model 2 or 4, the corrected bend, obtained after twiss_output or by setting the fitted FSE by hand, has exit slope x′ = 0 and, with FSE_CORRECTION=1, the nominal path 0.75 m. Tracking alone gives x′ = 5.031×10⁻⁶ and a path 4.441×10⁻⁵ m shorter, so adding a twiss_output command changes the tracked orbit.

Run `python TRK-39.py` with [_csbend_fse.py](../helpers/_csbend_fse.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-39.py"

    [Download TRK-39.py](TRK-39.py)

    ```python
    """TRK-39: CSBEND FSE_CORRECTION is applied only after a command computes the bend's matrix

    CSBEND fits the FSE_CORRECTION strength adjustment only when a command such as twiss_output computes
    the bend's transfer matrix; a run that only tracks applies no correction and prints no warning. For
    an error-free 0.75 m, 0.09 rad bend with edge model 2 or 4, the corrected bend, obtained after
    twiss_output or by setting the fitted FSE by hand, has exit slope x′ = 0 and, with FSE_CORRECTION=1,
    the nominal path 0.75 m. Tracking alone gives x′ = 5.031×10⁻⁶ and a path 4.441×10⁻⁵ m shorter, so
    adding a twiss_output command changes the tracked orbit.

    The script tracks an on-axis electron through an error-free CSBEND with FSE_CORRECTION=1 and edge
    models 2 and 4, once in a tracking-only run and once after twiss_output. The expected exit slope is
    that of the bend with the fitted FSE set by hand and refitting off, and the expected path is the
    nominal length that FSE_CORRECTION=1 imposes. The run with twiss_output is the control.
    """

    import sys

    from _csbend_fse import LENGTH, track
    from _elegant import modes, verdict

    TOL = 1e-10  # the fitted FSE is printed to 6 digits; that moves the exit slope by below 1e-12 rad

    defect = []
    for mode in modes():
        for model in (2, 4):
            with_twiss, fitted = track(mode, model, {"FSE_CORRECTION": 1}, twiss=True)
            if fitted is None:
                sys.exit(f"{mode} model {model}: twiss_output did not fit the correction: invalid")
            # Expected slope: the fitted correction applied by hand, independent of the fitting call.
            expected = track(mode, model, {"FSE_CORRECTION": 0, "FSE": fitted}, twiss=False)[0]["xp"]
            only, refit = track(mode, model, {"FSE_CORRECTION": 1}, twiss=False)
            label = f"{mode} EDGE_EFFECTS={model}"
            print(f"{label}: fitted FSE {fitted:.6g}; expected exit xp {expected:.6g}, path {LENGTH} m")
            for name, run_ in (
                ("twiss_output then track (control)", with_twiss),
                ("tracking only", only),
            ):
                print(
                    f"{label} {name}: xp={run_['xp']:.10g} path={run_['path']:.13g}"
                    f" (diff {run_['xp'] - expected:.4g} rad, {run_['path'] - LENGTH:.4g} m)"
                )
            if abs(with_twiss["xp"] - expected) > TOL or abs(with_twiss["path"] - LENGTH) > 1e-12:
                sys.exit(f"{label}: the control disagrees: invalid")
            print(f"{label} tracking only: correction fitted={refit is not None}")
            defect.append(abs(only["xp"] - expected) > TOL and refit is None)
    print("comparison: abs tolerance 1e-10 rad on xp, 1e-12 m on path")
    verdict(bool(defect) and all(defect))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

