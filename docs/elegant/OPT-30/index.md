---
title: "OPT-30: slice_analysis with final_values_only=1 writes no data and leaves its optimization symbols at zero"
hide:
  - toc
---

# OPT-30 — slice_analysis with final_values_only=1 writes no data and leaves its optimization symbols at zero

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`do_tracking`, per-element slice-analysis call, do_tracking.c:2895–2911](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2895-L2911)<br>[the end-of-beamline call after the element loop, do_tracking.c:3071–3090](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L3071-L3090)<br>[the tracking context passed to elements, do_tracking.c:1194](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L1194)<br>[`performSliceAnalysisOutput`, sliceAnalysis.c:394–484](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sliceAnalysis.c#L394-L484)<br>[`setupSliceAnalysis`, sliceAnalysis.c:168–175](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sliceAnalysis.c#L168-L175)<br>[`slice_analysis` places the final slice analysis at the end of the beamline in symbols for optimization equations, named as the output columns, elegant.tex:7566–7570](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7566-L7570)<br>[with `final_values_only` nonzero the slice quantities are computed only at the end of the beamline, elegant.tex:7593–7594](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7593-L7594)<br>[`slice_analysis not supported for parallel version`, elegant.c:2342–2345](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L2342-L2345) |

Elegant's `slice_analysis` with `final_values_only=1`, which the manual says computes the slice quantities only at the end of the beamline, never computes them in serial tracking: the run succeeds, but the output file holds no data page and the optimization symbols named after its columns stay at zero. For 128 particles with mean x = 2 mm and x′ = 0.4 mrad through drifts of 0.7 and 1.6 m, the drift map gives the final centroid Cx = 2.920 mm, which the default `final_values_only=0` writes and stores at the end of the line. With `final_values_only=1` the file has no rows, and the symbols Cx and particles read 0 instead of 2.920 mm and 128.

Run `python OPT-30.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: OPT-30.py"

    [Download OPT-30.py](OPT-30.py)

    ```python
    """OPT-30: slice_analysis with final_values_only=1 writes no data and leaves its optimization symbols at zero

    Elegant's `slice_analysis` with `final_values_only=1`, which the manual says computes the slice
    quantities only at the end of the beamline, never computes them in serial tracking: the run
    succeeds, but the output file holds no data page and the optimization symbols named after its
    columns stay at zero. For 128 particles with mean x = 2 mm and x′ = 0.4 mrad through drifts of 0.7
    and 1.6 m, the drift map gives the final centroid Cx = 2.920 mm, which the default
    `final_values_only=0` writes and stores at the end of the line. With `final_values_only=1` the file
    has no rows, and the symbols Cx and particles read 0 instead of 2.920 mm and 128.

    The script tracks a fixed beam of 128 particles through two drifts in serial Elegant with four
    slices, once with `final_values_only=0` as the control and once with `final_values_only=1`, and
    lists the RPN memories after tracking with `&rpn_expression expression=smem`. It compares the
    number of data pages, the whole-beam centroid Cx of the last row and the stored symbols Cx and
    particles with the drift-map value and the particle count. The tracked output beam of each run is
    a further control that the beam itself reaches the end of the line.
    """

    import re
    import sys

    import numpy as np
    from _elegant import run, verdict

    N = 128
    L1, L2 = 0.7, 1.6  # m, drift lengths
    X0, XP = 2e-3, 4e-4  # m and rad: mean x and the common slope of every particle
    # A drift maps x to x + L x', so the final centroid is X0 + (L1 + L2) XP = 2.92e-3 m.
    CX = X0 + (L1 + L2) * XP
    RTOL = 1e-12

    BEAM = {
        "x": np.linspace(X0 - 1e-3, X0 + 1e-3, N),  # mean X0 exactly (symmetric about it)
        "xp": np.full(N, XP),
        "y": 2e-4 * np.sin(np.arange(N)),
        "yp": 3e-5 * np.cos(np.arange(N)),
        "t": np.linspace(-1e-12, 1e-12, N),  # spread in time so that the four slices are populated
        "p": np.full(N, 3000.0),
    }
    LATTICE = f"D1: DRIF,L={L1!r}\nD2: DRIF,L={L2!r}\nBL: LINE=(D1,D2)\n"


    def commands(final_only):
        return (
            "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=3000, output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            f"&slice_analysis output=slices.sdds, n_slices=4, final_values_only={final_only} &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
            "&rpn_expression expression=smem &end\n"
        )


    def memory(log, name):
        """Value of an RPN memory in the `smem` listing (lines `name<TAB>value`)."""
        found = re.search(rf"^{re.escape(name)}\s+(\S+)$", log, re.MULTILINE)
        if found is None:
            sys.exit(f"RPN memory {name} missing from the smem listing: invalid run.")
        return float(found.group(1))


    def close(a, b):
        return bool(np.isclose(a, b, rtol=RTOL, atol=0))


    defect = False
    print(
        f"expected: one data page whose last row has s = {L1 + L2:.4g} m and Cx = {CX:.16g} m;"
        f" symbols Cx = {CX:.16g} m, particles = {N} (drift map x + L x', beam size)"
    )
    # Serial Elegant only: Pelegant does not run slice_analysis.
    for final_only in (0, 1):
        label = f"serial final_values_only={final_only}"
        result = run(
            {"lattice.lte": LATTICE, "run.ele": commands(final_only)},
            "serial",
            {"beam.sdds": BEAM},
            outputs=("slices.sdds", "out.sdds"),
        )
        out = result["out.sdds"]
        beam_cx = float(np.mean(out[0, "x"]))
        if len(out[0, "x"]) != N or not close(beam_cx, CX):
            sys.exit(f"{label}: tracked beam differs from the drift map: invalid run.")
        print(f"{label} output beam (control): particles={len(out[0, 'x'])} mean x={beam_cx:.16g}")
        slices = result["slices.sdds"]
        pages = slices.n_pages
        if pages:
            s, cx = np.asarray(slices[0, "s"]), np.asarray(slices[0, "Cx"])
            rows = f"rows={len(s)} last row s={s[-1]:.16g} Cx={cx[-1]:.16g}"
            written = pages == 1 and close(s[-1], L1 + L2) and close(cx[-1], CX)
        else:
            rows, written = "rows=0", False
        symbols = (memory(result["log"], "Cx"), memory(result["log"], "particles"))
        stored = close(symbols[0], CX) and symbols[1] == N
        print(f"{label} slice file: pages={pages} {rows}")
        print(f"{label} symbols: Cx={symbols[0]:.16g} particles={symbols[1]:.16g}")
        if final_only == 0 and not (written and stored):
            sys.exit(f"{label}: the control does not give the drift-map result: invalid run.")
        defect |= not (written and stored)
    print(f"comparison: rtol={RTOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

