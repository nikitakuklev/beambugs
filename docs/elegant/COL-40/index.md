---
title: "COL-40: CSRCSBEND with IGF=1 and BIN_SIZE writes past its Green-function arrays when the bunch lengthens"
hide:
  - toc
---

# COL-40 — CSRCSBEND with IGF=1 and BIN_SIZE writes past its Green-function arrays when the bunch lengthens

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Source | [rebinning, csbend.c:3286–3294](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3286-L3294)<br>[histogram and growth, csbend.c:3338–3353](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3338-L3353)<br>[IGF allocation, csbend.c:3443–3453](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3443-L3453)<br>[IGF fill, csbend.c:3456–3471](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3456-L3471)<br>[BIN_SIZE, track_data.c:2059](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2059)<br>[IGF, track_data.c:2071](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2071) |

With IGF=1 and BIN_SIZE, CSRCSBEND sizes its integrated-Green-function arrays for the bin count of its first slice and fills them up to the current bin count in every later slice. When a chirped bunch lengthens in the bend, the bin count grows and these writes run past the allocation. For 5001 electrons whose histogram grows from 89 to 2912 bins over the 24 slices of a 0.8 m, 0.4 rad bend, the run ends with a segmentation fault, and Valgrind reports an invalid write just past the 89-element array. The same bend with IGF=0, or with IGF=1 and a fixed bin count, completes.

Run `python COL-40.py` with [_csr.py](../helpers/_csr.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: COL-40.py"

    [Download COL-40.py](COL-40.py)

    ```python
    """COL-40: CSRCSBEND with IGF=1 and BIN_SIZE writes past its Green-function arrays when the bunch lengthens

    With IGF=1 and BIN_SIZE, CSRCSBEND sizes its integrated-Green-function arrays for the bin count of its first slice and fills them up to the current bin count in every later slice. When a chirped bunch lengthens in the bend, the bin count grows and these writes run past the allocation. For 5001 electrons whose histogram grows from 89 to 2912 bins over the 24 slices of a 0.8 m, 0.4 rad bend, the run ends with a segmentation fault, and Valgrind reports an invalid write just past the 89-element array. The same bend with IGF=0, or with IGF=1 and a fixed bin count, completes.

    Setup: `valgrind` on `PATH`; the script runs some cases of serial Elegant under Memcheck.

    The script tracks 5001 electrons (Gaussian quantiles, σ_ct = 12 μm, linear chirp of 1500 m⁻¹ in
    relative momentum about p = 1500 mₑc) through a 0.8 m, 0.4 rad, 24-slice CSRCSBEND with
    STEADY_STATE=1 and BIN_SIZE = 4 fs, which lengthens the bunch from slice to slice. The control with
    IGF=0 runs, and its wake output gives the bin count of every slice; with fixed BINS=128 and IGF=1
    the run also completes. With IGF=1 and BIN_SIZE, the script reports the exit of a plain run and the
    first invalid access Valgrind finds, with the allocation it hits.
    """

    import re

    from _csr import final, gaussian_beam, page, status, track, valgrind
    from _elegant import RunFailure, verdict

    P0 = 1500.0
    BEAM = gaussian_beam(5001, 12e-6, P0, chirp=1500.0)
    BEND = "L=0.8,ANGLE=0.4,N_SLICES=24,BINS=128,STEADY_STATE=1,SG_HALFWIDTH=1"
    CHARGE = "Q: CHARGE,TOTAL=1e-18\n"  # a weak bunch: the CSR kicks do not change the binning

    # Control: IGF=0 with BIN_SIZE completes; its wake file has one page per slice, whose row count is
    # that slice's bin count.
    control = track(
        CHARGE + f'B: CSRCSBEND,{BEND},IGF=0,BIN_SIZE=4e-15,OUTPUT_FILE="wake.sdds"',
        "Q,B",
        BEAM,
        P0,
        outputs=("out.sdds", "wake.sdds"),
    )
    if len(final(control)["p"]) != len(BEAM["p"]):
        raise SystemExit("The IGF=0 control lost particles.")
    wake = control["wake.sdds"]
    bins = [len(page(wake, i)[0]["DeltaGamma"]) for i in range(wake.n_pages)]
    print(
        f"serial IGF=0, BIN_SIZE=4e-15 (control): completed; bins per slice {bins[:4]} ... {bins[-1]}"
    )
    if not bins[-1] > bins[0]:
        raise SystemExit("The bunch does not lengthen in the control.")
    fixed = status(CHARGE + f"B: CSRCSBEND,{BEND},IGF=1", "Q,B", BEAM, P0)
    print(f"serial IGF=1, BINS=128 (control): exit status observed={fixed} expected=0")
    if fixed:
        raise SystemExit("The fixed-bin IGF control fails.")

    # The case: IGF=1 with BIN_SIZE. The Green-function arrays must follow the bin count; any
    # invalid access or abnormal exit is the defect.
    LATTICE = CHARGE + f"B: CSRCSBEND,{BEND},IGF=1,BIN_SIZE=4e-15"
    exit_status, log = 0, ""
    try:
        track(LATTICE, "Q,B", BEAM, P0, outputs=())
    except RunFailure as failure:  # keep the status and Elegant's termination message
        exit_status, log = failure.status, failure.log
    terminated = re.findall(r"Terminated by \S+", log)
    print(
        f"serial IGF=1, BIN_SIZE=4e-15: exit status observed={exit_status} "
        f"({terminated[0] if terminated else 'no signal'}) expected=0"
    )
    _, vlog = valgrind(LATTICE, "Q,B", BEAM, P0)
    first = re.search(
        r"(Invalid (?:write|read) of size \d+)\n==\d+==\s+at [^(]*\(([^)]*)\).*?\n"
        r"(?:==\d+==\s+by .*\n)*==\d+==\s+Address \S+ is (\d+ bytes after a block of size \d+) alloc'd\n"
        r"==\d+==\s+at .*\n==\d+==\s+by [^(]*\(([^)]*)\)",
        vlog,
    )
    if first:
        access, where, offset, allocated = first.groups()
        print(
            f"serial IGF=1, BIN_SIZE=4e-15 under Valgrind: first invalid access {access} at {where}, "
            f"{offset} allocated at {allocated} (= {int(offset.split()[-1]) // 8} doubles); "
            f"expected no invalid access"
        )
    else:
        print("serial IGF=1, BIN_SIZE=4e-15 under Valgrind: no invalid access found")
    verdict(bool(first) or exit_status != 0)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source reading, not executed for this claim |
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

