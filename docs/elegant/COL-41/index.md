---
title: "COL-41: CSRCSBEND with IGF=1 leaves dangling Green-function pointers, so a second step, pass or CSRCSBEND aborts"
hide:
  - toc
---

# COL-41 — CSRCSBEND with IGF=1 leaves dangling Green-function pointers, so a second step, pass or CSRCSBEND aborts

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Source | [declaration, csbend.c:2626](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2626)<br>[Makefile, Makefile:230](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/Makefile#L230)<br>[Makefile.mpi, Makefile.mpi:187](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/Makefile.mpi#L187)<br>[cleanup, csbend.c:4042–4055](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L4042-L4055)<br>[IGF allocation, csbend.c:3447–3452](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3447-L3452)<br>[IGF, track_data.c:2071](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2071) |

In the default build, a CSRCSBEND with IGF=1 frees its integrated-Green-function arrays at exit but keeps the static pointers to them, so the next CSRCSBEND of the run reallocates or frees the released memory again. For 5001 electrons in a 0.8 m, 0.4 rad bend with 128 bins, one step through one IGF bend completes, but a second step, a second pass, a second IGF bend or a following CSRCSBEND without IGF each abort with a double free, and Valgrind reports the invalid realloc or free. A single traversal of one IGF bend with no later CSRCSBEND is unaffected.

Run `python COL-41.py` with [_csr.py](../helpers/_csr.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: COL-41.py"

    [Download COL-41.py](COL-41.py)

    ```python
    """COL-41: CSRCSBEND with IGF=1 leaves dangling Green-function pointers, so a second step, pass or CSRCSBEND aborts

    In the default build, a CSRCSBEND with IGF=1 frees its integrated-Green-function arrays at exit but keeps the static pointers to them, so the next CSRCSBEND of the run reallocates or frees the released memory again. For 5001 electrons in a 0.8 m, 0.4 rad bend with 128 bins, one step through one IGF bend completes, but a second step, a second pass, a second IGF bend or a following CSRCSBEND without IGF each abort with a double free, and Valgrind reports the invalid realloc or free. A single traversal of one IGF bend with no later CSRCSBEND is unaffected.

    Setup: `valgrind` on `PATH`; the script runs some cases of serial Elegant under Memcheck.

    The script tracks 5001 electrons (Gaussian quantiles, σ_ct = 12 μm, p = 1500 mₑc, 10⁻¹⁸ C) through a
    0.8 m, 0.4 rad, 24-slice CSRCSBEND with STEADY_STATE=1, IGF=1 and BINS=128. One step through one
    bend is the control and must complete. The cases are two steps, two passes, a second IGF bend and a
    following IGF=0 bend; each must complete like the control. For each, the script prints the exit of a
    plain run and the first invalid heap operation Valgrind reports.
    """

    import re

    from _csr import gaussian_beam, track, valgrind
    from _elegant import RunFailure, verdict

    P0 = 1500.0
    BEAM = gaussian_beam(5001, 12e-6, P0)
    BEND = "L=0.8,ANGLE=0.4,N_SLICES=24,BINS=128,STEADY_STATE=1,SG_HALFWIDTH=1"
    IGF = f"B: CSRCSBEND,{BEND},IGF=1\n"
    CHARGE = "Q: CHARGE,TOTAL=1e-18\n"
    CASES = [
        ("one IGF bend, one step (control)", CHARGE + IGF, "Q,B", {}),
        ("one IGF bend, n_steps=2", CHARGE + IGF, "Q,B", {"steps": 2}),
        ("one IGF bend, n_passes=2", CHARGE + IGF, "Q,B", {"passes": 2}),
        ("two IGF bends", CHARGE + IGF + f"C: CSRCSBEND,{BEND},IGF=1", "Q,B,C", {}),
        ("IGF bend, then IGF=0 bend", CHARGE + IGF + f"C: CSRCSBEND,{BEND},IGF=0", "Q,B,C", {}),
    ]
    # First invalid heap operation in a Valgrind log: the kind, the call (free/realloc) and its caller.
    INVALID = re.compile(
        r"(Invalid free\(\) / delete / delete\[\] / realloc\(\))\n==\d+==\s+at [^:]*: (\w+) .*\n==\d+==\s+by [^(]*\(([^)]*)\)"
    )

    defect = False
    for label, elements, line, kw in CASES:
        exit_status, message = 0, "completed"
        try:
            track(elements, line, BEAM, P0, outputs=(), **kw)
        except RunFailure as failure:
            exit_status = failure.status
            found = re.findall(r"Terminated by \S+|double free[^\n]*|free\(\): [^\n]*", failure.log)
            message = ", ".join(dict.fromkeys(found)) or "failed"
        _, log = valgrind(elements, line, BEAM, P0, **kw)
        invalid = INVALID.search(log)
        heap = (
            f"{invalid.group(2)} of freed memory at {invalid.group(3)}"
            if invalid
            else "no invalid heap operation"
        )
        print(
            f"serial {label}: exit status observed={exit_status} ({message}); Valgrind: {heap}; expected=0, no invalid heap operation"
        )
        if label.endswith("(control)"):
            if exit_status or invalid:
                raise SystemExit("The single-step control fails.")
            continue
        defect |= exit_status != 0 or invalid is not None
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed<br><small>also an IGF bend followed by a bend without IGF</small> | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

