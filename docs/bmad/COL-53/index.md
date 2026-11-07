---
title: "COL-53: BBU ramping overwrites the circular bunch-queue index"
hide:
  - toc
---

# COL-53 — BBU ramping overwrites the circular bunch-queue index

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [`bbu_add_a_bunch`, bsim/code/bbu_track_mod.f90:575–612](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/code/bbu_track_mod.f90#L575-L612) |

Enabling the bunch-charge ramp in the beam-breakup simulation overwrites the circular queue slot with the chronological bunch number. The first insertion records slot zero even though the bunch occupies slot one. The next insertion reports a bunch-array overflow and aborts; the same lattice completes with ramping disabled.

Run `python COL-53.py` with [_bmad.py](../helpers/_bmad.py) and [_wake_tables_and_bbu.py](../helpers/_wake_tables_and_bbu.py) in the same directory.

??? example "Reproducer: COL-53.py"

    [Download COL-53.py](COL-53.py)

    ```python
    """COL-53: BBU ramping overwrites the circular bunch-queue index

    Enabling the bunch-charge ramp in the beam-breakup simulation overwrites the circular queue slot with the chronological bunch number. The first insertion records slot zero even though the bunch occupies slot one. The next insertion reports a bunch-array overflow and aborts; the same lattice completes with ramping disabled.

    The script runs the BBU executable with a fixed random seed on the same lattice with ramping disabled and enabled. It checks completion of the control and identifies the queue-overflow diagnostic and abort in the ramped case.
    """

    from _bmad import RunFailure, run, verdict
    from _wake_tables_and_bbu import BBU_FILES, bbu_control

    bbu_control()
    try:
        result = run("bbu", ["ramp.init"], files=BBU_FILES, outputs=["for_py.txt"])
    except RunFailure as exc:
        if exc.status != -4 or "BBU_BEAM%BUNCH ARRAY OVERFLOW" not in exc.log:
            raise
        print(f"ramping: observed=queue overflow, SIGILL ({exc.status}); expected=normal completion")
        defect = True
    else:
        if "v_gain" not in result.outputs["for_py.txt"].decode():
            raise RuntimeError(result.log)
        print("ramping: observed=normal completion; expected=normal completion")
        defect = False
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

