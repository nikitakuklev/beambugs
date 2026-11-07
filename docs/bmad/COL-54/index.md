---
title: "COL-54: BBU HOM-order filtering breaks mode-array allocation"
hide:
  - toc
---

# COL-54 — BBU HOM-order filtering breaks mode-array allocation

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [`HOM-order filtering`, bsim/bbu/bbu_program.f90:87–109](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/bbu/bbu_program.f90#L87-L109)<br>[`bbu_setup`, bsim/code/bbu_track_mod.f90:105–116](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/code/bbu_track_mod.f90#L105-L116)<br>[`track1_lr_wake`, bmad/multiparticle/wake_mod.f90:128–140](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L128-L140) |

Filtering higher-order modes in the beam-breakup simulation breaks the mode array when any modes are removed. Partial removal attempts to allocate an already allocated array, while complete removal leaves an unallocated array for downstream consumers. A cavity containing orders one and two fails with an allocation error at cutoff one, and a two-cavity case with one fully discarded mode crashes; retaining both orders completes.

Run `python COL-54.py` with [_bmad.py](../helpers/_bmad.py) and [_wake_tables_and_bbu.py](../helpers/_wake_tables_and_bbu.py) in the same directory.

??? example "Reproducer: COL-54.py"

    [Download COL-54.py](COL-54.py)

    ```python
    """COL-54: BBU HOM-order filtering breaks mode-array allocation

    Filtering higher-order modes in the beam-breakup simulation breaks the mode array when any modes are removed. Partial removal attempts to allocate an already allocated array, while complete removal leaves an unallocated array for downstream consumers. A cavity containing orders one and two fails with an allocation error at cutoff one, and a two-cavity case with one fully discarded mode crashes; retaining both orders completes.

    The script checks matching one-cavity and two-cavity controls that retain all modes, then runs partial-removal and complete-removal cases in separate processes. It requires the specific allocation failure after successful parsing or segmentation fault after tracking begins. All runs use seed 123.
    """

    from _bmad import RunFailure, run, verdict
    from _wake_tables_and_bbu import BBU_FILES, bbu_control

    bbu_control()
    bbu_control("control_all.init", "BBU two-cavity control HOM voltage gain")
    defect = False
    for name, status, diagnostic in [
        ("cutoff.init", 2, "Attempting to allocate already allocated variable"),
        ("cutoff_all.init", -11, "SIGSEGV"),
    ]:
        try:
            result = run("bbu", [name], files=BBU_FILES, outputs=["for_py.txt"])
        except RunFailure as exc:
            if (
                exc.status != status
                or diagnostic not in exc.log
                or "Lattice parse time" not in exc.log
                or (name == "cutoff_all.init" and "bbu_track_all loop begins" not in exc.log)
            ):
                raise
            print(f"{name}: observed=status {exc.status}, {diagnostic}; expected=normal completion")
            defect = True
        else:
            if "v_gain" not in result.outputs["for_py.txt"].decode():
                raise RuntimeError(result.log)
            print(f"{name}: observed=normal completion; expected=normal completion")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

