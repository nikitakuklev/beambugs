---
title: "TRK-47: Custom-tracker example registers an undeclared procedure pointer"
hide:
  - toc
---

# TRK-47 — Custom-tracker example registers an undeclared procedure pointer

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [Bmad Manual, “Custom and Hook Routines” (s:custom.hook)](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/misc-programming.tex#L21-L54)<br>[track1_custom pointer declaration](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bmad_routine_interface.f90#L4025-L4038)<br>[track1 custom dispatch](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/track1.f90#L264-L272) |

Following the custom-tracking registration example assigns the callback to track_custom_ptr, a name the interface does not declare. Compilation fails at that pointer assignment. The declared and dispatched pointer is track1_custom_ptr, which the same manual spells correctly in its declaration example. Substituting that name compiles and successfully associates the callback.

Run `python TRK-47.py` with [_bmad.py](../helpers/_bmad.py) and [_tracking_contracts_materials.py](../helpers/_tracking_contracts_materials.py) in the same directory.

??? example "Reproducer: TRK-47.py"

    [Download TRK-47.py](TRK-47.py)

    ```python
    """TRK-47: Custom-tracker example registers an undeclared procedure pointer

    Following the custom-tracking registration example assigns the callback to track_custom_ptr, a
    name the interface does not declare. Compilation fails at that pointer assignment. The declared
    and dispatched pointer is track1_custom_ptr, which the same manual spells correctly in its
    declaration example. Substituting that name compiles and successfully associates the callback.

    The script reads the callback assignment from BMAD_SOURCE and embeds it in a minimal Fortran
    program linked against BMAD_DIST. It first runs the corrected spelling as a control, then
    identifies the specific compiler diagnostic from the documented spelling.
    """

    import re

    from _bmad import RunFailure, fortran, source, verdict
    from _tracking_contracts_materials import rows

    text = (source() / "bmad/doc/misc-programming.tex").read_text()
    assignments = re.findall(r"^\s*(track\w+_ptr\s*=>\s*track1_custom)\s*$", text, re.MULTILINE)
    if len(assignments) != 1:
        raise ValueError("Expected one manual callback assignment")
    PROGRAM = """
    program callback_registration
    use bmad
    implicit none
    procedure(track1_custom_def) :: track1_custom
    ASSIGNMENT
    write(*,'(a,i3)') 'RESULT associated ',merge(1,0,associated(track1_custom_ptr))
    end program
    subroutine track1_custom(orbit,ele,param,err_flag,finished,track)
    use bmad
    implicit none
    type(coord_struct) orbit
    type(ele_struct) ele
    type(lat_param_struct) param
    type(track_struct),optional :: track
    logical err_flag,finished
    err_flag=.false.
    finished=.true.
    end subroutine
    """
    control = rows(fortran(PROGRAM.replace("ASSIGNMENT", "track1_custom_ptr => track1_custom")).log)
    if control["associated"].tolist() != [1]:
        raise AssertionError("Correct pointer did not associate")
    print("declared-pointer control: associated=1 expected=1")
    try:
        actual = rows(fortran(PROGRAM.replace("ASSIGNMENT", assignments[0])).log)
    except RunFailure as exc:
        if (
            exc.status != 1
            or "track_custom_ptr" not in exc.log
            or not any(
                message in exc.log
                for message in (
                    "has no IMPLICIT type",
                    "Non-POINTER in pointer association",
                    "Function 'track1_custom' requires an argument list",
                )
            )
        ):
            raise
        print(
            "manual assignment: compiler rejects undeclared track_custom_ptr; expected successful callback registration"
        )
        defect = True
    else:
        if actual["associated"].tolist() != [1]:
            raise AssertionError("Compiled example did not associate the callback")
        print("manual assignment: callback associated=1 expected=1")
        defect = False
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

