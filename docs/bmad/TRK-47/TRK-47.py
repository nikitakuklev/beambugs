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
