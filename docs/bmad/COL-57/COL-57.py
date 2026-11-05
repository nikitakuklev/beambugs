"""COL-57: Randomizing frequencies dereferences an absent wake

Calling the long-range wake frequency randomizer on an element without a wake crashes before returning its status. The routine’s contract permits this input and requires a normal return with the status false. An initialized element with no wake receives a segmentation fault, while an allocated wake with zero frequency spread returns normally.

The script calls the frequency randomizer in two fresh processes, first with an allocated zero-spread wake and then with no wake. It requires successful completion of the control and identifies the absent-wake segmentation fault. BMAD_DIST selects the build.
"""

from _bmad import RunFailure, fortran, verdict

PROGRAM = r"""
program randomize_wake
use bmad
use wake_mod
implicit none
type(ele_struct) ele
logical done
character(20) arg
call init_ele(ele,drift$)
call get_command_argument(1,arg)
if (trim(arg)=='allocated') call init_wake(ele%wake,0,0,0,1)
print *, 'associated=',associated(ele%wake)
flush(6)
done=.true.
call randomize_lr_wake_frequencies(ele,done)
print *, 'returned=',done
end program
"""
control = fortran(PROGRAM, ["allocated"]).log
if "associated= T" not in control or "returned= F" not in control:
    raise RuntimeError(control)
print("allocated zero-spread control: observed=normal return, set_done=false; expected=same")
try:
    result = fortran(PROGRAM, ["absent"])
except RunFailure as exc:
    if exc.status != -11 or "associated= F" not in exc.log or "SIGSEGV" not in exc.log:
        raise
    print("absent wake: observed=SIGSEGV (-11); expected=normal return, set_done=false")
    defect = True
else:
    if "associated= F" not in result.log or "returned= F" not in result.log:
        raise RuntimeError(result.log)
    print("absent wake: observed=normal return, set_done=false; expected=same")
    defect = False
verdict(defect)
