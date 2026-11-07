"""LAT-25: Overlay bookkeeping overruns its 20-attribute buffers

An overlay assigning more than twenty distinct real attributes of one element overruns fixed bookkeeping buffers. Assigning twenty normal multipole coefficients of a quadrupole succeeds, while adding the valid twenty-first coefficient crashes during lattice parsing. Bounds checking identifies an access to element 21 of a twenty-element pointer array. All twenty-one attributes are valid and should receive the requested value 0.001 without memory corruption.

The script parses overlays controlling twenty and twenty-one coefficients, checking every value in the successful control. It identifies the production crash only after the checkpoint before parsing the twenty-one-target case; the companion bounds script locates the overflowing write.
"""

import resource
import signal

from _bmad import RunFailure, compare, fortran, verdict
from _multipass_slicing import HEADER, control, rows

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
PROGRAM = r"""program probe
use bmad
implicit none
type(lat_struct), target :: lat
type(ele_struct), pointer :: ele, lord, slave
type(ele_struct) :: s1,s2
type(coord_struct) :: start, orb, ending, whole
type(coord_struct), allocatable :: orbit(:)
integer i, j, n
real(rp) v(6)
logical err
character(40) label, arg
call get_command_argument(1,arg)
read(arg,*) n
write(*,'(a,i0)') 'CHECKPOINT overlay attribute count ', n
flush(6)
call bmad_parser('lat.bmad',lat,err_flag=err)
if (err) stop 1
if (.not. associated(lat%ele(1)%b_pole)) stop 2
call row('assigned',lat%ele(1)%b_pole(0:n-1))
call deallocate_lat_pointers(lat)
contains
subroutine row(name,values)
character(*) name
real(rp) values(:)
write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
end subroutine
end program
"""


def lattice(count):
    targets = ", ".join(f"q[b{i}]:v" for i in range(count))
    return (
        HEADER
        + f"q:quadrupole,l=1,k1=0\no:overlay={{{targets}}},var={{v}},v=0.001\nbl:line=(q)\nuse,bl\n"
    )


normal = rows(fortran(PROGRAM, ["20"], files={"lat.bmad": lattice(20)}).log)
control(
    "twenty controlled coefficients, each v=0.001", normal["assigned"], [0.001] * 20, atol=1e-17
)
try:
    subject = rows(fortran(PROGRAM, ["21"], files={"lat.bmad": lattice(21)}).log)
except RunFailure as exc:
    if (
        exc.status != -signal.SIGSEGV
        or "CHECKPOINT overlay attribute count 21" not in exc.log
        or "Parsing lattice file" not in exc.log
        or "Program received signal SIGSEGV" not in exc.log
        or "RESULT assigned" in exc.log
    ):
        raise
    print(
        "twenty-one controlled coefficients: observed=SIGSEGV during parsing expected=21 assignments of 0.001"
    )
    defect = True
else:
    defect = compare(
        "twenty-one controlled coefficients", subject["assigned"], [0.001] * 21, atol=1e-17
    )
verdict(defect)
