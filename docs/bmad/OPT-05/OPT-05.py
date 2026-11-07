"""OPT-05: Backward transfer matrix dereferences an absent affine vector

Requesting a backward transfer matrix in an open lattice crashes when the optional affine-vector output is omitted. The matrix-only call should return the inverse of the forward matrix. Through a 2 m drift, the same backward call succeeds when the affine vector is supplied and returns the expected horizontal drift coefficient of −2 m and a zero vector.

The script checks forward matrix-only transport and backward transport with the affine output before omitting that output. An analytic 2 m drift supplies the complete six-dimensional inverse. It recognizes only a segmentation fault after successful parsing and the two passing controls as the defect;
"""

import resource
import signal

import numpy as np
from _bmad import RunFailure, compare, fortran, verdict
from _s3_14_optics_matrix_radiation import control, rows

PROGRAM = """
program reverse
use bmad
implicit none
type(lat_struct) lat
real(rp) mat(6,6), vec(6)
logical err
character(12) mode
call get_command_argument(1,mode)
call bmad_parser('lat.bmad',lat,err_flag=err)
if (err) error stop 'parse failed'
print *, 'PREREQUISITES PASSED'
flush(6)
if (mode == 'vector') then
 call transfer_matrix_calc(lat,mat,vec,ix1=1,ix2=0)
 call row('vec',vec)
else if (mode == 'forward') then
 call transfer_matrix_calc(lat,mat,ix1=0,ix2=1)
else
 call transfer_matrix_calc(lat,mat,ix1=1,ix2=0)
endif
call row('matrix',reshape(mat,[36]))
contains
subroutine row(name,values)
character(*) name
real(rp) values(:)
write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
end subroutine
end program
"""
LATTICE = "no_digested\nparameter[geometry] = open\nparameter[e_tot] = 3e9\nparameter[no_end_marker] = T\nbeginning[beta_a] = 10\nbeginning[beta_b] = 12\nd: drift, l = 2\nb: sbend, l = 2, g = 0.1, fringe_type = none\nbh: sbend, l = 1, g = 0.1, fringe_type = none\nq: quadrupole, l = 1, k1 = 0.5\nln: line = (d, b, bh, bh, q)\nuse, ln\n"

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
expected = np.eye(6)
expected[0, 1] = expected[2, 3] = -2
expected[4, 5] = -2 * (510998.95069 / 3e9) ** 2
defect = False
for mode in ("forward", "vector", "absent"):
    try:
        result = fortran(PROGRAM, [mode], files={"lat.bmad": LATTICE})
    except RunFailure as exc:
        if (
            mode != "absent"
            or exc.status != -signal.SIGSEGV
            or "PREREQUISITES PASSED" not in exc.log
            or "Program received signal SIGSEGV" not in exc.log
        ):
            raise
        print(
            "backward matrix only: observed=SIGSEGV in transfer_matrix_calc expected=inverse drift matrix"
        )
        defect = True
        continue
    r = rows(result.log)
    matrix = r["matrix"].reshape((6, 6), order="F")
    target = np.linalg.inv(expected) if mode == "forward" else expected
    if mode == "absent":
        defect |= compare("backward matrix only, inverse drift", matrix, target, atol=2e-14)
    else:
        control(f"{mode}, analytic drift matrix", matrix, target)
        if mode == "vector":
            control("backward affine vector", r["vec"], np.zeros(6), atol=0, rtol=0)
verdict(defect)
