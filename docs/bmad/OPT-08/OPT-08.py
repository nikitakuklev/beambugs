"""OPT-08: Matrix errors leave quantum fluctuations disabled globally

A handled transfer-matrix calculation error leaves the global radiation-fluctuation switch off. Matrix construction temporarily disables quantum fluctuations, but it must restore the incoming setting before returning an error. A deliberately failing custom matrix callback is called once with fluctuations off and returns with them still off, although they were enabled before the call. Subsequent tracking therefore omits quantum fluctuations after the caller recovers from the error. Successful matrix construction restores the switch.

The script runs the same custom matrix callback with successful and failing returns in separate processes. It checks that each callback ran once with fluctuations disabled, and that the successful call restored the switch before testing the error path. BMAD_DIST supplies the build.
"""

from _bmad import compare, fortran, verdict
from _s3_14_optics_matrix_radiation import control, rows

PROGRAM = """
module architecture_error_hook
use bmad
implicit none
integer calls, fluctuations_in_callback
logical fail_on_call
contains
subroutine fail_matrix(ele, param, start_orb, end_orb, err_flag)
type(ele_struct), target :: ele
type(lat_param_struct) param
type(coord_struct) start_orb, end_orb
logical err_flag
calls = calls + 1
fluctuations_in_callback = merge(1, 0, bmad_com%radiation_fluctuations_on)
end_orb = start_orb
call mat_make_unit(ele%mat6)
ele%vec0 = 0
err_flag = fail_on_call
end subroutine
end module

program architecture_error
use architecture_error_hook
implicit none
type(lat_struct), target :: lat
logical err
character(12) mode
call get_command_argument(1, mode)
select case(mode)
case('success')
  fail_on_call = .false.
case('failure')
  fail_on_call = .true.
case default
  error stop 'expected success or failure mode'
end select
call bmad_parser('lat.bmad', lat, make_mats6 = .false., err_flag = err)
if (err) stop 1
make_mat6_custom_ptr => fail_matrix
lat%ele(1)%mat6_calc_method = custom$
calls = 0
fluctuations_in_callback = -1
bmad_com%radiation_fluctuations_on = .true.
call make_mat6(lat%ele(1), lat%param, err_flag = err)
if (err .neqv. fail_on_call) stop 2
write(*, '(a,2i3)') 'RESULT callback', calls, fluctuations_in_callback
write(*, '(a,i3)') 'RESULT restored', merge(1, 0, bmad_com%radiation_fluctuations_on)
call deallocate_lat_pointers(lat)
end program
"""
LATTICE = "\nno_digested\nparameter[particle] = electron\nparameter[p0c] = 1e9\nparameter[geometry] = open\nbeginning[beta_a] = 10\nbeginning[beta_b] = 10\nq: quadrupole, l = 1.2, k1 = 0.7, fringe_type = none\nline1: line = (q)\n\nuse, line1\n"

defect = False
for mode in ("success", "failure"):
    r = rows(fortran(PROGRAM, [mode], files={"lat.bmad": LATTICE}).log)
    control(f"{mode} callback [count, fluctuations inside]", r["callback"], [1, 0], atol=0, rtol=0)
    if mode == "success":
        control("success fluctuations restored", r["restored"], [1], atol=0, rtol=0)
    else:
        defect = compare("fluctuations restored after handled error", r["restored"], [1], atol=0)
verdict(defect)
