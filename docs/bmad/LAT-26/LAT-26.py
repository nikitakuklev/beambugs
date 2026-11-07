"""LAT-26: A full-length one-lord slice loses its entrance aperture

A temporary full-length slice with one lord and apertures at both ends keeps only the exit check. In a 1.2 m quadrupole, a particle starting at x = 2 mm outside a 1 mm aperture is lost by the original element but survives its slice, reaching x = 65.56 μm at the exit. The analytic quadrupole map confirms the exit position; preserving both physical ends requires the entrance loss to remain in force.

The script creates a full-length quadrupole slice and compares its aperture selection and particle survival with the original element. It checks the surviving body's horizontal orbit against the analytic focusing map.
"""

import math

from _bmad import compare, fortran, verdict
from _multipass_slicing import HEADER, control, rows

PROGRAM = r"""program probe
use bmad
implicit none
type(lat_struct), target :: lat
type(ele_struct), pointer :: q
type(ele_struct), target :: sliced, src, dst
type(coord_struct) start, ending
type(all_pointer_struct) ptr
real(rp) v(6)
logical err
call bmad_parser('lat.bmad',lat,err_flag=err)
if (err) stop 1
q=>lat%ele(1)
q%aperture_at=both_ends$
q%value(x1_limit$)=0.001_rp; q%value(x2_limit$)=0.001_rp
v=0; v(1)=0.002_rp; v(2)=-0.001_rp
call init_coord(start,v,q,upstream_end$)
call track1(start,q,lat%param,ending,err_flag=err)
if (err) stop 2
call row('original_lost', [real(merge(1,0,ending%state==lost_pos_x$),rp)])
call create_element_slice(sliced,q,q%value(l$),0.0_rp,lat%param,.true.,.true.,err)
if (err) stop 3
call row('slice_setup', [real(sliced%n_lord,rp), sliced%value(l$), &
  sliced%value(x1_limit$), sliced%value(x2_limit$)])
call row('slice_aperture', [real(merge(1,0,sliced%aperture_at==both_ends$),rp), &
                          real(merge(1,0,sliced%aperture_at==exit_end$),rp)])
call track1(start,sliced,lat%param,ending,err_flag=err)
if (err) stop 4
call row('slice_alive', [real(merge(1,0,ending%state==alive$),rp)])
call row('slice_orbit', ending%vec)
call deallocate_ele_pointers(sliced)
call deallocate_lat_pointers(lat)
contains
subroutine row(name,values)
character(*) name
real(rp) values(:)
write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
end subroutine
end program
"""
data = rows(
    fortran(
        PROGRAM,
        files={
            "lat.bmad": HEADER
            + "q: quadrupole,l=1.2,k1=0.7,fringe_type=none\nbl:line=(q)\nuse,bl\n"
        },
    ).log
)
control("original particle lost at entrance", data["original_lost"], [1])
control(
    "slice [lord count, L (m), x limits (m)]",
    data["slice_setup"],
    [1, 1.2, 0.001, 0.001],
    atol=2e-16,
)
if data["slice_alive"] == [1]:
    # At delta=0, x''=-k1*x and px=x': use the exact quadrupole body map.
    k = math.sqrt(0.7)
    c, s = math.cos(k * 1.2), math.sin(k * 1.2)
    expected = [c * 0.002 - s * 0.001 / k, -k * s * 0.002 - c * 0.001]
    control(
        "surviving body [x (m), px], analytic map",
        data["slice_orbit"][:2],
        expected,
        atol=2e-16,
        rtol=2e-13,
    )
defect = compare(
    "slice [both ends, exit only, alive], entrance aperture contract",
    data["slice_aperture"] + data["slice_alive"],
    [1, 0, 0],
    atol=0,
)
verdict(defect)
