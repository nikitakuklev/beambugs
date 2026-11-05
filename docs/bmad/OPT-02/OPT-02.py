"""OPT-02: Radiation-map setup changes the original bend field model

Constructing a radiation map for a bend with exact multipoles silently switches exact multipoles off on the original element. An auxiliary map calculation must preserve the original element’s field model. In a 2 m combined-function bend at 3 GeV, identical radiation-disabled tracking before and after setup changes the exit horizontal position from 1.403 mm to 1.420 mm. The 16.91 μm shift follows a change from horizontally pure exact multipoles to the approximate field model.

The script tracks the same particle before and after radiation-map setup in a combined-function bend. Radiation is disabled for both tracks, so preserving the element must preserve the orbit. It also checks the exact-multipole selector directly; BMAD_DIST supplies the build.
"""

from _bmad import compare, fortran, verdict
from _s3_14_optics_matrix_radiation import control, rows

PROGRAM = """
program probe
use bmad
use rad_6d_mod
use radiation_mod
implicit none
type(lat_struct), target :: lat
type(coord_struct), allocatable :: orb(:)
type(coord_struct) o1, o2
type(ele_struct), pointer :: ele
real(rp) before(6), after(6)
logical err
integer stat
call bmad_parser('lat.bmad', lat, err_flag=err)
if (err) error stop 'parse failed'
bmad_com%radiation_damping_on = .false.
bmad_com%radiation_fluctuations_on = .false.
call twiss_and_track(lat, orb, stat)
if (stat /= ok$) error stop 'twiss failed'
  ele => lat%ele(2)
  ele%value(exact_multipoles$) = horizontally_pure$
  ele%value(k1$) = 0.5_rp
  call set_flags_for_changed_attribute(ele)
  call lattice_bookkeeper(lat)
  call init_coord(o1, [0.01_rp,0.0_rp,0.005_rp,0.0_rp,0.0_rp,0.0_rp], ele, upstream_end$)
  call make_mat6(ele, lat%param, o1, o2, err)
  if (err) error stop 'matrix failed'
  call track1(o1, ele, lat%param, o2, err_flag=err)
  if (err .or. o2%state /= alive$) error stop 'tracking failed'
  before = o2%vec
  call row('before_model', [ele%value(exact_multipoles$)])
  call radiation_map_setup(ele, err)
  if (err) error stop 'radiation setup failed'
  call row('after_model', [ele%value(exact_multipoles$)])
  call track1(o1, ele, lat%param, o2, err_flag=err)
  if (err .or. o2%state /= alive$) error stop 'tracking failed'
  after = o2%vec
  call row('before', before)
  call row('after', after)
  call row('selectors', [real(horizontally_pure$,rp),real(off$,rp)])
contains
subroutine row(name,values)
character(*) name
real(rp) values(:)
write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
end subroutine
end program
"""
LATTICE = "no_digested\nparameter[geometry] = open\nparameter[e_tot] = 3e9\nparameter[no_end_marker] = T\nbeginning[beta_a] = 10\nbeginning[beta_b] = 12\nd: drift, l = 2\nb: sbend, l = 2, g = 0.1, fringe_type = none\nbh: sbend, l = 1, g = 0.1, fringe_type = none\nq: quadrupole, l = 1, k1 = 0.5\nln: line = (d, b, bh, bh, q)\nuse, ln\n"

r = rows(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
control(
    "configured horizontally-pure selector", r["before_model"], r["selectors"][:1], atol=0, rtol=0
)
defect = compare(
    "selector after auxiliary setup, unchanged", r["after_model"], r["before_model"], atol=0
)
defect |= compare(
    "exit x (m), identical radiation-disabled tracking", r["after"][0], r["before"][0], atol=2e-14
)
print(f"exit x change (m)={r['after'][0] - r['before'][0]:.16g}")
verdict(defect)
