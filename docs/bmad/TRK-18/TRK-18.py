"""TRK-18: Reverse-time rfcavity matrices use the forward energy change

With backward-time tracking, the native RF cavity map returns the correct particle energy but the wrong longitudinal transfer matrix. It inverts a forward Jacobian evaluated at the reverse input, instead of differentiating the signed reverse kick there. For a thin 200 kV crest cavity at reference momentum 1 MeV/c, the derivative M₆₆ of exit fractional momentum with respect to entrance fractional momentum is 1.049 instead of 1.069. Differentiating the energy subtraction supplies the expected value.

The script compiles thin forward- and backward-time RF cavity cases using BMAD_DIST. It validates each signed energy change and compares M₆₆ with the derivative of E_out = E_in ± V. Forward-time tracking supplies a passing matrix control.
"""

from _bmad import compare, fortran, verdict
from _rf_and_spin import control, rows

PROGRAM = """program reverse_rf_probe
use bmad
implicit none
type(lat_struct), target :: lat
type(coord_struct) :: a,b
logical :: err
real(rp) :: mat(6,6),p,e,beta_out
integer :: td
call bmad_parser('reverse_rf.bmad',lat,err_flag=err)
if (err) error stop 'parse'
do td=1,-1,-2
call init_coord(a,ele=lat%ele(1),element_end=merge(upstream_end$,downstream_end$,td==1))
a%time_dir=td
b=a
call mat_make_unit(mat)
call track1_bmad(b,lat%ele(1),lat%param,err_flag=err,mat6=mat,make_matrix=.true.)
if (err .or. b%state /= alive$ .or. b%species /= a%species) error stop 'track'
e=hypot(a%p0c,mass_of(a%species))+td*lat%ele(1)%value(voltage$)
p=sqrt(e**2-mass_of(a%species)**2)
beta_out=p/e
write(*,'(a,i0,2es25.16)') 'RESULT M66_',td,mat(6,6),a%beta/beta_out
write(*,'(a,i0,2es25.16)') 'RESULT energy_',td,hypot(b%p0c*(1+b%vec(6)),mass_of(b%species)),e
enddo
end program
"""
LATTICE = """no_digested
parameter[particle] = electron
parameter[p0c] = 1e6
parameter[geometry] = open
rf: rfcavity, l = 0, rf_frequency = 1e9, voltage = 2e5, phi0 = 0.25
ln: line = (rf)
use, ln
"""
d = rows(fortran(PROGRAM, files={"reverse_rf.bmad": LATTICE}).log)
control("forward exit energy (eV), E_in plus V", *d["energy_1"], atol=2e-8)
control("forward M66, beta_in/beta_out", *d["M66_1"], atol=2e-13)
control("reverse exit energy (eV), E_in minus V", *d["energy_-1"], atol=2e-8)
# d(pc_out)/d(pc_in) = beta_in/beta_out follows by differentiating E_out=E_in-V.
verdict(compare("reverse M66, beta_in/beta_out", *d["M66_-1"], atol=2e-13))
