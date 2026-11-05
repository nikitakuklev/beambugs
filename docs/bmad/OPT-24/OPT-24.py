"""OPT-24: symp_lie_bmad applies each step's radiation twice

With radiation enabled in symp_lie_bmad wiggler or quadrupole tracking, every integration step applies two full-step radiation kicks. The mean energy loss and quantum-excitation variance therefore approach twice the values required by the radiation formulas. For an electron at Lorentz factor γ = 1000 in a ten-period, 1 m planar wiggler, the fractional momentum loss is 1.879×10⁻⁶ instead of 9.393×10⁻⁷. A 4000-particle fluctuation check gives 2.042 times the expected variance, and a quadrupole shows the same doubled mean loss.

The script tracks electrons and protons through a planar wiggler and a quadrupole and compares mean loss with analytic curvature integrals. It checks 4000 seeded fluctuation tracks in each wiggler and validates radiation-off and independent Runge–Kutta controls. BMAD_DIST selects the production build.
"""

from _bmad import compare, fortran, verdict
from _srdt_wiggler_maps import records

PROGRAM = """program radiation_steps
use bmad
use random_mod
implicit none
type(lat_struct), target :: lat
type(ele_struct), pointer :: ele
type(coord_struct) :: start, finish
real(rp) gamma, mc2, rc, peak, length, kq, i2, i3, expected, variance, total, squares, d
integer i
integer, parameter :: samples=4000
logical err
call bmad_parser('lat.bmad',lat,err_flag=err)
if(err) error stop 'Parsing failed'
ele=>lat%ele(1)
mc2=mass_of(lat%param%particle)
gamma=ele%value(e_tot$)/mc2
rc=classical_radius(lat%param%particle)
length=ele%value(l$)
call init_coord(start,lat%ele(0),downstream_end$)
if(ele%key==quadrupole$) then
  start%vec(1)=.01_rp
  kq=sqrt(ele%value(k1$))
  ! Paraxial quadrupole: x(s)=x0*cos(sqrt(k1)*s), curvature=k1*x(s).
  i2=ele%value(k1$)**2*start%vec(1)**2*(length/2+sin(2*kq*length)/(4*kq))
else
  peak=abs(c_light*ele%value(b_max$)*charge_of(lat%param%particle)/ele%value(p0c$))
  ! Complete planar periods: <g^2>=G^2/2, <|g|^3>=4*G^3/(3*pi).
  i2=peak**2*length/2
  i3=4*peak**3*length/(3*pi)
endif
expected=-2*rc*gamma**3*i2/3
bmad_com%radiation_damping_on=.false.
bmad_com%radiation_fluctuations_on=.false.
call track1(start,ele,lat%param,finish,err_flag=err)
if(err.or.finish%state/=alive$) error stop 'Off control failed'
write(*,'(a,es25.16)') 'OFF ',finish%vec(6)
bmad_com%radiation_damping_on=.true.
call track1(start,ele,lat%param,finish,err_flag=err)
if(err.or.finish%state/=alive$) error stop 'Damping track failed'
write(*,'(a,3es25.16)') 'LOSS ',finish%vec(6),expected,finish%vec(6)/expected
if(ele%key==quadrupole$) stop
! The independent map-based radiation route is a control for the mean loss.
ele%tracking_method=runge_kutta$
call track1(start,ele,lat%param,finish,err_flag=err)
if(err.or.finish%state/=alive$) error stop 'Runge-Kutta control failed'
write(*,'(a,3es25.16)') 'RK ',finish%vec(6),expected,finish%vec(6)/expected
ele%tracking_method=symp_lie_bmad$
bmad_com%radiation_damping_on=.false.
bmad_com%radiation_fluctuations_on=.true.
variance=55*rc*(h_bar_planck*c_light/mc2)*gamma**5*i3/(24*sqrt_3)
call ran_seed_put(12345)
total=0; squares=0
do i=1,samples
  call track1(start,ele,lat%param,finish,err_flag=err)
  if(err.or.finish%state/=alive$) error stop 'Fluctuation track failed'
  d=finish%vec(6)
  total=total+d; squares=squares+d*d
enddo
squares=(squares-total*total/samples)/(samples-1)
write(*,'(a,4es25.16)') 'VAR ',squares,variance,squares/variance,sqrt(2.0_rp/(samples-1))*squares/variance
end program
"""
LATTICE = """parameter[particle]=SPECIES
parameter[geometry]=open
beginning[e_tot]=1000*mass_of(SPECIES)
e: ELEMENT
line1: line=(e)
use,line1
"""
ELEMENTS = {
    "wiggler": "wiggler, l=1, n_period=10, num_steps=2000, tracking_method=symp_lie_bmad, b_max=sqrt(1000^2-1)*mass_of(SPECIES)/(c_light*charge_of(SPECIES))",
    "quadrupole": "quadrupole, l=1, k1=1, num_steps=2000, tracking_method=symp_lie_bmad",
}
defect = False
for species in ("electron", "proton"):
    for kind, element in ELEMENTS.items():
        lattice = LATTICE.replace("ELEMENT", element).replace("SPECIES", species)
        log = fortran(PROGRAM, files={"lat.bmad": lattice}).log
        if compare(f"{species} {kind}: radiation-off delta", records(log, "OFF")[0, 0], 0, atol=0):
            raise AssertionError("Radiation-off control failed")
        loss = records(log, "LOSS")[0]
        print(f"{species} {kind}: delta={loss[0]:.16g}, analytic delta={loss[1]:.16g}")
        defect |= compare(f"{species} {kind}: loss/analytic", loss[2], 1, atol=1e-5)
        if kind == "wiggler":
            rk = records(log, "RK")[0]
            # This separate route has a 3.5e-4 quadrature error for ten periods.
            if compare(f"{species}: Runge-Kutta loss/analytic control", rk[2], 1, atol=5e-4):
                raise AssertionError("Map-based radiation control failed")
            var = records(log, "VAR")[0]
            print(
                f"{species}: variance={var[0]:.16g}, analytic variance={var[1]:.16g}, ratio standard error={var[3]:.6g}"
            )
            # 4000 independent samples give a 2.24% relative standard error.
            # A 15% comparison band exceeds six expected standard errors.
            defect |= compare(f"{species}: variance/analytic", var[2], 1, atol=0.15)
verdict(defect)
