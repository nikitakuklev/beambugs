"""OPT-23: Charge-squared scaling of symp_lie_bmad loss and fluctuations

The script tracks proton and carbon-12 at the same relativistic factor and geometric curvature
through a planar wiggler and a quadrupole. Species-correct analytic rates normalize their mean
losses, and a common seeded sequence compares fluctuation variances; their ratios cancel the
separate common factor-of-two radiation defect. A Runge–Kutta radiation-map control checks the
species scaling independently. BMAD_DIST selects the build; the fluctuation sample has 4000
tracks per species and seed 12345.
"""

from _bmad import compare, verdict
from _dispersion_and_moments import execute, require

PROGRAM = r"""
program radiation_tracking
use bmad
use random_mod
implicit none
type(lat_struct),target :: lat
type(ele_struct),pointer :: ele
type(coord_struct) start,finish
real(rp) gamma,mc2,rc,gmax,len,kq,int_g2,int_g3,loss_expected,var_expected
real(rp) loss,s1,s2,d,rk
integer i
integer,parameter :: count=4000
call bmad_parser('lat.bmad',lat)
ele=>lat%ele(1)
mc2=mass_of(lat%param%particle)
gamma=ele%value(e_tot$)/mc2
rc=classical_radius(lat%param%particle)
len=ele%value(l$)
call init_coord(start,lat%ele(0),downstream_end$)
if (ele%key == quadrupole$) then
  kq=sqrt(ele%value(k1$))
  int_g2=ele%value(k1$)**2*1e-4_rp*(len/2+sin(2*kq*len)/(4*kq))
  start%vec(1)=1e-2_rp
else
  gmax=abs(c_light*ele%value(b_max$)*charge_of(lat%param%particle)/ele%value(p0c$))
  int_g2=gmax**2*len/2
  int_g3=4*gmax**3*len/(3*pi)
endif
! Compare the geometric orbit with radiation disabled before comparing species rates.
bmad_com%radiation_damping_on=.false.
bmad_com%radiation_fluctuations_on=.false.
call track1(start,ele,lat%param,finish)
if (finish%state /= alive$) error stop 'lost geometric control particle'
write(*,'(a,6es25.17)') 'RESULT geometric ',finish%vec
! Planar wiggler integrals: integral(g^2 ds)=gmax^2 L/2 and integral(|g|^3 ds)=4 gmax^3 L/(3 pi).
loss_expected=-2*rc*gamma**3*int_g2/3
bmad_com%radiation_damping_on=.true.
bmad_com%radiation_fluctuations_on=.false.
call track1(start,ele,lat%param,finish)
if (finish%state /= alive$) error stop 'lost damping particle'
loss=finish%vec(6)-start%vec(6)
write(*,'(a,3es25.17)') 'RESULT loss ',loss,loss_expected,loss/loss_expected
if (ele%key == wiggler$) then
  ele%tracking_method=runge_kutta$
  call track1(start,ele,lat%param,finish)
  if (finish%state /= alive$) error stop 'lost Runge-Kutta particle'
  rk=(finish%vec(6)-start%vec(6))/loss_expected
  ele%tracking_method=symp_lie_bmad$
  bmad_com%radiation_damping_on=.false.
  bmad_com%radiation_fluctuations_on=.true.
  var_expected=55*rc*(h_bar_planck*c_light/mc2)*gamma**5*int_g3/(24*sqrt(3.0_rp))
  call ran_seed_put(12345)
  s1=0; s2=0
  do i=1,count
    call track1(start,ele,lat%param,finish)
    if (finish%state /= alive$) error stop 'lost stochastic particle'
    d=finish%vec(6)-start%vec(6)
    s1=s1+d; s2=s2+d*d
  enddo
  s2=(s2-s1*s1/count)/(count-1)
  write(*,'(a,3es25.17)') 'RESULT variance ',s2,var_expected,s2/var_expected
  write(*,'(a,es25.17)') 'RESULT rk ',rk
endif
end program
"""


def lattice(species, element):
    text = (
        f"parameter[particle]={species}\nparameter[geometry]=open\n"
        f"beginning[e_tot]=1000*mass_of({species})\n"
    )
    if element == "wiggler":
        text += (
            "e: wiggler, l=1, n_period=10, num_steps=2000, tracking_method=symp_lie_bmad,\n"
            f" b_max=sqrt(1000^2-1)*mass_of({species})/(c_light*charge_of({species}))\n"
        )
    else:
        text += "e: quadrupole, l=1, k1=1, num_steps=2000, tracking_method=symp_lie_bmad\n"
    return text + "line1: line=(e)\nuse, line1\n"


defect = False
for element in ("wiggler", "quadrupole"):
    proton, carbon = [
        execute(PROGRAM, {"lat.bmad": lattice(species, element)})
        for species in ("proton", "#12C+6")
    ]
    require(carbon["geometric"], proton["geometric"], atol=2e-14, rtol=2e-12)
    if any(row["loss"][0] >= 0 or row["loss"][1] >= 0 for row in (proton, carbon)):
        raise RuntimeError("The observed and analytic damping losses must be negative")
    # Dividing each by its own species-correct analytic rate cancels masses and rigidity.
    # Taking their ratio cancels the separate, common two-kicks-per-step defect (OPT-24).
    defect |= compare(
        f"{element} carbon/proton normalized mean loss; charge-squared law",
        carbon["loss"][2] / proton["loss"][2],
        1.0,
        atol=0,
        rtol=2e-7,
    )
    print(
        f"{element} analytic-rate ratios: proton={proton['loss'][2]:.17g} carbon={carbon['loss'][2]:.17g}"
    )
    if element == "wiggler":
        # The independent radiation-map path scales with charge; its shared quadrature factor cancels.
        require(carbon["rk"] / proton["rk"], [1.0], atol=0, rtol=2e-8)
        print(
            f"Runge-Kutta analytic-rate ratios: proton={proton['rk'][0]:.17g} carbon={carbon['rk'][0]:.17g}"
        )
        if min(proton["variance"][:2]) <= 0 or min(carbon["variance"][:2]) <= 0:
            raise RuntimeError("Stochastic variance must be positive")
        defect |= compare(
            "wiggler carbon/proton normalized fluctuation variance, same 4000 random draws",
            carbon["variance"][2] / proton["variance"][2],
            1.0,
            atol=0,
            rtol=2e-6,
        )
        print(
            f"variance/analytic: proton={proton['variance'][2]:.17g} carbon={carbon['variance'][2]:.17g}"
        )
verdict(defect)
