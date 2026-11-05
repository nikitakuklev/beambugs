"""COL-31: IBS selector documentation names different algorithms from those executed

When users follow the ibs_ring description of the Bjorken–Mtingwa formula, its cimp selector runs Completely Integrated Modified Piwinski instead; Bjorken–Mtingwa uses bjmt. The Coulomb-log summary and driver comments also misidentify choices 3 and 4, which execute Bane and Kubo/Oide forms respectively. At one fixed electron-beam working point, cimp gives a longitudinal growth rate of 2.979 s⁻¹ while bjmt gives 2.513 s⁻¹. The input descriptions should name the algorithms selected by the existing branches.

The script runs both ibs1 selectors and calls their named kernels directly at identical fixed
optics. It also checks Coulomb-log choices 3 and 4 against separate analytic evaluations of
their tail-cut formulas. BMAD_DIST selects the production library and BMAD_SOURCE supplies the
manual.
"""

import math

from _bmad import compare, fortran, verdict
from _collective_contracts import rows, section, source_text

PROGRAM = """program selectors
use ibs_mod
implicit none
type(lat_struct), target :: lat
type(ibs_sim_param_struct) params
type(ibs_struct) rates
integer i
real(rp) lambda, clog
character(4) arg
call init_lat(lat,2)
lat%n_ele_track=1
lat%param%particle=electron$
lat%param%total_length=100
lat%param%n_part=1e10_rp
do i=0,2
 lat%ele(i)%key=drift$
 lat%ele(i)%value(l$)=100
 lat%ele(i)%value(e_tot$)=4000*m_electron
 lat%ele(i)%a%beta=10
 lat%ele(i)%b%beta=20
 lat%ele(i)%a%gamma=0.1_rp
 lat%ele(i)%b%gamma=0.05_rp
 lat%ele(i)%a%emit=1e-9_rp
 lat%ele(i)%b%emit=2e-11_rp
 lat%ele(i)%a%eta=0.1_rp
 lat%ele(i)%b%eta=0.01_rp
 lat%ele(i)%z%sigma=0.01_rp
 lat%ele(i)%z%sigma_p=0.001_rp
enddo
params%tau_a=0.01_rp
params%clog_to_use=1
lambda=lat%param%n_part/lat%ele(1)%z%sigma
call multi_coulomb_log(params,lat%ele(1),clog,lambda)
call cimp1(lat%ele(1),clog,rates,lambda)
write(*,'(a,3es25.16)') 'RESULT direct_cimp ',rates%inv_Tp,rates%inv_Ta,rates%inv_Tb
call bjmt1(lat%ele(1),clog,rates,lambda)
write(*,'(a,3es25.16)') 'RESULT direct_bjmt ',rates%inv_Tp,rates%inv_Ta,rates%inv_Tb
params%formula='cimp'
call ibs1(lat,params,rates,i=1)
write(*,'(a,3es25.16)') 'RESULT cimp ',rates%inv_Tp,rates%inv_Ta,rates%inv_Tb
params%formula='bjmt'
call ibs1(lat,params,rates,i=1)
write(*,'(a,3es25.16)') 'RESULT bjmt ',rates%inv_Tp,rates%inv_Ta,rates%inv_Tb
do i=3,4
 params%clog_to_use=i
 call multi_coulomb_log(params,lat%ele(1),clog,lambda)
 write(*,'(a,i0,es25.16)') 'RESULT log',i,clog
enddo
end program
"""
r = rows(fortran(PROGRAM).log)
for name in ["cimp", "bjmt"]:
    if compare(
        f"{name} dispatch rates (1/s), direct named routine",
        r[name],
        r["direct_" + name],
        atol=0,
        rtol=2e-14,
    ):
        raise RuntimeError("Selector control disagrees with direct routine")
manual = source_text("bsim/ibs_ring/doc/ibs_ring.tex")
passage = section(
    manual, r"\subsection{Bjorken \& Mtingwa's formula}", r"\subsection{Bane's approximation"
)
wrong_name = "ibs_formula=`cimp'" in passage
mismatch = compare(
    "documented Bjorken-Mtingwa selector cimp vs direct Bjorken-Mtingwa rates (1/s)",
    r["cimp"],
    r["direct_bjmt"],
    atol=0,
    rtol=1e-9,
)
# Independent forms: Bane uses the horizontal rest-frame speed; Kubo/Oide
# uses the root-sum-square speed in all three momentum dimensions.
gamma, tau, line_density, speed = 4000, 0.01, 1e12, 299792458
sx = math.sqrt(10 * 1e-9 + (0.1 * 0.001) ** 2)
sy = math.sqrt(20 * 2e-11 + (0.01 * 0.001) ** 2)
volume_per_length = (4 * math.pi) ** 1.5 * sx * sy * gamma
bmax = min(sx, sy, (volume_per_length / line_density) ** (1 / 3))
velocities = [math.sqrt(1e-9 / 10), math.sqrt(0.1 * 1e-9 + 0.05 * 2e-11 + 0.001**2 / gamma**2)]
for option, velocity in zip([3, 4], velocities, strict=True):
    bmin = math.sqrt(volume_per_length / line_density / math.pi / (tau / gamma)) / math.sqrt(
        speed * gamma * velocity
    )
    expected = math.log(bmax / bmin)
    if compare(
        f"Coulomb log option {option}, independent tail-cut formula",
        r[f"log{option}"],
        [expected],
        atol=2e-14,
    ):
        raise RuntimeError("Tail-cut algorithm control changed")

print(f"manual Bjorken-Mtingwa subsection names cimp={wrong_name}; expected bjmt")
wrong_log_summary = "3=kubo, 4=kubo w/vertical" in manual
wrong_log_driver = "3=Oide, 4=Bane" in source_text("bsim/ibs_ring/ibs_ring.f90")
print(
    f"Coulomb-log descriptions: stale manual={wrong_log_summary}, stale driver={wrong_log_driver}; expected 3=Bane, 4=Kubo/Oide"
)
verdict((wrong_name and mismatch) or wrong_log_summary or wrong_log_driver)
