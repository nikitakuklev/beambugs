"""COL-32: The documented ibs_ring Kubo selector and cache option are unavailable

When users select the documented ibs_ring Kubo method, the IBS dispatch stops because it has no kubo branch. The accompanying use_t6_cache option is also absent from the driver's namelist and causes an input error. A bjmt control reaches the rate calculation successfully. The manual should remove these driver options and direct covariance-based calculations to envelope_ibs.

The script calls ibs1 with a working bjmt control and the advertised kubo selector. It
separately invokes the production ibs_ring executable with use_t6_cache and requires that
option's specific namelist error. BMAD_DIST selects the build and BMAD_SOURCE supplies the
manual text.
"""

import numpy as np
from _bmad import RunFailure, fortran, run, verdict
from _collective_contracts import rows, source_text

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
call get_command_argument(1,arg)
params%formula=arg
write(*,'(a)') 'READY: calling ibs1'
call ibs1(lat,params,rates,i=1)
write(*,'(a,3es25.16)') 'RESULT rates ',rates%inv_Tp,rates%inv_Ta,rates%inv_Tb
end program
"""
control = rows(fortran(PROGRAM, ["bjmt"]).log)["rates"]
if control.shape != (3,) or not np.isfinite(control).all():
    raise RuntimeError("The bjmt control requires three finite rates")
print(f"bjmt control: rates={control.tolist()} 1/s")
log = fortran(PROGRAM, ["kubo"]).log
rejected = "Invalid IBS formula selected" in log and "READY: calling ibs1" in log
if not rejected:
    rows(log)["rates"]
print(f"ibs1 kubo: rejected={rejected}; expected documented selector accepted")
try:
    log = run("ibs_ring", ["run.in"], files={"run.in": "&parameters use_t6_cache=.true. /\n"}).log
except RunFailure as exc:
    if "Cannot match namelist object name use_t6_cache" not in exc.log:
        raise
    cache_rejected = True
else:
    if "Preparing lattice" not in log:
        raise RuntimeError(log)
    cache_rejected = False
print(
    f"ibs_ring use_t6_cache: rejected={cache_rejected}; expected documented namelist option accepted"
)
manual = source_text("bsim/ibs_ring/doc/ibs_ring.tex")
verdict(
    (rejected and "ibs_formula=`kubo'" in manual) or (cache_rejected and "use_t6_cache" in manual)
)
