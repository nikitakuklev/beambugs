"""COL-34: Envelope IBS argument comments call a damping time a rate

When callers follow the envelope IBS argument comments, they supply a damping rate where the tail-cut calculation requires a damping time in seconds. The implementation divides a ratio of volume to speed by that input to form a squared impact parameter, which requires time units. In a fixed diagonal beam covariance, passing 100 instead of a 0.01 s damping time increases the diffusion norm by a factor of 1.621. The envelope driver already passes the longest mode damping time; the argument comments should describe that convention.

The script evaluates the envelope IBS kernel on the same diagonal covariance with a damping time and
its reciprocal. It compares the finite diffusion outputs and reads the argument descriptions from
BMAD_SOURCE.
"""

import numpy as np
from _bmad import fortran, verdict
from _collective_contracts import rows, source_text

r = rows(
    fortran("""program damping_time
use envelope_mod
implicit none
real(rp) sigma(6,6),mat(6,6),tau(2)
integer i,j
sigma=0
sigma(1,1)=1e-6_rp
sigma(3,3)=1e-6_rp
sigma(5,5)=1e-6_rp
sigma(2,2)=1e-10_rp
sigma(4,4)=4e-10_rp
sigma(6,6)=9e-6_rp
tau=[0.01_rp,100.0_rp]
do j=1,2
 call beam_envelope_ibs(sigma,mat,.true.,tau(j),1000*m_electron,1e8_rp,electron$)
 write(*,'(a,i0,6es25.16)') 'RESULT tau',j,(mat(i,i),i=1,6)
enddo
end program
""").log
)
if not np.isfinite([r["tau1"], r["tau2"]]).all() or np.linalg.norm(r["tau1"]) == 0:
    raise RuntimeError("Nonfinite or zero diffusion fixture")
print(f"tau=0.01 s: diagonal diffusion={r['tau1'].tolist()} per m")
print(f"tau=100 (inverse of 0.01): diagonal diffusion={r['tau2'].tolist()} per m")
print(
    f"diffusion norm ratio, inverse-time input / time input={np.linalg.norm(r['tau2']) / np.linalg.norm(r['tau1']):.16g}"
)
text = source_text("bmad/multiparticle/envelope_mod.f90")
wrong_header = "horizontal betatron damping rate" in text
print(f"API headers label tau as damping rate={wrong_header}; tail-cut dimensions require seconds")
# vol/N has units m^3 and ptrans*c has units m/s. Dividing their
# ratio by tau in seconds gives bmin2^2 in m^2; a rate gives m^2*s^2.
verdict(wrong_header and not np.allclose(r["tau1"], r["tau2"], rtol=1e-12, atol=0))
