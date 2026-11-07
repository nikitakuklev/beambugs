"""TRK-40: The interior-start guard compares particle state with a location enum

With init_to_edge disabled, standard tracking accepts a live particle inside an element instead of reporting that this tracker requires an edge start. The guard compares the particle state with the inside location constant, so it cannot detect a live interior start. A particle starting halfway through a 1.2 m quadrupole receives the full 1.2 m map and returns no error. For entrance x = 1 mm and normalized horizontal momentum 0.0003, the returned x = 0.8394 mm and normalized horizontal momentum −0.0005447 match the full-element solution.

The script compares an edge-start control with the independent quadrupole body solution. It then sets location to inside at s = 0.6 m and disables automatic edge initialization, checking whether track1 reports an error.
"""

import numpy as np
from _bmad import compare, fortran, verdict
from _orbital_matrices_radiation import records

PROGRAM = """
program probe
use bmad

implicit none
type(lat_struct), target :: lat
type(coord_struct) a,b
real(rp) v(6)
logical err

call bmad_parser('lat.bmad',lat,err_flag=err)
if (err) error stop 'parse failed'
global_com%exit_on_error=.false.
v=0;v(1)=.001_rp;v(2)=.0003_rp
call init_coord(a,v,lat%ele(1),upstream_end$)
call track1(a,lat%ele(1),lat%param,b,err_flag=err)
if (err.or.b%state/=alive$) error stop 'edge control failed'
write(*,'(a,2es26.17)') 'RESULT edge ',b%vec(1:2)
a%location=inside$;a%s=lat%ele(1)%s_start+.6_rp
call track1(a,lat%ele(1),lat%param,b,err_flag=err,init_to_edge=.false.)
write(*,'(a,3i4,es26.17)') 'RESULT status ',merge(1,0,err),merge(1,0,b%state==alive$), &
 merge(1,0,a%location==inside$),a%s
write(*,'(a,2es26.17)') 'RESULT inside ',b%vec(1:2)
end program
"""
LATTICE = """
no_digested
parameter[geometry]=open
parameter[particle]=electron
beginning[p0c]=1e9
e: quadrupole,l=1.2,k1=.7,fringe_type=none
lat: line=(e)
use,lat
"""
rows = records(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
k = np.sqrt(0.7)
phase = k * 1.2
expected = [
    0.001 * np.cos(phase) + 0.0003 * np.sin(phase) / k,
    -0.001 * k * np.sin(phase) + 0.0003 * np.cos(phase),
]
if compare("edge-start x (m), px/p0, full quadrupole solution", rows["edge"], expected, atol=2e-16):
    raise RuntimeError("Edge control failed")
if compare("interior setup location=inside, s (m)", rows["status"][2:], [1, 0.6], atol=0):
    raise RuntimeError("Interior setup failed")
defect = compare("interior err_flag, expected rejection", rows["status"][0], 1, atol=0)
if defect:
    if rows["status"][1] != 1 or compare(
        "accepted interior x (m), px/p0, full-element signature",
        rows["inside"],
        expected,
        atol=2e-16,
    ):
        raise RuntimeError("Unexpected accepted-interior map")
verdict(defect)
