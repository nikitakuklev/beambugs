"""TRK-34: Moved wall-transition apertures compare different coordinate frames

A moved wall-transition aperture compares the current particle position in the aperture frame with the previous position in the laboratory frame. A particle remaining at the aperture centre can therefore appear to cross a wall and be marked lost. For a rectangular aperture with 10 mm half-widths and a 100 mm horizontal offset, identical old and current horizontal positions of 100 mm produce a loss. The centred-aperture control stays alive, as required for a stationary transverse point.

The script checks a stationary transverse point with zero and nonzero aperture offsets. It compares the returned state with alive, since neither case crosses the wall.
"""

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
type(coord_struct) old
integer j
call bmad_parser('lat.bmad',lat,err_flag=err)
if (err) error stop 'parse failed'
lat%ele(1)%aperture_type=rectangular$
lat%ele(1)%aperture_at=wall_transition$
lat%ele(1)%offset_moves_aperture=.true.
lat%ele(1)%value(x1_limit$)=.01_rp;lat%ele(1)%value(x2_limit$)=.01_rp
lat%ele(1)%value(y1_limit$)=.01_rp;lat%ele(1)%value(y2_limit$)=.01_rp
do j=0,1
 lat%ele(1)%value(x_offset_tot$)=.1_rp*j
 v=0;v(1)=.1_rp*j
 call init_coord(a,v,lat%ele(1),upstream_end$)
 old=a;old%s=.2_rp;a%s=.3_rp
 call check_aperture_limit(a,lat%ele(1),in_between$,lat%param,old)
 write(*,'(a,i0,2i5)') 'RESULT case',j,a%state,alive$
enddo
end program
"""
LATTICE = """
no_digested
parameter[geometry]=open
parameter[particle]=electron
beginning[p0c]=1e6
e: quadrupole,l=1,k1=.3,fringe_type=none
lat: line=(e)
use,lat
"""
rows = records(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
if compare("centred aperture, state=alive", rows["case0"][:1], rows["case0"][1:], atol=0):
    raise RuntimeError("Centred-aperture control failed")
verdict(compare("moved aperture, state=alive", rows["case1"][:1], rows["case1"][1:], atol=0))
