"""COL-68: Saved strong-beam centers are overwritten with the weak particle position

When beam–beam slice data are saved from element body coordinates, the saved strong-beam center is overwritten with the weak particle position. A supplied slice center at (3, −5) mm and element offsets (1, −2) mm require the laboratory center (4, −7) mm by coordinate translation. With the weak particle at (4, 6) mm in body coordinates, the routine instead records its laboratory position (5, 4) mm as the strong-beam center. Saving data already in laboratory coordinates preserves the supplied center.

The script calls save_a_step from BMAD_DIST with a parsed, translated beam–beam element and distinct particle and strong-beam positions. It first checks the laboratory-frame copy control, then compares the body-frame saved center with the analytic coordinate translation.
"""

import numpy as np
from _bmad import compare, fortran, verdict

LATTICE = """parameter[geometry]=open
parameter[e_tot]=1e9
parameter[particle]=electron
beginning[beta_a]=10
beginning[beta_b]=20
bb: beambeam, sig_x=.001, sig_y=.0005, beta_a_strong=10, beta_b_strong=20, x_offset=.001, y_offset=-.002
line: line=(bb)
use, line
"""
PROGRAM = """program probe
use bmad
implicit none
type(lat_struct), target :: lat
type(coord_struct) particle
type(track_struct) track
type(strong_beam_struct) strong
logical err, local
integer k
call bmad_parser('lat.bmad',lat,err_flag=err)
if(err) stop 1
call init_coord(particle,[.004_rp,0.0_rp,.006_rp,0.0_rp,0.0_rp,0.0_rp],lat%ele(1),upstream_end$)
strong%x_center=.003_rp
strong%y_center=-.005_rp
do k=1,2
  local=(k==2)
  track%n_pt=-1
  call save_a_step(track,lat%ele(1),lat%param,local,particle,0.0_rp,strong_beam=strong)
  if(track%n_pt/=0) stop 2
  write(*,'(a,i1,4es25.16)') 'CENTER ',k,track%pt(0)%strong_beam%x_center, &
      track%pt(0)%strong_beam%y_center,track%pt(0)%orb%vec(1),track%pt(0)%orb%vec(3)
enddo
end program
"""
output = fortran(PROGRAM, files={"lat.bmad": LATTICE}).log
rows = np.array(
    [
        [float(v) for v in line.split()[2:]]
        for line in output.splitlines()
        if line.startswith("CENTER ")
    ]
)
if rows.shape != (2, 4) or not np.isfinite(rows).all():
    raise RuntimeError("Expected two complete finite center records")
center_body = np.array([0.003, -0.005])
offset = np.array([0.001, -0.002])
particle_body = np.array([0.004, 0.006])
if compare("laboratory-frame supplied-center control (m)", rows[0, :2], center_body, atol=1e-16):
    raise RuntimeError("Laboratory-frame control failed")
# Pure translations give r_lab = r_body + offset for both beam and particle.
if compare("particle translation control (m)", rows[1, 2:], particle_body + offset, atol=1e-16):
    raise RuntimeError("Body-to-laboratory translation control failed")
verdict(
    compare(
        "saved strong-beam center (m), supplied center plus element offset",
        rows[1, :2],
        center_body + offset,
        atol=1e-16,
    )
)
