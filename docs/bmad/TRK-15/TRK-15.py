"""TRK-15: PTC lcavity updates convert the RF phase to radians twice

When lattice bookkeeping updates an lcavity's existing PTC fibre, its element representation,
the phase conversion multiplies turns by 2π twice. An update with unchanged attributes must
preserve the constructed phase and particle map. With phase φ₀ = 0.13 turns, the correct lag π/2
− 2πφ₀ = 0.7540 rad becomes π/2 − (2π)²φ₀ = −3.561 rad. In a 1 MV cavity, the reference
electron's energy gain changes from 684.5 keV to 407.6 keV after that unchanged bookkeeping
pass. Bmad standard tracking avoids this PTC update path.

The script compiles a Fortran program that builds a PTC layout for a lattice holding one
traveling-wave lcavity, tracks the reference particle with PTC, marks the cavity's attributes stale
without changing them, reruns the lattice bookkeeping and tracks again. It compares the lag and
energy gain before and after the update with π/2 − 2π φ0 and V cos(2π φ0). The first track, on the
fibre as constructed, is the control.
"""

import numpy as np
from _bmad import compare, fortran, verdict

LATTICE = """no_digested
parameter[particle] = electron
parameter[p0c] = 5e6
parameter[geometry] = open
lc: lcavity, l = 0, rf_frequency = 1e9, voltage = 1e6, phi0 = 0.13, n_rf_steps = 4, cavity_type = traveling_wave
ln: line = (lc)
use, ln
"""

PROGRAM = """
program ptc_track_probe
use bmad
use ptc_layout_mod, only: lat_to_ptc_layout, ptc_track_all
implicit none
type(lat_struct), target :: lat
type(ele_struct), pointer :: ele
type(coord_struct), allocatable :: orb(:)
integer :: i
real(rp) :: e0, e1
call bmad_parser('lat.bmad', lat)
call lat_to_ptc_layout(lat)
ele => lat%ele(1)
allocate(orb(0:lat%n_ele_track))
do i = 1, 2
  call init_coord(orb(0), ele=lat%ele(0), element_end=downstream_end$)
  call ptc_track_all(lat%branch(0), orb)
  if (orb(1)%state /= alive$) error stop 'particle lost'
  ! Physical total energies before and after the cavity, eV.
  e0 = hypot(orb(0)%p0c * (1 + orb(0)%vec(6)), mass_of(orb(0)%species))
  e1 = hypot(orb(1)%p0c * (1 + orb(1)%vec(6)), mass_of(orb(1)%species))
  print '(a,i0,4es25.16e3)', 'RESULT ', i, ele%ptc_fibre%mag%lag, e1 - e0, ele%value(voltage$), ele%value(phi0$)
  ! An unchanged attribute group: the bookkeeper updates the existing fibre.
  call set_ele_status_stale(ele, attribute_group$)
  call lattice_bookkeeper(lat)
enddo
end program
"""

log = fortran(PROGRAM, files={"lat.bmad": LATTICE}).log
rows = {}
for line in log.splitlines():
    words = line.split()
    if len(words) == 6 and words[0] == "RESULT":
        rows[int(words[1])] = np.array([float(w) for w in words[2:]])
if set(rows) != {1, 2}:
    raise RuntimeError(f"Missing results:\n{log}")

voltage, phi0 = rows[1][2], rows[1][3]
np.testing.assert_array_equal(rows[1][2:], [1e6, 0.13])
np.testing.assert_array_equal(rows[2][2:], rows[1][2:])
# Bmad's lcavity phase in PTC: lag = pi/2 - 2 pi phi0 (phi0 in turns, 0 on crest), so the
# reference particle gains V cos(2 pi phi0).
lag = np.pi / 2 - 2 * np.pi * phi0
gain = voltage * np.cos(2 * np.pi * phi0)
# Reconstructing PTC's energies with Bmad's electron mass leaves the constructed fibre's gain
# 5.6e-5 eV from V cos(2 pi phi0); 1e-3 eV covers it and is far below the error under test.
control = compare(
    "constructed fibre: lag (rad), expected pi/2 - 2 pi phi0", rows[1][0], lag, atol=1e-12
)
control |= compare(
    "constructed fibre: energy gain (eV), expected V cos(2 pi phi0)", rows[1][1], gain, atol=1e-3
)
if control:
    raise RuntimeError("The constructed fibre disagrees; the comparison is not valid.")
defect = compare(
    "after unchanged bookkeeping: lag (rad), expected pi/2 - 2 pi phi0", rows[2][0], lag, atol=1e-12
)
defect |= compare(
    "after unchanged bookkeeping: energy gain (eV), expected the constructed fibre's gain",
    rows[2][1],
    rows[1][1],
    atol=1e-6,
)
verdict(defect)
