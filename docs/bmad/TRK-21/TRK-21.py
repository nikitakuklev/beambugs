"""TRK-21: PTC-to-Bmad cavity update inverts voltage and lcavity phase wrongly

Synchronizing an unchanged PTC cavity back into Bmad changes its voltage and linear-cavity phase. The inverse conversion omits the active length in the voltage and places the phase parentheses incorrectly. A 0.5 m standing-wave linear cavity specified with 1 MV and phase 0.13 turns becomes 3.336 MV and 1.451 turns. Inverting the forward conversion requires the original voltage and phase to be preserved.

The script builds a standing-wave PTC cavity and immediately reads its unchanged fibre back into Bmad, using BMAD_DIST. Voltage and phase must survive this conversion round trip.
"""

from _bmad import compare, fortran, verdict
from _rf_and_spin import control, rows

PROGRAM = """program rev
use bmad
use ptc_layout_mod, only: lat_to_ptc_layout, update_ele_from_fibre
implicit none
type(lat_struct), target :: lat
type(ele_struct), pointer :: c
character(200) file
logical err
file='cavity.bmad'
call bmad_parser(trim(file), lat, err_flag=err)
if (err) error stop 'parse'
call lat_to_ptc_layout(lat)
c => lat%ele(1)
write(*,'(a,3es25.16)') 'RESULT before ', c%value(voltage$), c%value(phi0$), c%value(l_active$)
call update_ele_from_fibre(c)
write(*,'(a,3es25.16)') 'RESULT after ', c%value(voltage$), c%value(phi0$), c%value(l_active$)
end program
"""
LATTICE = """no_digested
parameter[particle] = electron
parameter[p0c] = 1e8
parameter[geometry] = open
c: lcavity, l = 0.5, voltage = 1e6, phi0 = 0.13, rf_frequency = 5e8, ds_step = 0.5, tracking_method = symp_lie_ptc, autoscale_phase = F, autoscale_amplitude = F
beamline: line = (c)
use, beamline
"""
d = rows(fortran(PROGRAM, files={"cavity.bmad": LATTICE}).log)
control("initial voltage (V) and phase (turns)", d["before"][:2], [1e6, 0.13], atol=1e-12)
control("active length preserved (m)", d["after"][2], d["before"][2], atol=1e-15)
# The forward map uses volt=2e-6*V/l_active and lag=pi/2-2*pi*phi0.
voltage_bad = compare("voltage round trip (V)", d["after"][0], d["before"][0], atol=1e-8)
phase_bad = compare("phase round trip (turns)", d["after"][1], d["before"][1], atol=2e-14)
verdict(voltage_bad or phase_bad)
