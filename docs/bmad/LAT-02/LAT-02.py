"""LAT-02: ASCII reading consumes the next bunch’s first header line

When a modern ASCII particle file contains several bunches, the reader discards the first header line of every bunch after the first. An electron bunch followed by a proton bunch reads back species codes −1 and −999 instead of −1 and 2, although the later charge headers survive. Each recognized header assignment must apply to its own bunch. Inserting a comment before the second species header preserves both species.

The script reads an electron bunch followed by a proton bunch from inline ASCII. A second run adds a comment before the proton header as a control; both runs check that the later charge assignments survive.
"""

from _bmad import compare, verdict
from _s3_11_particle_io import HEADER_FILE, execute

body = """
call read_beam_file('beam.txt', beam, bi, err)
if (err) error stop 'ASCII read failed'
if (size(beam%bunch) /= 2) error stop 'wrong bunch count'
do i = 1, 2
  if (size(beam%bunch(i)%particle) /= 1) error stop 'wrong particle count'
enddo
call emit('species', real([beam%bunch(1)%particle(1)%species, beam%bunch(2)%particle(1)%species],rp))
call emit('charges', beam%bunch%charge_tot)
"""
control = execute(
    body,
    files={
        "beam.txt": HEADER_FILE.replace("# species = proton", "# next bunch\n# species = proton")
    },
)
observed = execute(body, files={"beam.txt": HEADER_FILE})
for name, result in [("comment control", control), ("leading species", observed)]:
    if compare(f"{name}: charge_tot; header values", result["charges"], [1, 12], atol=0):
        raise RuntimeError("Charge header control failed")
if compare("comment control: species; electron/proton codes", control["species"], [-1, 2], atol=0):
    raise RuntimeError("Species control failed")
verdict(compare("leading species: electron/proton codes", observed["species"], [-1, 2], atol=0))
