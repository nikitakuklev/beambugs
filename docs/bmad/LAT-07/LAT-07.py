"""LAT-07: Modern particle-file readers ignore the requested bunch count

Reading a modern ASCII or HDF5 particle file ignores a nonzero requested bunch count. Requesting one bunch from a two-bunch file returns both with no error. The documented contract requires exactly the requested number, with an error when the file contains too few. A zero request correctly returns all bunches.

The script reads a two-bunch ASCII file and a two-bunch HDF5 file with requests for zero and one bunch. Zero selects all bunches as a control; the nonzero request is checked against the documented exact count.
"""

from _bmad import compare, verdict
from _s3_11_particle_io import HEADER_FILE, execute

h5_setup = """
call reallocate_beam(beam, 2, 2)
do i = 1, 2
  beam%bunch(i)%particle%state = alive$
  beam%bunch(i)%particle%species = electron$
  beam%bunch(i)%particle%p0c = 1e9_rp
  beam%bunch(i)%particle%beta = 1
  beam%bunch(i)%particle%charge = 1
enddo
call hdf5_write_beam('beam.h5', beam%bunch, .false., err)
if (err) error stop 'HDF5 write failed'
"""
defect = False
for kind in ("ASCII", "HDF5"):
    for request in (0, 1):
        setup = h5_setup if kind == "HDF5" else ""
        filename = "beam.h5" if kind == "HDF5" else "beam.txt"
        result = execute(
            setup
            + f"""
bi%n_bunch = {request}
call read_beam_file('{filename}', back, bi, err)
if (err) error stop 'beam read failed'
call emit('count', [real(size(back%bunch),rp)])
""",
            files={
                "beam.txt": HEADER_FILE.replace(
                    "# species = proton", "# next bunch\n# species = proton"
                )
            },
        )
        expected = 2 if request == 0 else request
        mismatch = compare(
            f"{kind} n_bunch={request}; documented selection", result["count"][0], expected, atol=0
        )
        if request == 0 and mismatch:
            raise RuntimeError("All-bunches control failed")
        defect |= mismatch
verdict(defect)
