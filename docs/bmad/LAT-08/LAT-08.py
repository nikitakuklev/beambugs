"""LAT-08: One short HDF5 bunch reduces later bunches’ particle limits

When an HDF5 file contains a bunch shorter than the requested particle count, the reader carries that smaller limit into later bunches. A file with two and five particles, read with a request for four per bunch, returns two and two after a warning. The documented contract requires a shortage error. If reading continues, applying the request independently to each bunch would retain two and four.

The script writes HDF5 bunches containing two and five particles, then requests either two or four per bunch. It checks the documented error flag for the shortage case and, when the reader continues, compares its counts with independent per-bunch limits.
"""

from _bmad import compare, verdict
from _s3_11_particle_io import execute


def case(request):
    return execute(f"""
call reallocate_beam(beam, 2, 5)
call reallocate_bunch(beam%bunch(1), 2)
do i = 1, 2
  beam%bunch(i)%particle%state = alive$
  beam%bunch(i)%particle%species = electron$
  beam%bunch(i)%particle%p0c = 1e9_rp
  beam%bunch(i)%particle%beta = 1
  beam%bunch(i)%particle%charge = 1
enddo
call hdf5_write_beam('beam.h5', beam%bunch, .false., err)
if (err) error stop 'HDF5 write failed'
bi%n_particle = {request}
call read_beam_file('beam.h5', back, bi, err)
call emit('error', [real(merge(1,0,err),rp)])
if (.not. err) then
  if (size(back%bunch) /= 2) error stop 'wrong bunch count'
  call emit('counts', real([size(back%bunch(1)%particle), size(back%bunch(2)%particle)],rp))
endif
""")


control = case(2)
if compare("request=2 error flag", control["error"][0], 0, atol=0) or compare(
    "request=2 counts", control["counts"], [2, 2], atol=0
):
    raise RuntimeError("No-shortage control failed")
shortage = case(4)
defect = compare(
    "request=4 error flag; manual requires shortage rejection", shortage["error"][0], 1, atol=0
)
if shortage["error"][0] == 0:
    # This is the conditional continuation result, not the manual's required error.
    defect |= compare(
        "request=4 counts if continuing; min(file_count, request) per bunch",
        shortage["counts"],
        [2, 4],
        atol=0,
    )
verdict(defect)
