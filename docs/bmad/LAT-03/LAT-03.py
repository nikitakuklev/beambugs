"""LAT-03: Final initialization counts lost particles and leaves live charge stale

Initializing a beam from a particle file can return live counts and charge that disagree with the final particles. An ASCII bunch with states alive, lost, alive and final charges 2, 4, 6 returns three live particles and live charge 12 instead of two and 8. An all-live HDF5 bunch rescaled from total charge 6 to 12 keeps live charge 6. Live metadata must be computed from the final charges of particles whose state is alive.

The script initializes an ASCII bunch containing a lost particle and an all-live HDF5 bunch whose charge is doubled. It derives the expected live metadata by masking the returned final particle charges with the alive state.
"""

from _bmad import compare, verdict
from _s3_11_particle_io import COUNT_FILE, execute

ascii_result = execute(
    """
bi%position_file = 'counts.txt'
bi%bunch_charge = 12
call init_beam_distribution(ele, param, bi, beam, err)
if (err) error stop 'ASCII initialization failed'
call emit('states', real(beam%bunch(1)%particle%state,rp))
call emit('charges', beam%bunch(1)%particle%charge)
call emit('live', [real(beam%bunch(1)%n_live,rp), beam%bunch(1)%charge_live])
""",
    files={"counts.txt": COUNT_FILE},
)
if compare("ASCII final states", ascii_result["states"], [1, 2, 1], atol=0) or compare(
    "ASCII final charges (C)", ascii_result["charges"], [2, 4, 6], atol=0
):
    raise RuntimeError("ASCII particle control failed")
alive = ascii_result["states"] == 1
expected = [alive.sum(), ascii_result["charges"][alive].sum()]
defect = compare(
    "ASCII n_live, charge_live (C); final alive mask", ascii_result["live"], expected, atol=0
)
h5 = execute("""
call reallocate_beam(beam, 1, 2)
beam%bunch(1)%particle%state = alive$
beam%bunch(1)%particle%species = electron$
beam%bunch(1)%particle%p0c = 1e9_rp
beam%bunch(1)%particle%beta = 1
beam%bunch(1)%particle%charge = 3
beam%bunch(1)%charge_tot = 6
beam%bunch(1)%charge_live = 6
call hdf5_write_beam('beam.h5', beam%bunch, .false., err)
if (err) error stop 'HDF5 write failed'
bi%position_file = 'beam.h5'
bi%bunch_charge = 12
call init_beam_distribution(ele, param, bi, back, err)
if (err) error stop 'HDF5 initialization failed'
call emit('states', real(back%bunch(1)%particle%state,rp))
call emit('charges', back%bunch(1)%particle%charge)
call emit('totals', [back%bunch(1)%charge_live, back%bunch(1)%charge_tot])
""")
if compare("HDF5 states", h5["states"], [1, 1], atol=0) or compare(
    "HDF5 charges (C)", h5["charges"], [6, 6], atol=0
):
    raise RuntimeError("HDF5 particle control failed")
if compare("HDF5 charge_tot (C); requested charge", h5["totals"][1], 12, atol=0):
    raise RuntimeError("HDF5 total-charge control failed")
defect |= compare(
    "HDF5 charge_live (C); final alive charge sum",
    h5["totals"][0],
    h5["charges"][h5["states"] == 1].sum(),
    atol=0,
)
verdict(defect)
