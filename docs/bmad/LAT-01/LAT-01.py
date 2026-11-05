"""LAT-01: Live-only ASCII output corrupts retained particles

When live-only ASCII output filters a bunch whose first particle is lost, it can replace the retained particles’ charge, direction and field with values inferred from the discarded particle. A two-particle example retains one particle but reads back charge 1 instead of 2, direction +1 instead of −1, and zero field instead of (0.6, 0.8). Filtering should preserve the supported data of every retained particle. Writing the same bunch without filtering preserves these values.

The script writes the same two particles with and without alive_only, then reads both files. It compares the retained particle’s charge, direction and field with the input values and checks the unfiltered control.
"""

from _bmad import compare, verdict
from _s3_11_particle_io import execute

setup = """
call reallocate_beam(beam, 1, 2)
beam%bunch(1)%particle = coord_struct()
beam%bunch(1)%particle%species = electron$
beam%bunch(1)%particle%p0c = 1e9_rp
beam%bunch(1)%particle%beta = 1
beam%bunch(1)%particle(1)%state = lost$
beam%bunch(1)%particle(1)%charge = 1
beam%bunch(1)%particle(2)%state = alive$
beam%bunch(1)%particle(2)%charge = 2
beam%bunch(1)%particle(2)%direction = -1
beam%bunch(1)%particle(2)%field = [0.6_rp, 0.8_rp]
"""


def case(filtered):
    return execute(
        setup
        + f"""
call write_ascii_beam_file('beam.txt', beam, .true., {".true." if filtered else ".false."})
call read_beam_file('beam.txt', back, bi, err)
if (err) error stop 'ASCII read failed'
if (size(back%bunch) /= 1) error stop 'bunch count changed'
if (size(back%bunch(1)%particle) /= {1 if filtered else 2}) error stop 'wrong filter count'
p => back%bunch(1)%particle({1 if filtered else 2})
if (p%state /= alive$) error stop 'retained state changed'
call emit('retained', [p%charge, real(p%direction,rp), p%field])
"""
    )["retained"]


expected = [2, -1, 0.6, 0.8]
if compare(
    "unfiltered control: charge, direction, field; input identity",
    case(False),
    expected,
    atol=2e-15,
):
    raise RuntimeError("Unfiltered control failed")
verdict(
    compare(
        "alive_only: charge, direction, field; retained input identity",
        case(True),
        expected,
        atol=2e-15,
    )
)
