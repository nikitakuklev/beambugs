"""Small native particle-file cases and strict result parsing for this family of reproducers."""

from collections.abc import Mapping

import numpy as np
from _bmad import fortran

HEADER_FILE = """# species = electron
# charge_tot = 1
#! x charge state
0 1 alive
# species = proton
# charge_tot = 12
#! x charge state
0 1 alive
"""
COUNT_FILE = """# species = electron
# p0c = 1e9
# charge_tot = 6
#! x charge state
0 1 alive
0 2 lost
0 3 alive
"""


def program(body: str) -> str:
    return (
        """
program particle_io
use beam_utils
use random_mod
implicit none
type(ele_struct) ele
type(lat_param_struct) param
type(beam_init_struct) bi
type(beam_struct), target :: beam, back
type(coord_struct), pointer :: p
type(random_state_struct) rs, initial
real(rp) x, v(6), scalar(6), beta0
integer i, j, k
logical err
character(40) label
call ran_seed_put(12345)
global_com%exit_on_error = .false.
call init_ele(ele, drift$)
ele%value(p0c$) = 1e9_rp
ele%value(p0c_start$) = 1e9_rp
ele%value(e_tot$) = hypot(1e9_rp, mass_of(electron$))
ele%value(e_tot_start$) = ele%value(e_tot$)
ele%a%beta = 1
ele%b%beta = 1
ele%orientation = 1
param%particle = electron$
bi%n_particle = 0
"""
        + body
        + """
print *, 'COMPLETED'
contains
subroutine emit(name, values)
character(*) name
real(rp) values(:)
write(*, '(3a, *(es26.17e3))') 'RESULT ', name, ' ', values
end subroutine
end program
"""
    )


def rows(log: str) -> dict[str, np.ndarray]:
    result = {}
    for line in log.splitlines():
        if line.startswith("RESULT "):
            _, name, *values = line.split()
            if name in result or not values:
                raise RuntimeError(f"Duplicate or empty result: {line}")
            values_array = np.array([float(v) for v in values])
            if not np.isfinite(values_array).all():
                raise RuntimeError(f"Nonfinite result: {line}")
            result[name] = values_array
    if not result:
        raise RuntimeError(f"No results:\n{log}")
    return result


def execute(body: str, *, files: Mapping[str, str | bytes] | None = None) -> dict[str, np.ndarray]:
    log = fortran(program(body), files=files).log
    if "COMPLETED" not in log or any(s in log for s in ("[ERROR", "[FATAL", "[ABORT")):
        raise RuntimeError(log)
    return rows(log)


# Each read uses a fresh process, so reader global state cannot leak between policies.
POLICY_SETUP = """
ele%value(p0c$) = 1e6_rp
ele%value(p0c_start$) = 1e6_rp
ele%value(e_tot$) = hypot(1e6_rp, m_electron)
call reallocate_beam(beam, 1, 1)
p => beam%bunch(1)%particle(1)
call init_coord(p, [0.001_rp, 0.04_rp, -0.002_rp, -0.02_rp, 0.01_rp, 0.1_rp], &
                ele, upstream_end$, electron$)
p%charge = 1e-12_rp
p%t = 4.1e-9_rp
beam%bunch(1)%charge_tot = 1e-12_rp
beam%bunch(1)%charge_live = 1e-12_rp
beam%bunch(1)%n_live = 1
call emit('input', [p%vec, p%p0c, p%beta, real(p%t,rp), m_electron])
call hdf5_write_beam('beam.h5', beam%bunch, .false., err)
if (err) error stop 'HDF5 write failed'
ele%value(p0c$) = 1.7e6_rp
"""


def policy(mode: str) -> dict[str, np.ndarray]:
    options = {
        "omitted": "",
        "true": ", conserve_momentum=.true.",
        "false": ", conserve_momentum=.false.",
    }
    return execute(
        POLICY_SETUP
        + f"""
call read_beam_file('beam.h5', back, bi, err, ele{options[mode]})
if (err) error stop 'HDF5 read failed'
if (size(back%bunch) /= 1) error stop 'bunch count changed'
if (size(back%bunch(1)%particle) /= 1) error stop 'particle count changed'
p => back%bunch(1)%particle(1)
call emit('output', [p%vec, p%p0c, p%beta, real(p%t,rp)])
"""
    )
