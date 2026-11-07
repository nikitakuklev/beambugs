"""TAO-06: RMS reduction integrates squared endpoints instead of the linear interpolant

Tao’s RMS range reduction squares the endpoint samples and applies trapezoidal quadrature,
instead of integrating the square of the documented linear interpolation. An exactly linear,
centered orbit over a one-metre drift returns 0.5000 mm instead of 0.2887 mm, making the
variance three times too large. Its mean and integral are zero. Integrating a linear segment
with endpoint values a and b requires a second moment proportional to (a² + ab + b²)/3.

The script centers an orbit in a drift and requests RMS, average, and integral constraints. It
checks the endpoints and zero mean, then compares RMS with direct integration of the linear orbit.
"""

import math

from _bmad import compare, verdict
from _tao_optimization_io import LINE, RMS_INIT, block, control, data_values, numbers, tao

# Over a one-metre drift x(s)=dx*(s-.5), so integral(x^2 ds)=dx^2/12.
dx = 0.001 / math.sqrt(1 - 0.001**2)
commands = [
    f"set particle_start x = {-dx / 2:.17g}",
    "pipe lat_list beginning,d1 orbit.vec.1",
    "pipe data_parameter range.x model_value",
]
result = tao(commands, lattice=LINE, init=RMS_INIT)
control(
    "linear orbit endpoints (m)",
    numbers(block(result.log, commands[1])).ravel(),
    [-dx / 2, dx / 2],
    atol=1e-16,
)
values = data_values(result.log, commands[2])
control("centered mean (m) and integral (m^2)", values[1:], [0, 0], atol=1e-16)
verdict(
    compare("centered linear-orbit RMS (m), dx/sqrt(12)", values[0], dx / math.sqrt(12), atol=1e-15)
)
