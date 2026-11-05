"""TAO-07: Four a-mode chromatic pipe selectors read b-mode fields

The a-mode chromatic derivative selectors in Tao’s pipe interface return b-mode values. With a/b
beta derivatives set to 11/22 and alpha derivatives set to 33/44, lat_list reports [22, 44] for
the a mode, and ele:param returns 22 for its beta derivative. The ele:twiss response correctly
returns the a-mode values 11 and 33. The four a-mode beta, alpha, dispersion, and
dispersion-slope selectors must read their matching a-mode fields.

The script initializes distinct a- and b-mode beta and alpha derivatives, then queries lat_list,
ele:param, and ele:twiss. It uses ele:twiss and the lattice initial conditions as controls.
BMAD_DIST supplies Tao.
"""

from _bmad import compare, verdict
from _tao_optimization_io import ABS_INIT, LINE, block, control, numbers, parameters, tao

commands = [
    "pipe lat_list beginning ele.a.dbeta_dpz,ele.a.dalpha_dpz",
    "pipe ele:param beginning ele.a.dbeta_dpz",
    "pipe ele:twiss beginning",
]
result = tao(commands, lattice=LINE, init=ABS_INIT)
twiss = parameters(block(result.log, commands[2]))
control(
    "ele:twiss a-mode beta/alpha derivatives",
    [float(twiss["dbeta_dpz_a"]), float(twiss["dalpha_dpz_a"])],
    [11, 33],
    atol=5e-12,
)
defect = compare(
    "lat_list a-mode beta/alpha derivatives",
    numbers(block(result.log, commands[0])).ravel(),
    [11, 33],
    atol=5e-12,
)
param = parameters(block(result.log, commands[1]))
defect |= compare(
    "ele:param a-mode beta derivative", float(param["ele.a.dbeta_dpz"]), 11, atol=5e-12
)
verdict(defect)
