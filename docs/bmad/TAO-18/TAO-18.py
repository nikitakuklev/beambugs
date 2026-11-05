"""TAO-18: Disabled lattice calculation makes pipe merit report exactly zero

Querying pipe merit after disabling Tao lattice calculation reports zero even when the objective has a nonzero minimum. A weighted steering fit has analytic minimum 0.0003200, which Tao returns while calculation is enabled; the next query returns zero after lattice_calc_on is set false. Zero is presented as a valid scalar rather than an unavailable objective. The merit routine returns before evaluating the objective or setting its optional calculation-status argument.

The script solves a one-kicker weighted least-squares problem and checks the enabled result against its analytic minimum. It then disables lattice calculation and repeats the scalar merit query without changing the fit. Reenabling calculation checks that the objective is recovered.
"""

from _bmad import compare, verdict
from _tao_contracts_normal_form import tao

LATTICE = """parameter[geometry] = open
parameter[particle] = electron
parameter[p0c] = 1e6
beginning[beta_a] = 3
beginning[beta_b] = 4
k: hkicker, kick=0.003
d: drift, l=2
lat: line=(k,d)
use, lat
"""
INIT = """&tao_design_lattice
 n_universes = 1
 design_lattice(1)%file = 'lat.bmad'
/
&tao_params
 global%plot_on = F
 global%n_opti_cycles = 30
 global%n_opti_loops = 2
 global%opt_with_ref = F
 global%opt_with_base = F
/
&tao_d2_data
 d2_data%name = 'match'
 n_d1_data = 1
/
&tao_d1_data
 ix_d1_data = 1
 d1_data%name = 'px'
 default_data_type = 'orbit.px'
 default_merit_type = 'target'
 default_weight = 4
 datum(1)%ele_name = 'END'
 datum(1)%meas = 0.02
/
&tao_var
 v1_var%name = 'steer'
 default_attribute = 'kick'
 default_step = 1e-6
 default_merit_type = 'target'
 default_meas = 0
 default_weight = 1
 var(1)%ele_name = 'K'
/
"""
result = tao(
    LATTICE,
    [
        "run lm",
        "pipe -write enabled.txt merit",
        "set global lattice_calc_on = F",
        "pipe -write disabled.txt merit",
        "set global lattice_calc_on = T",
        "pipe -write restored.txt merit",
    ],
    options=["-init", "opt.init"],
    files={"opt.init": INIT},
    outputs=["enabled.txt", "disabled.txt", "restored.txt"],
)
enabled, disabled = [float(result.outputs[name]) for name in ("enabled.txt", "disabled.txt")]
# M(k)=4*(k-.02)**2+k**2, so k*=.016 and M(k*)=.00032.
expected = 4 * (0.016 - 0.02) ** 2 + 0.016**2
assert not compare(
    "enabled optimum merit, analytic weighted least squares", enabled, expected, atol=1e-18
)
assert not compare(
    "restored calculation merit, unchanged analytic objective",
    float(result.outputs["restored.txt"]),
    expected,
    atol=1e-18,
)
verdict(
    compare(
        "disabled scalar merit; a valid objective retains the nonzero minimum",
        disabled,
        expected,
        atol=1e-18,
    )
)
