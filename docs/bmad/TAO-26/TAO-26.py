"""TAO-26: Ad hoc pipe srdt expressions return uncomputed values as valid

Evaluating an ad hoc srdt expression through Tao’s pipe interface returns a numerical value before the resonance-driving terms have been calculated. In a ring with either zero sextupole strength or K₂L = 10 m⁻², the ad hoc h₂₁₀₀₀ magnitude is the same 0.01000 m⁻¹ᐟ², while initialized data and an analytic sextupole formula give 0 and 3.990 m⁻¹ᐟ² respectively. The ad hoc h₂₂₀₀₀ real part is −1.000 m⁻¹ in both cases. Calculation is enabled by initialized data, but the expression path reads the stored fields and marks them valid without enabling it.

The script compares ad hoc queries with initialized srdt data in separate fresh processes for zero and nonzero sextupole strength. It checks the initialized values against an analytic thin-sextupole term using independent periodic optics, and prints the second uncomputed term as well.
"""

from _bmad import compare, verdict
from _tao_contracts_normal_form import THIN_RING, evaluated, tao, thin_beta

expression = "lat::srdt.h21000.a"
init = """&tao_d2_data
 d2_data%name = 'rdt'
 n_d1_data = 1
/
&tao_d1_data
 ix_d1_data = 1
 d1_data%name = 'h'
 datum(1)%data_type = 'srdt.h21000.a'
/
"""
bad = False
for strength in (0, 10):
    # One sextupole at phase zero: |h21000|=K2L*beta**1.5/16.
    expected = strength * thin_beta() ** 1.5 / 16
    lat = THIN_RING.format(strength=strength)
    control = tao(
        lat,
        ["pipe evaluate " + expression],
        files={"rdt.init": init},
        options=["-lat", "lat.bmad", "-init", "rdt.init"],
    )
    assert not compare(
        f"K2L={strength}, initialized srdt control",
        evaluated(control.log, expression),
        expected,
        atol=2e-13,
    )
    result = tao(lat, ["pipe evaluate " + expression, "pipe evaluate lat::srdt.h22000.r"])
    bad |= compare(
        f"K2L={strength}, ad hoc srdt marked valid",
        evaluated(result.log, expression),
        expected,
        atol=2e-13,
    )
    print(
        f"K2L={strength}, ad hoc h22000.r marked valid: {evaluated(result.log, 'lat::srdt.h22000.r'):.16g}"
    )
verdict(bad)
