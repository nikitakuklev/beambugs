"""OPT-33: Periodic-wiggler matrices omit the leading momentum response of focusing

For an off-axis particle in an averaged periodic wiggler, the native matrix omits the leading change of focusing with particle momentum. Differentiating the tracked Hill rotation gives a positive momentum response that the matrix replaces with a much smaller, wrong-sign term. In a 0.48 m, 1.1 T planar wiggler at 0.7 GeV/c, M₄₆/y₀ is −2.264×10⁻⁴ m⁻¹ instead of +0.05259 m⁻¹, where M₄₆ is the derivative of normalized vertical momentum with respect to fractional momentum offset and y₀ is entrance displacement. The omission persists toward weak field and also occurs in the helical model.

The script compares planar weak-field cases and an equal-plane helical case against direct differentiation of the averaged split map. Native transverse blocks and two trajectory stencils serve as controls, including zero field. BMAD_DIST selects the production build.
"""

import numpy as np
from _bmad import compare, verdict
from _srdt_wiggler_maps import matrices, split_jacobian, split_map

LATTICE = """parameter[geometry]=open
parameter[particle]=electron
parameter[p0c]=.7e9
beginning[beta_a]=9
beginning[beta_b]=11
w: wiggler, l=.48, l_period=.16, b_max=FIELD, field_calc=MODEL, tracking_method=bmad_standard
line1: line=(w)
use,line1
"""
defect = False
for model, amplitude in (("planar_model", 2e-8), ("helical_model", 0.0006)):
    fields = (1.1, 0.11, 0.011, 0) if model == "planar_model" else (1.1, 0)
    for field in fields:
        helical = model == "helical_model"
        v = [amplitude if helical else 0, 0, amplitude, 0, 0, 0.19 if helical else 0]
        lattice = LATTICE.replace("FIELD", str(field)).replace("MODEL", model)
        row, m, fd = matrices(lattice, [(1, v)])[0]
        args = (0.48, 0.16, field, 0.7e9, helical)
        jac = split_jacobian(v, *args)
        np.testing.assert_allclose(row[6:10], split_map(v, *args).real, rtol=0, atol=5e-17)
        np.testing.assert_allclose(m[0, :4, :4], jac[:, :4], rtol=0, atol=2e-13)
        np.testing.assert_allclose(
            fd[:, 3, 5] / amplitude, jac[3, 5] / amplitude, rtol=1e-8, atol=2e-11
        )
        bad = compare(
            f"{model}, B={field} T: M46/y0 (1/m)",
            m[0, 3, 5] / amplitude,
            jac[3, 5] / amplitude,
            atol=2e-11,
        )
        if field == 0 and bad:
            raise AssertionError("Zero-field control failed")
        defect |= bad
verdict(defect)
