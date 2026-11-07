"""OPT-32: Helical-wiggler horizontal octupole derivatives use vertical coordinates

In the averaged helical-wiggler model, the native horizontal Jacobian differentiates the cubic horizontal kick using the vertical coordinate. The horizontal derivative must depend on the square of the horizontal coordinate. For a 0.48 m, 1.1 T wiggler at 0.7 GeV/c with horizontal position x = 2.4 mm, vertical position y = 0 and fractional momentum offset δ = 0.19, M₂₁ is −0.04463 m⁻¹ instead of −0.04541 m⁻¹, a 1.720% focusing error. Here M₂₁ is the derivative of horizontal normalized momentum with respect to x. The vertical block, equal-plane coordinates and zero-field controls give the correct transverse derivatives.

The script checks unequal-plane and equal-plane helical-wiggler orbits and zero-field controls. It compares native and auto matrices with a split-map analytic model, complex-step derivatives and two fourth-order trajectory stencils.
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
for field in (1.1, 0):
    lattice = LATTICE.replace("FIELD", str(field)).replace("MODEL", "helical_model")
    vectors = [
        (1, [0.0024, 0, 0, 0, 0, 0.19]),
        (1, [0, 0, 0.0024, 0, 0, 0.19]),
        (1, [0.0006, 0, 0.0006, 0, 0, 0.19]),
    ]
    for (_, v), (row, m, fd) in zip(vectors, matrices(lattice, vectors), strict=True):
        args = (0.48, 0.16, field, 0.7e9, True)
        jac = split_jacobian(v, *args)
        np.testing.assert_allclose(row[6:10], split_map(v, *args).real, rtol=0, atol=5e-17)
        np.testing.assert_allclose(
            fd[:, :4, :4], np.tile(jac[:, :4], (2, 1, 1)), rtol=0, atol=4e-11
        )
        np.testing.assert_allclose(m[0, 2:4, :4], jac[2:4, :4], rtol=0, atol=2e-13)
        bad = compare(
            f"B={field} T, x={v[0]} m, y={v[2]} m: horizontal Jacobian",
            m[0, :2, :4],
            jac[:2, :4],
            atol=2e-13,
        )
        if (field == 0 or v[0] == v[2]) and bad:
            raise AssertionError("Equal-plane or zero-field control failed")
        defect |= bad
verdict(defect)
