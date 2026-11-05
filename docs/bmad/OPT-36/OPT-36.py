"""OPT-36: Radiation integrals omit displaced sextupoles and octupoles

Radiation-integral calculations skip sextupoles and octupoles when both steering attributes are zero, even if the supplied orbit passes through a nonzero field. The curvature integrals I₂ = ∫g²ds and I₃ = ∫|g|³ds must be positive on these displaced trajectories, where g is curvature. A 650 MeV/c electron entering a 6 cm sextupole with normalized strength K₂ = 160 m⁻³ at x = 8 mm gives zero for both integrals, instead of 1.570×10⁻⁶ m⁻¹ and 8.035×10⁻⁹ m⁻². Octupoles and magnet offsets show the same omission; on-axis and zero-strength controls return the correct zero.

The script integrates a symmetry-plane Lorentz trajectory independently for displaced sextupoles and octupoles and compares Bmad radiation integrals in three cache modes. Magnet-offset, zero-strength and on-axis cases are included, and a negligible steering attribute activates the otherwise omitted path as a control. BMAD_DIST selects the production build.
"""

import math

import numpy as np
from _bmad import compare, verdict
from _srdt_wiggler_maps import radiation
from scipy.integrate import solve_ivp

LATTICE = """parameter[geometry]=open
parameter[particle]=electron
parameter[p0c]=650e6
beginning[beta_a]=7
beginning[alpha_a]=.3
beginning[beta_b]=11
beginning[alpha_b]=-.2
particle_start[x]=XSTART
m: SPEC
line1: line=(PIECES*m)
use,line1
"""
defect = False
for order, strength, kind in ((2, 160, "sextupole"), (3, 24000, "octupole")):
    for x, offset, k in (
        (0.008, 0, strength),
        (0, 0.006, strength),
        (0, 0, strength),
        (0.008, 0, 0),
    ):
        length = 0.06
        coefficient = k / math.factorial(order)

        # On the symmetry plane at delta=0, px'= -h and x'=px/sqrt(1-px^2).
        # Integrate I2'=h^2 and I3'=|h|^3 along the same Lorentz trajectory.
        def rhs(_s, state, coefficient=coefficient, order=order):
            u, px = state[:2]
            curvature = coefficient * u**order
            return [px / math.sqrt(1 - px * px), -curvature, curvature**2, abs(curvature) ** 3]

        solution = solve_ivp(
            rhs, (0, length), [x - offset, 0, 0, 0], method="DOP853", rtol=2e-13, atol=1e-22
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        expected = solution.y[2:, -1]
        # A positive input-only bound also excludes zero without an ODE solver.
        radius = abs(x - offset)
        max_force = abs(coefficient) * (2 * radius) ** order
        displacement = max_force * length**2 / (2 * math.sqrt(1 - (max_force * length) ** 2))
        if radius and displacement >= radius / 2:
            raise AssertionError("Displacement bound is outside its domain")
        min_curvature = abs(coefficient) * (radius - displacement) ** order
        lower = length * np.array([min_curvature**2, abs(min_curvature) ** 3])
        np.testing.assert_array_less(lower * (1 - 1e-14) - 1e-30, expected + 1e-30)
        for kick in (0, 1e-24):
            spec = f"{kind}, l={length}, k{order}={k}, x_offset={offset}, fringe_type=none, ds_step=.0002, vkick={kick}"
            lat = LATTICE.replace("XSTART", str(x)).replace("SPEC", spec).replace("PIECES", "1")
            meta, total, end = radiation(lat)
            # Reference energy and rigidity come from the stated input, rather
            # than from radiation_integrals or its dependent element attributes.
            mass, p0c = 510998.95069, 650e6
            energy = math.hypot(p0c, mass)
            np.testing.assert_allclose(
                meta[:4], [p0c, mass, energy, 299792458.0], rtol=2e-15, atol=0
            )
            np.testing.assert_allclose(
                end[:2], solution.y[:2, -1] + [offset, 0], rtol=0, atol=6e-12
            )
            actual = total[2:4]
            if kick:
                # The enabled curvature sampler differentiates over at most 1 mm.
                np.testing.assert_allclose(actual, expected, rtol=1.1e-4, atol=1e-22)
                print(
                    f"{kind}, x={x}, offset={offset}, k={k}: steering control I2,I3={actual.tolist()}; expected={expected.tolist()}"
                )
            else:
                bad = compare(
                    f"{kind}, x={x}, offset={offset}, k={k}: I2 (1/m), I3 (1/m^2)",
                    actual,
                    expected,
                    atol=1e-22,
                    rtol=1.1e-4,
                )
                if (x == offset or k == 0) and bad:
                    raise AssertionError("Zero-curvature control failed")
                # Open-machine sums use gamma^4 I2 and gamma^7 I3 at constant energy.
                gamma = energy / mass
                np.testing.assert_allclose(
                    total[4:6], actual * [gamma**4, gamma**7], rtol=2e-14, atol=1e-22
                )
                print(f"positive bounds I2,I3={lower.tolist()}")
                defect |= bad
verdict(defect)
