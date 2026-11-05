"""OPT-16: Normalized Twiss propagation corrupts ordinary dispersion

Enabling normalize_twiss while propagating an off-momentum open lattice applies an
angle-normalized map to ordinary dispersion coordinates. A 1.7 m drift at fractional momentum
offset δ = 0.2 and transverse normalized momenta (0.03, −0.02) gives horizontal dispersion
−0.04256 m instead of the exact tracking derivative −0.03546 m. A thin quadrupole with
integrated strength 0.3 m⁻¹ returns horizontal momentum dispersion −0.004 instead of −0.006;
both elements give the expected values with normalization disabled.

The script tracks an off-momentum drift and thin quadrupole with each normalization setting. It
checks the orbit and entrance dispersion, then compares exit dispersion with derivatives of the
exact drift and thin-lens maps. BMAD_DIST selects the build.
"""

import numpy as np
from _bmad import compare, verdict
from _dispersion_and_moments import OPEN_LINE, execute, lattice, require

defect = False
for element in ("drift", "thin_quad"):
    for normalized in (False, True):
        initial = np.array([0, 0.03, 0, -0.02, 0, 0.2])
        q = 1 + initial[5]
        mat = np.eye(4)
        column = np.zeros(4)
        entrance = np.zeros(4)
        if element == "drift":
            definition = "e: drift, l=1.7"
            # Exact drift: x_out=x+L*px/sqrt(q^2-px^2-py^2).
            ps = np.sqrt(q * q - initial[1] ** 2 - initial[3] ** 2)
            column[[0, 2]] = -1.7 * initial[[1, 3]] * q / ps**3
            orbit = initial[:4].copy()
            orbit[[0, 2]] += 1.7 * initial[[1, 3]] / ps
        else:
            definition = "e: multipole, k1l=0.3"
            initial[[0, 2]] = [0.01, -0.007]
            entrance = np.array([0.04, 0.006, -0.02, 0.003])
            mat[1, 0], mat[3, 2] = -0.3, 0.3
            orbit = mat @ initial[:4]
        expected = mat @ entrance + column
        extra = f"bmad_com[normalize_twiss] = {'T' if normalized else 'F'}\n"
        extra += "\n".join(
            f"beginning[{name}] = {value:.17g}"
            for name, value in zip(("eta_x", "etap_x", "eta_y", "etap_y"), entrance, strict=True)
        )
        values = execute(OPEN_LINE, {"lat.bmad": lattice(definition, start=initial, extra=extra)})
        require(values["entrance"], entrance)
        require(values["orbit"][:4], orbit)
        if not normalized:
            require(values["exit"], expected)
        defect |= compare(
            f"{element}, normalize_twiss={normalized}, dispersion from tracking derivative",
            values["exit"],
            expected,
            atol=1e-15,
            rtol=3e-14,
        )
verdict(defect)
