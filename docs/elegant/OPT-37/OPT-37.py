"""OPT-37: MULT radiation and coupling integrals ignore FACTOR

MULT radiation integrals and the linear coupling integral ignore FACTOR, although tracking and transfer matrices apply it, so a MULT and the identical field with FACTOR folded into its strength report different optics. For a 0.2000 m quadrupole MULT with integrated strength 0.3000 m⁻¹, horizontal offset 3.000 mm and FACTOR=2, I₂ is 4.050×10⁻⁶ instead of 1.620×10⁻⁵ m⁻¹ and I₃ is eight times too small; energy loss and damping times inherit the error. Centred and rolled by 1.000 mrad, the same magnet gives a coupling integral of 4.685×10⁻⁴ instead of 9.370×10⁻⁴. FACTOR=0 still gives nonzero radiation and coupling. Sextupole MULTs and pole-tip input fail the same way; the curvature and skew gradient of the scaled field give the correct values.

Run one 0.2000 m MULT at p/(mc) = 3000 through twiss_output from fixed entrance optics (βx = 4 m, βy = 6 m, ηx = 0.2 m), once with FACTOR = 1, 2 and 0 and once with that factor folded into KNL or BTIPL, and require bit-identical transfer matrices for each pair. Radiation cases compare I₂ = Lh² and I₃ = L|h|³ for the entrance-orbit curvature h of the scaled field, plus I₁, I₄, I₅ and U0 against the folded MULT; coupling cases compare the coupling integral and emittance ratio with closed-form thick-quadrupole optics, with FACTOR=1 as the control.
"""

from _slicing import close, discrepancy

# isort: split
import math

import _elegant as native
import numpy as np

P0 = 3000.0  # p/(mc)
L = 0.2
BORE = 0.02
DX = 0.003
DY = 0.002
TILT = 0.001
BETA = (4.0, 6.0)  # entrance betas with alpha = 0
# B*rho from SDDS constants.h, the values the executed build uses for BTIPL conversion.
BRHO = P0 * 9.1093897e-31 * 2.99792458e8 / 1.60217733e-19
RADIATION = ("I1", "I2", "I3", "I4", "I5", "U0")


def strength(order, knl, form):
    if form == "KNL":
        return f"KNL={knl:.17g}"
    # KnL = n! BTIPL / (B*rho * BORE^n), the manual's pole-tip relation.
    return f"BORE={BORE},BTIPL={knl * BRHO * BORE**order / math.factorial(order):.17g}"


def twiss(order, knl, form, factor, placement, options):
    deck = (
        f"&run_setup lattice=line.lte,use_beamline=BL,p_central={P0} &end\n"
        f"&twiss_output filename=twiss.sdds,matched=0,beta_x={BETA[0]},beta_y={BETA[1]},{options} &end\n"
        "&matrix_output SDDS_output=matrix.sdds,SDDS_output_order=1 &end\n"
    )
    element = (
        f"Q:MULT,L={L},ORDER={order},{strength(order, knl, form)},FACTOR={factor:.17g},{placement}"
    )
    result = native.run(
        {"line.lte": f"{element}\nBL:LINE=(Q)\n", "run.ele": deck},
        outputs=("twiss.sdds", "matrix.sdds"),
    )
    tw, mat = result["twiss.sdds"], result["matrix.sdds"]
    names = (
        RADIATION if "radiation_integrals=1" in options else ("couplingIntegral", "emittanceRatio")
    )
    values = np.array([float(tw[name][0]) for name in names])
    matrix = np.array([[mat[0, f"R{i}{j}"][-1] for j in range(1, 7)] for i in range(1, 7)])
    if not np.isfinite(values).all() or not np.isfinite(matrix).all():
        raise RuntimeError("Nonfinite native optics")
    return values, matrix


def pair(order, knl, form, factor, placement, options):
    """FACTOR=f and the same field with f folded into the strength; their maps must be identical."""
    scaled = twiss(order, knl, form, factor, placement, options)
    folded = twiss(order, knl * factor, form, 1.0, placement, options)
    if not np.array_equal(scaled[1].view(np.uint64), folded[1].view(np.uint64)):
        raise RuntimeError("FACTOR and folded strength give different transfer matrices")
    return scaled[0], folded[0], folded[1]


def quadrupole(k):
    """Closed-form thick quadrupole 2x2 matrix of gradient k over length L."""
    w = math.sqrt(abs(k))
    if k > 0:
        return np.array(
            [[math.cos(w * L), math.sin(w * L) / w], [-w * math.sin(w * L), math.cos(w * L)]]
        )
    if k < 0:
        return np.array(
            [[math.cosh(w * L), math.sinh(w * L) / w], [w * math.sinh(w * L), math.cosh(w * L)]]
        )
    return np.array([[1.0, L], [0.0, 1.0]])


def coupling_oracle(skew, k1, tilt):
    """Single-source coupling integral and emittance ratio from closed-form uncoupled optics.

    A roll t mixes the planes; the diagonal blocks of R(-t) Q R(t) are cos²t Q(k) + sin²t Q(-k),
    independent of the sign convention of t. Exit betas and phases follow from those blocks with
    alpha = 0 at the entrance, as in the uncoupled twiss propagation. For one source the phase
    factor has unit modulus, so |kappa| = |K_skew L| sqrt(<beta_x><beta_y>) / (2 pi), with entrance/exit
    averaged betas, and the emittance ratio is kappa² / (kappa² + Delta²), where Delta is the
    tune difference minus its nearest integer.
    """
    c2, s2 = math.cos(tilt) ** 2, math.sin(tilt) ** 2
    blocks = (
        c2 * quadrupole(k1) + s2 * quadrupole(-k1),
        c2 * quadrupole(-k1) + s2 * quadrupole(k1),
    )
    # A coupled block is not symplectic; the uncoupled propagation divides by its determinant.
    beta_exit = [
        (m[0, 0] ** 2 * b + m[0, 1] ** 2 / b) / np.linalg.det(m)
        for m, b in zip(blocks, BETA, strict=True)
    ]
    tunes = [
        math.atan2(m[0, 1], b * m[0, 0]) / (2 * math.pi) for m, b in zip(blocks, BETA, strict=True)
    ]
    kappa = (
        abs(skew)
        * math.sqrt((BETA[0] + beta_exit[0]) * (BETA[1] + beta_exit[1]) / 4)
        / (2 * math.pi)
    )
    delta = tunes[0] - tunes[1] - math.floor(tunes[0] - tunes[1] + 0.5)
    ratio = kappa**2 / (kappa**2 + delta**2) if kappa else 0.0
    return np.array([kappa, ratio]), blocks


def local_radiation(order, knl):
    """I2 and I3 of Elegant's model for an off-axis quadrupole or sextupole.

    The element is treated as a bend whose curvature is the field at the entrance orbit (x = 0):
    h = K_n (x - DX)^n / n! with K_n = KnL / L, giving I2 = L h² and I3 = L |h|³.
    """
    h = knl / L * (-DX) ** order / math.factorial(order)
    return np.array([L * h**2, L * abs(h) ** 3])


def source(order, knl, factor):
    """Integrated skew gradient, normal gradient and roll of the coupling source."""
    if order == 1:
        # Rolled quadrupole: skew gradient K1 sin(2 TILT).
        return factor * knl * math.sin(2 * TILT), factor * knl / L, TILT
    # Vertically displaced sextupole on the axis: skew gradient K2 (y - DY) at y = 0; its
    # uncoupled blocks are drifts.
    return factor * knl * DY, 0.0, 0.0


def main():
    faults = []
    radiation = "eta_x=0.2,radiation_integrals=1"
    for order, knl, form in (
        (1, 0.3, "KNL"),
        (1, 0.3, "BTIPL"),
        (2, 30.0, "KNL"),
        (2, 30.0, "BTIPL"),
    ):
        unit = local_radiation(order, knl)  # unscaled strength; tolerances are relative to it
        for factor in (1.0, 2.0, 0.0):
            observed, folded, _ = pair(order, knl, form, factor, f"DX={DX}", radiation)
            expected = local_radiation(order, factor * knl)
            close(folded[1:3] / unit, expected / unit, 1e-14, "folded-strength I2/I3")
            label = f"MULT ORDER={order} {form} FACTOR={factor:g}"
            if factor == 1:
                close(observed[1:3] / unit, expected / unit, 1e-14, "FACTOR=1 control")
                if not np.array_equal(observed, folded):
                    raise RuntimeError("FACTOR=1 control differs from the folded strength")
            else:
                # The observed law is the unscaled strength's integrals.
                faults.append(discrepancy(observed[1:3] / unit, expected / unit, np.ones(2), 1e-14))
                faults.append(not np.array_equal(observed, folded))
            print(
                f"{label}: I2={observed[1]:.17g}, I3={observed[2]:.17g}; expected I2={expected[0]:.17g}, "
                f"I3={expected[1]:.17g} (L*h^2, L*|h|^3 at h=FACTOR*KnL*(-DX)^n/(n!*L))"
            )
            print(
                f"{label}: I1, I4, I5, U0={observed[[0, 3, 4, 5]].tolist()}; "
                f"folded-strength MULT={folded[[0, 3, 4, 5]].tolist()} (identical field and matrix)"
            )
    for order, form in ((1, "KNL"), (1, "BTIPL"), (2, "KNL")):
        knl = 0.3
        placement = f"TILT={TILT}" if order == 1 else f"DY={DY}"
        unit = coupling_oracle(*source(order, knl, 1.0))[0]
        for factor in (1.0, 2.0, 0.0):
            observed, folded, matrix = pair(
                order, knl, form, factor, placement, "radiation_integrals=0"
            )
            expected, blocks = coupling_oracle(*source(order, knl, factor))
            # The observed law: the unscaled skew strength weighted by the optics of the scaled field.
            _, k1, tilt = source(order, knl, factor)
            faulty, _ = coupling_oracle(source(order, knl, 1.0)[0], k1, tilt)
            close(matrix[:2, :2], blocks[0], 2e-16, "closed-form x block")
            close(matrix[2:4, 2:4], blocks[1], 2e-16, "closed-form y block")
            close(folded / unit, expected / unit, 1e-14, "folded-strength coupling")
            label = f"MULT ORDER={order} {form} {placement} FACTOR={factor:g}"
            if factor == 1:
                close(observed / unit, expected / unit, 1e-14, "FACTOR=1 coupling control")
            else:
                faults.append(discrepancy(observed / unit, expected / unit, faulty / unit, 1e-14))
            print(
                f"{label}: couplingIntegral={observed[0]:.17g}, emittanceRatio={observed[1]:.17g}; "
                f"expected {expected[0]:.17g}, {expected[1]:.17g} "
                "(|K_skew L| sqrt(<beta_x><beta_y>)/(2 pi), kappa^2/(kappa^2+Delta^2), closed-form optics)"
            )
    native.verdict(any(faults))


if __name__ == "__main__":
    main()
