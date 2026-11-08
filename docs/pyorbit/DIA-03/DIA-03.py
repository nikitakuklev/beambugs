"""DIA-03: Weighted `computeBunchMoments` raises the weights to the moment order and adds an unnormalized copy

With a per-particle `macrosize` attribute, `BunchTwissAnalysis.computeBunchMoments` multiplies the
weight w into every power of a particle's offsets, so a moment of order i + j carries wⁱ⁺ʲ, and it
adds both the √β-normalized and the unnormalized product before dividing by the total weight once.
The weighted central moment Σw(x−⟨x⟩)ⁱ(y−⟨y⟩)ʲ/Σw, divided by βx^(i/2)βy^(j/2), does not change when
every weight is scaled, but these values grow with the weight and mix two units. For 32 particles
with ⟨x²⟩ = 1 mm² and βx = 10 m, the order-2 moment (2, 0) is 1.000e-7 m without the attribute,
1.100e-6 with weight 1 and 3.300e-6 with weight 3; with weight 1, (4, 0) is 1.010e-12 instead of
1.000e-14 m².

The script builds the 32-particle sign-factorial bunch of x = ±1 mm, x′ = ±0.1 mrad, y = ±2 mm,
y′ = ±0.1 mrad and dE = ±1 MeV with a `macrosize` attribute of 1, 3, and alternately 2 and 1,
computes order-4 moments without dispersion subtraction or emittance normalization, and compares
(2, 0) and (4, 0) with the weighted central moments Σw·xⁱ/Σw divided by βx^(i/2), computed
with NumPy from the same coordinates. The bunch without the attribute is the control. Each line
also gives the closed form of the code's accumulation, which the observed values match.
"""

import itertools

import numpy as np
from orbit.core.bunch import Bunch, BunchTwissAnalysis  # ty: ignore[unresolved-import]

ROWS = np.array(
    [
        (sx * 1e-3, sxp * 1e-4, sy * 2e-3, syp * 1e-4, 0.0, se * 1e-3)
        for sx, sxp, sy, syp, se in itertools.product((1, -1), repeat=5)
    ]
)


def moments(weights):
    b = Bunch()
    b.getSyncParticle().kinEnergy(1.0)
    for r in ROWS:
        b.addParticle(*r)
    if weights is not None:
        b.addPartAttr("macrosize")
        for i, w in enumerate(weights):
            b.partAttrValue("macrosize", i, 0, float(w))
    t = BunchTwissAnalysis()
    t.computeBunchMoments(b, 4, -1, -1)
    return t.getBunchMoment(2, 0), t.getBunchMoment(4, 0), t.getBeta(0)


def weighted_central(w, i):
    """Correct weighted central moment sum(w (x - xbar)^i) / sum(w) of x."""
    x = ROWS[:, 0]
    xbar = (w * x).sum() / w.sum()
    return (w * (x - xbar) ** i).sum() / w.sum()


n = len(ROWS)
ok = True
cases = (
    ("none (control)", None),
    ("1", np.ones(n)),
    ("3", np.full(n, 3.0)),
    ("2, 1 alternating", np.where(np.arange(n) % 2 == 0, 2.0, 1.0)),
)
for label, w in cases:
    m20, m40, beta_x = moments(w)
    wv = np.ones(n) if w is None else w
    # Correct: the weighted central moment divided by beta_x^(i/2); beta_x = 10 m for every case,
    # because each weight pattern keeps the weighted x-x' moments of the factorial set.
    exp20 = weighted_central(wv, 2) / beta_x
    exp40 = weighted_central(wv, 4) / beta_x**2
    print(
        f"macrosize {label}: (2,0) observed {m20:.6g}, expected {exp20:.6g}; "
        f"(4,0) observed {m40:.6g}, expected {exp40:.6g}"
    )
    if w is None:
        ok &= abs(m20 - exp20) < 1e-12 * exp20 and abs(m40 - exp40) < 1e-12 * exp40
    else:
        # Code: each power of the offset carries one factor w, and the normalized and raw products
        # are both added, then divided by sum(w) once:
        # (i,0) = sum(w^i (x^i / beta^(i/2) + x^i)) / sum(w).
        code20 = (w**2 * (ROWS[:, 0] ** 2 / beta_x + ROWS[:, 0] ** 2)).sum() / w.sum()
        code40 = (w**4 * (ROWS[:, 0] ** 4 / beta_x**2 + ROWS[:, 0] ** 4)).sum() / w.sum()
        print(
            f"macrosize {label}: code accumulation sum(w^i (x^i/beta^(i/2) + x^i))/sum(w) gives "
            f"(2,0) {code20:.6g}, (4,0) {code40:.6g}"
        )
        ok &= abs(m20 - code20) < 1e-12 * code20 and abs(m40 - code40) < 1e-12 * code40
        ok &= abs(m20 - exp20) > 0.5 * exp20

print("DEFECT REPRODUCED" if ok else "NOT REPRODUCED")
