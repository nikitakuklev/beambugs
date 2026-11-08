"""COL-02: `LSpaceChargeCalc.setUseGrad(1)` scales the kick as q³ and reverses it for negative ions

With `setUseGrad(1)`, `LSpaceChargeCalc` multiplies the binned line density by the charge number q and then by the classical radius r₀, which already contains q², so the energy kick scales as q³ instead of q² (one q for the source, one for the kicked particle). A negative ion (q = −1), such as H⁻, therefore gets a reversed kick and q = 2 a doubled one, while the default Fourier branch scales as q². For a Gaussian bunch in a 250 m ring, the gradient-branch kick at z = σ, in units of the analytic q = +1 kick, is −1.006 for q = −1 and 8.051 for q = 2, instead of +1 and 4.

The script tracks one deterministic Gaussian bunch (rms length 12.5 m, 250 m ring, 128 bins) with charge numbers +1, −1, +2 and +3 through the default Fourier branch and through the smoothed gradient branch, and prints the kick of the particle nearest z = σ relative to the analytic q = +1 kick, against the expected ratio q². The q = +1 rows and the Fourier-branch rows are the controls.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from orbit.core.bunch import Bunch
from orbit.core.spacecharge import LSpaceChargeCalc
from orbit.utils.consts import mass_proton
from scipy.special import ndtri

C, NBINS, N, SIG, NPHYS, B_A = 250.0, 128, 20000, 12.5, 2e14, 3.0
U = SIG * ndtri((np.arange(N) + 0.5) / N)  # deterministic Gaussian sample, mean 0
I_PROBE = int(np.argmin(np.abs(U - SIG)))


def run(q, grad):
    b = Bunch()
    b.mass(mass_proton)
    b.charge(q)
    b.macroSize(NPHYS / N)
    b.getSyncParticle().kinEnergy(1.0)
    for zi in U:
        b.addParticle(0.0, 0.0, 0.0, 0.0, zi, 0.0)
    calc = LSpaceChargeCalc(B_A, C, 0, 1, NBINS)
    if grad:
        calc.setUseGrad(1)
        calc.setSmoothGrad(1)
    calc.trackBunch(b)
    return b.dE(I_PROBE)


# Reference: analytic q = +1 kick dE = -C g r0_p m_p / gamma^2 * dlambda/dz at z = U[I_PROBE], g = 1 + 2 ln(b/a).
ref = Bunch()
ref.mass(mass_proton)
ref.getSyncParticle().kinEnergy(1.0)
g = 1.0 + 2.0 * np.log(B_A)
z = U[I_PROBE]
lam = NPHYS / (np.sqrt(2 * np.pi) * SIG) * np.exp(-0.5 * (z / SIG) ** 2)
an1 = (
    -C
    * g
    * ref.classicalRadius()
    * ref.mass()
    / ref.getSyncParticle().gamma() ** 2
    * (-z / SIG**2)
    * lam
)

bad, ctl_ok = False, True
for grad in (False, True):
    for q in (1.0, -1.0, 2.0, 3.0):
        ratio = run(q, grad) / an1
        # Expected ratio q^2: source charge q times kicked charge q. The kick factor contains
        # r0 = r_p (m_p/m) q^2 once, so any further factor q is surplus.
        branch = (
            "gradient (setUseGrad(1), setSmoothGrad(1))" if grad else "Fourier (default, control)"
        )
        print(
            f"{branch} q = {q:+.0f}: dE(z = sigma)/analytic(q = +1) observed {ratio:+.4g}, "
            f"expected {q * q:+.4g} (kick scales as q^2)"
        )
        ok = abs(ratio - q * q) < 0.05 * q * q
        if grad and q != 1.0:
            bad |= not ok
        else:
            ctl_ok &= ok
print("DEFECT REPRODUCED" if bad and ctl_ok else "NOT REPRODUCED")
