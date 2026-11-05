"""COL-34: Transient CSRCSBEND trapezoid integration gives its first regular sample weight 3/4 instead of 1/2

With CSRCSBEND's default transient CSR (STEADY_STATE=0, TRAPAZOID_INTEGRATION=1), the trapezoid end correction halves the first sample behind each bin twice, giving it weight 3/4 instead of 1/2. Once the slippage length exceeds the bunch, the transient wake must equal the steady-state wake. For a 35 μm rms Gaussian bunch in a 0.6 m, 0.3 rad bend with 400 bins, the steady-state wake matches the analytic Gaussian wake within 0.1470% of its peak, while the transient wake differs from it by 1.243% of the peak, exactly the extra quarter sample, and from the analytic wake by 1.376%. The excess falls slowly with the bin count: 1.968% at 200 bins, 0.7894% at 800.

The script tracks 240001 electrons (Gaussian quantiles, σ_ct = 35 μm, p = 4000 mₑc, 10⁻¹⁶ C so
the beam does not change) through a one-slice 0.6 m, 0.3 rad CSRCSBEND with 200, 400 and 800
bins, once with STEADY_STATE=0 (transient, the default) and once with STEADY_STATE=1, and reads
both exported wakes. The slippage length 2.250 mm exceeds the binned bunch, so both must equal the
analytic steady-state Gaussian wake, which the steady-state control does within 0.15%. The
script also rebuilds the transient-minus-steady difference from the exported density derivative
as a quarter of the first regular trapezoid sample.
"""

import numpy as np
from _csr import E_CHARGE, R_E, C, gaussian_beam, page, steady_state_wake, track
from _elegant import verdict

P0, SIGMA, CHARGE = 4000.0, 35e-6, 1e-16
L, ANGLE = 0.6, 0.3
RHO = L / ANGLE
BEND = (
    f"L={L},ANGLE={ANGLE},N_SLICES=1,INTEGRATION_ORDER=6,"
    "SG_HALFWIDTH=1,SGDERIV_HALFWIDTH=3,SGDERIV_ORDER=3"
)
BEAM = gaussian_beam(240001, SIGMA, P0)
# Steady-state CSR strength of the whole bend (see _csr.steady_state_wake): N electrons in a bend
# of length L and radius rho.
K = 2 * (CHARGE / E_CHARGE) * R_E * L / (3 * RHO**2) ** (1 / 3)
TOLERANCE = 0.003  # of the analytic peak; the steady-state control is within 0.0015


def wake(bins: int, steady: int):
    lattice = (
        f"Q: CHARGE,TOTAL={CHARGE!r}\n"
        f'B: CSRCSBEND,{BEND},BINS={bins},STEADY_STATE={steady},OUTPUT_FILE="wake.sdds"'
    )
    return page(track(lattice, "Q,B", BEAM, P0, outputs=("wake.sdds",))["wake.sdds"])


defect = False
for bins in (200, 400, 800):
    (transient, info), (steady, _) = wake(bins, 0), wake(bins, 1)
    # Slippage rho angle^3/24 = 2.25 mm exceeds the histogram, so the transient boundary terms
    # vanish and the transient wake is the steady-state integral (Stupakov and Emma, case A/B).
    if not info["SlippageLength"] > info["TotalBunchLength"]:
        raise SystemExit("The slippage does not exceed the binned bunch.")
    expected = steady_state_wake(steady["s"], SIGMA, K)
    peak = np.max(np.abs(expected))
    error_t = np.max(np.abs(transient["DeltaGamma"] - expected)) / peak
    error_s = np.max(np.abs(steady["DeltaGamma"] - expected)) / peak
    difference = transient["DeltaGamma"] - steady["DeltaGamma"]
    print(
        f"serial {bins} bins: max |transient - steady| / analytic peak observed={np.max(np.abs(difference)) / peak:.6%} "
        f"expected=0; max |wake - analytic| / peak transient={error_t:.6%} steady-state={error_s:.6%}"
    )
    if error_s > TOLERANCE:
        raise SystemExit("The steady-state control disagrees with the analytic wake.")
    # The difference is a quarter of the first sample behind each bin, the regular part of the
    # integrand k lambda'(z + h) h^(-1/3) times the bin width h: k lambda'(z + h) h^(2/3) / 4.
    # LinearDensityDeriv is Q c^2 lambda'. The last two bins have fewer than two samples behind
    # them and no endpoint correction.
    h = info["BinSize"]
    lambda_prime = transient["LinearDensityDeriv"] / (CHARGE * C**2)
    quarter = K * lambda_prime[1:] * h ** (2 / 3) / 4
    mismatch = np.max(np.abs(difference[:-2] - quarter[:-1])) / np.max(np.abs(difference))
    print(
        f"serial {bins} bins: difference rebuilt as k lambda'(z+h) h^(2/3)/4, max relative residual {mismatch:.3g}"
    )
    defect |= error_t > TOLERANCE and mismatch < 1e-9
print(f"tolerance {TOLERANCE:.1%} of the analytic peak")
verdict(defect)
