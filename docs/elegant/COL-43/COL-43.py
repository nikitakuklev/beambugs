"""COL-43: CSRCSBEND and CSRDRIFT apply the CSR wake half a bin toward the bunch tail

CSRCSBEND, and CSRDRIFT in the default and Stupakov models, interpolate the CSR wake as if each value sat at its bin's lower edge, although the histogram and the wake output place it at the bin center, so every particle receives the wake half a bin toward the bunch tail. For a 10 pC, 35 μm rms Gaussian bunch in a 0.6 m, 0.3 rad bend with 400 bins, the kicks deviate from the wake at the particles by up to 1.262% of its peak, and the mean energy loss is 0.5104% above the closed-form Gaussian value, whereas the wake sampled at the particles is within 0.03107%. The bias halves with each doubling of the bin count.

The script tracks 240001 electrons (Gaussian quantiles, σ_ct = 35 μm, p = 4000 mₑc, 10 pC)
through a one-slice 0.6 m, 0.3 rad CSRCSBEND (STEADY_STATE=1) with 200, 400 and 800 bins and
rebuilds every particle's kick from the exported wake, placing wake value i at the bin center the
output declares and, alternatively, at the bin's lower edge. It compares the mean energy change
with the closed-form Gaussian mean, which the wake sampled at bin centers matches. It repeats the
rebuild for a following 20 mm CSRDRIFT with one kick, in the default model (the bend's wake) and
with USE_STUPAKOV=1, from the coordinates a WATCH records at the bend exit.
"""

import numpy as np
from _csr import E_CHARGE, R_E, C, final, gaussian_beam, page, track
from _elegant import particles, verdict
from scipy.special import gamma

P0, SIGMA, CHARGE = 4000.0, 35e-6, 1e-11
L, ANGLE = 0.6, 0.3
RHO = L / ANGLE
DRIFT = 0.02
BEND = (
    f"L={L},ANGLE={ANGLE},N_SLICES=1,INTEGRATION_ORDER=6,SG_HALFWIDTH=1,SGDERIV_HALFWIDTH=3,"
    "SGDERIV_ORDER=3,STEADY_STATE=1"
)
BEAM = gaussian_beam(240001, SIGMA, P0)
K = 2 * (CHARGE / E_CHARGE) * R_E * L / (3 * RHO**2) ** (1 / 3)  # see _csr.steady_state_wake
# Mean steady-state change of gamma over a Gaussian bunch: averaging k int lambda'(z+u) u^(-1/3) du
# over lambda gives -k Gamma(5/6) / (2^(4/3) sqrt(pi) sigma^(4/3)) (density autocorrelation).
MEAN = -K * gamma(5 / 6) / (2 ** (4 / 3) * np.sqrt(np.pi) * SIGMA ** (4 / 3))
ATOL = 1e-11  # m_e c; the rebuild with the right placement reproduces every kick within ~1e-12


def reconstruct(position: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of the wake at `position` (in bins from the grid's lower end), with
    value i placed at the declared bin center i + 1/2 and, alternatively, at the lower edge i."""
    index = np.arange(len(values))
    return np.interp(position, index + 0.5, values), np.interp(position, index, values)


def compare(label: str, observed: np.ndarray, position: np.ndarray, values: np.ndarray) -> bool:
    center, edge = reconstruct(position, values)
    peak = np.max(np.abs(values))
    e_center, e_edge = np.max(np.abs(observed - center)), np.max(np.abs(observed - edge))
    print(
        f"serial {label}: max |kick - wake at the particle| observed={e_center:.4g} m_e c "
        f"({e_center / peak:.4%} of the peak) expected=0; with value i at the lower edge {e_edge:.3g} m_e c"
    )
    if e_edge > ATOL:
        raise SystemExit(f"{label}: the lower-edge rebuild does not reproduce the kicks.")
    return e_center > ATOL


defect = False
for bins in (200, 400, 800):
    lattice = f'Q: CHARGE,TOTAL={CHARGE!r}\nB: CSRCSBEND,{BEND},BINS={bins},OUTPUT_FILE="wake.sdds"'
    result = track(lattice, "Q,B", BEAM, P0, outputs=("out.sdds", "wake.sdds"))
    out, (wake, info) = final(result), page(result["wake.sdds"])
    # The kick is Delta gamma(ct)/p0 (1 + x/rho) in relative momentum (csbend.c); dividing out
    # the path factor leaves Delta gamma at the particle. The output s is ct measured from the
    # middle of the histogram, which for this symmetric bunch is the middle of its ct range.
    kick = (out["p"] - P0) / (1 + out["x"] / RHO)
    ct = C * out["t"]
    s = ct - (ct.max() + ct.min()) / 2
    h = info["BinSize"]
    position = (s - (wake["s"][0] - h / 2)) / h
    defect |= compare(f"bend, {bins} bins", kick, position, wake["DeltaGamma"])
    center, _ = reconstruct(position, wake["DeltaGamma"])
    print(
        f"serial bend, {bins} bins: mean Delta gamma observed={np.mean(kick):.10g} "
        f"expected={MEAN:.10g} (ratio {np.mean(kick) / MEAN:.7f}); wake at the particle "
        f"{np.mean(center):.10g} (ratio {np.mean(center) / MEAN:.7f})"
    )

# Drifts: one kick at the middle of a 20 mm CSRDRIFT after the 400-bin bend.
for model, extra, output in (
    ("default", "", "wake.sdds"),
    ("Stupakov", ',USE_STUPAKOV=1,STUPAKOV_OUTPUT="drift.sdds"', "drift.sdds"),
):
    lattice = (
        f'Q: CHARGE,TOTAL={CHARGE!r}\nB: CSRCSBEND,{BEND},BINS=400,OUTPUT_FILE="wake.sdds"\n'
        f'W: WATCH,FILENAME="exit.sdds"\nD: CSRDRIFT,L={DRIFT},N_KICKS=1{extra}'
    )
    result = track(lattice, "Q,B,W,D", BEAM, P0, outputs=("out.sdds", "exit.sdds", output))
    exit_beam, out = particles(result["exit.sdds"]), final(result)
    wake, info = page(result[output])
    beta = exit_beam["p"] / np.sqrt(1 + exit_beam["p"] ** 2)
    ct_exit = C * exit_beam["t"]
    # ct at the kick: half the drift travelled along the particle's direction at its speed.
    ct = ct_exit + DRIFT / 2 * np.sqrt(1 + exit_beam["xp"] ** 2 + exit_beam["yp"] ** 2) / beta
    if model == "default":
        # The bend's wake, frozen, scaled by the drift step over the bend length; its grid is the
        # bend-exit histogram (range 1.2 times the ct spread, centered) moving with the light
        # front, so its lower end has advanced by the distance travelled.
        h = info["BinSize"]
        lower = (ct_exit.max() + ct_exit.min()) / 2 - 0.6 * np.ptp(ct_exit) + DRIFT / 2
        values = wake["DeltaGamma"] * DRIFT / L
    else:
        # The Stupakov model rebins the bunch at the kick over 1.2 times its ct spread.
        h = 1.2 * np.ptp(ct) / len(wake["s"])
        lower = (ct.max() + ct.min()) / 2 - 0.6 * np.ptp(ct)
        values = wake["DeltaGamma"]
        if not np.allclose(np.diff(wake["s"]), h, rtol=1e-9):
            raise SystemExit("The Stupakov grid spacing differs from the rebinned range.")
    defect |= compare(f"{model} CSRDRIFT", out["p"] - exit_beam["p"], (ct - lower) / h, values)
print(f"rebuild tolerance {ATOL:g} m_e c")
verdict(defect)
