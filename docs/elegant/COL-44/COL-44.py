"""COL-44: An accelerating RFCW without a longitudinal wake evaluates its longitudinal space charge at the entrance momentum

With LSC=1 and no longitudinal wake updating the reference between kicks, an accelerating RFCW
evaluates its longitudinal space-charge impedance at the entrance momentum throughout the
cavity; increasing N_KICKS does not update that momentum. For a 0.8 m cavity accelerating an
electron bunch of 160 μm rms length and 240 μm transverse rms size from γ = 6 to 30, the +1σ
energy kick is 3.590 times the local-momentum result. Splitting it into 480 one-kick RFCWs with
CHANGE_P0=1 gives a result within 0.3468%. An enabled longitudinal wake with CHANGE_P0=1 also
updates the reference between kicks, avoiding the frozen entrance momentum.

The script tracks 65536 electrons (Gaussian quantiles, sigma_ct = 160 um, rms sizes 240 um,
1 fC) through a 0.8 m RFCW with FREQ=0 and PHASE=90, which accelerates every particle uniformly
from gamma = 6 to 30, with LSC=1, 1024 LSC bins and 480 kicks, and through the same cavity split
into 480 one-kick RFCWs with CHANGE_P0=1. It rebuilds both results in NumPy from Elegant's
periodic LSC kick, summed over the 480 kicks with the reference momentum each one uses, and
compares the single cavity with the same sum at the local momentum of each kick. The control runs
the cavity with LSC=0.
"""

import numpy as np
from _elegant import verdict
from _lsc import ME_EV, RADIUS_FACTOR, C, at, energy, periodic_kick, round_beam, track

COUNT, SIGMA, SIGMA_R, CHARGE, BINS = 65536, 160e-6, 240e-6, 1e-15, 1024
LENGTH, KICKS, GAMMA_IN, GAMMA_OUT = 0.8, 480, 6.0, 30.0
RB = RADIUS_FACTOR * SIGMA_R
VOLT = (GAMMA_OUT - GAMMA_IN) * ME_EV  # FREQ=0, PHASE=90: every kick adds VOLT/KICKS
P_IN = float(np.sqrt(GAMMA_IN**2 - 1))
RTOL = 1e-3  # rebuilt sums against Elegant; 1 fC makes the bunch's own redistribution negligible
MIN_RATIO = 1.1  # defect if the single cavity exceeds the local-momentum sum by more than 10%

BEAM = round_beam(COUNT, SIGMA, SIGMA_R, P_IN)
Z = C * BEAM["t"]


def cavity(segments: int, lsc: int) -> str:
    """`segments` equal RFCWs of KICKS/segments kicks each, together the 0.8 m cavity."""
    return (
        f"R: RFCW,L={LENGTH / segments!r},CELL_LENGTH={LENGTH / segments!r},"
        f"VOLT={VOLT / segments!r},FREQ=0,PHASE=90,CHANGE_P0=1,N_KICKS={KICKS // segments},"
        f"LSC={lsc},LSC_BINS={BINS},ZWAKE=0,TRWAKE=0"
    )


def lsc_kick(segments: int, lsc: int = 1) -> np.ndarray:
    """Energy change (eV) of each particle beyond the common gain VOLT.

    The bunch is symmetric about z = 0 and the LSC kick antisymmetric, so pairing particle i
    with its mirror image removes the rounding of the 12 MeV gain.
    """
    out = track(
        f"Q: CHARGE,TOTAL={CHARGE!r}\n{cavity(segments, lsc)}",
        f"Q,{segments}*R",
        BEAM,
        P_IN,
    )
    gain = energy(out["p"]) - energy(P_IN) - VOLT
    return (gain - gain[::-1]) / 2


def rebuilt(momenta: np.ndarray) -> np.ndarray:
    """Sum of Elegant's periodic LSC kick over the kicks, each of length LENGTH/KICKS, with the
    momentum in the impedance taken from `momenta` (one per kick)."""
    total = np.zeros(COUNT)
    for p in np.unique(momenta):
        count = int(np.sum(momenta == p))
        total += count * periodic_kick(Z, float(p), RB, CHARGE, LENGTH / KICKS, BINS)[0]
    return total


# Momentum at each kick. FREQ=0 makes gamma rise by (GAMMA_OUT - GAMMA_IN)/KICKS per kick, each
# RF kick preceding its LSC kick. One cavity: the reference stays at the entrance momentum until
# CHANGE_P0 acts at the exit. 480 one-kick cavities: each LSC kick uses the momentum at the
# entrance of its cavity (the previous cavity's CHANGE_P0). Local physics: the impedance
# integrated along the cavity with the beam's momentum at each point, here by the midpoint rule,
# gamma = GAMMA_IN + (k + 1/2) dgamma for kick k, whose error is second order in dgamma.
step = (GAMMA_OUT - GAMMA_IN) / KICKS
frozen = np.full(KICKS, P_IN)
split = np.sqrt((GAMMA_IN + step * np.arange(KICKS)) ** 2 - 1)
local = np.sqrt((GAMMA_IN + step * (np.arange(KICKS) + 0.5)) ** 2 - 1)
expected = at(Z, rebuilt(local), SIGMA)

control = np.max(np.abs(lsc_kick(1, lsc=0)))
print(f"serial one RFCW, LSC=0 (control): max |antisymmetric energy change| = {control:.3g} eV")
if control > 1e-4:
    raise SystemExit("the cavity without LSC does not give every particle the same gain")

defect = False
for label, segments, momenta in (
    ("one RFCW, 480 kicks", 1, frozen),
    ("480 one-kick RFCWs with CHANGE_P0=1", KICKS, split),
):
    observed = at(Z, lsc_kick(segments), SIGMA)
    model = at(Z, rebuilt(momenta), SIGMA)
    print(
        f"serial {label}: LSC kick at +1 sigma observed={observed:.9g} eV "
        f"expected (local momentum)={expected:.9g} eV ratio={observed / expected:.6f}; "
        f"rebuilt with the momenta Elegant uses {model:.9g} eV"
    )
    if abs(observed / model - 1) > RTOL:
        raise SystemExit(f"{label}: the rebuild does not reproduce Elegant")
    if segments == 1:
        defect = observed / expected > MIN_RATIO
print(f"rebuild tolerance {RTOL:g} relative; defect if the one-cavity ratio exceeds {MIN_RATIO}")
verdict(defect)
