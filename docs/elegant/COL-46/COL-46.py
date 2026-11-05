"""COL-46: WAKE gives each particle the full W(0) of its own bin, so full-origin wakes such as CORGPIPE's overstate the loss

WAKE gives every particle the table's W(0) times all charge in its own time bin, its own included.
When W(0) is the wake just behind a charge, W(0⁺), as in CORGPIPE's table, built from Bane's model,
a charge should receive W(0⁺)/2 from itself and nothing from charge behind it (beam-loading
theorem), so the mean loss is too large by W(0⁺)/2 times the average charge of a particle's bin. For
a uniform 0.8 nC, 4 ps electron bunch in a 0.1 m CORGPIPE with the default time step, the mean loss
is 4.288% above the exact particle sum; a quarter step leaves 1.058%. The manual does not say
whether a table's W(0) is W(0⁺) or W(0⁺)/2; halving it removes the excess.

The script tracks a uniform 0.8 nC electron bunch through a WAKE whose table is a constant
3e13 V/C from t = 0, with the full value and with half of it at t = 0, and compares the mean loss
with the beam-loading result Q W/2, which holds for any bunch shape. It then tracks a uniform 4 ps
bunch through a 0.1 m CORGPIPE with its default time step and with a quarter of it, compares the
mean loss with the loss of Bane's cosine wake summed exactly over the particles, and reproduces
the CORGPIPE result with a WAKE table of the same cosine, full and halved at t = 0.
"""

import numpy as np
from _elegant import verdict
from _wake import beam, loss, table, track

P0, CHARGE = 2000.0, 0.8e-9  # m_e c, C
C, Z0 = 2.99792458e8, 4e-7 * np.pi * 2.99792458e8  # as in wake.c
CONTROL_RTOL = 1e-3  # half-origin controls against the exact expectations (binning error ~3e-4)
MIN_EXCESS = 0.01  # defect if the default CORGPIPE loss exceeds the expectation by more than 1%


def wake_loss(times: np.ndarray, t: np.ndarray, w: np.ndarray, extra: str = "") -> np.ndarray:
    """Loss (eV) of each particle through CHARGE + WAKE reading the table (t, w)."""
    files = {"wake.sdds": table({"t": t, "W": w}, {"t": "s", "W": "V/C"})}
    out, _ = track(
        f'Q: CHARGE,TOTAL={CHARGE!r}\nD: DRIFT,L=0.05\nW1: WAKE,INPUTFILE="wake.sdds",'
        f'TCOLUMN="t",WCOLUMN="W"{extra}',
        "Q,D,W1,D",
        beam(times, P0),
        P0,
        files,
    )
    return loss(out, P0)


def report(label: str, observed: float, expected: float) -> float:
    print(
        f"serial {label}: mean loss observed={observed:.10g} eV expected={expected:.10g} eV "
        f"(ratio {observed / expected:.7f})"
    )
    return observed / expected - 1


# 1. Constant wake W+ = 3e13 V/C on 0 <= t <= 40 ps, step 0.2 ps; 8192 particles spread
# uniformly over 32 ps. Each particle receives W+ from every particle ahead of it and, by the
# beam-loading theorem, W+/2 from itself, so the mean loss is q W+ [N(N-1)/2 + N/2]/N = Q W+/2.
# A convolution that gives each bin's whole charge, own particle included, the full W(0) adds
# Q W+/2 * sum_b (n_b/N)^2 over the bins; halving W(0) removes it exactly.
W_FLAT, DT_FLAT, N_FLAT = 3e13, 0.2e-12, 8192
times = (np.arange(N_FLAT) + 0.5) / N_FLAT * 32e-12
grid = np.arange(201) * DT_FLAT
flat = np.full(len(grid), W_FLAT)
half = flat.copy()
half[0] /= 2
expected_flat = CHARGE * W_FLAT / 2
# Elegant's bins: centers at min(t) - dt + i dt, each particle in its nearest center.
counts = np.bincount(np.floor((times - (times.min() - DT_FLAT)) / DT_FLAT + 0.5).astype(int))
print(
    f"flat wake: full-origin prediction Q W+/2 (1 + sum (n_b/N)^2) = "
    f"{expected_flat * (1 + np.sum((counts / N_FLAT) ** 2)):.10g} eV"
)
report("WAKE flat, full W(0)", wake_loss(times, grid, flat).mean(), expected_flat)
control = report(
    "WAKE flat, W(0) halved (control)", wake_loss(times, grid, half).mean(), expected_flat
)
if abs(control) > CONTROL_RTOL:
    raise SystemExit("the half-origin flat-wake control misses Q W+/2")

# 2. CORGPIPE: Bane's single-mode wake W(t) = 2 kappa L cos(omega t) for t > 0, with
# kappa = Z0 c/(2 pi a^2) and omega = c sqrt(2 p/(a h g)) (radius a, period p, depth h, gap g).
# Summing the pair wakes with the self term W(0+)/2 gives the mean loss of N equal particles
#   (q/2N) sum_ij W(|t_i - t_j|) = Q kappa L |<exp(i omega t)>|^2.
RADIUS, PERIOD, DEPTH, GAP, LENGTH = 0.01, 1e-4, 2e-4, 5e-5, 0.1
kappa = Z0 * C / (2 * np.pi * RADIUS**2)
omega = C * np.sqrt(2 * PERIOD / (RADIUS * DEPTH * GAP))
times = (np.arange(32768) + 0.5) / 32768 * 4e-12
expected_pipe = CHARGE * kappa * LENGTH * abs(np.mean(np.exp(1j * omega * times))) ** 2
pipe = f"L={LENGTH},RADIUS={RADIUS},PERIOD={PERIOD},DEPTH={DEPTH},GAP={GAP}"
# The default step is min(0.1/omega, bunch duration/30), here 4 ps/30, with 1.5*30 = 45 fixed
# bins centered on the mean time (wake.c, track_through_corgpipe).
step = float(np.ptp(times)) / 30
out, log = track(f"Q: CHARGE,TOTAL={CHARGE!r}\nC1: CORGPIPE,{pipe}", "Q,C1", beam(times, P0), P0)
pipe_loss = loss(out, P0)
excess = report("CORGPIPE, default step", pipe_loss.mean(), expected_pipe)
out, _ = track(
    f"Q: CHARGE,TOTAL={CHARGE!r}\nC1: CORGPIPE,{pipe},DT={step / 4!r}", "Q,C1", beam(times, P0), P0
)
report("CORGPIPE, a quarter of the default step", loss(out, P0).mean(), expected_pipe)
cosine = 2 * kappa * LENGTH * np.cos(omega * np.arange(45) * step)
same = wake_loss(times, np.arange(45) * step, cosine, ",N_BINS=45")
print(
    f"WAKE with the same cosine table, full W(0): max |loss - CORGPIPE loss| = {np.max(np.abs(same - pipe_loss)):.3g} eV"
)
cosine[0] /= 2
control = report(
    "WAKE with the same cosine table, W(0) halved (control)",
    wake_loss(times, np.arange(45) * step, cosine, ",N_BINS=45").mean(),
    expected_pipe,
)
if abs(control) > CONTROL_RTOL:
    raise SystemExit("the half-origin cosine control misses the exact expectation")
print(
    f"control tolerance {CONTROL_RTOL:g} relative; defect if the default CORGPIPE loss is more than "
    f"{MIN_EXCESS:.0%} above the expectation"
)
verdict(excess > MIN_EXCESS)
