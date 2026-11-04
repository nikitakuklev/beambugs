"""COL-22: Constant RFMODE frequency waveforms repeatedly scale stored voltage

In single-bunch RFMODE tracking, a constant frequency multiplier a supplied through FREQ_WAVEFORM,
with FREQ divided by a to preserve the resonant frequency, scales the stored beam-induced voltage by
√a on every traversal. A stationary passive cavity should follow the same damped response as the
direct-frequency setting, which agrees with an independent parallel-RLC circuit calculation. For
quality factor Q = 11.00, shunt impedance RA = 200.0 kΩ, frequency 0.8300 GHz and a 0.3000 nC
electron bunch represented by 18 macroparticles, the fifth-traversal tail energy loss is 9366 eV for
a = 1.440 or 1.247×10⁴ eV for a = 0.2500, instead of 1.015×10⁴ eV. Multi-turn cavity simulations
using this waveform option are affected.

The script tracks the bunch through RFMODE and a drift five times, exciting the cavity by rigid
tracking until the last traversal, and compares the fifth traversal's energy losses with a lumped
parallel-RLC circuit driven by the same charge impulses, cross-checked against the closed-form
resonator wake. The controls are the direct frequency, a unit waveform, a single traversal, and
SINGLE_PASS = 1, which discards the stored field. It also shows that scaling the circuit state by √a
at each traversal reproduces the observed losses.
"""

import sys

import numpy as np
from _elegant import modes, particles, run, verdict
from scipy.linalg import expm

# SDDS constants.h, as Elegant uses them.
C, REST_EV = 299792458.0, 510999.06
N = 18  # electron macroparticles, spaced 2 ps
FREQ, Q, RA = 0.83e9, 11.0, 2e5  # actual resonant frequency (Hz), Q, shunt impedance (ohm)
CHARGE, P0 = 0.3e-9, 64.0  # C, m_e c
FLIGHT = 2.7e-9  # s between traversals, supplied by a drift
TIMES = 0.43e-9 + (np.arange(N) - (N - 1) / 2) * 2e-12  # s, arrival times of the first traversal
LENGTH = float(C * P0 / np.sqrt(1 + P0**2) * FLIGHT)  # drift length for FLIGHT, m

# (traversals, constant waveform factor a or None for no waveform, SINGLE_PASS, label)
CASES = (
    (5, None, 0, "direct frequency, 5 passes"),
    (5, 1.0, 0, "unit waveform, 5 passes"),
    (5, 1.44, 0, "factor 1.44, 5 passes"),
    (5, 0.25, 0, "factor 0.25, 5 passes"),
    (3, 1.44, 0, "factor 1.44, 3 passes"),
    (1, 1.44, 0, "factor 1.44, first pass"),
    (5, 1.44, 1, "factor 1.44, single pass"),
)


def circuit_losses(times, multiplier=1.0, single=False):
    """Energy loss in eV of each particle at each traversal in a parallel RLC circuit.

    The resonator is R = RA/2 in parallel with C = 2Q/(RA omega) and L = 1/(omega^2 C). Its
    state z = (V, sqrt(L/C) I_L) has voltage units and obeys dz/dt = omega [[-1/Q, -1], [1, 0]] z.
    A particle of charge q deposits q/C into V, and loses energy q (V + half its own deposit)
    (the fundamental theorem of beam loading). `multiplier` scales the state at the start of
    each traversal; the physical circuit has multiplier 1, and `single` clears it each time.
    """
    omega = 2 * np.pi * FREQ
    capacitance = 2 * Q / (RA * omega)
    generator = omega * np.array([[-1 / Q, -1], [1, 0]])
    deposit = CHARGE / N / capacitance
    state, last = np.zeros(2), float(times[0, 0])
    loss = np.zeros_like(times)
    for traversal, arrivals in enumerate(times):
        state = np.zeros(2) if single else state * multiplier
        for i in np.argsort(arrivals):
            state = expm(generator * (float(arrivals[i]) - last)) @ state
            loss[traversal, i] = state[0] + deposit / 2  # volts, times the unit charge e
            state[0] += deposit
            last = float(arrivals[i])
    return loss


def green_losses(arrivals, previous):
    """Closed-form energy loss in eV of the last traversal from the resonator wake of `previous`.

    W(t) = RA omega/(2Q) exp(-alpha t) [cos(omega_d t) - alpha/omega_d sin(omega_d t)],
    alpha = omega/(2Q), omega_d = sqrt(omega^2 - alpha^2), with half weight at coincidence.
    """
    omega = 2 * np.pi * FREQ
    alpha = omega / (2 * Q)
    damped = np.sqrt(omega**2 - alpha**2)
    delays = arrivals[:, None] - previous[None, :]
    wake = (
        RA * omega / (2 * Q) * np.exp(-alpha * delays)
        * (np.cos(damped * delays) - alpha / damped * np.sin(damped * delays))
    )  # fmt: skip
    weight = (delays > 1e-18) + 0.5 * (np.abs(delays) <= 1e-18)
    return CHARGE / N * np.sum(wake * weight, axis=1)


def last_traversal_losses(mode, binless, passes, factor, single):
    """Observed energy loss (eV) of each particle on the last traversal, and all arrival times."""
    waveform = ',FREQ_WAVEFORM="frequency.sdds=time+factor"' if factor is not None else ""
    scale = 1.0 if factor is None else factor
    # Actual resonant frequency FREQ = FREQ_parameter * factor.
    lattice = (
        f"CH: CHARGE,TOTAL={CHARGE!r}\n"
        'B: WATCH,FILENAME="before.sdds",MODE="coordinates"\n'
        f"R: RFMODE,FREQ={FREQ / scale!r},Q={Q},RA={RA!r},BIN_SIZE=2e-12,N_BINS=64,"
        f"BINLESS={binless},RIGID_UNTIL_PASS={passes - 1},SINGLE_PASS={single},"
        f"PRELOAD=0,INITIAL_V=0,DRIVE_FREQUENCY=0{waveform}\n"
        'A: WATCH,FILENAME="after.sdds",MODE="coordinates"\n'
        f"D: DRIF,L={LENGTH!r}\n"
        "BL: LINE=(CH,B,R,A,D)\n"
    )
    table = (
        "SDDS1\n"
        "&column name=time, units=s, type=double &end\n"
        "&column name=factor, type=double &end\n"
        f"&data mode=ascii, no_row_counts=1 &end\n-1e-06 {scale!r}\n1e-06 {scale!r}\n"
    )
    commands = (
        "&global_settings echo_namelists=0 &end\n"
        f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0!r},output=out.sdds &end\n"
        f"&run_control n_steps=1,n_passes={passes} &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track soft_failure=0 &end\n"
    )
    beam = {"t": TIMES, "p": np.full(N, P0), "particleID": np.arange(1, N + 1)}
    result = run(
        {"lattice.lte": lattice, "run.ele": commands, "frequency.sdds": table},
        mode,
        {"beam.sdds": beam},
        outputs=("before.sdds", "after.sdds"),
    )
    before = [particles(result["before.sdds"], k) for k in range(passes)]
    after = [particles(result["after.sdds"], k) for k in range(passes)]
    if any(len(page["p"]) != N for page in before + after):
        sys.exit(f"{mode}: particles lost")
    for k in range(passes - 1):  # rigid tracking until the last traversal
        if not np.allclose(after[k]["p"], before[k]["p"], rtol=0, atol=2e-13):
            sys.exit(f"{mode}: the cavity acted before the last traversal")
    energy = lambda p: REST_EV * np.sqrt(1 + p**2)
    loss = energy(before[-1]["p"]) - energy(after[-1]["p"])
    return loss, np.array([page["t"] for page in before])


def fmt(values):
    return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"


RTOL, ATOL = 1e-10, 2e-8  # eV
defect = False
for mode in modes():
    for binless in (0, 1) if mode == "serial" else (0,):  # MPI Pelegant has no BINLESS
        for passes, factor, single, label in CASES:
            name = f"{mode} {'binless' if binless else 'binned'} {label}"
            observed, arrivals = last_traversal_losses(mode, binless, passes, factor, single)
            # Expected: the physical circuit, which the Green-function sum cross-checks.
            expected = circuit_losses(arrivals, single=bool(single))[-1]
            histories = arrivals[-1:] if single else arrivals
            green = sum(green_losses(arrivals[-1], past) for past in histories)
            if not np.allclose(green, expected, rtol=2e-13, atol=2e-9):
                sys.exit(f"{name}: the two oracles disagree")
            # Cause: the circuit state scaled by sqrt(a) at the start of every traversal.
            scale = 1.0 if factor is None else factor
            predicted = circuit_losses(arrivals, np.sqrt(scale), bool(single))[-1]
            error = np.max(np.abs(observed - expected))
            fault = scale != 1.0 and not single and passes > 1
            print(
                f"{name}: observed={fmt(observed[-1])} expected={fmt(expected[-1])} "
                f"max error={error:.6g} eV, residual of sqrt(a) prediction="
                f"{np.max(np.abs(observed - predicted)):.3g} eV"
            )
            physical = np.allclose(observed, expected, rtol=RTOL, atol=ATOL)
            if fault and not physical:
                # A mismatch must be explained by the sqrt(a) scaling to count as the defect.
                if not np.allclose(observed, predicted, rtol=RTOL, atol=ATOL):
                    sys.exit(f"{name}: losses follow neither the circuit nor the sqrt(a) scaling")
                defect |= error > 100  # eV; the physical circuit differs by under 1e-7
            elif not physical:
                sys.exit(f"{name}: control disagrees with the circuit; the comparison is invalid")
print(f"comparison: rtol={RTOL:g}, atol={ATOL:g} eV")
verdict(defect)
