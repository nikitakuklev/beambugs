---
title: "COL-46: WAKE gives each particle the full W(0) of its own bin, so full-origin wakes such as CORGPIPE's overstate the loss"
hide:
  - toc
---

# COL-46 — WAKE gives each particle the full W(0) of its own bin, so full-origin wakes such as CORGPIPE's overstate the loss

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Tags | <span class="bb-tag">contract-unsettled</span> <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [binTimeDistribution, wake.c:539–545](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L539-L545)<br>[convolution and kick, wake.c:273–290](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L273-L290)<br>[wake.c:514–529](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L514-L529)<br>[WAKE manual, wake.tex:1–12](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/wake.tex#L1-L12)<br>[CORGPIPE table, wake.c:678–686](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L678-L686)<br>[CORGPIPE manual, corgpipe.tex:1–5](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/corgpipe.tex#L1-L5)<br>[CORGPLATES table, wake.c:816–820](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L816-L820)<br>[cwake.c:399](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/cwake.c#L399) |

With default binning and gather settings, WAKE uses the full table value W(0) for charge in a particle’s own bin. If W(0) is the positive-delay limit W(0⁺), this adds an endpoint excess; the beam-loading value is W(0⁺)/2. CORGPIPE generates full-origin tables: for a uniform 0.8 nC, 4 ps bunch in a 0.1 m pipe, the mean loss exceeds the exact particle sum by 4.288% at the default step and 1.058% at a quarter step. Halving the origin leaves 0.02645% binning error in the default-step cosine control. WAKE’s manual leaves the origin convention unspecified; tables already containing W(0⁺)/2 avoid this endpoint excess.

Run `python COL-46.py` with [_elegant.py](../helpers/_elegant.py) and [_wake.py](../helpers/_wake.py) in the same directory.

??? example "Reproducer: COL-46.py"

    [Download COL-46.py](COL-46.py)

    ```python
    """COL-46: WAKE gives each particle the full W(0) of its own bin, so full-origin wakes such as CORGPIPE's overstate the loss

    With default binning and gather settings, WAKE uses the full table value W(0) for charge in a
    particle’s own bin. If W(0) is the positive-delay limit W(0⁺), this adds an endpoint excess; the
    beam-loading value is W(0⁺)/2. CORGPIPE generates full-origin tables: for a uniform 0.8 nC, 4 ps
    bunch in a 0.1 m pipe, the mean loss exceeds the exact particle sum by 4.288% at the default
    step and 1.058% at a quarter step. Halving the origin leaves 0.02645% binning error in the
    default-step cosine control. WAKE’s manual leaves the origin convention unspecified; tables
    already containing W(0⁺)/2 avoid this endpoint excess.

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

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed<br><small>endpoint excess for full-origin WAKE and CORGPIPE data</small> | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

