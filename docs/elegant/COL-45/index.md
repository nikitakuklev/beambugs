---
title: "COL-45: Longitudinal space charge uses an unpadded FFT, so an isolated bunch gets the field of a train spaced by its own length"
hide:
  - toc
---

# COL-45 — Longitudinal space charge uses an unpadded FFT, so an isolated bunch gets the field of a train spaced by its own length

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [LSCDRIFT window, lsc.c:145](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L145)<br>[forward FFT, lsc.c:246](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L246)<br>[impedance and inverse FFT, lsc.c:337–358](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L337-L358)<br>[CSRDRIFT calls, csbend.c:5212](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L5212)<br>[5229](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L5229)<br>[window, lsc.c:446](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L446)<br>[FFT, lsc.c:543](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L543)<br>[impedance and inverse FFT, lsc.c:625–642](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L625-L642)<br>[allocation, lsc.c:96–101](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L96-L101)<br>[LSCDRIFT manual, lscdrift.tex:1–6](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/lscdrift.tex#L1-L6)<br>[rfcw.tex:14–19](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/rfcw.tex#L14-L19) |

LSCDRIFT, and the longitudinal space charge of RFCW and CSRDRIFT, apply the impedance with a discrete Fourier transform over a window three bins longer than the bunch's sampled extent, without zero padding, so the field is that of a periodic train of bunch copies. For short, wide, low-energy bunches the images cancel much of the field. At Lorentz factor γ = 4, a 1 pC Gaussian bunch of 80 μm rms length and 2 mm rms size receives at +1σ a kick 21.43% weaker than the isolated-bunch field from a real-space Coulomb integral, and 6.004% weaker at γ = 15. More bins do not help; stretching the window to 80σ brings the kick within 0.09324% of the isolated value.

Run `python COL-45.py` with [_elegant.py](../helpers/_elegant.py) and [_lsc.py](../helpers/_lsc.py) in the same directory.

??? example "Reproducer: COL-45.py"

    [Download COL-45.py](COL-45.py)

    ```python
    """COL-45: Longitudinal space charge uses an unpadded FFT, so an isolated bunch gets the field of a train spaced by its own length

    LSCDRIFT, and the longitudinal space charge of RFCW and CSRDRIFT, apply the impedance with a
    discrete Fourier transform over a window three bins longer than the bunch's sampled extent, without
    zero padding, so the field is that of a periodic train of bunch copies. For short, wide, low-energy
    bunches the images cancel much of the field. At Lorentz factor γ = 4, a 1 pC Gaussian bunch of 80 μm
    rms length and 2 mm rms size receives at +1σ a kick 21.43% weaker than the isolated-bunch field from
    a real-space Coulomb integral, and 6.004% weaker at γ = 15. More bins do not help; stretching the
    window to 80σ brings the kick within 0.09324% of the isolated value.

    The script tracks 262144 electrons (Gaussian quantiles, sigma_ct = 80 um, rms sizes 2 mm,
    1 pC) through one thin LSCDRIFT (L=0, LEFFECTIVE=0.2 m, 2048 bins) at gamma = 4 and 15, and at
    gamma = 4 through a 0.2 m RFCW with VOLT=0 and LSC=1, and compares the energy kick at +1 sigma
    with the open-boundary field of the same uniform-disk model, an independent real-space integral
    with no FFT. A NumPy rebuild of Elegant's periodic transform reproduces every kick. The control
    adds two spectator particles at +-40 sigma, which barely change the open field but stretch the
    FFT period, and shows the kick converging to the open value.
    """

    import numpy as np
    from _elegant import verdict
    from _lsc import (
        EPSILON_0,
        RADIUS_FACTOR,
        C,
        at,
        energy,
        open_kick,
        periodic_kick,
        round_beam,
        track,
    )

    COUNT, SIGMA, SIGMA_R, CHARGE, LEFF = 262144, 80e-6, 2e-3, 1e-12, 0.2
    RB = RADIUS_FACTOR * SIGMA_R
    REBUILD_ATOL = 1e-6  # eV; the NumPy rebuild of the periodic FFT matches Elegant to ~1e-9 eV
    CONTROL_RTOL = 2e-3  # the long-period control must approach the open-boundary kick this closely
    MIN_DEFICIT = 0.01  # defect if the gamma = 4 kick misses the open-boundary kick by more than 1%


    def lsc_kick(beam: dict, p: float, element: str, charge: float) -> np.ndarray:
        """Energy change (eV) of each particle through the CHARGE + LSC element."""
        out = track(f"Q: CHARGE,TOTAL={charge!r}\nL1: {element}", "Q,L1", beam, p)
        return energy(out["p"]) - energy(p)


    def case(
        label: str,
        beam: dict,
        p: float,
        element: str,
        bins: int,
        open_value: float,
        charge: float = CHARGE,
    ) -> float:
        """Print the kick at +1 sigma against the open-boundary value; return observed/open - 1."""
        z = C * beam["t"]
        kick = lsc_kick(beam, p, element, charge)
        rebuilt, period = periodic_kick(z, p, RB, charge, LEFF, bins)
        rebuild_error = float(np.max(np.abs(kick - rebuilt)))
        if rebuild_error > REBUILD_ATOL:
            raise SystemExit(f"{label}: the periodic rebuild misses Elegant by {rebuild_error:g} eV")
        observed = at(z, kick, SIGMA)
        print(
            f"serial {label}: kick at +1 sigma observed={observed:.9g} eV "
            f"expected (open boundary)={open_value:.9g} eV ratio={observed / open_value:.6f}; "
            f"FFT period {period / SIGMA:.4f} sigma, periodic rebuild within {rebuild_error:.2g} eV"
        )
        return observed / open_value - 1


    defect = False
    for gamma in (4.0, 15.0):
        p = float(np.sqrt(gamma**2 - 1))
        beam = round_beam(COUNT, SIGMA, SIGMA_R, p)
        open_value = open_kick(1.0, SIGMA, p, RB, CHARGE, LEFF)
        deficit = case(
            f"LSCDRIFT gamma={gamma:g}",
            beam,
            p,
            f"LSCDRIFT,L=0,LEFFECTIVE={LEFF},BINS=2048,INTERPOLATE=1,SMOOTHING=0",
            2048,
            open_value,
        )
        if gamma == 4.0:
            defect = deficit < -MIN_DEFICIT
            # The RF cavity's LSC (addLSCKick) uses the same window. VOLT=0 and zero slopes leave the
            # bunch unchanged over the 0.2 m, so its one kick of length 0.2 m is the same thin kick;
            # 1 fC keeps the plasma length long enough for one kick (the cavity refuses a kick
            # spacing above a tenth of it), and the kicks scale linearly with charge.
            case(
                "RFCW gamma=4 (VOLT=0, one kick)",
                beam,
                p,
                f"RFCW,L={LEFF},CELL_LENGTH={LEFF},VOLT=0,FREQ=0,N_KICKS=1,LSC=1,LSC_BINS=2048,"
                "ZWAKE=0,TRWAKE=0",
                2048,
                open_value * 1e-3,
                charge=CHARGE * 1e-3,
            )
            # Control: two spectators at -+40 sigma, total charge and rms sizes unchanged, 16384
            # bins to keep the bin width. Their open-space field at +1 sigma is the single-disk
            # kernel sign(v) - v/sqrt(v^2 + R^2), v = 1 -+ 40 in units of sigma, R = r_b/(p sigma);
            # the Gaussian part keeps COUNT of the COUNT + 2 macroparticles.
            spect = {
                "x": np.r_[0.0, beam["x"], 0.0],
                "y": np.r_[0.0, beam["y"], 0.0],
                "t": np.r_[-40 * SIGMA / C, beam["t"], 40 * SIGMA / C],
                "p": np.full(COUNT + 2, p),
                "particleID": np.arange(1, COUNT + 3),
            }
            for axis in ("x", "y"):
                spect[axis] = spect[axis] / spect[axis].std() * SIGMA_R
            big_r = RB / (p * SIGMA)
            kernel = sum(np.sign(v) - v / np.hypot(v, big_r) for v in (41.0, -39.0))
            scale = -CHARGE * LEFF / (2 * np.pi * EPSILON_0 * RB**2)
            open_spect = (COUNT * open_value + scale * kernel) / (COUNT + 2)
            control = case(
                "LSCDRIFT gamma=4, spectators at -+40 sigma (control)",
                spect,
                p,
                f"LSCDRIFT,L=0,LEFFECTIVE={LEFF},BINS=16384,INTERPOLATE=1,SMOOTHING=0",
                16384,
                open_spect,
            )
            if abs(control) > CONTROL_RTOL:
                raise SystemExit("the long-period control does not approach the open-boundary kick")
    print(
        f"defect if the gamma=4 LSCDRIFT kick is more than {MIN_DEFICIT:.0%} below the open-boundary "
        f"kick; control tolerance {CONTROL_RTOL:g} relative"
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-04</span> | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed<br><small class="bb-scope">LSCDRIFT for isolated bunches with appreciable image fields</small> | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-04</span> | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

