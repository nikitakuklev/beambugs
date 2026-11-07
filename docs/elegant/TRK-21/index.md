---
title: "TRK-21: RFCA subtracts its CHANGE_T clock-wrap interval once per kick"
hide:
  - toc
---

# TRK-21 — RFCA subtracts its CHANGE_T clock-wrap interval once per kick

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`trackRfCavityWithWakes`, simple_rfca.c:720–722](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L720-L722)<br>[`add_to_particle_energy`, simple_rfca.c:1137](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1137)<br>[SRS matrix branch, simple_rfca.c:1001–1003](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1001-L1003)<br>[clock receives the interval, simple_rfca.c:629](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L629)<br>[RFCA manual, rfca.tex:19–32](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/rfca.tex#L19-L32)<br>[clock definition, simple_rfca.c:63–75](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L63-L75) |

In an RFCA with CHANGE_T = 1, a nonzero interval T of whole RF periods is subtracted at each of N integration kicks but added to the clock only once. Laboratory arrival time, stored time plus clock, shifts by −(N−1)T although the energy is unchanged, so downstream time-dependent elements act at the wrong time. For a 0.135 m, 24 kV, 0.8 GHz cavity with N = 3 and electron momentum p = 7 mₑc, T = 6.250 ns and the shift is −12.50 ns. A downstream BUMPER gives horizontal slope x′ = 1.475×10⁻³ instead of 1.388×10⁻³, as checked by an independent laboratory-time recurrence. One effective kick is unaffected.

Run `python TRK-21.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-21.py"

    [Download TRK-21.py](TRK-21.py)

    ```python
    """TRK-21: RFCA subtracts its CHANGE_T clock-wrap interval once per kick

    In an RFCA with CHANGE_T = 1, a nonzero interval T of whole RF periods is subtracted at each of N
    integration kicks but added to the clock only once. Laboratory arrival time, stored time plus clock,
    shifts by −(N−1)T although the energy is unchanged, so downstream time-dependent elements act at the
    wrong time. For a 0.135 m, 24 kV, 0.8 GHz cavity with N = 3 and electron momentum p = 7 mₑc, T =
    6.250 ns and the shift is −12.50 ns. A downstream BUMPER gives horizontal slope x′ = 1.475×10⁻³
    instead of 1.388×10⁻³, as checked by an independent laboratory-time recurrence. One effective kick
    is unaffected.

    The script sends 24 electrons with a 6 ps arrival-time spread through an RFCA with CHANGE_T = 0 and
    1, for 1, 3 and 5 kicks, the ordinary and the standing-wave (SRS) body models and two momentum and
    frequency pairs. It checks the unwrapped run against a laboratory-time integration of the cavity,
    then compares the laboratory arrival times of the two runs, which must agree, and the slope given by
    a downstream BUMPER with a linear time waveform with its value for the true laboratory time. The
    one-kick cavities are the control.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h, as Elegant uses them.
    C, ME_MEV = 299792458.0, 0.51099906
    COUNT = 24
    LENGTH, VOLT, PHASE = 0.135, 24000.0, 23.0  # m, V, degrees
    KICK_ANGLE = 0.0007  # downstream BUMPER, rad
    WAVE_SLOPE = 1e7  # 1/s: BUMPER waveform A(u) = 2 - WAVE_SLOPE*u
    P_ATOL, T_ATOL, XP_ATOL = 6e-13, 6e-23, 5e-18
    # (reference momentum p0 in m_e c, RF frequency in Hz, removed RF periods, residual time in s)
    BEAMS = ((7.0, 0.8e9, 5, 0.10e-9), (65.0, 1.6e9, 22, 0.07e-9))
    KICKS = (1, 3, 5)

    WAVEFORM = (
        "SDDS1\n"
        '&column name=t, type=double, units="s" &end\n'
        "&column name=A, type=double &end\n"
        "&data mode=ascii &end\n"
        "2\n"
        "-1e-06 12\n"
        "1e-06 -8\n"
    )  # A(t) = 2 - t/(100 ns)


    def track(mode, p0, frequency, t_in, model, kicks, change_t):
        """Laboratory arrival times, momenta and downstream BUMPER slopes of the beam, and tOffset.

        An entrance BUMPER fixes the waveform origin at the mean input time. The cavity has no end
        focusing, so that the energy gain does not depend on focusing models.
        """
        lattice = (
            'R0: BUMPER,L=0,ANGLE=0,N_KICKS=1,WAVEFORM="wave.sdds=t+A",PHASE_REFERENCE=903\n'
            f"RF: RFCA,L={LENGTH},VOLT={VOLT!r},FREQ={frequency!r},PHASE={PHASE!r},T_REFERENCE=0,"
            f"CHANGE_T={change_t},N_KICKS={kicks},BODY_FOCUS_MODEL={model},"
            f"STANDING_WAVE={int(model == 'srs')},END1_FOCUS=0,END2_FOCUS=0\n"
            f'K: BUMPER,L=0,ANGLE={KICK_ANGLE!r},N_KICKS=1,WAVEFORM="wave.sdds=t+A",PHASE_REFERENCE=903\n'
            "BL: LINE=(R0,RF,K)\n"
        )
        commands = (
            "&change_particle name=electron &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, output=out.sdds, "
            "default_order=2, random_number_seed=94821 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        files = {"lattice.lte": lattice, "run.ele": commands, "wave.sdds": WAVEFORM}
        result = run(files, mode, {"beam.sdds": {"t": t_in, "p": np.full(COUNT, p0)}})["out.sdds"]
        out = particles(result)
        if len(out["p"]) != COUNT or not all(np.all(np.isfinite(v)) for v in out.values()):
            sys.exit(f"{mode} p0={p0} {model} kicks={kicks}: particles were lost or not finite")
        t_offset = float(result["tOffset"][0])
        t_lab = out["t"] + t_offset
        if not np.isfinite(t_offset) or not np.all(np.isfinite(t_lab)):
            sys.exit(
                f"{mode} p0={p0} {model} kicks={kicks}: the clock or laboratory times are not finite"
            )
        return t_lab, out["p"], out["xp"], t_offset


    def ordinary_cavity(p0, t_in, frequency, kicks):
        """Laboratory-time integration of the on-axis cavity with no clock: the independent oracle.

        Each of the N kicks is preceded and followed by a drift of L/(2N) at the current velocity,
        t += L/(2 N beta c), and changes the energy by
        dgamma = V/N/(m c^2/e) sin(2 pi f (t - k L/(N c)) + phi), k the kick index (the field
        phase is referenced to the light-speed arrival at the kick), p = sqrt(gamma^2 - 1).
        """
        p, t = np.full(COUNT, p0), t_in.copy()
        for k in range(kicks):
            t = t + LENGTH / (2 * kicks * C) * np.hypot(p, 1) / p
            phase = 2 * np.pi * frequency * (t - k * LENGTH / (kicks * C)) + np.deg2rad(PHASE)
            gamma = np.hypot(p, 1) + VOLT / kicks / (1e6 * ME_MEV) * np.sin(phase)
            p = np.sqrt((gamma - 1) * (gamma + 1))
            t = t + LENGTH / (2 * kicks * C) * np.hypot(p, 1) / p
        return p, t


    def downstream_slope(p0, t_lab, t_in, p_out):
        """BUMPER slope: ANGLE*A(t - mean(t_in))*p0/p, with A(u) = 2 - WAVE_SLOPE*u."""
        return KICK_ANGLE * (2 - WAVE_SLOPE * (t_lab - t_in.mean())) * p0 / p_out


    def show(label, observed, expected):
        print(f"{label}: observed=[{float(observed)!r}] expected=[{float(expected)!r}]")


    def main():
        defect = False
        for mode in modes():
            for p0, frequency, cycles, residual in BEAMS:
                removed = cycles / frequency  # CHANGE_T removes whole RF periods from the time
                t_in = removed + residual + np.linspace(-3e-12, 3e-12, COUNT)
                for model in ("none", "srs"):
                    for kicks in KICKS:
                        plain = track(mode, p0, frequency, t_in, model, kicks, 0)
                        wrapped = track(mode, p0, frequency, t_in, model, kicks, 1)
                        label = f"{mode} p0={p0:g} f={frequency:g} {model} kicks={kicks}"
                        if model == "none":
                            oracle_p, oracle_t = ordinary_cavity(p0, t_in, frequency, kicks)
                            if np.any(np.abs(plain[1] - oracle_p) > P_ATOL) or np.any(
                                np.abs(plain[0] - oracle_t) > T_ATOL
                            ):
                                sys.exit(f"{label}: the unwrapped run disagrees with the oracle")
                        # The clock register receives the removed interval once; every sampled time is
                        # reduced by it, so the laboratory time t + tOffset must not change.
                        if abs(wrapped[3] - removed) > T_ATOL / 3:
                            sys.exit(f"{label}: the clock does not hold one removed interval")
                        if np.any(np.abs(wrapped[1] - plain[1]) > P_ATOL):
                            sys.exit(f"{label}: wrapping changes the momentum")
                        shift = wrapped[0] - plain[0]
                        # Removing T at each of N kicks and banking it once leaves -(N - 1) T.
                        predicted = -(kicks - 1) * removed
                        if np.any(np.abs(shift - predicted) > T_ATOL):
                            sys.exit(f"{label}: the time shift is not -(N - 1) T")
                        show(label + " lab-time shift wrapped - unwrapped", shift[0], 0.0)
                        # Downstream kick from the laboratory time: the oracle where it exists, else
                        # the unwrapped run, which the one-kick and oracle cases validate.
                        ref_t, ref_p = (oracle_t, oracle_p) if model == "none" else plain[:2]
                        expected_slope = downstream_slope(p0, ref_t, t_in, ref_p)
                        if np.any(np.abs(plain[2] - expected_slope) > XP_ATOL):
                            sys.exit(f"{label}: the unwrapped downstream kick is wrong")
                        show(label + " downstream kick", wrapped[2][0], expected_slope[0])
                        bad = bool(
                            np.any(np.abs(shift) > T_ATOL)
                            or np.any(np.abs(wrapped[2] - expected_slope) > XP_ATOL)
                        )
                        if kicks == 1:
                            if bad:
                                sys.exit(f"{label}: the one-kick control is not invariant")
                            continue
                        defect |= bad
        print(f"comparison: atol p={P_ATOL:g}, t={T_ATOL:g} s, xp={XP_ATOL:g}")
        verdict(defect)


    main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |

