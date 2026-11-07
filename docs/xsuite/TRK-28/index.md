---
title: "TRK-28: Absolute-time RF phase loses particle arrival offsets at long clocks"
hide:
  - toc
---

# TRK-28 — Absolute-time RF phase loses particle arrival offsets at long clocks

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack main (`f09e0fb`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`track_rf_kick_single_particle`, track_rf.h:51–66](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_rf.h#L51-L66)<br>[track_rf.h:138–146](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_rf.h#L138-L146)<br>[harmonic branch, lines 45–49](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_rf.h#L45-L49)<br>[Cavity wrapper, cavity.h:11–48](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/cavity.h#L11-L48)<br>[CrabCavity wrapper, crab_cavity.h:11–48](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/crab_cavity.h#L11-L48)<br>[thin](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/thin_slice_cavity.h#L15-L51)<br>[thick](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/thick_slice_cavity.h#L15-L51)<br>[thin](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/thin_slice_crab_cavity.h#L15-L51)<br>[thick](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/thick_slice_crab_cavity.h#L15-L51)<br>[Cavity docstring, cavity.py:40–54](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/cavity.py#L40-L54)<br>[CrabCavity docstring, crab_cavity.py:34–42](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/crab_cavity.py#L34-L42)<br>[Particles docstring, particles.py:247–252](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/particles.py#L247-L252)<br>[`_advance_particle_time_s_and_reference_turns`, tracker.py:2068–2077](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/tracker.py#L2068-L2077)<br>[absolute-time RF kick, xtrack.tex:5621–5650](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L5621-L5650)<br>[numerical-reproducibility page](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/numericalreproducibility.rst)<br>[setter, particles.py:1206–1211](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/particles.py#L1206-L1211) |

With `absolute_time=True`, Cavity and CrabCavity add each particle's small arrival-time offset to an RF phase formed from the full simulation clock, so these offsets lose resolution as the clock grows. For 450.0 GeV/c protons and a 1.000 MV, 400.8 MHz cavity at a clock of one hour, energy gains are wrong by up to 340.5 eV, and the kick difference between particles 1.000 mm apart in the longitudinal coordinate ζ is 6.995% too small. At 18.20 hours that difference is zero instead of −8147 eV. A 50-digit sine with whole RF periods removed exactly, and the integer-harmonic identity with relative-time kicks, establish the expected values. Relative-time cavities and recent clock origins remain accurate.

Run `python TRK-28.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-28.py"

    [Download TRK-28.py](TRK-28.py)

    ```python
    """TRK-28: Absolute-time RF phase loses particle arrival offsets at long clocks

    With `absolute_time=True`, Cavity and CrabCavity add each particle's small arrival-time offset to an RF phase formed from the full simulation clock, so these offsets lose resolution as the clock grows. For 450.0 GeV/c protons and a 1.000 MV, 400.8 MHz cavity at a clock of one hour, energy gains are wrong by up to 340.5 eV, and the kick difference between particles 1.000 mm apart in the longitudinal coordinate ζ is 6.995% too small. At 18.20 hours that difference is zero instead of −8147 eV. A 50-digit sine with whole RF periods removed exactly, and the integer-harmonic identity with relative-time kicks, establish the expected values. Relative-time cavities and recent clock origins remain accurate.

    A 1.000 MV cavity at 400.8 MHz, with fixed frequency, tracks 450.0 GeV/c protons at ζ = 0 and 1.000 mm with simulation clocks that are exact whole numbers of RF periods, so each event must equal the same event at clock zero. The expected kicks come from a 50-digit evaluation of the RF sine with the whole periods removed exactly, and a second set of fixtures uses an exactly representable revolution period at an integer harmonic, where the physics manual's identity makes the absolute-time kick equal to the relative-time kick on every turn. CrabCavity is checked the same way. Clock zero, relative-time cavities at the long clocks and a quarter-period origin change with compensated phase are passing controls; installed serial prebuilt kernels run with compilation forbidden.
    """

    import mpmath as mp
    from _prebuilt import CTX, identity, kernels, np, xt

    mp.mp.dps = 50
    C = 299792458  # exact SI value of c
    P0C = 450e9
    VOLTAGE = 1e6
    FREQUENCY = 400789000.0
    PHASE = 0.25
    ZETA = (0.0, 1e-3)
    # A comparison bound of 1e-4 eV is 1e-10 of the voltage; correct binary64
    # evaluation at a recent clock origin is below 1e-9 eV for these kicks.
    ATOL_EV = 1e-4
    failures = []


    def exact(value):
        # Binary64 inputs convert exactly; the oracle uses the stored values.
        return mp.mpf(float(value))


    def whole_periods(frequency, time_s):
        # f*t as an exact rational from the stored binary64 values.
        fn, fd = float(frequency).as_integer_ratio()
        tn, td = float(time_s).as_integer_ratio()
        return (fn * tn) % (fd * td) == 0


    def rf_sine(frequency, zeta, beta0, phase, cycles=0):
        # Delta E = V sin(2 pi f t + phase) with t = time_s - zeta/(beta0 c) at s = 0;
        # cycles is frac(f*time_s), removed before the small zeta term is added.
        return mp.sin(
            2 * mp.pi * (mp.mpf(cycles) - exact(frequency) * exact(zeta) / (exact(beta0) * C))
            + exact(phase)
        )


    def cavity_gains(frequency, time_s, *, absolute=True, phase=PHASE):
        p = xt.Particles(_context=CTX, p0c=P0C, zeta=list(ZETA), time_s=time_s)
        cavity = xt.Cavity(_context=CTX, voltage=VOLTAGE, frequency=frequency,
                           phase=phase, absolute_time=absolute)
        cavity.track(p)
        # Initial ptau is zero, so the energy gain is p0c*ptau with no cancellation.
        return np.asarray(p.ptau) * P0C, float(p.beta0[0])


    def expected_gains(frequency, beta0, phase=PHASE, cycles=0):
        return np.array([float(VOLTAGE * rf_sine(frequency, z, beta0, phase, cycles))
                         for z in ZETA])


    def compare(label, observed, expected, derivation, control=False):
        observed, expected = np.asarray(observed), np.asarray(expected)
        error = float(np.max(np.abs(observed - expected)))
        passed = error <= ATOL_EV
        if control:
            assert passed, (label, observed, expected)
        else:
            failures.append(not passed)
        diff_obs = float(observed[1] - observed[0])
        diff_exp = float(expected[1] - expected[0])
        print(f"{label}: observed={observed.tolist()} expected={expected.tolist()} "
              f"max_error={error:.7g} eV kick_difference observed={diff_obs!r} "
              f"expected={diff_exp!r} relative_error={diff_obs / diff_exp - 1:.6g} "
              f"derivation={derivation} match={passed}")


    def fixed_frequency():
        gains, beta0 = cavity_gains(FREQUENCY, 0.0)
        reference = expected_gains(FREQUENCY, beta0)
        compare("clock 0 s (control)", gains, reference,
                "50-digit V*sin(phase-2*pi*f*zeta/(beta0*c))", control=True)
        for time_s in (60.0, 600.0, 3600.0, 36000.0, 65536.0):
            assert whole_periods(FREQUENCY, time_s)
            gains, _ = cavity_gains(FREQUENCY, time_s)
            compare(f"clock {time_s:g} s", gains, reference,
                    "f*time_s is an exact integer: same event as clock 0")
        for time_s in (3600.0, 65536.0):
            gains, _ = cavity_gains(FREQUENCY, time_s, absolute=False)
            compare(f"relative time, clock {time_s:g} s (control)", gains, reference,
                    "relative-time cavity ignores the clock", control=True)
        # Quarter-period origin change: phase at time zero shifts by -pi/2, so the
        # event at time_s = 1/(4f) equals the event at clock zero with the stored phase.
        quarter = 0.25 / FREQUENCY
        shifted = PHASE - np.pi / 2
        gains, _ = cavity_gains(FREQUENCY, quarter, phase=shifted)
        cycles = mp.mpf(float(FREQUENCY)) * exact(quarter) % 1
        compare("quarter-period origin, compensated phase (control)", gains,
                expected_gains(FREQUENCY, beta0, phase=shifted, cycles=cycles),
                "50-digit sine of the exactly reduced clock", control=True)


    def harmonic_identity():
        # Revolution period 2^-13 s and harmonic 48925 (f = 400.7936 MHz): turn N has
        # the exact clock N*2^-13 s, and f*t = 48925*N whole periods, so the manual's
        # integer-harmonic identity makes every turn's kick equal to the relative-time
        # cavity's kick at the same zeta.
        t_rev, harmonic = 2.0**-13, 48925
        frequency = harmonic / t_rev
        relative, beta0 = cavity_gains(frequency, 0.0, absolute=False)
        compare("harmonic turn 0, relative time (control)", relative,
                expected_gains(frequency, beta0), "50-digit sine", control=True)
        for turns in (1_000_000, 30_000_000):
            time_s = turns * t_rev
            assert whole_periods(frequency, time_s)
            gains, _ = cavity_gains(frequency, time_s)
            compare(f"harmonic turn {turns} (clock {time_s:.6g} s)", gains, relative,
                    "integer-harmonic identity: relative-time kick")


    def crab_cavity():
        kicks = {}
        for time_s in (0.0, 3600.0, 65536.0):
            p = xt.Particles(_context=CTX, p0c=P0C, zeta=list(ZETA), time_s=time_s)
            crab = xt.CrabCavity(_context=CTX, crab_voltage=VOLTAGE, frequency=FREQUENCY,
                                 phase=PHASE, absolute_time=True)
            crab.track(p)
            # At x = 0 the deflection is (V/p0c)*sin(2*pi*f*t+phase); report it in
            # eV as p0c*px so the same bound applies.
            kicks[time_s] = (np.asarray(p.px) * P0C, float(p.beta0[0]))
        reference = expected_gains(FREQUENCY, kicks[0.0][1])
        compare("CrabCavity clock 0 s (control)", kicks[0.0][0], reference,
                "50-digit (V/p0c)*sin(phase-2*pi*f*zeta/(beta0*c)), times p0c", control=True)
        for time_s in (3600.0, 65536.0):
            compare(f"CrabCavity clock {time_s:g} s", kicks[time_s][0], reference,
                    "f*time_s is an exact integer: same deflection as clock 0")


    identity()
    fixed_frequency()
    harmonic_identity()
    crab_cavity()
    kernels()
    print(f"COMPARISONS failing={sum(failures)} of {len(failures)} long-clock cases; "
          f"bound {ATOL_EV:g} eV")
    print("DEFECT REPRODUCED" if any(failures) else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed<br><small>integer-harmonic turn identity</small> | `claude-opus-5-5` | high | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

