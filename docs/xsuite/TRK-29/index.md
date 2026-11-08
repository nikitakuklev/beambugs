---
title: "TRK-29: Absolute-time RF clock accumulates rounding once per turn"
hide:
  - toc
---

# TRK-29 — Absolute-time RF clock accumulates rounding once per turn

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`increment_at_turn`, local_particle_common.h:281–283](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/headers/local_particle_common.h#L281-L283)<br>[local_particle_scalar.h:115–123](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/headers/local_particle_scalar.h#L115-L123)<br>[tracker.py:1741–1790](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/tracker.py#L1741-L1790)<br>[`_advance_particle_time_s_and_reference_turns`, tracker.py:2054–2077](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/tracker.py#L2054-L2077)<br>[`track_rf_kick_single_particle`, track_rf.h:45–56](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_rf.h#L45-L56)<br>[lines 64–66](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_rf.h#L64-L66)<br>[lines 138–146](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_rf.h#L138-L146)<br>[thin](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/thin_slice_cavity.h#L15-L51)<br>[thick](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/thick_slice_cavity.h#L15-L51)<br>[thin](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/thin_slice_crab_cavity.h#L15-L51)<br>[thick](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/thick_slice_crab_cavity.h#L15-L51)<br>[Cavity docstring, cavity.py:34–54](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/cavity.py#L34-L54)<br>[CrabCavity docstring, crab_cavity.py:34–42](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/crab_cavity.py#L34-L42)<br>[Particles docstring, particles.py:245–258](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/particles.py#L245-L258)<br>[absolute-time RF kick, xtrack.tex:5621–5650](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L5621-L5650)<br>[numerical-reproducibility page](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/numericalreproducibility.rst) |

With `absolute_time=True`, Cavity and CrabCavity read a simulation clock that tracking advances by adding the revolution period once per turn, so its rounding error accumulates and depends on how the turns are split into calls. In a 26.66 km ring with 450.0 GeV/c protons and a 400.8 MHz (harmonic 35640) cavity, one 60.00 s tracking call drifts the clock by 5.096×10⁻¹⁰ s, an RF phase error of 1.283 rad. An initially synchronous particle in a 6.000 MV longitudinal model ends that call at longitudinal coordinate ζ = 0.1528 m instead of 0, and resuming tracking kicks it by 5.754 MeV instead of 0. Exact integer-turn clocks establish these values; relative-time cavities and an exactly representable revolution period remain accurate.

Run `python TRK-29.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-29.py"

    [Download TRK-29.py](TRK-29.py)

    ```python
    """TRK-29: Absolute-time RF clock accumulates rounding once per turn

    With `absolute_time=True`, Cavity and CrabCavity read a simulation clock that tracking advances by adding the revolution period once per turn, so its rounding error accumulates and depends on how the turns are split into calls. In a 26.66 km ring with 450.0 GeV/c protons and a 400.8 MHz (harmonic 35640) cavity, one 60.00 s tracking call drifts the clock by 5.096×10⁻¹⁰ s, an RF phase error of 1.283 rad. An initially synchronous particle in a 6.000 MV longitudinal model ends that call at longitudinal coordinate ζ = 0.1528 m instead of 0, and resuming tracking kicks it by 5.754 MeV instead of 0. Exact integer-turn clocks establish these values; relative-time cavities and an exactly representable revolution period remain accurate.

    A 26658.8832 m ring at 450.0 GeV/c carries a harmonic-35640 absolute-time Cavity followed by a one-turn map. With a 1 V cavity and frozen longitudinal motion, the kick on the last turn reads the RF phase the kernel used; the expected phase is computed in exact rational arithmetic from the integer turn count times the stored revolution period, with whole RF periods removed exactly, and a binary64 replay of the per-turn addition attributes the difference to the clock. One long call, the same turns in 1000-turn calls, one-turn calls and a CrabCavity are compared; a 6 MV cavity with momentum slip shows the synchronous particle displaced and kicked on resumption. A relative-time cavity, an exactly representable revolution period and a fresh event at the reconstructed clock are passing controls; installed serial prebuilt kernels run with compilation forbidden.
    """

    import math
    from fractions import Fraction

    from _prebuilt import CTX, identity, kernels, np, xt

    C = 299792458.0  # exact SI value of c, as C_LIGHT in the kernel
    P0C = 450e9
    LENGTH = 26658.8832
    HARMONIC = 35640
    TURNS = 674_729  # 60.00 s of LHC revolutions
    failures = []


    def exact_phase(frequency, time):
        # 2*pi*frac(f*t) in [-pi, pi): f and t are exact rationals of the stored
        # binary64 values, so the only rounding is the final conversion.
        cycles = Fraction(frequency) * time
        cycles -= (cycles + Fraction(1, 2)) // 1
        return 2 * math.pi * float(cycles)


    def replay_clock(turns, calls, period):
        # Clock of the last event (turn turns-1) as the pinned tracker forms it.
        # Each line.track launches its first turn alone and then the remaining
        # turns in one kernel; the kernel adds t_sim to its local clock after every
        # turn (local_particle_common.h), and after each launch Python stores
        # time_s + count*t_sim (tracker.py), discarding the kernel's local clock.
        stored, done, event = 0.0, 0, 0.0
        for count in calls:
            event = stored
            stored += 1 * period
            local = stored
            for _ in range(count - 1):
                event = local
                local += period
            stored += (count - 1) * period
            done += count
        assert done == turns
        return event, stored


    def bound(frequency, time):
        # Four ulps of the unreduced phase 2*pi*f*t that track_rf.h forms: the
        # phase-resolution loss of a large clock, separate from this defect,
        # stays inside it. For a 1 V kick the bound in eV equals the bound in rad.
        return 4 * math.ulp(2 * math.pi * frequency * time) + 1e-12


    def compare(label, observed, expected, tol, derivation, extra="", control=False):
        passed = abs(observed - expected) <= tol
        if control:
            assert passed, (label, observed, expected)
        else:
            failures.append(not passed)
        print(f"{label}: observed={observed!r} expected={expected!r} "
              f"error={observed - expected:.4g} bound={tol:.3g} {extra}"
              f"derivation={derivation} match={passed}")


    def build(elements):
        line = xt.Line(elements=elements)
        line.particle_ref = xt.Particles(p0c=P0C)
        line.build_tracker(_context=CTX)
        return line


    def unit_kick(label, turns, block=None, *, kind="Cavity", absolute=True,
                  length=LENGTH, control=False):
        block = block or turns
        calls = [min(block, turns - start) for start in range(0, turns, block)]
        beta0 = float(xt.Particles(p0c=P0C).beta0[0])
        period = length / (beta0 * C)  # t_sim and the kernel's t_rev0
        frequency = HARMONIC / period  # track_rf.h: harmonic / t_rev0
        if kind == "Cavity":
            rf = xt.Cavity(voltage=1.0, harmonic=HARMONIC, phase=0.0, absolute_time=absolute)
        else:  # CrabCavity takes a fixed frequency only
            rf = xt.CrabCavity(crab_voltage=1.0, frequency=frequency, phase=0.0,
                               absolute_time=absolute)
        # Zero tunes make the one-turn map the identity, so px keeps the crab kicks.
        line = build([rf, xt.LineSegmentMap(length=length, qx=0.0, qy=0.0,
                                            longitudinal_mode="frozen")])
        p = xt.Particles(_context=CTX, p0c=P0C, zeta=0.0)
        monitor = xt.ParticlesMonitor(_context=CTX, start_at_turn=turns - 2,
                                      stop_at_turn=turns, num_particles=1)
        for count in calls:
            line.track(p, num_turns=count, turn_by_turn_monitor=monitor)
        assert float(p.t_sim) == period and int(p.state[0]) == 1
        # Kick of the last turn n = turns-1 at s = zeta = 0: 1 V gives
        # Delta E = sin(2*pi*f*t_n) eV, with Delta(ptau) for Cavity and Delta(px)
        # for CrabCavity, whose -cos(phi + pi/2) deflection equals sin(phi).
        attr = "ptau" if kind == "Cavity" else "px"
        observed = (float(getattr(p, attr)[0]) - float(getattr(monitor, attr)[0, -1])) * P0C
        n = turns - 1
        event, stored = replay_clock(turns, calls, period)
        assert float(p.time_s) == stored
        if absolute:
            expected = math.sin(exact_phase(frequency, n * Fraction(period)))
            drift = float(Fraction(event) - n * Fraction(period))
            extra = (f"clock_drift={drift:.4g} s beta0*c*drift={beta0 * C * drift:.4g} m "
                     f"replayed_kick={math.sin(exact_phase(frequency, Fraction(event)))!r} "
                     f"stored_time_s_error={float(Fraction(stored) - turns * Fraction(period)):.3g} s ")
            tol = bound(frequency, n * period)
            derivation = "sin(2*pi*frac(f*n*T)), exact rational n*T"
        else:
            expected, extra, tol = 0.0, "", 1e-12
            derivation = "relative-time kick at zeta=0 is sin(0)"
        compare(f"{label} ({len(calls)} call(s), last turn {n}, t={n * period:.6g} s)",
                observed, expected, tol, derivation, extra, control)
        return period


    def reconstructed_event():
        # Fresh one-turn call at the final event's clock, set from the exact
        # integer-turn product n*T rounded once to binary64.
        beta0 = float(xt.Particles(p0c=P0C).beta0[0])
        period = LENGTH / (beta0 * C)
        frequency = HARMONIC / period
        n = TURNS - 1
        clock = float(n * Fraction(period))
        line = build([xt.Cavity(voltage=1.0, harmonic=HARMONIC, phase=0.0, absolute_time=True),
                      xt.LineSegmentMap(length=LENGTH, longitudinal_mode="frozen")])
        p = xt.Particles(_context=CTX, p0c=P0C, at_turn=n, time_s=clock, reference_turns=n)
        line.track(p, num_turns=1)
        compare(f"one-turn call at reconstructed clock n*T, n={n} (control)",
                float(p.ptau[0]) * P0C, math.sin(exact_phase(frequency, n * Fraction(period))),
                bound(frequency, clock), "sin(2*pi*frac(f*n*T)), exact rational n*T",
                control=True)


    def synchronous_particle(absolute):
        # 6 MV cavity at phase pi followed by an explicit momentum-slip map,
        # dzeta = -(alpha_p - 1/gamma0^2)*C*delta, without internal RF. With an
        # integer harmonic the manual's turn identity removes 2*pi*h per turn, so
        # zeta = delta = 0 is an exact synchronous solution with zero kick.
        beta0 = float(xt.Particles(p0c=P0C).beta0[0])
        frequency = HARMONIC / (LENGTH / (beta0 * C))
        line = build([
            xt.Cavity(voltage=6e6, harmonic=HARMONIC, phase=math.pi, absolute_time=absolute),
            xt.LineSegmentMap(length=LENGTH, qx=0.31, qy=0.32, longitudinal_mode="nonlinear",
                              voltage_rf=0, frequency_rf=frequency,
                              momentum_compaction_factor=3.225e-4),
        ])
        p = xt.Particles(_context=CTX, p0c=P0C, zeta=0.0, delta=0.0)
        line.track(p, num_turns=TURNS)
        zeta = float(np.asarray(p.zeta)[0])
        before = float(p.ptau[0])
        line.track(p, num_turns=1)  # resume in a second call
        kick = (float(p.ptau[0]) - before) * P0C
        assert int(np.asarray(p.state)[0]) == 1
        mode = "absolute time" if absolute else "relative time (control)"
        compare(f"synchronous particle, {mode}, zeta after one {TURNS}-turn call [m]",
                zeta, 0.0, 1e-12, "exact synchronous solution", control=not absolute)
        compare(f"synchronous particle, {mode}, energy kick on resuming [eV]",
                kick, 0.0, 1e-3, "sin(pi) at the synchronous position", control=not absolute)


    identity()
    unit_kick("Cavity, one call", TURNS)
    unit_kick("Cavity, 1000-turn calls", TURNS, 1000)
    unit_kick("Cavity, one call", 10_000)
    unit_kick("Cavity, one-turn calls", 10_000, 1)
    unit_kick("CrabCavity, one call", TURNS, kind="CrabCavity")
    unit_kick("relative-time Cavity, one call (control)", TURNS, absolute=False, control=True)
    # Exactly representable revolution period 2^-13 s: every repeated addition is
    # exact, so the long call must agree with the integer-turn clock.
    binary_length = float(xt.Particles(p0c=P0C).beta0[0]) * C * 2.0**-13
    period = unit_kick("Cavity, one call, period 2^-13 s (control)", TURNS,
                       length=binary_length, control=True)
    assert period == 2.0**-13
    reconstructed_event()
    synchronous_particle(True)
    synchronous_particle(False)
    kernels()
    print(f"COMPARISONS failing={sum(failures)} of {len(failures)} absolute-time cases")
    print("DEFECT REPRODUCED" if any(failures) else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed<br><small class="bb-scope">integer-turn clock and call partitions</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

