---
title: "TRK-22: Absolute reference clocks corrupt thick RF phase after many turns"
hide:
  - toc
---

# TRK-22 — Absolute reference clocks corrupt thick RF phase after many turns

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on ImpactX development (`f9c2a82`, head build) |
| Tags | <span class="bb-tag">known-limitation</span> <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Issue | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1391]] |
| Discussed in | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1195]] |
| Source | [coordinate contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/theory/coordinates_units.rst#L6-L17)<br>[`ReferenceParticle.H:34–42`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/particles/ReferenceParticle.H#L34-L42)<br>[`RFCavity.H:347–353`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/RFCavity.H#L347-L353)<br>[`464–481`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/RFCavity.H#L464-L481)<br>[`414–428`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/RFCavity.H#L414-L428)<br>[`193–204`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/RFCavity.H#L193-L204)<br>[`Sbend.H:262–277`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/Sbend.H#L262-L277)<br>[`Drift.H:175–185`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/Drift.H#L175-L185)<br>[thick-cavity contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L1748-L1780)<br>[Fourier representation](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/theory/softedge_elements.rst#L13-L27)<br>[`RF_Efield`, 304–317](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/RFCavity.H#L304-L317)<br>[Defaults are ten map steps and one slice](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L1816-L1824)<br>[positive-step validation](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/RFCavity.H#L559-L563)<br>[second-order split](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/particles/integrators/Integrators.H#L93-L112)<br>[`RFCavity.H:601–612`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/RFCavity.H#L601-L612)<br>[reference RF example](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/examples/rfcavity/run_rfcavity_ref_part.py#L44-L54)<br>[Repeated lattice periods](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L301-L306)<br>[`common.H:149–166`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/tracking/common.H#L149-L166)<br>[synchronous phase](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L1950-L1958)<br>[`ShortRF.H:199–220`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/ShortRF.H#L199-L220) |

After many turns, rounding of the absolute reference clock changes the phase of thick RF cavities. For protons at Lorentz factor γ₀=2, sampling a 4.000 cm, 1.300 GHz gap at 28.15 MV/m after 10⁶ turns of a 998.6 m magnetic ring gives −787.7 keV instead of +587.4 keV with the default ten integration steps. Exact flight time and a high-precision split map establish the expected gain; independent analytic-invariant quadrature gives +588.1 keV before finite-step error. The stored clock is ahead by 176.2 ps. Refining the cavity integration can introduce further clock error. ShortRF uses synchronous phase and is independent of this absolute clock.

Run `python TRK-22.py` with [_precision.py](../helpers/_precision.py) in the same directory.

??? example "Reproducer: TRK-22.py"

    [Download TRK-22.py](TRK-22.py)

    ```python
    """TRK-22: Absolute reference clocks corrupt thick RF phase after many turns

    After many turns, rounding of the absolute reference clock changes the phase of thick RF cavities. For protons at Lorentz factor γ₀=2, sampling a 4.000 cm, 1.300 GHz gap at 28.15 MV/m after 10⁶ turns of a 998.6 m magnetic ring gives −787.7 keV instead of +587.4 keV with the default ten integration steps. Exact flight time and a high-precision split map establish the expected gain; independent analytic-invariant quadrature gives +588.1 keV before finite-step error. The stored clock is ahead by 176.2 ps. Refining the cavity integration can introduce further clock error. ShortRF uses synchronous phase and is independent of this absolute clock.

    Setup: run pinned to one CPU (`taskset -c <n>`); the shared `_precision` helper refuses to run on more, and child processes inherit that CPU.

    Track four native sector bends per turn and sample a constant-profile RF gap after 10⁴ and 10⁶ turns against exact flight time and a 65-digit local-clock split map. An invariant quadrature independently checks the continuum limit, while initialized clocks and zero-origin, field-free, Drift and ShortRF controls separate accumulation from integration error. Two fresh simulations check full particle tracking, with fixed particles, disabled collective effects, radiation, spin and diagnostics, and explicit finalization. The shared _precision helper caps numerical threads.
    """

    import hashlib
    import json
    import math
    import subprocess
    import sys
    from importlib.metadata import version
    from pathlib import Path

    # Apply the shared thread/affinity guard before native numerical imports.
    from _precision import CPU

    # isort: split

    import mpmath as mp
    import numpy as np

    C = 299792458
    MASS = 938.27208816
    FREQUENCY = 1.3e9
    LENGTH = 0.04
    FIELD = 0.03
    mp.mp.dps = 65


    def phase_at_entry(clock, phase):
        return (2 * mp.pi * mp.mpf(FREQUENCY) * clock / C + mp.mpf(phase) * mp.pi / 180) % (2 * mp.pi)


    def compensated_phase(clock):
        return float((20 - 360 * mp.mpf(FREQUENCY) * clock / C + 180) % 360 - 180)


    def split_solution(theta, steps, slices):
        gamma, elapsed = mp.mpf(2), mp.mpf(0)
        # The source is a kick/drift/kick split. Keeping elapsed time local removes
        # large-clock addition loss without changing the finite integration order.
        dz = mp.mpf(LENGTH) / (steps * slices)
        kick = mp.mpf(FIELD) * dz / 2
        wave = 2 * mp.pi * mp.mpf(FREQUENCY) / C
        for _ in range(steps * slices):
            gamma += kick * mp.cos(theta + wave * elapsed)
            elapsed += dz * gamma / mp.sqrt(gamma * gamma - 1)
            gamma += kick * mp.cos(theta + wave * elapsed)
        return gamma, elapsed


    def invariant_solution(theta):
        wave = 2 * mp.pi * mp.mpf(FREQUENCY) / C

        # With b=sqrt(gamma²-1), db/dT=E*cos(theta+k*T). Integrate that
        # invariant first, then solve L=integral beta(T) dT, independently of
        # the native s-stepping integrator. Input binary64 values stay exact.
        def momentum(elapsed):
            return mp.sqrt(3) + mp.mpf(FIELD) / wave * (mp.sin(theta + wave * elapsed) - mp.sin(theta))

        def distance(elapsed):
            return mp.quad(lambda u: momentum(u) / mp.sqrt(1 + momentum(u) ** 2), [0, elapsed])

        guess = mp.mpf(LENGTH) * 2 / mp.sqrt(3)
        elapsed = mp.findroot(
            lambda u: distance(u) - mp.mpf(LENGTH),
            (guess * mp.mpf("0.99"), guess * mp.mpf("1.01")),
            tol=mp.mpf("1e-58"),
        )
        assert abs(distance(elapsed) - mp.mpf(LENGTH)) < mp.mpf("1e-55")
        return mp.sqrt(1 + momentum(elapsed) ** 2), elapsed


    def reference(clock=0.0):
        import impactx

        ref = impactx.RefPart().set_charge_qe(1).set_mass_MeV(MASS).set_kin_energy_MeV(MASS)
        assert ref.gamma == 2
        ref.t = clock
        return ref


    def cavity(phase, steps, slices, field=FIELD):
        from impactx import elements

        return elements.RFCavity(
            ds=LENGTH,
            escale=field,
            freq=FREQUENCY,
            phase=phase,
            cos_coefficients=[2.0],
            sin_coefficients=[0.0],
            mapsteps=steps,
            nslice=slices,
        )


    def native_gap(clock, phase, steps, slices, field=FIELD):
        import impactx

        ref, elem = reference(clock), cavity(phase, steps, slices, field)
        # Public reference push advances one slice. Fresh spatial coordinates
        # isolate time arithmetic from the separate s-sedge profile subtraction.
        for _ in range(slices):
            impactx.push(ref, elem)
        return ref.gamma, ref.t


    def gap_case(name, clock, phase, steps, slices, expected_clock):
        observed, observed_clock = native_gap(clock, phase, steps, slices)
        theta = phase_at_entry(expected_clock, phase)
        gamma, elapsed = split_solution(theta, steps, slices)
        energy_error = float((mp.mpf(observed) - gamma) * MASS * 1e6)
        clock_error = float((mp.mpf(clock) - expected_clock) / C * mp.mpf(1e12))
        print(
            f"{name}: gain={((observed - 2) * MASS * 1000):.10f} keV "
            f"expected={float((gamma - 2) * MASS * 1000):.10f} keV "
            f"error={energy_error:.10g} eV input_clock_error={clock_error:.10g} ps "
            f"elapsed={observed_clock - clock:.17g} m expected_elapsed={float(elapsed):.17g} m; "
            "expected from exact arrival phase and 65-digit same-step local-clock splitting"
        )
        return observed, gamma, elapsed, energy_error


    def beam_child(spec):
        from impactx import ImpactX

        sim = ImpactX()
        try:
            for key, value in dict(
                particle_shape=1,
                space_charge="off",
                strang_split=False,
                diagnostics=False,
                slice_step_diagnostics=False,
                tiny_profiler=False,
                memory_profiler=False,
                csr=False,
                isr=False,
                spin=False,
                verbose=0,
            ).items():
                setattr(sim, key, value)
            sim.init_grids()
            ref = sim.beam.ref
            ref.set_charge_qe(1).set_mass_MeV(MASS).set_kin_energy_MeV(MASS)
            ref.t = spec["clock"]
            rays = np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0.001, 0.002, -0.001, 0.001, 0.002, -0.001],
                    [-0.001, -0.002, 0.001, -0.001, -0.002, 0.001],
                ]
            )
            sim.beam.add_n_particles(
                **dict(zip(("x", "px", "y", "py", "t", "pt"), rays.T.copy(), strict=True)),
                qm=ref.qm_ratio_SI,
                w=np.ones(3),
            )
            sim.lattice.append(cavity(spec["phase"], 100, 4))
            sim.track_particles()
            columns = [
                "position_x",
                "momentum_x",
                "position_y",
                "momentum_y",
                "position_t",
                "momentum_t",
            ]
            particles = sim.beam.to_df(local=True).sort_index()[columns].to_numpy()
            assert particles.shape == (3, 6) and np.all(np.isfinite(particles))
            return dict(gamma=ref.gamma, particles=particles.tolist())
        finally:
            sim.finalize()


    def run_beam(clock, phase):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "child"],
            input=json.dumps(dict(clock=clock, phase=phase)),
            text=True,
            capture_output=True,
            check=True,
            timeout=290,
        )
        rows = [line[7:] for line in result.stdout.splitlines() if line.startswith("RESULT ")]
        if len(rows) != 1:
            raise RuntimeError(result.stdout + result.stderr)
        return json.loads(rows[0])


    def main():
        import impactx
        import impactx.impactx_pybind as native

        config = impactx.Config.to_dict()
        assert config["precision"] == config["precision_particles"] == "DOUBLE"
        assert not any(config[key] for key in ("have_mpi", "have_omp", "have_gpu"))
        print(
            f"Python {sys.version.split()[0]} "
            + " ".join(f"{name}={version(name)}" for name in ("impactx-noacc", "numpy", "mpmath"))
            + f" CPU={CPU} numerical_threads=1; fixed protons, no random seed"
        )
        print("binary_sha256=" + hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest())
        print("config=" + json.dumps(config, sort_keys=True))
        defects = []
        beta = mp.sqrt(3) / 2
        circumference = float(5000 * beta * C / mp.mpf(FREQUENCY))
        quarter = circumference / 4
        bend = impactx.elements.Sbend(ds=quarter, rc=circumference / (2 * float(mp.pi)))
        ref = reference()
        for turn in range(1, 1000001):
            for _ in range(4):
                impactx.push(ref, bend)
            if turn not in (10000, 1000000):
                continue
            # Four exact binary64 bend lengths divided by analytic beta give
            # flight time independently of the accumulated native clock.
            true_clock = turn * 4 * mp.mpf(quarter) / beta
            phase = compensated_phase(true_clock)
            for steps, slices in ((10, 1), (100, 4)):
                row = gap_case(
                    f"ring_{turn}_turns_{steps}x{slices}", ref.t, phase, steps, slices, true_clock
                )
                if turn == 1000000:
                    defects.append(abs(row[3]) > 1e4)
            if turn == 1000000:
                row = gap_case("ring_once_rounded_10x1", float(true_clock), phase, 10, 1, true_clock)
                assert abs(row[3]) < 2

        clock = float(mp.mpf(1000000) * 5000 * C / mp.mpf(FREQUENCY))
        phase = compensated_phase(mp.mpf(clock))
        initialized = gap_case("initialized_clock_100x4", clock, phase, 100, 4, mp.mpf(clock))
        theta = phase_at_entry(mp.mpf(clock), phase)
        continuum, elapsed = invariant_solution(theta)
        with mp.workdps(85):
            refined, refined_time = invariant_solution(theta)
        assert float(continuum) == float(refined) and float(elapsed) == float(refined_time)
        truncation = float((initialized[1] - continuum) * MASS * 1e6)
        print(
            f"invariant: gain={float((continuum - 2) * MASS * 1000):.10f} keV "
            f"expected_elapsed={float(elapsed):.17g} m finite_step_error={truncation:.10g} eV; "
            "expected from db/dT=E*cos(theta+k*T), scalar quadrature/root, 65/85 digits agree"
        )
        defects.append(abs(initialized[3]) > 100 * abs(truncation))
        local_phase = float(theta * 180 / mp.pi)
        local = gap_case("local_clock_100x4", 0.0, local_phase, 100, 4, mp.mpf(0))
        # Each local split has O(n*u) arithmetic error; this generous bound
        # includes energy kicks and elementary-function rounding, not truncation.
        bound = 128 * 400 * 2**-53 * (2 + FIELD * LENGTH)
        assert abs(mp.mpf(local[0]) - local[1]) < bound
        for distance in (1000, 100000):
            short_clock = distance / (math.sqrt(3) / 2)
            row = gap_case(
                f"linac_{distance}m_100x4",
                short_clock,
                compensated_phase(mp.mpf(short_clock)),
                100,
                4,
                mp.mpf(short_clock),
            )
            assert abs(row[3]) < 1e-3

        frozen = float(2**40)
        _, observed_clock = native_gap(frozen, 20, 10000, 1, field=0)
        exact_advance = mp.mpf(LENGTH) * 2 / mp.sqrt(3)
        rounded = float(mp.mpf(frozen) + exact_advance)
        print(
            f"field_free_10000x1: elapsed={observed_clock - frozen:.17g} m "
            f"expected={float(exact_advance):.17g} m once_rounded={rounded - frozen:.17g} m; "
            "expected from L/beta, rounded once only at the final absolute clock"
        )
        defects.append(observed_clock == frozen)
        drift = reference(frozen)
        impactx.push(drift, impactx.elements.Drift(ds=LENGTH))
        assert drift.t == rounded
        print(
            f"drift: elapsed={drift.t - frozen:.17g} m expected={rounded - frozen:.17g} m; L/beta rounded once"
        )
        expected_thin = 2 + LENGTH * FIELD * math.cos(math.radians(20))
        for origin in (0.0, frozen):
            thin = reference(origin)
            impactx.push(thin, impactx.elements.ShortRF(V=LENGTH * FIELD, freq=FREQUENCY, phase=20))
            assert abs(thin.gamma - expected_thin) < 5e-16 and thin.t == origin
            print(
                f"ShortRF_clock_{origin:.0f}: gamma={thin.gamma:.17g} expected={expected_thin:.17g}; 2+V*cos(20 degrees)"
            )

        beam_local, beam_large = run_beam(0.0, local_phase), run_beam(clock, phase)
        assert abs(beam_large["gamma"] - initialized[0]) < 1e-14
        delta = np.array(beam_large["particles"]) - np.array(beam_local["particles"])
        assert np.max(np.abs(delta[0])) < 1e-15
        print(
            f"particles: large_gamma={beam_large['gamma']:.17g} "
            f"expected_direct_reference_gamma={initialized[0]:.17g} "
            f"delta_px={delta[1, 1]:.10g} delta_pt={delta[1, 5]:.10g} delta_t={delta[1, 4]:.10g} m "
            "expected_origin_delta=0; compensated clock-origin equivalence, three fixed rays"
        )
        print("DEFECT REPRODUCED" if any(defects) else "NOT REPRODUCED")


    if __name__ == "__main__":
        if sys.argv[1:] == ["child"]:
            print("RESULT " + json.dumps(beam_child(json.load(sys.stdin))))
        elif sys.argv[1:]:
            raise RuntimeError("Run without arguments")
        else:
            main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD |

