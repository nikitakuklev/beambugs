---
title: "API-19: MAD-X beam import retains the preceding species’ magnetic anomaly"
hide:
  - toc
---

# API-19 — MAD-X beam import retains the preceding species’ magnetic anomaly

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on ImpactX development (`f9c2a82`, head build) |
| Tags | <span class="bb-tag">experimental</span> [?](../../methodology.md#tags) |
| Source | [`src/python/impactx/madx_to_impactx.py:1707–1754`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/impactx/madx_to_impactx.py#L1707-L1754)<br>[__init__.py:22](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/impactx/__init__.py#L22-L22)<br>[62](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/python/impactx/__init__.py#L62-L62)<br>[`src/particles/ReferenceParticle.H:111–122`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/particles/ReferenceParticle.H#L111-L122)<br>[`src/particles/ReferenceParticle.H:139–141`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/particles/ReferenceParticle.H#L139-L141)<br>[`src/elements/Sol.H:116–123`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/Sol.H#L116-L123)<br>[`src/elements/Sol.H:284–290`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/Sol.H#L284-L290)<br>[zero-anomaly guard](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/tracking/particles.cpp#L109-L115)<br>[named-species contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/python.rst#L846-L854)<br>[load_file contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/python.rst#L888-L898)<br>[spin contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L2792-L2805) |

With experimental spin tracking, importing a named electron beam through RefPart.load_file() after selecting protons retains the proton magnetic anomaly. At 150.0 MeV total energy, a 0.3500 T, 0.6000 m solenoid rotates an initially vertical unit spin to (−0.9216, 0.3881, 0) instead of (−0.4079, 0.9130, 0), a 0.5249 component error. An independent Thomas–BMT integration establishes the electron result. Explicitly setting the electron anomaly restores it, and imported orbital tracking remains unchanged.

Run `python API-19.py`.

??? example "Reproducer: API-19.py"

    [Download API-19.py](API-19.py)

    ```python
    """API-19: MAD-X beam import retains the preceding species’ magnetic anomaly

    With experimental spin tracking, importing a named electron beam through RefPart.load_file() after selecting protons retains the proton magnetic anomaly. At 150.0 MeV total energy, a 0.3500 T, 0.6000 m solenoid rotates an initially vertical unit spin to (−0.9216, 0.3881, 0) instead of (−0.4079, 0.9130, 0), a 0.5249 component error. An independent Thomas–BMT integration establishes the electron result. Explicitly setting the electron anomaly restores it, and imported orbital tracking remains unchanged.

    Setup: run pinned to one CPU (`taskset -c <n>`); the script refuses to run on more.

    Two fixed axial particles traverse a 0.6 m solenoid at 25 and 150 MeV with both field signs after
    seven species/import histories. A Cartesian SI-unit Thomas–BMT integration and constant-axis
    exponential supply independent spin expectations, with explicit-anomaly, native-electron and
    spin-disabled controls. Temporary MAD-X decks live beside this script and are removed on exit.
    """

    import contextlib
    import importlib.metadata
    import io
    import math
    import os
    import platform
    import warnings
    from pathlib import Path
    from tempfile import TemporaryDirectory

    THREADS = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    )
    for name in THREADS:
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError("Launch pinned to one CPU, for example taskset -c 5")

    import numpy as np
    import scipy
    from impactx import Config, ImpactX, ImpactXParIter, RefPart, elements, read_beam
    from scipy.integrate import solve_ivp

    ROOT = Path(__file__).resolve().parent
    C = 299792458.0
    E = 1.602176634e-19
    G_ELECTRON = 0.00115965218062
    LENGTH = 0.6
    SPIN0 = np.array([0.0, 1.0, 0.0])
    PHASE = ("position_x", "position_y", "position_t", "momentum_x", "momentum_y", "momentum_t")
    SPIN = ("spin_x", "spin_y", "spin_z")


    def oracle(ref, field, anomaly):
        """Cartesian Thomas-BMT: dS/dt = Omega x S, E=0, beta along z."""
        gamma = ref.gamma
        beta = np.array([0.0, 0.0, ref.beta])
        magnetic = np.array([0.0, 0.0, field])
        omega = -(ref.charge / ref.mass) * (
            (anomaly + 1 / gamma) * magnetic
            - anomaly * gamma / (gamma + 1) * np.dot(beta, magnetic) * beta
        )
        duration = LENGTH / (ref.beta * C)
        # Integrate in dimensionless time u=t/T to avoid an SI time absolute tolerance.
        result = solve_ivp(
            lambda _u, spin: duration * np.cross(omega, spin),
            (0.0, 1.0),
            SPIN0,
            method="DOP853",
            rtol=3e-14,
            atol=3e-15,
        )
        assert result.success
        numeric = result.y[:, -1]
        # Independent exact exponential for a constant axial Cartesian angular velocity.
        angle = float(omega[2] * duration)
        exact = np.array([-math.sin(angle), math.cos(angle), 0.0])
        assert np.max(np.abs(numeric - exact)) <= 2e-13
        return numeric, exact, angle


    def view(pc):
        tiles = [tile.soa().to_numpy(copy=False).real for tile in ImpactXParIter(pc, level=0)]
        assert len(tiles) == 1
        return tiles[0]


    def run(directory):
        assert len(os.sched_getaffinity(0)) == 1
        config = Config.to_dict()
        if any(config[k] for k in ("have_mpi", "have_gpu", "have_omp")):
            raise RuntimeError("This reproducer requires a serial CPU build")
        if config["precision"] != "DOUBLE" or config["precision_particles"] != "DOUBLE":
            raise RuntimeError("This reproducer requires double precision")
        print(
            f"Python {platform.python_version()} impactx-noacc="
            f"{importlib.metadata.version('impactx-noacc')} numpy={np.__version__} "
            f"scipy={scipy.__version__}; CPU double serial SIMD={config['simd_size']}; "
            "two fixed axial particles, no random seed"
        )
        assert RefPart().set_species("electron").gyromagnetic_anomaly == G_ELECTRON
        sim = ImpactX()
        try:
            sim.tiny_profiler = False
            sim.memory_profiler = False
            sim.verbose = 0
            sim.particle_shape = 2
            sim.space_charge = "off"
            sim.csr = False
            sim.isr = False
            sim.isr_on_ref_part = False
            sim.diagnostics = False
            sim.slice_step_diagnostics = False
            sim.spin = False
            sim.diag_file_prefix = str(directory / "spin_files")
            sim.init_grids()
            pc = sim.beam
            ref = pc.ref
            ref.set_species("electron").set_kin_energy_MeV(150)
            zeros = np.zeros(2)
            pc.add_n_particles(
                zeros, zeros, zeros, zeros, zeros, zeros, qm=ref.qm_ratio_SI, w=np.ones(2)
            )
            results = []
            for energy in (25.0, 150.0):
                deck = directory / f"electron_{energy:g}.madx"
                deck.write_text(f"BEAM, PARTICLE=electron, ENERGY={energy / 1000:.17g};\n")
                for field in (0.35, -0.35):
                    orbit_controls = []
                    for history in (
                        "fresh",
                        "electron",
                        "proton",
                        "proton_explicit_G",
                        "proton_read_beam",
                        "proton_spin_off",
                        "native_electron",
                    ):
                        ref.reset()
                        if history != "fresh":
                            ref.set_species(
                                "electron" if history in ("electron", "native_electron") else "proton"
                            )
                        if history == "native_electron":
                            mass_mev = ref.mass * C * C / E / 1e6
                            ref.set_kin_energy_MeV(energy - mass_mev)
                        elif history == "proton_read_beam":
                            with (
                                warnings.catch_warnings(record=True) as messages,
                                contextlib.redirect_stdout(io.StringIO()),
                            ):
                                read_beam(ref, str(deck))
                            assert messages and all(
                                "anomal" not in str(m.message).lower() for m in messages
                            )
                        else:
                            with (
                                warnings.catch_warnings(record=True) as messages,
                                contextlib.redirect_stdout(io.StringIO()),
                            ):
                                RefPart.load_file(ref, str(deck))
                            assert messages and all(
                                "anomal" not in str(m.message).lower() for m in messages
                            )
                        if history == "proton_explicit_G":
                            ref.set_gyromagnetic_anomaly(G_ELECTRON)
                        mass_mev = ref.mass * C * C / E / 1e6
                        observed_energy = ref.gamma * mass_mev
                        assert abs(observed_energy - energy) <= 6e-14
                        assert ref.charge == -E
                        real = view(pc)
                        for key in PHASE:
                            real[key][:] = 0.0
                        for key, value in zip(SPIN, SPIN0, strict=True):
                            real[key][:] = value
                        real["qm"][:] = ref.qm_ratio_SI
                        expected, exact, angle = oracle(ref, field, G_ELECTRON)
                        inherited, _, inherited_angle = oracle(ref, field, ref.gyromagnetic_anomaly)
                        # Sol.ks = q B / P0. P0 = beta*gamma*m*c, in SI.
                        ks = ref.charge * field / (ref.beta_gamma * ref.mass * C)
                        sim.lattice.clear()
                        sim.lattice.append(elements.Sol(ds=LENGTH, ks=ks))
                        sim.spin = history != "proton_spin_off"
                        record = dict(
                            energy_MeV=energy,
                            B_T=field,
                            L_m=LENGTH,
                            ks_per_m=ks,
                            history=history,
                            anomaly=ref.gyromagnetic_anomaly,
                            expected=expected.tolist(),
                            exact=exact.tolist(),
                            expected_angle=angle,
                            inherited_angle=inherited_angle,
                        )
                        try:
                            sim.track_particles()
                        except RuntimeError as exc:
                            assert history == "fresh" and "gyromagnetic anomaly" in str(exc)
                            record.update(status="ZERO_ANOMALY_GUARD", error=str(exc))
                        else:
                            real = view(pc)
                            spins = np.column_stack([real[key] for key in SPIN])
                            phases = np.column_stack([real[key] for key in PHASE])
                            assert np.max(np.abs(phases)) == 0.0
                            orbit_controls.append((history, phases.copy(), ref.t, ref.z))
                            observed = spins[0]
                            assert np.array_equal(spins[0], spins[1])
                            error = float(np.max(np.abs(observed - expected)))
                            if history == "proton_spin_off":
                                assert np.array_equal(observed, SPIN0)
                                status = "SPIN_DISABLED"
                            elif history in ("proton", "proton_read_beam"):
                                assert np.max(np.abs(observed - inherited)) < 3e-13
                                if error > 0.1:
                                    status = "STALE_ANOMALY"
                                else:
                                    assert error < 3e-13
                                    status = "ELECTRON_BMT_PASS"
                            else:
                                assert error < 3e-13
                                status = "ELECTRON_BMT_PASS"
                            record.update(
                                status=status,
                                observed=observed.tolist(),
                                max_error=error,
                                inherited_prediction_error=float(np.max(np.abs(observed - inherited))),
                            )
                        results.append(record)
                        label = f"E={energy:g} MeV B={field:+g} T {history} G={ref.gyromagnetic_anomaly:.14g}"
                        if record["status"] == "ZERO_ANOMALY_GUARD":
                            print(
                                f"{label}: observed=ZERO_ANOMALY_GUARD expected=electron BMT rotation"
                            )
                        else:
                            vector = lambda a: "(" + ",".join(f"{x:.16g}" for x in a) + ")"
                            target = SPIN0 if history == "proton_spin_off" else expected
                            print(
                                f"{label}: observed={vector(record['observed'])} "
                                f"expected={vector(target)} status={record['status']} "
                                f"error={np.max(np.abs(np.array(record['observed']) - target)):.4g}; "
                                "expected from SI Thomas-BMT/identity when spin off"
                            )
                    imports = [item for item in orbit_controls if item[0] != "native_electron"]
                    assert all(
                        np.array_equal(item[1], imports[0][1]) and item[2:] == imports[0][2:]
                        for item in imports
                    )
                    print(
                        f"ORBIT energy={energy} B={field}: all imported histories bit-identical",
                        flush=True,
                    )
            stale = sum(r["status"] == "STALE_ANOMALY" for r in results)
            guarded = sum(r["status"] == "ZERO_ANOMALY_GUARD" for r in results)
            print(
                f"histories: stale={stale}/8 zero_anomaly_guard={guarded}/4 expected=0,0; "
                "12 electron BMT controls and 4 spin-disabled controls pass at 3e-13"
            )
            print("DEFECT REPRODUCED" if stale or guarded else "NOT REPRODUCED")
        finally:
            sim.finalize()


    def main():
        with TemporaryDirectory(prefix="API-19-", dir=ROOT) as directory:
            run(Path(directory))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD |
| Found<br><small>named-electron import and spin on development</small> | `gpt-6-astra` | xhigh | 2026-11-06 | development (`f9c2a82e24b7`), source, CPU NOACC, double, serial, FFT, no openPMD or SIMD |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD |
| Confirmed<br><small>independent SI Thomas–BMT confirmation on development</small> | `gpt-6.1-sol` | xhigh | 2026-11-06 | development (`f9c2a82e24b7`), source, CPU NOACC, double, serial, FFT, no openPMD or SIMD |

