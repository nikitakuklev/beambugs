---
title: "IO-06: Source cannot reload empty BeamMonitor populations"
hide:
  - toc
---

# IO-06 — Source cannot reload empty BeamMonitor populations

| | |
|---|---|
| Code | [ImpactX](../index.md) · Particles and I/O |
| Priority | P3 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on ImpactX commit [`f9c2a82`](https://github.com/BLAST-ImpactX/impactx/commit/f9c2a82e24b7), committed 2026-11-05 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Pull request | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1694]], fix |
| Source | [`src/elements/Source.cpp:133–160`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/Source.cpp#L133-L160)<br>[`src/elements/Source.cpp:182–208`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/elements/Source.cpp#L182-L208)<br>[Source contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L2110-L2118)<br>[reference-restoration option](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L2136-L2149) |

Loading a BeamMonitor file with zero saved particles through Source raises “Unallocated pointer passed during chunk loading.” instead of injecting an empty population. The reader passes a null buffer to openPMD even for a zero-length read. The same call is reached on MPI ranks assigned no particles when the file contains fewer particles than ranks. HDF5, JSON and TOML empty files fail in serial tests; one- and three-particle files preserve all tested coordinates and weights exactly.

Run `python IO-06.py`.

??? example "Reproducer: IO-06.py"

    [Download IO-06.py](IO-06.py)

    ```python
    """IO-06: Source cannot reload empty BeamMonitor populations

    Loading a BeamMonitor file with zero saved particles through Source raises “Unallocated pointer passed during chunk loading.” instead of injecting an empty population. The reader passes a null buffer to openPMD even for a zero-length read. The same call is reached on MPI ranks assigned no particles when the file contains fewer particles than ranks. HDF5, JSON and TOML empty files fail in serial tests; one- and three-particle files preserve all tested coordinates and weights exactly.

    Setup: run pinned to one CPU (`taskset -c <n>`); the script refuses to run on more.

    Directly constructed zero-, one- and three-particle proton populations are written and reloaded
    through HDF5, JSON and TOML in both encodings and restoration modes. Independent HDF5 inspection and
    exact coordinate/weight sums establish the expected reads, while an existing sentinel checks empty
    injection. Temporary files live beside this script and are removed on exit.
    """

    import importlib.metadata
    import os
    import platform
    import tomllib
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

    import h5py
    import numpy as np
    from impactx import Config, ImpactX, ImpactXParIter, elements, push

    HERE = Path(__file__).resolve().parent
    E_CHARGE = 1.602176634e-19
    ERROR = "Unallocated pointer passed during chunk loading."
    FIELDS = (
        "position_x",
        "position_y",
        "position_t",
        "momentum_x",
        "momentum_y",
        "momentum_t",
        "weighting",
    )


    def arrays(pc):
        rows = []
        for tile in ImpactXParIter(pc, level=0):
            real = tile.soa().to_numpy(copy=False).real
            rows.extend(np.column_stack([real[k] for k in FIELDS]).tolist())
        data = np.asarray(rows).reshape((-1, len(FIELDS)))
        return data[data[:, 0].argsort()]


    def load(pc, path, **kwargs):
        try:
            push(pc, elements.Source("openPMD", str(path), **kwargs))
        except RuntimeError as exc:
            if str(exc) != ERROR:
                raise
            return "NULL_BUFFER_ERROR"
        return "LOADED"


    def run(directory):
        config = Config.to_dict()
        if any(config[k] for k in ("have_mpi", "have_gpu", "have_omp")):
            raise RuntimeError("This reproducer requires a serial CPU build")
        if config["precision"] != "DOUBLE" or config["precision_particles"] != "DOUBLE":
            raise RuntimeError("This reproducer requires double precision")
        if not config["have_openpmd"] or not all(
            config["openpmd_backends"][k] for k in ("hdf5", "json", "toml")
        ):
            raise RuntimeError("This reproducer requires openPMD with HDF5, JSON and TOML")
        print(
            f"Python {platform.python_version()} impactx-noacc="
            f"{importlib.metadata.version('impactx-noacc')} numpy={np.__version__} "
            f"h5py={h5py.__version__}; CPU double serial SIMD={config['simd_size']}; "
            "fixed particles, no random seed"
        )
        sim = ImpactX()
        try:
            for key, value in dict(
                tiny_profiler=False,
                memory_profiler=False,
                verbose=0,
                particle_shape=2,
                space_charge="off",
                csr=False,
                isr=False,
                isr_on_ref_part=False,
                diagnostics=False,
                slice_step_diagnostics=False,
                spin=False,
                diag_file_prefix=str(directory),
            ).items():
                setattr(sim, key, value)
            sim.init_grids()
            pc, ref = sim.beam, sim.beam.ref
            failures = appends = reads = 0
            for backend in ("h5", "json", "toml"):
                for encoding in ("g", "f"):
                    for count in (0, 1, 3):
                        pc.clear_particles()
                        ref.reset()
                        ref.set_species("proton").set_kin_energy_MeV(250)
                        z = np.zeros(count)
                        x = np.array([-0.003, 0.001, 0.004])[:count]
                        weights = np.array([2.0, 3.0, 5.0])[:count]
                        if count:
                            pc.add_n_particles(x, z, z, z, z, z, qm=ref.qm_ratio_SI, w=weights)
                        expected = arrays(pc)
                        expected_ref = (ref.mass, ref.charge, ref.pt, ref.gyromagnetic_anomaly)
                        name = f"{backend}_{encoding}_{count}"
                        monitor = elements.BeamMonitor(
                            name,
                            backend=backend,
                            encoding=encoding,
                            particles=True,
                            beam_moments=False,
                        )
                        try:
                            push(pc, monitor, step=7)
                        finally:
                            monitor.finalize()
                        suffix = "_%T" if encoding == "f" else ""
                        path = directory / "openPMD" / f"{name}{suffix}.{backend}"
                        (physical,) = path.parent.glob(name + "*." + backend)
                        if backend == "h5":
                            with h5py.File(physical) as file:
                                beam = file["data/7/particles/beam"]
                                ids = beam["id"]
                                extent = (
                                    ids.shape[0]
                                    if isinstance(ids, h5py.Dataset)
                                    else int(ids.attrs["shape"][0])
                                )
                                assert extent == count
                                if count:
                                    assert np.array_equal(beam["weighting"][...], weights)
                        stored_ref = expected_ref
                        if backend == "toml":
                            attrs = tomllib.loads(physical.read_text())["data"]["7"]["particles"][
                                "beam"
                            ]["attributes"]
                            stored_ref = tuple(
                                attrs[k]
                                for k in (
                                    "mass_ref",
                                    "charge_ref",
                                    "pt_ref",
                                    "gyromagnetic_anomaly_ref",
                                )
                            )
                            # Text serialization can round by one ulp; restoration must equal
                            # the stored attributes, rather than the original binary values.
                            ulps = np.abs(
                                (np.array(stored_ref) - expected_ref)
                                / np.spacing(np.array(expected_ref))
                            )
                            assert ulps.max() <= 1
                        statuses = []
                        for restore in (False, True):
                            pc.clear_particles()
                            ref.mass, ref.charge, ref.pt, ref.gyromagnetic_anomaly = expected_ref
                            status = load(pc, path, load_step=7, load_ref_particle=restore)
                            reads += 1
                            statuses.append(status)
                            if status == "NULL_BUFFER_ERROR":
                                assert count == 0 and len(arrays(pc)) == 0
                                failures += 1
                            else:
                                loaded = arrays(pc)
                                assert pc.total_number_of_particles() == count
                                assert np.array_equal(loaded, expected)
                                assert (ref.mass, ref.charge, ref.pt, ref.gyromagnetic_anomaly) == (
                                    stored_ref if restore else expected_ref
                                )
                                # w counts physical particles: electrical charge is e*sum(w).
                                charge = float(loaded[:, -1].sum() * ref.charge)
                                assert abs(charge - weights.sum() * E_CHARGE) <= 3e-33
                        print(
                            f"{name}: restore=False/True observed={','.join(statuses)} "
                            f"expected=LOADED,LOADED count={count} charge={weights.sum() * E_CHARGE:.12g} C; "
                            "expected from saved arrays and e*sum(w), successful arrays exact"
                        )
                        if count == 0:
                            pc.clear_particles()
                            one, zero = np.ones(1), np.zeros(1)
                            pc.add_n_particles(
                                one * 0.007,
                                zero,
                                zero,
                                zero,
                                zero,
                                zero,
                                qm=ref.qm_ratio_SI,
                                w=one * 11,
                            )
                            before = arrays(pc)
                            status = load(pc, path, load_ref_particle=False)
                            assert np.array_equal(arrays(pc), before)
                            appends += status == "NULL_BUFFER_ERROR"
                            print(
                                f"{name} append: observed={status} expected=LOADED; "
                                "empty injection is identity, weight-11 sentinel unchanged"
                            )
            assert reads == 36
            print(
                f"empty reads: observed_null_errors={failures}/12 expected=0; "
                f"empty appends: observed_null_errors={appends}/6 expected=0; "
                "24 nonempty controls exact, HDF5 extents independently inspected"
            )
            print("DEFECT REPRODUCED" if failures or appends else "NOT REPRODUCED")
        finally:
            sim.finalize()


    def main():
        with TemporaryDirectory(prefix="IO-06-", dir=HERE) as directory:
            run(Path(directory))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD/HDF5 |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU NOACC, double, serial, SIMD 4, FFT, openPMD/HDF5/JSON/TOML |

