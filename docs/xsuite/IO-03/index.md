---
title: "IO-03: Checkpoint loading restores RNG words onto reordered particle coordinates"
hide:
  - toc
---

# IO-03 — Checkpoint loading restores RNG words onto reordered particle coordinates

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack main (`3cd8bfc`) |
| Source | [`Particles.from_dict`, particles.py:485–498](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/particles.py#L485-L498) |

On CPU, loading a checkpoint with `Particles.from_dict(load_rng_state=True)` and default options reorders the coordinates into active-then-lost order but restores the saved random-number-generator (RNG) words in their original slot order. In checkpoints with lost rows before live rows, some live particles therefore continue another particle's saved random stream, while the persistence contract and an uninterrupted copy require each particle to keep its own. With seeds 11, 22, 33 and 44 and particle 1 lost, live particle 2 draws 0.1720 instead of 0.6269, and particle 3 draws 0.6269 instead of 0.1897. Coordinates keep their particle IDs, but restarted stochastic trajectories change; all-active checkpoints and loading with `_no_reorganize=True` preserve the associations.

??? example "Reproducer: IO-03.py"

    [Download IO-03.py](IO-03.py)

    ```python
    """IO-03: Checkpoint loading restores RNG words onto reordered particle coordinates

    On CPU, loading a checkpoint with `Particles.from_dict(load_rng_state=True)` and default options
    reorders the coordinates into active-then-lost order but restores the saved random-number-generator
    (RNG) words in their original slot order. In checkpoints with lost rows before live rows, some live
    particles therefore continue another particle's saved random stream, while the persistence contract
    and an uninterrupted copy require each particle to keep its own. With seeds 11, 22, 33 and 44 and
    particle 1 lost, live particle 2 draws 0.1720 instead of 0.6269, and particle 3 draws 0.6269 instead
    of 0.1897. Coordinates keep their particle IDs, but restarted stochastic trajectories change;
    all-active checkpoints and loading with `_no_reorganize=True` preserve the associations.

    The script gives four particles with IDs 0–3 the explicit RNG seeds 11, 22, 33 and 44, saves them to
    JSON with `to_dict(keep_rng_state=True)` and loads them with `from_dict(load_rng_state=True)`. It
    compares the restored RNG words and the next `RandomUniform` value of each particle ID exactly with
    the saved record and with an uninterrupted copy. An all-active round trip is the control; with
    particle 1 lost, loading reorders the coordinates to [0, 2, 3, 1] but leaves the RNG words in [0, 1,
    2, 3] order.
    """

    import json

    import numpy as np
    import xobjects as xo
    import xtrack as xt

    RTOL, ATOL = 0, 0
    SEEDS = np.array([11, 22, 33, 44], dtype=np.uint32)


    def evaluate():
        particles = xt.Particles(p0c=1e9, x=[0.01, 0.02, 0.03, 0.04])
        particles._init_random_number_generator(seeds=SEEDS)
        rng = xt.RandomUniform()
        names = [f"_rng_s{index}" for index in range(1, 5)]
        cases = {}
        for interleaved_loss in (False, True):
            particles.state[:] = [1, 0 if interleaved_loss else 1, 1, 1]
            source_words = np.array([getattr(particles, name).copy() for name in names]).T
            # Actual JSON serialization exercises the documented persistence format.
            saved = json.loads(json.dumps(particles.to_dict(keep_rng_state=True), cls=xo.JEncoder))
            np.testing.assert_array_equal(saved["particle_id"], particles.particle_id)
            np.testing.assert_array_equal(np.array([saved[name] for name in names]).T, source_words)
            control = particles.copy()
            expected_samples = rng.generate(n_samples=4, particles=control).ravel()
            restored = xt.Particles.from_dict(saved, load_rng_state=True)
            np.testing.assert_array_equal(restored.x, 0.01 * (restored.particle_id + 1))
            restored_words = np.array([getattr(restored, name).copy() for name in names]).T
            expected_words = source_words[restored.particle_id]
            actual_samples = rng.generate(n_samples=4, particles=restored).ravel()
            if not interleaved_loss:
                np.testing.assert_array_equal(restored_words, expected_words)
                np.testing.assert_array_equal(actual_samples, expected_samples)
            else:
                cases["checkpoint RNG words after coordinate reordering"] = (
                    restored_words,
                    expected_words,
                )
                cases["next random value by particle ID after checkpoint restore"] = (
                    actual_samples,
                    expected_samples,
                )
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare(random_seed=SEEDS.tolist())
        print(
            "Expected: JSON save/load preserves RNG state and stochastic continuation per particle ID, including interleaved losses."
        )
        print(
            "An all-active round trip and the saved JSON contents are checked before the loss-layout case."
        )
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

??? example "Helper: _common.py"

    [Download _common.py](_common.py)

    ```python
    """Shared setup and output for the defect reproducers.

    `prepare()` moves to a fresh temporary working directory, where CFFI compiles the kernels, and
    prints the package versions and context; `report()` prints the observed and expected rows and the
    verdict line.
    """

    import atexit
    import importlib
    import importlib.metadata
    import os
    import platform
    import shutil
    import tempfile
    from pathlib import Path

    import numpy as np
    import xobjects as xo


    def prepare(*, radiation="off", random_seed=None):
        # CFFI compiles into cwd; keep generated files outside the source trees.
        scratch = Path(tempfile.mkdtemp(prefix="xsuite-repro-"))
        atexit.register(shutil.rmtree, scratch, ignore_errors=True)
        os.chdir(scratch)
        xo.settings.allow_kernel_compilation = True
        print(f"Python {platform.python_version()}; {platform.platform()}")
        for name in ("xsuite", "xtrack", "xobjects", "xdeps", "xpart", "xfields", "xcoll"):
            module = importlib.import_module(name)
            print(f"{name} {module.__version__}: {module.__file__}")
        for name in ("numpy", "scipy", "madng-tpsa", "cffi"):
            print(f"{name} {importlib.metadata.version(name)}")
        randomness = (
            "deterministic, no random seed"
            if random_seed is None
            else f"fixed RNG seed(s)={random_seed}"
        )
        print(f"ContextCpu, serial; radiation={radiation}; {randomness}")
        print(
            f"allow_kernel_compilation={xo.settings.allow_kernel_compilation}; "
            f"force_kernel_compilation={xo.settings.force_kernel_compilation}"
        )


    def report(cases, *, rtol, atol):
        np.set_printoptions(precision=15, linewidth=140)
        mismatch = False
        for label, (observed, expected) in cases.items():
            print(f"{label}: observed={np.asarray(observed)} expected={np.asarray(expected)}")
            mismatch |= not np.allclose(observed, expected, rtol=rtol, atol=atol)
        print(f"comparison: rtol={rtol:g}, atol={atol:g}")
        print("DEFECT REPRODUCED" if mismatch else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

