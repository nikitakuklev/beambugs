---
title: "COL-05: The 3D PIC beta interface loses the supplied Lorentz factor"
hide:
  - toc
---

# COL-05 — The 3D PIC beta interface loses the supplied Lorentz factor

| | |
|---|---|
| Code | [ImpactX](../index.md) · Collective effects |
| Priority | P5 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Tags | <span class="bb-tag">extreme-regime</span> [?](../../methodology.md#tags) |
| Discussed in | [[gh:https://github.com/BLAST-WarpX/warpx/pull/7178]]<br>[[gh:https://github.com/BLAST-ImpactX/impactx/pull/1621]] |
| Source | [`PoissonSolve.cpp#L52-L61`](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/particles/spacecharge/PoissonSolve.cpp#L52-L61)<br>[`PoissonSolver.H#L270-L284`](https://github.com/BLAST-WarpX/warpx/blob/ec1660d961696c865c87c26728b7e248b5454082/Source/ablastr/fields/PoissonSolver.H#L270-L284)<br>[integrated Green function](https://github.com/BLAST-WarpX/warpx/blob/ec1660d961696c865c87c26728b7e248b5454082/Source/ablastr/fields/IntegratedGreenFunctionSolver.H#L83-L96)<br>[kick normalization](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/src/particles/spacecharge/GatherAndPush.cpp#L98-L119)<br>[3D model contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L2456-L2460)<br>[equation](https://github.com/BLAST-WarpX/warpx/blob/ec1660d961696c865c87c26728b7e248b5454082/Source/ablastr/fields/PoissonSolver.H#L157-L172) |

For 3D FFT particle-in-cell space charge, ImpactX passes only the rounded velocity `β` to the WarpX/ABLASTR Poisson solver, which reconstructs γ from it, while ImpactX's own coordinate transforms keep the original γ. Lorentz similarity requires the kick scaled by `(βγ)²` to stay unchanged for a fixed rest-frame source and mesh. At γ = 6e7 (30.66 TeV electrons) the solver uses γ = 4.745e7 and the rescaled transverse kick is 9.642% low, on a normalized kick of only 1.735e-20. At γ = 1e4 the error is 2.716e-9.

## Reproducer

Requires scipy.

Uniform-sphere Lorentz similarity, exact continuum interior field, and matched low-energy squeezed-source controls isolate the beta reconstruction loss. Fresh serial CPU/double child per native case, timeout 290 s. Requires Python 3.13 math.fma; writes no files.

??? example "COL-05.py"

    [Download COL-05.py](COL-05.py)

    ```python
    """COL-05: the 3D PIC beta interface loses the supplied Lorentz factor.

    Uniform-sphere Lorentz similarity, exact continuum interior field, and matched
    low-energy squeezed-source controls isolate the beta reconstruction loss.
    Fresh serial CPU/double child per native case, timeout 290 s. Requires Python
    3.13 math.fma; writes no files.
    """

    import os

    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[variable] = "1"
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

    import importlib.metadata
    import json
    import math
    import subprocess
    import sys
    from pathlib import Path

    import amrex.space3d as amr
    import impactx
    import numpy as np
    from impactx import Config, ImpactX, RefPart, elements
    from scipy.constants import c, epsilon_0
    from scipy.integrate import quad


    def simulation(**options):
        print(
            f"Python {sys.version.split()[0]} numpy={np.__version__} scipy={importlib.metadata.version('scipy')}"
        )
        config = Config.to_dict()
        assert all(not config[k] for k in ("have_mpi", "have_omp", "have_gpu"))
        assert config["precision"] == config["precision_particles"] == "DOUBLE"
        print(
            json.dumps(
                {
                    "impactx": impactx.__version__,
                    "impactx_file": impactx.__file__,
                    "impactx-noacc": importlib.metadata.version("impactx-noacc"),
                    "pyamrex": amr.__version__,
                    "config": config,
                }
            )
        )
        sim = ImpactX()
        settings = dict(
            particle_shape=1,
            diagnostics=False,
            slice_step_diagnostics=False,
            tiny_profiler=False,
            memory_profiler=False,
            csr=False,
            isr=False,
            spin=False,
            strang_split=False,
            verbose=0,
        )
        settings.update(options)
        for key, value in settings.items():
            setattr(sim, key, value)
        try:
            sim.init_grids()
        except BaseException:
            sim.finalize()
            raise
        return sim


    def insert(sim, ref, positions, weights):
        phase = np.zeros((len(positions), 6))
        phase[:, :3] = positions
        sim.beam.set_ref_particle(ref)
        sim.beam.add_n_particles(*phase.T.copy(), qm=ref.qm_ratio_SI, w=weights)


    def momenta(sim, count):
        frame = sim.beam.to_df(local=True).reset_index(drop=True).sort_values("idcpu")
        return frame[["momentum_x", "momentum_y", "momentum_t"]].to_numpy()[-count:]


    def case(cells, gamma, squeeze=1.0, charge_factor=1.0):
        ref = RefPart().set_species("electron")
        ref.set_kin_energy_MeV((gamma - 1) * ref.mass_MeV)
        gamma, bg = ref.gamma, ref.beta_gamma
        radius, charge, length = 0.0013, 7e-13 * charge_factor, 0.019
        axis = (np.arange(cells) + 0.5) * 2 / cells - 1
        xyz = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        xyz = xyz[(xyz**2).sum(axis=1) < 1] * radius
        # All tracers are in the transverse central plane; the longitudinal kick is zero
        # by reflection symmetry. The uniform sphere has an exact linear interior field.
        probes = radius * np.array(
            [[0.25, 0, 0], [0.5, 0, 0], [1.25, 0, 0], [-0.5, 0, 0], [0, 0.5, 0], [0.3, 0.4, 0]]
        )
        positions = np.vstack([xyz, probes])
        positions[:, 2] *= -squeeze / bg
        weights = np.r_[np.full(len(xyz), charge / abs(ref.charge) / len(xyz)), np.zeros(len(probes))]
        sim = simulation(space_charge="3D", poisson_solver="fft", n_cell=[cells] * 3, max_level=0)
        try:
            insert(sim, ref, positions, weights)
            half = 2 * radius
            sim.domain = amr.RealBox(
                [-half, -half, -half * squeeze / gamma], [half, half, half * squeeze / gamma]
            )
            sim.lattice.append(elements.Drift(ds=length))
            sim.track_particles()
            assert sim.beam.ref.gamma == gamma
            observed = momenta(sim, len(probes))[:, :2]
            assert np.isfinite(observed).all()
            scaled = observed * bg**2
            # Exact continuum field of a uniform sphere, including exterior probes.
            r = np.linalg.norm(probes, axis=1)
            field = (
                charge / (4 * np.pi * epsilon_0) * probes[:, :2] / np.maximum(radius, r)[:, None] ** 3
            )
            spherical_oracle = abs(ref.charge) * length / (ref.mass * c * c) * field
            beta_solver = math.sqrt(1 - 1 / gamma**2)
            # This wheel contracts 1-beta*beta. The gamma=1e4 paired surrogate
            # distinguishes fused and separately rounded arithmetic experimentally.
            reconstructed = 1 / math.sqrt(math.fma(-beta_solver, beta_solver, 1.0))
            result = dict(
                settings=dict(cells=cells, gamma=gamma, squeeze=squeeze, charge_factor=charge_factor),
                gamma=gamma,
                bg=bg,
                beta_solver=beta_solver,
                gamma_solver=reconstructed,
                gamma_solver_separate=1 / math.sqrt(1 - beta_solver**2),
                a=reconstructed / gamma,
                particles=len(positions),
                observed=observed.tolist(),
                scaled=scaled.tolist(),
                sphere_oracle=spherical_oracle.tolist(),
            )
            print("RESULT " + json.dumps(result), flush=True)
        finally:
            sim.finalize()


    def run(cells, gamma, squeeze=1.0, charge_factor=1.0):
        spec = dict(cells=cells, gamma=gamma, squeeze=squeeze, charge_factor=charge_factor)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--case", json.dumps(spec)],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=290,
            check=False,
        )
        print(completed.stdout, end="", flush=True)
        completed.check_returncode()
        rows = [
            line.removeprefix("RESULT ")
            for line in completed.stdout.splitlines()
            if line.startswith("RESULT ")
        ]
        assert len(rows) == 1
        return json.loads(rows[0])


    def main():
        reproduced = True
        for cells, gamma in ((32, 6e7), (48, 6e7), (64, 6e7), (32, 2e7), (32, 4e7), (32, 1e4)):
            low, high = run(cells, 7.0), run(cells, gamma)
            # Match both the effective stretched geometry and source charge. Include
            # the low-energy reconstruction error when constructing the surrogate.
            squeeze = high["a"] / low["a"]
            surrogate = run(cells, 7.0, squeeze, squeeze)
            base, elevated, mimicked = (np.asarray(r["scaled"]) for r in (low, high, surrogate))
            valid = np.abs(base) > np.max(np.abs(base)) * 1e-10
            residual = float(np.max(np.abs((elevated[valid] - mimicked[valid]) / base[valid])))
            assert residual < 5e-13
            # Lorentz similarity requires (beta*gamma)^2*delta_px to stay constant.
            # A sphere squeezed to aR with charge aQ gives an ellipsoid whose
            # interior transverse field ratio is 3*Ax, Ax=a/2*integral below.
            a = high["a"]
            integral, error = quad(
                lambda u, a=a: 1 / ((1 + u) ** 2 * math.sqrt(a * a + u)),
                0,
                np.inf,
                epsabs=2e-13,
                epsrel=2e-13,
            )
            assert error < 1e-12
            wrong_limit = 1.5 * a * integral
            ratio = elevated[1, 0] / base[1, 0]
            oracle = np.asarray(low["sphere_oracle"])
            print(
                f"cells={cells} gamma={gamma:g} sources={low['particles'] - 6} "
                f"gamma_solver={high['gamma_solver']:.16g} stretch_ratio={a:.16g} "
                f"low_scaled={base[1, 0]:.16g} high_scaled={elevated[1, 0]:.16g} "
                f"ratio={ratio:.16g} required_ratio=1 wrong_ellipsoid_limit={wrong_limit:.16g} "
                f"surrogate_residual={residual:.12g} "
                f"low_continuum_error={base[1, 0] / oracle[1, 0] - 1:.12g} "
                f"raw_high={high['observed'][1][0]:.16g} "
                f"same_mesh_expected={base[1, 0] / high['bg'] ** 2:.16g}",
                flush=True,
            )
            if gamma == 6e7:
                reproduced &= abs(ratio - 1) > 0.08
                assert abs(ratio - wrong_limit) < 4e-4
            if gamma == 1e4:
                separate_squeeze = (high["gamma_solver_separate"] / high["gamma"]) / (
                    low["gamma_solver_separate"] / low["gamma"]
                )
                separate = np.asarray(run(cells, 7.0, separate_squeeze, separate_squeeze)["scaled"])
                separate_residual = float(
                    np.max(np.abs((elevated[valid] - separate[valid]) / base[valid]))
                )
                print(
                    f"gamma10000_fused_residual={residual:.12g} separate_residual={separate_residual:.12g}",
                    flush=True,
                )
                assert separate_residual > 1e-10
        print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")
        if not reproduced:
            raise AssertionError("Expected the beta-interface similarity failure")


    if __name__ == "__main__":
        if len(sys.argv) == 1:
            main()
        elif len(sys.argv) == 3 and sys.argv[1] == "--case":
            case(**json.loads(sys.argv[2]))
        else:
            raise ValueError("Run without arguments, or with --case JSON for a native child")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

