---
title: "IO-04: Thermal negative-charge initialization returns NaN positions"
hide:
  - toc
---

# IO-04 — `Thermal` negative-charge initialization returns NaN positions

| | |
|---|---|
| Code | [ImpactX](../index.md) · Particles and I/O |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Source | [`Thermal.H#L99-L108`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/particles/distribution/Thermal.H#L99-L108)<br>[`matched_scale_radius` at L162-L172](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/particles/distribution/Thermal.H#L162-L172)<br>[caller](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/initialization/InitDistribution.cpp#L310-L312)<br>[L354-L356](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/initialization/InitDistribution.cpp#L354-L356)<br>[sampler contract](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/particles/distribution/Thermal.H#L277-L290) |

Initializing a `Thermal` distribution for a negatively charged species, such as electrons or H⁻, with positive focusing and nonzero bunch charge returns NaN for every particle position, while the momenta are finite and the call returns normally. The radius calculation raises a self-field coefficient computed with the signed species charge to a fractional power. Reversing the charge sign leaves the repulsive self-force unchanged, so the distribution must equal that of the opposite charge; an independent Gauss-law integration gives that finite solution. In the tested electron and H⁻ cases all 2049 positions are NaN; at zero bunch charge both signs give identical results.

## Reproducer

Requires scipy.

Fresh serial child per sample, with seed 1 and one simulation per process. Temporary arrays beside this script support paired checks and are removed on exit. The independent Poisson-Boltzmann ODE checks existence and charge symmetry.

??? example "IO-04.py"

    [Download IO-04.py](IO-04.py)

    ```python
    """IO-04: Thermal negative-charge initialization returns NaN positions.

    Fresh serial child per sample, with seed 1 and one simulation per process.
    Temporary arrays beside this script support paired checks and are removed on exit.
    The independent Poisson-Boltzmann ODE checks existence and charge symmetry.
    """

    import os

    for _name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[_name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

    import importlib.metadata
    import json
    import math
    import subprocess
    import sys
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import numpy as np
    from scipy.integrate import solve_ivp

    HERE = Path(__file__).resolve().parent
    COLS = [
        "position_x",
        "position_y",
        "position_t",
        "momentum_x",
        "momentum_y",
        "momentum_t",
    ]


    def configure(sim):
        from impactx import Config

        config = Config.to_dict()
        assert not any(config[k] for k in ("have_mpi", "have_gpu", "have_omp"))
        assert config["precision"] == config["precision_particles"] == "DOUBLE"
        sim.tiny_profiler = False
        sim.memory_profiler = False
        sim.particle_shape = 1
        sim.space_charge = "off"
        sim.csr = False
        sim.isr = False
        sim.spin = False
        sim.diagnostics = False
        sim.slice_step_diagnostics = False
        sim.verbose = 0
        sim.init_grids()


    def environment():
        from amrex import space3d as amr
        from impactx import Config

        print(
            f"Python {sys.version.split()[0]} numpy={np.__version__} "
            f"scipy={importlib.metadata.version('scipy')} "
            f"impactx-noacc={importlib.metadata.version('impactx-noacc')} "
            f"pyamrex={amr.__version__}",
            flush=True,
        )
        print("build", Config.to_dict(), flush=True)


    def thermal_case(sim, spec, output):
        from impactx import distribution

        ref = sim.beam.ref
        if "species" in spec:
            ref.set_species(spec["species"])
        else:
            ref.set_mass_MeV(938.27208816).set_charge_qe(spec["z"])
        ref.set_kin_energy_MeV(spec.get("energy", ref.mass_MeV * (0.1 / 938.27208816)))
        k = spec.get("k", 2 * math.pi)
        temp = spec.get("temperature", 36e-6)
        charge = spec.get("charge", 1.4285714285714286e-10)
        if spec.get("scale_mass"):
            charge *= ref.mass_MeV / 938.27208816
        n = spec.get("n", 2049)
        distr = distribution.Thermal(
            k=k,
            kT=temp,
            kT_halo=temp,
            normalize=spec.get("norm", 0.41604661),
            normalize_halo=spec.get("norm", 0.41604661),
            halo=spec.get("halo", 0.0),
        )
        sim.add_particles(bunch_charge=charge, distr=distr, npart=n)
        data = sim.beam.to_df(local=True).sort_index()[COLS].to_numpy()
        bg = ref.beta_gamma
        # Force per rest-energy: q * rho / (m c^2); Q is a magnitude.
        epsilon0 = 8.8541878188e-12
        signed_c = ref.charge_qe * charge / (bg * bg * ref.mass_MeV * 1e6 * epsilon0)
        physical_c = abs(signed_c)
        radius = math.sqrt(temp + (physical_c * k / (20 * math.pi * math.sqrt(5))) ** (2 / 3)) / k
        np.savez(output / f"{spec['name']}.npz", data=data)
        finite = np.isfinite(data)
        return {
            **spec,
            "n": n,
            "mass": ref.mass_MeV,
            "z": ref.charge_qe,
            "bunch_charge": charge,
            "gamma": -ref.pt,
            "beta_gamma": bg,
            "signed_C": signed_c,
            "physical_C": physical_c,
            "physical_radius_scale": radius,
            "finite_rows": int(np.all(finite, axis=1).sum()),
            "finite_components": finite.sum(axis=0).tolist(),
            "nan_components": np.isnan(data).sum(axis=0).tolist(),
            "all_finite": bool(finite.all()),
            "r_rms": float(np.sqrt(np.mean(data[:, 0] ** 2 + data[:, 1] ** 2 + (bg * data[:, 2]) ** 2)))
            if finite[:, :3].all()
            else None,
            "p_rms": np.sqrt(np.mean(data[:, 3:] ** 2, axis=0)).tolist(),
        }


    def cases():
        specs = []
        for species in ("proton", "Hminus", "positron", "electron"):
            specs.append(dict(name=f"species_{species}", species=species, scale_mass=True))
        for z in (-2, -1, 1, 2):
            specs.append(dict(name=f"charge_Z{z}", z=z, charge=1.4285714285714286e-10 / abs(z)))
        for z in (-1, 1):
            for charge in (0.0, 1e-20):
                specs.append(dict(name=f"intensity_Z{z}_Q{charge:g}", z=z, charge=charge))
            specs.append(
                dict(
                    name=f"params_Z{z}",
                    z=z,
                    k=3.0,
                    temperature=10e-6,
                    norm=0.7,
                    halo=0.3,
                    n=513,
                    charge=1e-12,
                )
            )
        return specs


    def radial_oracle(case, rtol):
        """Integrate probability enclosed F and potential psi=rscale*phi.

        Spherical Gauss law gives phi'=-F/(4*pi*r^2). Boltzmann density
        supplies F'=4*pi*r^2*A*exp[-(k^2*r^2/2+C*phi)/T]. C is positive
        for any single-species beam since force density contains q*rho.
        The dimensionless radius s=r/rscale keeps these equations well-scaled.
        """
        temp = case.get("temperature", 36e-6)
        k = case.get("k", 2 * math.pi)
        scale = case["physical_radius_scale"]
        c = case["physical_C"]
        prefactor = case.get("norm", 0.41604661) / (2 * math.pi) ** 1.5
        a = (k * scale) ** 2 / (2 * temp)
        b = c / (scale * temp)

        def ode(s, y):
            f, psi, _ = y
            density = prefactor * np.exp(-a * s * s - b * psi)
            df = 4 * math.pi * s * s * density
            return [df, -f / (4 * math.pi * s * s), s * s * df]

        start = 1e-8
        sol = solve_ivp(
            ode,
            (start, 10.0),
            [4 * math.pi * prefactor * start**3 / 3, -prefactor * start**2 / 6, 0.0],
            method="DOP853",
            rtol=rtol,
            atol=rtol * 1e-2,
            dense_output=True,
        )
        assert sol.success
        grid = np.linspace(start, 10, 10001)
        values = sol.sol(grid)
        assert np.all(np.isfinite(values))
        assert np.min(np.diff(values[0])) > -1e-10
        return {
            "name": case["name"],
            "rtol": rtol,
            "C": c,
            "F_endpoint": float(values[0, -1]),
            "r_rms": scale * math.sqrt(values[2, -1] / values[0, -1]),
        }


    def main():
        from impactx import ImpactX

        if len(sys.argv) > 1:
            if len(sys.argv) != 4 or sys.argv[1] != "child":
                raise ValueError("Expected child <JSON case> <output directory>")
            spec, output = json.loads(sys.argv[2]), Path(sys.argv[3])
            sim = ImpactX()
            try:
                configure(sim)
                result = thermal_case(sim, spec, output)
            finally:
                sim.finalize()
            (output / f"{spec['name']}.json").write_text(json.dumps(result, allow_nan=False))
            print(json.dumps(result, allow_nan=False), flush=True)
            return

        environment()
        print(
            "Oracle: charge reversal preserves q*rho; C=abs(Z)*Q/(bg^2*Erest[eV]*epsilon0). "
            "Every supported sample should have finite positions and momenta.",
            flush=True,
        )
        print("AMReX serial default seed 1 in each fresh child process.", flush=True)
        with TemporaryDirectory(prefix="io-04-", dir=HERE) as directory:
            output = Path(directory)
            results = {}
            for spec in cases():
                child = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "child",
                        json.dumps(spec),
                        directory,
                    ],
                    cwd=output,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=290,
                    check=False,
                )
                print(child.stdout.decode(), end="", flush=True)
                child.check_returncode()
                row = json.loads((output / f"{spec['name']}.json").read_text())
                results[row["name"]] = row
                expected_finite = row["z"] > 0 or row["bunch_charge"] == 0
                assert row["all_finite"] == expected_finite
                assert row["finite_components"][3:] == [row["n"]] * 3
                if not expected_finite:
                    assert row["nan_components"][:3] == [row["n"]] * 3
                print(
                    f"{row['name']}: expected_finite_positions={row['n']} "
                    f"actual={row['finite_components'][0]} finite_momenta={row['finite_components'][3]}",
                    flush=True,
                )

            for left, right in (
                ("charge_Z1", "charge_Z2"),
                ("species_positron", "species_electron"),
                ("charge_Z1", "charge_Z-1"),
                ("intensity_Z1_Q0", "intensity_Z-1_Q0"),
            ):
                with np.load(output / f"{left}.npz") as data:
                    a = data["data"].copy()
                with np.load(output / f"{right}.npz") as data:
                    b = data["data"].copy()
                assert np.array_equal(a[:, 3:], b[:, 3:])
                if left in ("charge_Z1", "intensity_Z1_Q0") and right != "charge_Z-1":
                    assert np.array_equal(a, b)
                print(
                    f"PAIRED {left} / {right}: exact_momenta={np.array_equal(a[:, 3:], b[:, 3:])} "
                    f"exact_all={np.array_equal(a, b)}",
                    flush=True,
                )

            for left, right in (
                ("charge_Z1", "charge_Z-1"),
                ("params_Z1", "params_Z-1"),
                ("intensity_Z1_Q0", "intensity_Z-1_Q0"),
                ("intensity_Z1_Q1e-20", "intensity_Z-1_Q1e-20"),
            ):
                pair = []
                for name in (left, right):
                    coarse = radial_oracle(results[name], 1e-12)
                    fine = radial_oracle(results[name], 3e-14)
                    assert abs(coarse["r_rms"] - fine["r_rms"]) < 2e-11
                    for row in (coarse, fine):
                        print("POISSON_BOLTZMANN", json.dumps(row, allow_nan=False), flush=True)
                    pair.append(fine)
                assert pair[0]["r_rms"] == pair[1]["r_rms"]
            print(
                f"Checked {len(results)} Thermal cases and charge-symmetric continuum oracles.",
                flush=True,
            )
        print("DEFECT REPRODUCED", flush=True)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

