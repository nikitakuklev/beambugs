---
title: "TRK-18: Low-energy ExactDrift corrupts co-moving arrival times"
hide:
  - toc
---

# TRK-18 — Low-energy `ExactDrift` corrupts co-moving arrival times

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Issue | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1576]] |
| Source | [`ExactDrift.H#L95-L96`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/ExactDrift.H#L95-L96)<br>[particle map](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/ExactDrift.H#L140-L153)<br>[`ReferenceParticle.H#L169-L185`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/particles/ReferenceParticle.H#L169-L185)<br>[public model](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/python.rst#L1999-L2001)<br>[coordinate contract](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/docs/source/theory/coordinates_units.rst#L5-L13) |

At low energy, `ExactDrift` shifts the arrival time `t` of particles at the reference energy and direction, because its longitudinal-momentum radical subtracts the separately rounded terms `(1/β−pt)²` and `1/(βγ)²`. A 90-digit evaluation shows that these particles keep their relative arrival times. For a proton (rest energy 938.27208816 MeV) at 1 keV kinetic energy, a 1 m drift adds 0.002379 m (7.9 ps) to `t` and shifts the bunch centroid; off-energy and inclined particles get timing errors up to 0.002454 m and transverse errors up to 1.013e-7 m. At 1000 MeV the errors are below 1e-16 m; at low energy their sign and size depend on floating-point rounding.

## Reproducer

One serial, double-precision simulation; deterministic particles; no output files. The explicit rest mass is part of the fixture because stored gamma controls rounding.

??? example "TRK-18.py"

    [Download TRK-18.py](TRK-18.py)

    ```python
    """TRK-18: Low-energy ExactDrift corrupts co-moving arrival times.

    One serial, double-precision simulation; deterministic particles; no output files.
    The explicit rest mass is part of the fixture because stored gamma controls rounding.
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
    import math
    import sys
    from decimal import Decimal, localcontext

    import numpy as np

    COLS = [
        "position_x",
        "position_y",
        "position_t",
        "momentum_x",
        "momentum_y",
        "momentum_t",
    ]


    def expected_drift(phase, gamma, length):
        """Rationalized velocity difference; exact zero invariant stays exact.

        d = P_s/P0, A = gamma_particle/(beta0*gamma0).
        Delta t/L = (A - d/beta0)/d. Rationalize the numerator to
        avoid the difference of two large inverse velocities at low energy.
        """
        result = phase.copy()
        with localcontext() as ctx:
            ctx.prec = 90
            g = Decimal.from_float(gamma)
            bg2 = (g - 1) * (g + 1)
            beta = bg2.sqrt() / g
            dl = Decimal.from_float(length)
            for row, out in zip(phase, result, strict=True):
                x, y, t, px, py, u = map(Decimal.from_float, row)
                pxy2 = px * px + py * py
                radical = 1 - 2 * u / beta + u * u - pxy2
                assert radical > 0 and g - bg2.sqrt() * u > 1
                d = radical.sqrt()
                a = 1 / beta - u
                numerator = (2 * u / beta - u * u) / bg2 + pxy2 / (beta * beta)
                dt = dl * numerator / (d * (a + d / beta))
                out[:3] = [float(x + dl * px / d), float(y + dl * py / d), float(t + dt)]
        return result


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


    def drift_case(sim, spec):
        from impactx import elements

        sim.beam.clear_particles()
        sim.lattice.clear()
        ref = sim.beam.ref
        ref.reset()
        if "species" in spec:
            ref.set_species(spec["species"])
        else:
            ref.set_mass_MeV(spec.get("mass", 938.27208816)).set_charge_qe(1)
        ref.set_kin_energy_MeV(spec["energy"])
        gamma = -ref.pt
        n = spec.get("n", 9)
        phase = np.zeros((n, 6))
        phase[:, 0] = np.arange(n) * 1e-5
        phase[:, 1] = np.arange(n) * -2e-5
        phase[:, 2] = np.arange(n) * 1e-4
        stable_beta = math.sqrt((gamma - 1) * (gamma + 1)) / gamma
        if spec.get("offsets"):
            phase[:, 3] = np.linspace(-0.03, 0.02, n)
            phase[:, 4] = np.linspace(0.005, -0.009, n)
            phase[:, 5] = stable_beta * np.linspace(-0.01, 0.01, n)
        sim.beam.add_n_particles(*[col.copy() for col in phase.T], qm=ref.qm_ratio_SI, w=np.ones(n))
        before = sim.beam.to_df(local=True).sort_index()[COLS].to_numpy()
        assert np.array_equal(before, phase)
        length = spec.get("length", 1.0)
        slices = spec.get("slices", 1)
        element = getattr(elements, spec.get("element", "ExactDrift"))(ds=length, nslice=slices)
        ref_before = {key: getattr(ref, key) for key in ("pt", "pz", "t", "s")}
        if spec.get("direct"):
            for _ in range(slices):
                element.push(sim.beam)
        else:
            sim.lattice.append(element)
            sim.track_particles()
        observed = sim.beam.to_df(local=True).sort_index()[COLS].to_numpy()
        expected = expected_drift(phase, gamma, length)
        assert np.isfinite(observed).all()
        assert ref.pt == ref_before["pt"]
        error = observed - expected
        return {
            **spec,
            "mass_MeV": ref.mass_MeV,
            "gamma": gamma,
            "beta": ref.beta,
            "beta_gamma": ref.beta_gamma,
            "dt": (observed[:, 2] - before[:, 2]).tolist(),
            "t_errors": error[:, 2].tolist(),
            "max_t_error": float(abs(error[:, 2]).max()),
            "max_xy_error": float(abs(error[:, :2]).max()),
            "max_momentum_error": float(abs(error[:, 3:]).max()),
            "ref_before": ref_before,
            "ref_after": {key: getattr(ref, key) for key in ("pt", "pz", "t", "s")},
            "oracle_first": expected[0].tolist(),
            "observed_first": observed[0].tolist(),
        }


    def cases():
        specs = []
        for energy in (1e-6, 0.001, 0.004, 0.01, 0.1, 1.0, 1000.0):
            specs.append(dict(name=f"zero_E{energy:g}", energy=energy))
        for energy in (0.001, 0.1):
            specs.append(dict(name=f"builtin_E{energy:g}", energy=energy, species="proton"))
        for energy in (0.001 - 2e-13, 0.001 + 2e-13):
            specs.append(dict(name=f"neighbor_E{energy:.13g}", energy=energy))
        for length in (0.125, 3.7, -0.25, 0.0):
            specs.append(dict(name=f"zero_L{length:g}", energy=0.001, length=length))
        for n in (1, 4, 5):
            specs.append(dict(name=f"zero_n{n}", energy=0.001, n=n))
        for slices in (7, 128):
            specs.append(dict(name=f"zero_s{slices}", energy=0.001, slices=slices))
        specs.append(dict(name="zero_direct", energy=0.001, direct=True))
        specs.append(dict(name="zero_mass", energy=0.001 * 0.51099895 / 938.27208816, mass=0.51099895))
        for element in ("Drift", "ChrDrift"):
            specs.append(dict(name=f"zero_{element}", energy=0.001, element=element))
        for energy in (0.001, 0.004, 1000.0):
            for slices in (1, 17):
                specs.append(
                    dict(
                        name=f"spread_E{energy:g}_s{slices}",
                        energy=energy,
                        slices=slices,
                        offsets=True,
                    )
                )
        return specs


    def main():
        from impactx import ImpactX

        environment()
        print(
            "Oracle: 90-digit relativistic velocity with rationalized arrival-time difference; "
            "px=py=pt=0 implies Delta t=0 exactly.",
            flush=True,
        )
        results = {}
        sim = ImpactX()
        try:
            configure(sim)
            for spec in cases():
                row = drift_case(sim, spec)
                results[row["name"]] = row
                print(
                    f"{row['name']}: t0_expected={row['oracle_first'][2]:.17g} "
                    f"t0_observed={row['observed_first'][2]:.17g} "
                    f"max_t_error={row['max_t_error']:.17g} "
                    f"max_xy_error={row['max_xy_error']:.17g} "
                    f"max_momentum_error={row['max_momentum_error']:.17g}",
                    flush=True,
                )
        finally:
            sim.finalize()
        base = results["zero_E0.001"]["dt"][0]
        assert 0.002 < base < 0.003
        assert results["zero_Drift"]["max_t_error"] == 0
        assert results["zero_E1000"]["max_t_error"] == 0
        assert results["zero_L0"]["max_t_error"] == 0
        assert all(row["max_momentum_error"] == 0 for row in results.values())
        for name in (
            "zero_s7",
            "zero_s128",
            "zero_direct",
            "zero_mass",
            "zero_n1",
            "zero_n4",
            "zero_n5",
        ):
            assert abs(results[name]["dt"][0] - base) < 1e-16
        for length in (0.125, 3.7, -0.25):
            assert abs(results[f"zero_L{length:g}"]["dt"][0] - length * base) < 1e-16
        for slices in (1, 17):
            assert 0.002 < results[f"spread_E0.001_s{slices}"]["max_t_error"] < 0.003
            assert results[f"spread_E1000_s{slices}"]["max_t_error"] < 1e-16
            assert results[f"spread_E1000_s{slices}"]["max_xy_error"] < 1e-17
        print(f"Checked {len(results)} drift cases; momenta unchanged in every case.", flush=True)
        print("DEFECT REPRODUCED", flush=True)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

