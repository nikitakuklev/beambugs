---
title: "ImpactX shared helpers"
---

# ImpactX shared helpers

These modules support the [ImpactX reproducers](../index.md).
Each entry links all the helpers it needs, including those imported by another helper.
Save the linked helpers in the same directory as the downloaded reproducer scripts.

## `_precision.py` { #_precision }

[Download _precision.py](_precision.py)

??? example "Source: _precision.py"

    ```python
    """Shared serial ImpactX harness for mesh and zero-source precision reproducers."""

    import hashlib
    import importlib.metadata
    import itertools
    import json
    import math
    import os
    import subprocess
    import sys
    from decimal import Decimal, localcontext
    from pathlib import Path

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError("Launch on a single CPU, for example with taskset -c 2")
    # Child processes inherit this single-CPU affinity.
    (CPU,) = os.sched_getaffinity(0)
    ROOT = Path(__file__).resolve().parent
    COLS = ["position_x", "position_y", "position_t", "momentum_x", "momentum_y", "momentum_t"]
    E_CHARGE = 1.602176634e-19


    def run(spec):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "child"],
            input=json.dumps(spec),
            text=True,
            capture_output=True,
            timeout=290,
            check=False,
            cwd=ROOT,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        rows = [line[7:] for line in result.stdout.splitlines() if line.startswith("RESULT ")]
        if len(rows) != 1:
            raise RuntimeError(result.stdout + result.stderr)
        return json.loads(rows[0])


    def environment():
        versions = {
            name: importlib.metadata.version(name) for name in ("impactx-noacc", "numpy", "pandas")
        }
        print("environment", versions, f"CPU={CPU} numerical_threads=1 AMReX_seed=1")


    def native(spec):
        import impactx
        import numpy as np
        from amrex.space3d import RealBox
        from impactx import Config, ImpactX, ImpactXParIter, distribution, elements

        config = Config.to_dict()
        assert config["precision"] == config["precision_particles"] == "DOUBLE"
        assert not any(config[k] for k in ("have_mpi", "have_gpu", "have_omp"))
        binary = next(Path(impactx.__file__).parent.rglob("impactx_pybind*.so"))
        identity = dict(
            binary=str(binary),
            sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            config=config,
            affinity=sorted(os.sched_getaffinity(0)),
        )
        if "EXPECTED_IMPACTX_SHA256" in os.environ:
            assert identity["sha256"] == os.environ["EXPECTED_IMPACTX_SHA256"]
        sim = ImpactX()
        try:
            options = dict(
                particle_shape=spec.get("shape", 1),
                space_charge=spec.get("mode", "3D"),
                strang_split=spec.get("strang", False),
                diagnostics=False,
                slice_step_diagnostics=False,
                tiny_profiler=False,
                memory_profiler=False,
                csr=False,
                isr=False,
                spin=False,
                verbose=0,
                poisson_solver="fft",
            )
            for key, val in options.items():
                setattr(sim, key, val)
            cells = spec.get("cells", 16)
            sim.n_cell = [cells] * 3
            for axis in "xyz":
                setattr(sim, "blocking_factor_" + axis, [1])
                setattr(sim, "max_grid_size_" + axis, [cells])
            if "factor" in spec:
                sim.prob_relative = [spec["factor"]]
            if spec.get("static"):
                sim.domain = RealBox([-1.0] * 3, [1.0] * 3)
            sim.space_charge_apply_longitudinal_kick = spec.get("longitudinal", True)
            sim.init_grids()
            ref = sim.beam.ref
            sign = spec.get("sign", 1)
            ref.set_charge_qe(sign).set_mass_MeV(938.27208816).set_kin_energy_MeV(938.27208816)
            kind = spec["kind"]
            charge = spec.get("charge", 1e-9)
            if kind in ("zero", "initialize"):
                n = spec.get("n", 128)
                distr = distribution.Gaussian(
                    lambdaX=0.0008,
                    lambdaY=0.0012,
                    lambdaT=0.008,
                    lambdaPx=1e-4,
                    lambdaPy=2e-4,
                    lambdaPt=3e-4,
                )
                if n == 1:
                    sim.beam.add_n_particles(
                        [0.001],
                        [0.002],
                        [0.01],
                        [1e-4],
                        [2e-4],
                        [3e-4],
                        qm=ref.qm_ratio_SI,
                        bunch_charge=charge,
                    )
                else:
                    sim.add_particles(bunch_charge=charge, distr=distr, npart=n)
                if kind == "zero" and spec.get("weights_after"):
                    # Exercise zeroing an existing charged population through documented tile access.
                    for pti in ImpactXParIter(sim.beam, level=0):
                        pti.soa().to_xp().real["weighting"][:] = 0.0
                    charge = 0.0
            else:
                if kind in ("grid", "boundary"):
                    pos = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=3)))
                    if kind == "boundary":
                        pos = (pos + 1.0) / 2.0
                    pos *= np.array([2.0**-10, 2.0**-9, 2.0**-7])
                    if spec.get("translate"):
                        pos[:, 0] += 0.003
                    mom = np.zeros_like(pos)
                elif kind == "track":
                    rng = np.random.default_rng(20261106)
                    pos = rng.normal(size=(8192, 3)) * [0.0008, 0.0012, 0.008]
                    mom = np.zeros_like(pos)
                else:
                    raise ValueError(kind)
                n = len(pos)
                sim.beam.add_n_particles(
                    *[np.ascontiguousarray(pos[:, j]) for j in range(3)],
                    *[np.ascontiguousarray(mom[:, j]) for j in range(3)],
                    qm=ref.qm_ratio_SI,
                    bunch_charge=charge,
                )
            before = sim.beam.to_df().sort_index()
            initial = before[COLS].to_numpy()
            q_initial = sign * E_CHARGE * math.fsum(before["weighting"].tolist())
            result = dict(
                expected_retained=n,
                spec=spec,
                identity=identity,
                requested=n,
                before=len(before),
                q_requested=sign * charge,
                q_before=q_initial,
            )
            if kind == "boundary":
                # Fixed-s 2D mode avoids coordinate conversions: no transport or deposition.
                sim.resize_mesh()
                geom = sim.Geom(0)
                hi = sim.domain.hi()
                accepted = sum(geom.insideRoundOffDomain(*row) for row in pos)
                with localcontext() as context:
                    context.prec = 90
                    maximum = Decimal.from_float(2.0**-10)
                    padding = maximum * (Decimal.from_float(spec["factor"]) - 1) / 2
                    result.update(exact_padding_x=str(padding), exact_hi_x=str(maximum + padding))
                result.update(rounded_hi_x=float(hi[0]), predicted_accepted=accepted)
                sim.beam.redistribute()
            elif kind == "grid":
                sim.deposit_charge()
                lo, hi = np.array(sim.domain.lo()), np.array(sim.domain.hi())
                result["bounds"] = [lo.tolist(), hi.tolist()]
                geom = sim.Geom(0)
                dx = np.array(geom.data().CellSize())
                result["cell_size"] = dx.tolist()
                mf = sim.rho(0)
                tiles = mf.to_numpy()
                assert len(tiles) == 1
                ng = tuple(mf.n_grow_vect)
                interior = tiles[0][tuple(slice(g, -g if g else None) for g in ng) + (0,)]
                result["q_deposited"] = float(np.sum(interior, dtype=np.float64) * np.prod(dx))
                result["upper_cell"] = np.floor((hi - lo) / dx).astype(int).tolist()
            elif kind in ("track", "zero"):
                length = 0.01 if kind == "track" else 0.037
                sim.lattice.append(elements.Drift(ds=length, nslice=100 if kind == "track" else 1))
                sim.track_particles()
                result["length"] = length
            after = sim.beam.to_df().sort_index()
            final = after[COLS].to_numpy()
            result.update(
                retained=len(after),
                q_after=sign * E_CHARGE * math.fsum(after["weighting"].tolist()),
                nonfinite=int(np.count_nonzero(~np.isfinite(final))),
            )
            if kind == "zero":
                # At zero source density, linear drift has x'=px, y'=py,
                # t'=pt/(beta*gamma)^2. gamma=2, hence (beta*gamma)^2=3.
                expected = initial.copy()
                expected[:, :3] += result["length"] * initial[:, 3:] / [1.0, 1.0, 3.0]
                result["expected_nonfinite"] = int(np.count_nonzero(~np.isfinite(expected)))
                assert len(final) == len(expected) == n
                result["nonfinite_by_coordinate"] = np.count_nonzero(
                    ~np.isfinite(final), axis=0
                ).tolist()
                if result["nonfinite"] == 0:
                    result["drift_max_abs_error"] = float(np.max(np.abs(final - expected)))
                    if charge == 0.0:
                        errors = np.max(np.abs(final - expected), axis=0)
                        scales = np.max(np.abs(initial), axis=0)
                        scales[5] += 2.0 / math.sqrt(3.0)
                        bounds = 16.0 * (2.0**-53) * scales
                        result["drift_error_by_coordinate"] = errors.tolist()
                        result["drift_bound_by_coordinate"] = bounds.tolist()
                        assert np.all(errors <= bounds), (errors, bounds)
                result["initial_first"] = initial[0].tolist()
                result["expected_first"] = expected[0].tolist()
                result["final_first"] = final[0].tolist()
            return result
        finally:
            sim.finalize()


    if __name__ == "__main__":
        if sys.argv[1:] != ["child"]:
            raise RuntimeError("Use an entry reproducer")
        print("RESULT " + json.dumps(native(json.load(sys.stdin))))

    ```

## `_slicing.py` { #_slicing }

[Download _slicing.py](_slicing.py)

??? example "Source: _slicing.py"

    ```python
    """Shared serial ImpactX/MAD-X harness for geometry and radiation reproducers."""

    import hashlib
    import importlib.metadata
    import json
    import math
    import os
    import subprocess
    import sys
    import tempfile
    import warnings
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent
    COLS = ["position_x", "momentum_x", "position_y", "momentum_y", "position_t", "momentum_t"]
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
        raise RuntimeError("Launch on a single CPU, for example with taskset -c 3")
    # Child processes inherit this single-CPU affinity.
    (CPU,) = os.sched_getaffinity(0)

    import numpy as np


    def run(config):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "child"],
            input=json.dumps(config),
            text=True,
            capture_output=True,
            timeout=290,
            check=False,
            cwd=ROOT,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        rows = [line[7:] for line in result.stdout.splitlines() if line.startswith("RESULT ")]
        if len(rows) != 1:
            raise RuntimeError(result.stdout + result.stderr)
        return json.loads(rows[0])


    def madx_check(deck, rays):
        from cpymad.madx import Madx

        # MAD-X T,PT are both the negatives of ImpactX t,pt.
        with (
            tempfile.TemporaryDirectory(dir=ROOT, prefix="madx-") as folder,
            Madx(stdout=False, cwd=folder) as mad,
        ):
            mad.input(deck)
            tw = mad.twiss(betx=2, bety=3, rmatrix=True)
            matrix = np.array([[tw[f"re{i + 1}{j + 1}"][-1] for j in range(6)] for i in range(6)])
            sign = np.diag([1, 1, 1, 1, -1, -1])
            survey = mad.survey()
            out = {
                # Element names without occurrence suffix and their exit positions.
                "names": [str(n).split(":")[0] for n in tw.name],
                "positions": [float(v) for v in tw.s],
                "s": float(tw.s[-1]),
                "matrix": (sign @ matrix @ sign).tolist(),
                "survey_theta": float(survey.theta[-1]),
            }
            mad.input("track,onepass=true,onetable=true;")
            for q in rays:
                mad.command.start(x=q[0], px=q[1], y=q[2], py=q[3], t=-q[4], pt=-q[5])
            mad.input("run,turns=1;endtrack;")
            table = mad.table.trackone
            selection = np.asarray(table.turn) == 1
            values = (
                np.column_stack(
                    [np.asarray(table[k])[selection] for k in ("x", "px", "y", "py", "t", "pt")]
                )
                @ sign
            )
            out["rays"] = values[np.argsort(np.asarray(table.number)[selection])].tolist()
            return out


    def native(config):
        from impactx import ImpactX, elements

        sim = ImpactX()
        try:
            settings = dict(
                verbose=0,
                diagnostics=False,
                slice_step_diagnostics=False,
                tiny_profiler=False,
                memory_profiler=False,
                particle_shape=2,
                space_charge=False,
                csr=False,
                isr=False,
                spin=False,
                strang_split=False,
            )
            if config["kind"] == "radiation":
                settings.update(
                    isr=config.get("enabled", True),
                    isr_order=config.get("order", 1),
                    isr_on_ref_part=config.get("ref_loss", False),
                    strang_split=config.get("strang", False),
                )
            for key, value in settings.items():
                setattr(sim, key, value)
            sim.init_grids()
            ref = sim.beam.ref
            gamma = config.get("gamma", 3.0)
            ref.set_species("electron").set_kin_energy_MeV((gamma - 1) * ref.mass_MeV)
            beta, bg, mass = ref.beta, ref.beta_gamma, ref.mass
            if config["kind"] != "radiation":
                q = np.zeros((3, 6))
                if config["kind"] == "sequence":
                    q[1, :2] = [0.0002, 0.0007]
                    q[2, 3] = -0.0003
                else:
                    q[1, 0], q[2, 5] = 0.0008, 0.0002
                deck = config["deck"]
                native_result = madx_check(deck, q)
                with tempfile.TemporaryDirectory(dir=ROOT, prefix="deck-") as folder:
                    path = Path(folder) / "case.madx"
                    path.write_text(deck)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        sim.lattice.load_file(str(path), min_model=config.get("model", "linear"))
                if config.get("native_bend"):
                    sim.lattice.clear()
                    sim.lattice.append(elements.ThinDipole(theta=math.degrees(config["angle"]), rc=10))
                matrix = sim.lattice.transfer_map(ref).to_numpy().copy()
                records = sim.lattice.to_dicts()
                info = dict(
                    native=native_result,
                    matrix=matrix.tolist(),
                    records=records,
                    warnings=[str(w.message) for w in caught],
                    initial=q.tolist(),
                )
            else:
                q = np.zeros((8192, 6))
                q[:, 0] = config["x"]
                radius, length = 10.0, 1.0
                edge = config.get("edge", 0.0)
                sim.lattice.append(elements.DipEdge(psi=edge, rc=radius, g=0, location="entry"))
                for _ in range(config.get("pieces", 1)):
                    size = length / config.get("pieces", 1)
                    if config.get("body", "Sbend") == "ExactSbend":
                        bend = elements.ExactSbend(
                            ds=size, phi=math.degrees(size / radius), nslice=config["n"]
                        )
                    elif config.get("body") == "CFbend":
                        bend = elements.CFbend(ds=size, rc=radius, k=0, nslice=config["n"])
                    else:
                        bend = elements.Sbend(ds=size, rc=radius, nslice=config["n"])
                    sim.lattice.append(bend)
                sim.lattice.append(elements.DipEdge(psi=edge, rc=radius, g=0, location="exit"))
                info = {}
            sim.beam.add_n_particles(
                *q[:, [0, 2, 4, 1, 3, 5]].T.copy(), qm=ref.qm_ratio_SI, w=np.ones(len(q))
            )
            sim.track_particles()
            frame = sim.beam.to_df(local=True).reset_index(drop=True).sort_values("idcpu")
            final = frame[COLS].to_numpy()
            if config["kind"] == "radiation":
                pt = final[:, 5]
                info.update(
                    mean_pt=float(pt.mean()),
                    var_pt=float(pt.var(ddof=1)),
                    pt_sha256=hashlib.sha256(pt.tobytes()).hexdigest(),
                    max_pt=float(np.max(np.abs(pt))),
                    mean_t=float(final[:, 4].mean()),
                    mean_x=float(final[:, 0].mean()),
                )
            else:
                info["rays"] = final.tolist()
            info.update(
                beta=beta,
                bg=bg,
                mass=mass,
                s=ref.s,
                ct=ref.t,
                heading=math.atan2(ref.px, ref.pz),
                ref_gamma=ref.gamma,
            )
            return info
        finally:
            sim.finalize()


    def environment():
        print(
            "environment",
            {
                name: importlib.metadata.version(name)
                for name in ("impactx-noacc", "cpymad", "numpy", "scipy")
            },
            f"CPU={CPU} numerical_threads=1 AMReX_seed=1",
        )


    if __name__ == "__main__":
        if sys.argv[1:] != ["child"]:
            raise RuntimeError("Use an entry reproducer")
        print("RESULT " + json.dumps(native(json.load(sys.stdin))))

    ```
