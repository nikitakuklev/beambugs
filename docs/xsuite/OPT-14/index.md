---
title: "OPT-14: Full radiation equilibrium covariance omits accumulated diffusion"
hide:
  - toc
---

# OPT-14 — Full radiation equilibrium covariance omits accumulated diffusion

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xtrack main (`f09e0fb`) |
| Source | [`_get_equilibrium_emittance_full`, radiation.py:244–357](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/radiation.py#L244-L357)<br>[public analysis caller](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_postprocess_and_complem_results.py#L168-L231)<br>[equilibrium balance](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L6064-L6094)<br>[FCC-ee local beam-size comparison](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/examples/fcc_ee/001_match_emittance.py#L83-L110) |

Full radiation analysis transports the reported equilibrium beam covariance with damping alone, omitting the accumulated photon diffusion, so beam variances differ at the two ends of the same periodic boundary. At equilibrium, damped transport plus diffusion must reproduce the starting covariance, whatever its absolute normalization. In a 144 m, 24-cell electron ring at 7 GeV/c with 48 bends and a 50 MV cavity, the horizontal variance ends at 7.660 × 10⁻⁵ m² instead of its starting 7.695 × 10⁻⁵ m², a 0.4538% deficit. The defect affects `eq_beam_covariance_matrix` from `radiation_method='full'`, also with lattice functions disabled.

??? example "Reproducer: OPT-14.py"

    [Download OPT-14.py](OPT-14.py)

    ```python
    """OPT-14: Full radiation equilibrium covariance omits accumulated diffusion

    Full radiation analysis transports the reported equilibrium beam covariance with damping alone,
    omitting the accumulated photon diffusion, so beam variances differ at the two ends of the same
    periodic boundary. At equilibrium, damped transport plus diffusion must reproduce the starting
    covariance, whatever its absolute normalization. In a 144 m, 24-cell electron ring at 7 GeV/c with
    48 bends and a 50 MV cavity, the horizontal variance ends at 7.660 × 10⁻⁵ m² instead of its starting
    7.695 × 10⁻⁵ m², a 0.4538% deficit. The defect affects `eq_beam_covariance_matrix` from
    `radiation_method='full'`, also with lattice functions disabled.

    The script builds that ring of FODO cells (thick bends, thin quadrupoles, one harmonic-36 cavity,
    mean radiation with compensated energy loss) at 3, 5 and 7 GeV/c and compares the last row of the
    covariance table with the first. Element Jacobians measured from tracked rays show that the reported
    covariances follow damped transport alone, and a Lyapunov solve with the SI photon second moment
    checks that a correct equilibrium closes. Across a radiation-free drift the reported covariance
    transforms correctly, and disabling lattice functions returns the same covariance.
    """

    import json

    from _prebuilt import CTX, identity, kernels, np, xt
    from scipy.constants import c, elementary_charge, epsilon_0, hbar, m_e
    from scipy.linalg import solve_discrete_lyapunov

    identity(radiation="mean")


    def ring(p0c):
        elements: list[xt.BeamElement] = [xt.Marker()]
        names = ["origin"]
        ncell = 24
        for i in range(ncell):
            cell = [
                xt.Multipole(knl=[0.0, 0.26]),
                xt.Drift(length=0.6),
                xt.Bend(length=1.4, angle=np.pi / ncell),
                xt.Drift(length=1.0),
                xt.Multipole(knl=[0.0, -0.26]),
                xt.Drift(length=1.0),
                xt.Bend(length=1.4, angle=np.pi / ncell),
                xt.Drift(length=0.6),
            ]
            for label, el in zip(("qf", "d1", "ba", "d2", "qd", "d3", "bb", "d4"), cell):
                elements.append(el)
                names.append(f"{label}{i}")
        beta = p0c / np.hypot(p0c, xt.ELECTRON_MASS_EV)
        elements += [xt.Cavity(voltage=50e6, frequency=36 * beta * c / 144.0, phase=np.pi), xt.Marker()]
        names += ["rf", "closure"]
        line = xt.Line(elements=elements, element_names=names)
        line.particle_ref = xt.Particles(_context=CTX, p0c=p0c, mass0=xt.ELECTRON_MASS_EV, q0=-1)
        line.build_tracker(_context=CTX)
        line.configure_radiation(model="mean")
        line.compensate_radiation_energy_loss()
        return line


    results = []
    for energy in (3e9, 5e9, 7e9):
        line = ring(energy)
        tw = line.twiss(
            radiation_method="full",
            radiation_analysis=True,
            compute_R_element_by_element=True,
            chrom=False,
        )
        sigma = tw.eq_beam_covariance_matrix.Sigma
        orbit = np.array([tw[k] for k in ("x", "px", "y", "py", "zeta", "pzeta")]).T
        orbit_closure = float(np.max(np.abs(orbit[-1] - orbit[0])))
        assert orbit_closure < 2e-11
        assert np.all(tw.damping_constants_turns > 0)
        assert np.all(np.diag(sigma[0])[[0, 4, 5]] > 0)
        # Equilibrium: Sigma0 = A_turn Sigma0 A_turn.T + Q_turn, so the last row must equal the first.
        closure = (np.diag(sigma[-1]) - np.diag(sigma[0]))[[0, 4, 5]] / np.diag(sigma[0])[[0, 4, 5]]
        assert np.max(np.abs(closure)) > 1e-4
        # Measure every element's Jacobian about the same closed orbit with +/- rays.
        # ptau = beta0*pzeta is exact; use the public linked property at construction.
        errs = []
        matrices = []
        for step in (2e-7, 1e-7):
            rays = np.vstack([step * np.eye(6), -step * np.eye(6)])
            u = rays + orbit[0]
            p = xt.Particles(
                _context=CTX,
                p0c=energy,
                mass0=xt.ELECTRON_MASS_EV,
                q0=-1,
                x=u[:, 0],
                px=u[:, 1],
                y=u[:, 2],
                py=u[:, 3],
                zeta=u[:, 4],
                ptau=tw.beta0 * u[:, 5],
            )
            line.track(p, turn_by_turn_monitor="ONE_TURN_EBE")
            assert np.all(p.state == 1)
            mon = line.record_last_track
            coordinates = np.array([mon.x, mon.px, mon.y, mon.py, mon.zeta, mon.ptau / tw.beta0])
            rr = ((coordinates[:, :6, :] - coordinates[:, 6:, :]) / (2 * step)).transpose(2, 0, 1)
            matrices.append(rr)
            transported = rr @ sigma[0] @ rr.transpose(0, 2, 1)
            active = [0, 4, 5]
            transport_err = float(
                np.max(
                    np.abs(
                        np.diagonal(transported, axis1=1, axis2=2)[:, active]
                        / np.diagonal(sigma, axis1=1, axis2=2)[:, active]
                        - 1
                    )
                )
            )
            errs.append(transport_err)
            assert transport_err < 2e-7
        rr = matrices[-1]
        matrix_convergence = float(np.max(np.abs(matrices[0] - rr)))
        # Independently integrate positive photon second moments at bend exits.
        # This is a specified quadrature of a stochastic linear model. Its exact
        # equilibrium must be periodic for every normalization of the source term.
        re = elementary_charge**2 / (4 * np.pi * epsilon_0 * m_e * c**2)
        lambdabar = hbar / (m_e * c)
        gamma = np.hypot(energy, xt.ELECTRON_MASS_EV) / xt.ELECTRON_MASS_EV
        variance_per_bend = (
            55 / (24 * np.sqrt(3)) * re * lambdabar * gamma**5 * (np.pi / 24 / 1.4) ** 3 * 1.4
        )
        A = rr[-1]
        Q = np.zeros((6, 6))
        for i, name in enumerate(line.element_names):
            if isinstance(line[name], xt.Bend):
                # Photon recoil is along the orbit momentum; it leaves direction
                # unchanged in the ultrarelativistic approximation.
                v = np.zeros(6)
                v[1] = orbit[i + 1, 1] / (1 + tw.delta[i + 1])
                v[3] = orbit[i + 1, 3] / (1 + tw.delta[i + 1])
                v[5] = 1
                q = variance_per_bend * np.outer(v, v)
                tail = np.linalg.solve(rr[i + 1].T, A.T).T
                Q += tail @ q @ tail.T
        # Scale coordinates before solving to avoid length/momentum conditioning.
        scale = np.sqrt(np.maximum(np.diag(sigma[0]), 1e-18))
        scaled_A = A * scale[None, :] / scale[:, None]
        scaled_Q = Q / scale[:, None] / scale[None, :]
        S = solve_discrete_lyapunov(scaled_A, scaled_Q, method="direct")
        lyap = float(np.linalg.norm(S - scaled_A @ S @ scaled_A.T - scaled_Q) / np.linalg.norm(S))
        assert lyap < 5e-14
        assert np.min(np.linalg.eigvalsh(S)) > -1e-10
        sigma0 = S * scale[:, None] * scale[None, :]
        correct_end = A @ sigma0 @ A.T + Q
        periodic_error = float(np.max(np.abs(correct_end - sigma0) / scale[:, None] / scale[None, :]))
        assert periodic_error < 1e-10
        # A non-radiating drift has exactly zero inhomogeneous covariance source.
        idx = line.element_names.index("d20")
        local = np.linalg.solve(rr[idx].T, rr[idx + 1].T).T
        drift_residual = float(np.max(np.abs(local @ sigma[idx] @ local.T - sigma[idx + 1])))
        assert drift_residual < 2e-13
        bend_idx = line.element_names.index("ba0")
        local_bend = np.linalg.solve(rr[bend_idx].T, rr[bend_idx + 1].T).T
        bend_source55 = float((sigma[bend_idx + 1] - local_bend @ sigma[bend_idx] @ local_bend.T)[5, 5])
        assert abs(bend_source55) / variance_per_bend < 2e-4
        no_lattice_difference = None
        if energy == 5e9:
            slim = line.twiss(
                radiation_method="full",
                radiation_analysis=True,
                compute_lattice_functions=False,
                chrom=False,
            )
            assert "betx" not in slim
            no_lattice_difference = float(np.max(np.abs(slim.eq_beam_covariance_matrix.Sigma - sigma)))
            assert no_lattice_difference < 2e-13
        rec = dict(
            p0c=energy,
            circumference=tw.line_length,
            energy_loss=tw.energy_loss,
            damping=tw.damping_constants_turns.tolist(),
            orbit_closure=orbit_closure,
            start_diag=np.diag(sigma[0]).tolist(),
            end_diag=np.diag(sigma[-1]).tolist(),
            relative_closure=closure.tolist(),
            native_transport_errors=errs,
            matrix_convergence=matrix_convergence,
            lyapunov_residual=lyap,
            periodic_error=periodic_error,
            diffusion_trace=float(np.trace(Q)),
            variance_per_bend=variance_per_bend,
            drift_residual=drift_residual,
            bend_covariance_source55=bend_source55,
            no_lattice_difference=no_lattice_difference,
        )
        results.append(rec)
        print("RADIATION", json.dumps(rec))
        print(
            f"CLOSURE p0c={energy:g} xx_expected={sigma[0, 0, 0]:.17g} "
            f"xx_actual={sigma[-1, 0, 0]:.17g} "
            f"pzeta_expected={sigma[0, 5, 5]:.17g} "
            f"pzeta_actual={sigma[-1, 5, 5]:.17g}"
        )

    kernels()
    print("DEFECT REPRODUCED")

    ```

??? example "Helper: _prebuilt.py"

    [Download _prebuilt.py](_prebuilt.py)

    ```python
    """Serial CPU setup that runs Xsuite's prebuilt kernels with kernel compilation forbidden.

    `identity()` prints the package versions and context; `kernels()` asserts that a prebuilt kernel
    library was loaded and prints it.
    """

    import importlib
    import importlib.metadata
    import os
    import platform
    from pathlib import Path

    # Configure thread pools before importing NumPy or Xsuite.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[name] = "1"
    os.environ["XSUITE_ALLOW_KERNEL_COMPILATION"] = "0"
    os.environ["XSUITE_FORCE_KERNEL_COMPILATION"] = "0"
    os.environ["XSUITE_CFFI_FORBID_COMPILE"] = "1"

    import numpy as np
    import xcoll as xc
    import xobjects as xo
    import xtrack as xt
    from _common import report

    __all__ = ["CTX", "identity", "kernels", "np", "particles", "report", "xc", "xt"]

    np.set_printoptions(precision=17)
    xo.settings.allow_kernel_compilation = False
    xo.settings.force_kernel_compilation = False
    CTX = xo.ContextCpu(omp_num_threads=0)


    def identity(*, radiation="off", random_seed=None):
        print(f"Python {platform.python_version()}; {platform.platform()}")
        for name in ("numpy", "scipy", "cffi"):
            print(name, importlib.metadata.version(name))
        for name in ("xsuite", "xtrack", "xobjects", "xpart", "xfields", "xdeps", "xcoll"):
            mod = importlib.import_module(name)
            print("PACKAGE", name, mod.__version__, mod.__file__)
        randomness = (
            "deterministic, no random seed"
            if random_seed is None
            else f"fixed RNG seed(s)={random_seed}"
        )
        print(
            f"CONTEXT CPU serial; radiation {radiation}; {randomness}; "
            "kernel compilation forbidden; prebuilt kernels"
        )


    def kernels():
        libraries = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "/xsuite/lib/" in line and ".so" in line
            }
        )
        assert libraries, "No prebuilt kernel library loaded"
        print("KERNEL_LIBRARIES", libraries)


    def particles(**kwargs):
        return xt.Particles(_context=CTX, p0c=1e9, **kwargs)

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
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, prebuilt kernels (compilation forbidden), CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, prebuilt kernels (compilation forbidden), CPU, serial |

