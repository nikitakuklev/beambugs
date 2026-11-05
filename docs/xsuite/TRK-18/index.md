---
title: "TRK-18: Radiation tracking omits the field wedges of angled dipole faces"
hide:
  - toc
---

# TRK-18 — Radiation tracking omits the field wedges of angled dipole faces

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on xtrack 0.118.2 (`ad5e808`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`WITH_RADIATION`, track_magnet.h:96–182](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet.h#L96-L182)<br>[`magnet_radiation`, track_magnet_radiation.h:233–267](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_radiation.h#L233-L267)<br>[entry](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet.h#L406-L449)<br>[exit](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet.h#L561-L592)<br>[`track_magnet_edge_particles`](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_edge.h#L79-L180)<br>[`Wedge_single_particle`, track_wedge.h:13–58](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_wedge.h#L13-L58)<br>[track_magnet.h:278–288](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet.h#L278-L288)<br>[`Bend` entry slice, thin_slice_bend_entry.h:41–68](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/thin_slice_bend_entry.h#L41-L68)<br>[`_add_radiation_analysis_to_twiss_result`, twiss_postprocess_and_complem_results.py:168–200](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_postprocess_and_complem_results.py#L168-L200)<br>[`configure_radiation`, line.py:5019–5024](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L5019-L5024)<br>[bend edge parameters, _common.py:116–124](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/_common.py#L116-L124)<br>[`RBend`, rbend.py:30–59](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/rbend.py#L30-L59)<br>[physics guide, xtrack.tex:2682–2686](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L2682-L2686)<br>[emitted power, xtrack.tex:5882–5885](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L5882-L5885) |

With synchrotron radiation, tracking through a bend with angled hard-edge faces, including a rectangular `RBend`, radiates only in the curved body and omits the field wedge at each face, whose length varies with horizontal position x. In a rectangular magnet a displaced ray slides along parallel faces, so its energy loss U cannot depend on x; for a 1.5 m, π/32 rad bend at 3 GeV/c, U⁻¹dU/dx is 0.06550 m⁻¹ instead of zero, and quantum emission shifts likewise. In a 132 m, 3 GeV electron ring of rectangular bends with a 5 MV cavity, the tracked horizontal damping partition number is 0.7879 instead of 0.9906 from an independent radiation integration, and the equilibrium emittance is 25.63% too large. Perpendicular faces are unaffected.

??? example "Reproducer: TRK-18.py"

    [Download TRK-18.py](TRK-18.py)

    ```python
    """TRK-18: Radiation tracking omits the field wedges of angled dipole faces

    With synchrotron radiation, tracking through a bend with angled hard-edge faces, including a
    rectangular `RBend`, radiates only in the curved body and omits the field wedge at each face, whose
    length varies with horizontal position x. In a rectangular magnet a displaced ray slides along
    parallel faces, so its energy loss U cannot depend on x; for a 1.5 m, π/32 rad bend at 3 GeV/c,
    U⁻¹dU/dx is 0.06550 m⁻¹ instead of zero, and quantum emission shifts likewise. In a 132 m, 3 GeV
    electron ring of rectangular bends with a 5 MV cavity, the tracked horizontal damping partition
    number is 0.7879 instead of 0.9906 from an independent radiation integration, and the equilibrium
    emittance is 25.63% too large. Perpendicular faces are unaffected.

    The script tracks 3 GeV/c electrons through one rectangular 1.5 m bend at x = 0 and ±0.1 mm with
    mean radiation, for `Bend` and curved-body `RBend` with default, `linear` and `full` edges, unsliced
    and Teapot-sliced, and compares the loss slope with zero, the value that translation along parallel
    faces requires; perpendicular faces and the straight-body `RBend` are controls. It repeats the
    displacement test with 200,000 seeded particles in `quantum` mode. It then runs
    `twiss(radiation_analysis=True)` on a 132 m ring of rectangular bends at two RF settings and
    compares the horizontal partition number and emittance with an independent linear integration of
    radiation damping and photon diffusion that includes the face terms; a sector-face ring and the ring
    with the derived face term inserted as thin maps are controls.
    """

    from _prebuilt import CTX, identity, kernels, np, xt
    from scipy.constants import c, hbar, m_e, physical_constants
    from scipy.linalg import expm, solve_discrete_lyapunov

    identity(radiation="mean/quantum", random_seed="314159+arange(200000)")

    P0C = 3e9
    MASS = xt.ELECTRON_MASS_EV
    R_E = physical_constants["classical electron radius"][0]
    GAMMA = np.hypot(P0C, MASS) / MASS
    BETA0 = P0C / np.hypot(P0C, MASS)
    # Mean radiated power over E c is A h^2 with A = (2/3) r_e gamma^3 (ultrarelativistic Larmor).
    A = 2 / 3 * R_E * GAMMA**3
    C_Q = 55 / (32 * np.sqrt(3)) * hbar / (m_e * c)
    defects = []


    def particles(**kwargs):
        return xt.Particles(_context=CTX, p0c=P0C, mass0=MASS, q0=-1, **kwargs)


    def bend(kind, edge, e, length, theta):
        """A uniform dipole whose physical faces are rotated by e from the radial planes."""
        kwargs = dict(k0=theta / length)
        if edge != "default":
            kwargs.update(edge_entry_model=edge, edge_exit_model=edge)
        if kind == "Bend":
            return xt.Bend(length=length, angle=theta, edge_entry_angle=e, edge_exit_angle=e, **kwargs)
        # RBend edge angles are measured from the rectangular faces, which lie at theta/2.
        kwargs.update(edge_entry_angle=e - theta / 2, edge_exit_angle=e - theta / 2)
        if kind == "RBend-straight":
            kwargs["rbend_model"] = "straight-body"
        return xt.RBend(length_straight=2 * length / theta * np.sin(theta / 2), angle=theta, **kwargs)


    def build(elements, slices):
        line = xt.Line(elements=elements)
        line.particle_ref = particles()
        if slices:
            line.slice_thick_elements(
                [
                    xt.Strategy(slicing=None),
                    xt.Strategy(slicing=xt.Teapot(slices), element_type=xt.Bend),
                    xt.Strategy(slicing=xt.Teapot(slices), element_type=xt.RBend),
                ]
            )
        line.build_tracker(_context=CTX)
        return line


    # Single magnet: 1.5 m, angle pi/32, uniform field, no gap, no multipoles.
    LB1, TH1 = 1.5, np.pi / 32
    DX = 1e-4
    cases = [(k, edge, TH1 / 2) for k in ("Bend", "RBend") for edge in ("default", "linear", "full")]
    cases += [("Bend", "default", 0.0), ("RBend-straight", "default", TH1 / 2)]
    for kind, edge, e in cases:
        for slices in (0, 64):
            line = build([bend(kind, edge, e, LB1, TH1)], slices)
            line.configure_radiation(model="mean")
            p = particles(x=[-DX, 0.0, DX])
            line.track(p)
            assert np.all(p.state == 1)
            loss = -np.asarray(p.ptau) * P0C
            slope = (loss[2] - loss[0]) / (2 * DX * loss[1])
            # At fixed energy and field the loss is proportional to the path length in the field.
            # With x(s) = x0 (cos hs + tan e sin hs) after the entrance lens, the body adds
            # x0 (sin th + tan e (1 - cos th)) through the (1 + h x) metric; each face rotated by e
            # removes x tan e of field length at its displacement (x0 at entry, x(L) at exit).
            body = (np.sin(TH1) + np.tan(e) * (1 - np.cos(TH1))) / LB1
            face = -np.tan(e) * (1 + np.cos(TH1) + np.tan(e) * np.sin(TH1)) / LB1
            if kind == "RBend-straight":
                # Radiation between the parallel body planes already holds the rectangular geometry.
                expected, label = 0.0, "straight body, parallel planes"
                assert abs(slope) < 1e-5
            elif e == 0:
                expected, label = body, "sector faces, no face term"
                assert abs(slope - expected) < 1e-5
            else:
                # For e = th/2 body + face = 2 tan(th/2)/L - 2 tan(th/2)/L = 0: translation invariance.
                expected, label = body + face, "parallel faces, translation invariance"
                assert abs(expected) < 1e-15 and abs(slope - body) < 1e-5
                defects.append(abs(slope - expected) > 0.9 * abs(face))
            print(
                f"LOCAL {kind} edge={edge} e={e:.6g} slices={slices}: U0={loss[1]:.10g} eV "
                f"slope observed={slope:.10g} expected={expected:.10g} 1/m ({label}); "
                f"body only={body:.10g}"
            )

    # Quantum mode: paired seeds at x = -20 mm and +20 mm; translation invariance needs equal losses.
    N = 200_000
    seeds = (314159 + np.arange(N)).astype(np.uint32)
    for kind in ("Bend", "RBend-straight"):
        line = build([bend(kind, "full", TH1 / 2, LB1, TH1)], 0)
        line.configure_radiation(model="quantum")
        losses = []
        for x in (-0.02, 0.02):
            p = particles(x=np.full(N, x))
            p._init_random_number_generator(seeds=seeds)
            line.track(p)
            assert np.all(p.state == 1)
            losses.append(-np.asarray(p.ptau) * P0C)
        diff = losses[1] - losses[0]
        sem = np.std(diff, ddof=1) / np.sqrt(N)
        if kind == "Bend":
            defects.append(abs(diff.mean()) > 6 * sem)
        else:
            assert np.all(diff == 0)
        print(
            f"QUANTUM {kind} full edge: mean loss at -20 mm={losses[0].mean():.10g} eV "
            f"at +20 mm={losses[1].mean():.10g} eV difference={diff.mean():.10g}+-{sem:.4g} eV "
            "expected=0 (translation invariance)"
        )

    # Ring: 24 cells QF(0.25, +1.2) D(0.7) B(1.4) D(0.4) QD(0.25, -1.2) D(0.4) B(1.4) D(0.7).
    NC, LB = 24, 1.4
    TH = np.pi / NC
    H = TH / LB
    CELL = [("q", 0.25, 1.2), ("d", 0.7, 0.0), ("b", LB, H), ("d", 0.4, 0.0),
            ("q", 0.25, -1.2), ("d", 0.4, 0.0), ("b", LB, H), ("d", 0.7, 0.0)]
    CIRC = NC * sum(row[1] for row in CELL)
    I2 = NC * 2 * LB * H**2


    def independent(e, harmonic, voltage, with_faces):
        """Linear one-turn map with radiation damping and photon diffusion in (x, px, zeta, delta).

        Body generator: x' = px, px' = -k x + h delta - A h^2 px, zeta' = -h x + delta/gamma^2,
        delta' = -A h^2 (2 delta + h x) (power ~ E^2 B^2 and the (1 + h x) path). A face rotated by
        e is the lens px += h tan(e) x and, with with_faces, delta += A h^2 tan(e) x from the field
        length -x tan e it removes. Diffusion d<delta^2>/ds = 2 A C_q gamma^2 |h|^3. The reference
        loss is compensated (tapered design ray); the RF slope at nonzero dispersion is kept.
        """
        total, noise = np.eye(4), np.zeros((4, 4))
        lens = np.eye(4)
        lens[1, 0] = H * np.tan(e)
        if with_faces:
            lens[3, 0] = A * H**2 * np.tan(e)
        for _ in range(NC):
            for kind, length, strength in CELL:
                h = H if kind == "b" else 0.0
                k = strength if kind == "q" else h**2
                gen = np.zeros((4, 4))
                gen[0, 1], gen[1, 0], gen[1, 3] = 1.0, -k, h
                gen[2, 0], gen[2, 3] = -h, 1 / GAMMA**2
                gen[1, 1], gen[3, 0], gen[3, 3] = -A * h**2, -A * h**3, -2 * A * h**2
                rate = np.zeros((4, 4))
                rate[3, 3] = 2 * A * C_Q * GAMMA**2 * abs(h) ** 3
                # Van Loan block exponential: exact map and integrated diffusion of a uniform body.
                block = np.block([[gen, rate], [np.zeros((4, 4)), -gen.T]])
                ex = expm(block * length)
                mats = [(ex[:4, :4], ex[:4, 4:] @ ex[:4, :4].T)]
                if kind == "b":
                    mats = [(lens, 0.0)] + mats + [(lens, 0.0)]
                for mat, diffusion in mats:
                    noise = mat @ noise @ mat.T + diffusion
                    total = mat @ total
        loss = P0C * A * I2
        rf = np.eye(4)
        rf[3, 2] = np.sqrt(voltage**2 - loss**2) / P0C * 2 * np.pi * harmonic / CIRC
        noise, total = rf @ noise @ rf.T, rf @ total
        vals = np.linalg.eigvals(total)
        vals = vals[vals.imag > 0]
        vals = vals[np.argsort(np.angle(vals))]  # synchrotron mode first, then horizontal
        partitions = -2 * np.log(abs(vals)) / (A * I2)
        sigma = solve_discrete_lyapunov(total, noise)
        assert np.max(abs(sigma - total @ sigma @ total.T - noise)) < 1e-15
        # Mode emittances are the moduli of the eigenvalues of J Sigma.
        j = np.kron(np.eye(2), [[0.0, 1.0], [-1.0, 0.0]])
        emit = np.sort(abs(np.linalg.eigvals(j @ sigma)))[::2]
        return partitions[1], emit[0]


    def ring(kind, edge, e, harmonic, voltage, face_maps=False):
        elements = []
        for _ in range(NC):
            for what, length, strength in CELL:
                if what == "q":
                    elements.append(xt.Quadrupole(length=length, k1=strength))
                elif what == "d":
                    elements.append(xt.Drift(length=length))
                else:
                    b = bend(kind, edge, e, LB, TH)
                    if face_maps:
                        # Diagnostic: the derived face loss term as zero-length linear maps on ptau.
                        m1 = np.eye(6)
                        m1[5, 0] = BETA0 * A * H**2 * np.tan(e)
                        elements += [xt.FirstOrderTaylorMap(m1=m1), b, xt.FirstOrderTaylorMap(m1=m1)]
                    else:
                        elements.append(b)
        elements.append(xt.Cavity(voltage=voltage, frequency=harmonic * BETA0 * c / CIRC, phase=np.pi))
        line = build(elements, 128)
        line.configure_radiation(model="mean")
        line.compensate_radiation_energy_loss(verbose=False)
        tw = line.twiss(radiation_analysis=True, chrom=False)
        return float(tw.partition_numbers[0]), float(tw.eq_gemitt_x)


    ER = TH / 2
    for kind, edge, e, harmonic, voltage, face_maps in (
        ("Bend", "default", ER, 1, 1.5e6, False),
        ("Bend", "default", ER, 220, 5e6, False),
        ("Bend", "full", ER, 220, 5e6, False),
        ("RBend", "default", ER, 220, 5e6, False),
        ("Bend", "default", 0.0, 220, 5e6, False),
        ("Bend", "default", ER, 220, 5e6, True),
    ):
        jx, emit = ring(kind, edge, e, harmonic, voltage, face_maps)
        jx_exp, emit_exp = independent(e, harmonic, voltage, True)
        jx_omit, emit_omit = independent(e, harmonic, voltage, False)
        if e == 0 or face_maps:
            # Controls: no face term, or the face term supplied by hand. The agreement is limited by
            # the O(U/E) = 2e-4 energy sawtooth the linear model neglects and by thin slicing.
            assert abs(jx - jx_exp) < 1e-3 and abs(emit / emit_exp - 1) < 2e-3
            label = "sector faces" if e == 0 else "with derived face maps"
        else:
            # Tracking agrees with the model in which the faces do not radiate.
            assert abs(jx - jx_omit) < 1e-3 and abs(emit / emit_omit - 1) < 2e-3
            defects.append(abs(jx - jx_exp) > 0.1 and emit / emit_exp - 1 > 0.1)
            label = "rectangular"
        print(
            f"RING {kind} edge={edge} {label} RF h={harmonic} V={voltage:.3g} V: "
            f"Jx observed={jx:.10g} expected={jx_exp:.10g} (faces omitted {jx_omit:.10g}); "
            f"eq_gemitt_x observed={emit:.10g} expected={emit_exp:.10g} m "
            f"(faces omitted {emit_omit:.10g}) excess={emit / emit_exp - 1:+.4%}"
        )

    kernels()
    print("DEFECT REPRODUCED" if defects and all(defects) else "NOT REPRODUCED")

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
| Found | `claude-opus-5-5` | high | 2026-11-05 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, serial |

