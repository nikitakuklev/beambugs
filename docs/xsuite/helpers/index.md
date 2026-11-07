---
title: "Xsuite shared helpers"
---

# Xsuite shared helpers

These modules support the [Xsuite reproducers](../index.md).
Each entry links all the helpers it needs, including those imported by another helper.
Save the linked helpers in the same directory as the downloaded reproducer scripts.

## `_common.py` { #_common }

[Download _common.py](_common.py)

??? example "Source: _common.py"

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

## `_ibs.py` { #_ibs }

[Download _ibs.py](_ibs.py)

??? example "Source: _ibs.py"

    ```python
    """Periodic LineSegmentMap optics and an independent reference for analytical IBS growth rates.

    The reference integrates the Gaussian collision matrix exactly with Carlson elliptic integrals.
    """

    import warnings
    from typing import TypedDict

    from _prebuilt import CTX, np, xt
    from scipy.constants import c
    from scipy.special import elliprd
    from xfields.ibs import BjorkenMtingwaIBS


    class BeamParams(TypedDict):
        gemitt_x: float
        gemitt_y: float
        sigma_delta: float
        bunch_length: float
        total_beam_intensity: int


    PARAMS: BeamParams = dict(
        gemitt_x=3e-6,
        gemitt_y=0.8e-6,
        sigma_delta=7e-4,
        bunch_length=0.35,
        total_beam_intensity=4_000_000_000,
    )


    def make_line(beta, *, dx=1.7, dpx=0.11, dy=0.0, dpy=0.0, varying=False):
        base = dict(betx=8.0, bety=14.0, alfx=0.45, alfy=-0.25, dx=dx, dpx=dpx, dy=dy, dpy=dpy)
        optics = [base]
        lengths = [59.0]
        if varying:
            optics += [
                dict(betx=17.0, bety=6.0, alfx=-0.3, alfy=0.6, dx=0.8, dpx=-0.07, dy=0.4, dpy=0.035),
                dict(betx=11.0, bety=9.0, alfx=0.1, alfy=-0.2, dx=2.2, dpx=0.03, dy=-0.2, dpy=-0.015),
            ]
            lengths = [13.0, 29.0, 17.0]
        elements = []
        for i, length in enumerate(lengths):
            end = optics[(i + 1) % len(optics)]
            kwargs = {key: [value, end[key]] for key, value in optics[i].items()}
            elements.append(
                xt.LineSegmentMap(length=length, qx=0.12, qy=0.09, longitudinal_mode="frozen", **kwargs)
            )
        line = xt.Line(elements=elements)
        gamma = 1 / np.sqrt(1 - beta**2)
        line.particle_ref = xt.Particles(
            _context=CTX, mass0=xt.PROTON_MASS_EV, p0c=xt.PROTON_MASS_EV * beta * gamma
        )
        line.build_tracker(_context=CTX)
        tw = line.twiss(method="4d")
        for key, value in base.items():
            np.testing.assert_allclose(getattr(tw, key)[0], value, rtol=2e-9, atol=1e-13)
        # Confirm every dispersion column against tracked closed orbits at 18 offsets.
        offsets = np.linspace(-8e-4, 8e-4, 18)
        particles = xt.Particles(
            _context=CTX,
            mass0=xt.PROTON_MASS_EV,
            p0c=line.particle_ref.p0c[0],
            delta=offsets,
            x=dx * offsets,
            px=dpx * offsets,
            y=dy * offsets,
            py=dpy * offsets,
        )
        max_orbit_error = 0.0
        for element, endpoint in zip(elements, optics[1:] + optics[:1], strict=True):
            element.track(particles)
            for coord, disp in [("x", "dx"), ("px", "dpx"), ("y", "dy"), ("py", "dpy")]:
                err = float(np.max(abs(getattr(particles, coord) - endpoint[disp] * offsets)))
                max_orbit_error = max(max_orbit_error, err)
                assert err < 2e-15
        return tw, max_orbit_error


    def local_oracle(tw, row, *, dispersion_scale=1.0, params: BeamParams = PARAMS):
        """Integrate the matrix collision kernel analytically after diagonalizing L.

        For eigenvalues l_i, put c_i=1/l_i and R_i=c_i RD(c_j,c_k,c_i).
        Substituting lambda=1/t in the matrix integral gives
        J_i = 2 (sum(R)-3 R_i)/(3 sqrt(det L)). The amplitude rate is
        C/2 tr(L_u J). No B&M polynomial coefficients or quadrature are used.
        """
        g = tw.gamma0
        parts = []
        for axis, emit, trans in [("x", params["gemitt_x"], 0), ("y", params["gemitt_y"], 2)]:
            beta = tw["bet" + axis][row]
            alpha = tw["alf" + axis][row]
            disp = dispersion_scale * tw["d" + axis][row]
            prime = dispersion_scale * tw["dp" + axis][row]
            phi = prime + alpha * disp / beta
            block = np.zeros((3, 3))
            block[trans, trans] = beta / emit
            block[1, 1] = g * g * (disp * disp / beta + beta * phi * phi) / emit
            block[trans, 1] = block[1, trans] = -g * beta * phi / emit
            parts.append(block)
        parts.append(np.diag([0.0, g * g / params["sigma_delta"] ** 2, 0.0]))
        matrix = sum(parts)
        vals, vecs = np.linalg.eigh(matrix)
        assert np.min(vals) > 0
        inv = 1 / vals
        rs = np.array([inv[i] * elliprd(inv[(i + 1) % 3], inv[(i + 2) % 3], inv[i]) for i in range(3)])
        js = 2 * (sum(rs) - 3 * rs) / (3 * np.sqrt(np.prod(vals)))
        kernel = (vecs * js) @ vecs.T
        contractions = np.array([np.trace(part @ kernel) for part in parts])
        return contractions, vals


    def oracle(tw, *, dispersion_scale=1.0, params: BeamParams = PARAMS, log=None, left_average=False):
        if log is None:
            log = BjorkenMtingwaIBS(tw).coulomb_log(**params)
        r0 = tw.particle_on_co.get_classical_particle_radius0()
        prefactor = (
            params["total_beam_intensity"]
            * r0
            * r0
            * c
            * log
            / (
                16
                * np.pi
                * tw.beta0**3
                * tw.gamma0**4
                * params["gemitt_x"]
                * params["gemitt_y"]
                * params["sigma_delta"]
                * params["bunch_length"]
            )
        )
        rows = [
            local_oracle(tw, i, dispersion_scale=dispersion_scale, params=params)[0]
            for i in range(len(tw.s))
        ]
        if left_average:
            return (
                prefactor
                * np.sum(np.array(rows[:-1]) * np.diff(tw.s)[:, None], axis=0)
                / tw.line_length
            )
        return prefactor * np.trapezoid(rows, tw.s, axis=0) / tw.line_length


    def run_case(label, beta, *, params: BeamParams = PARAMS, **optics):
        tw, orbit_error = make_line(beta, **optics)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = np.array(tw.get_ibs_growth_rates(formalism="B&M", **params).as_tuple())
        expected = oracle(tw, params=params)
        scaled = oracle(tw, dispersion_scale=tw.beta0, params=params)
        logs = BjorkenMtingwaIBS(tw).coulomb_log(**params)
        rec = dict(
            label=label,
            beta=beta,
            optics=optics,
            parameters=params,
            actual=actual.tolist(),
            oracle=expected.tolist(),
            beta_scaled_oracle=scaled.tolist(),
            relative_error=((actual - expected) / np.maximum(abs(expected), 1e-25)).tolist(),
            orbit_max_abs_error=orbit_error,
            coulomb_log=logs,
            warnings=sorted({str(w.message) for w in caught}),
        )
        rec["twiss"] = {
            name: tw[name].tolist()
            for name in ("s", "betx", "bety", "alfx", "alfy", "dx", "dpx", "dy", "dpy")
        }
        if np.all(np.isfinite(actual)):
            coefficient_error = 0.0
            bm = BjorkenMtingwaIBS(tw)
            args = (params["gemitt_x"], params["gemitt_y"], params["sigma_delta"])
            aa, bb, cc = (np.asarray(getattr(bm, "_" + key)(*args)) for key in ("a", "b", "c"))
            plane_a = [np.asarray(getattr(bm, "_a" + key)(*args)) for key in ("x", "y", "z")]
            plane_b = [np.asarray(getattr(bm, "_b" + key)(*args)) for key in ("x", "y", "z")]
            _, bracket_x, bracket_y, bracket_z = bm._constants(**params)
            for row in range(len(tw.s)):
                # Rebuild quadratic forms independently; compare the integrand pointwise.
                g = tw.gamma0
                parts = []
                for axis, emit, trans in [("x", args[0], 0), ("y", args[1], 2)]:
                    beta_u = tw["bet" + axis][row]
                    disp = tw.beta0 * tw["d" + axis][row]
                    phi = tw.beta0 * tw["dp" + axis][row] + tw["alf" + axis][row] * disp / beta_u
                    part = np.zeros((3, 3))
                    part[trans, trans] = beta_u / emit
                    part[trans, 1] = part[1, trans] = -g * beta_u * phi / emit
                    part[1, 1] = g * g * (disp * disp / beta_u + beta_u * phi * phi) / emit
                    parts.append(part)
                parts.append(np.diag([0.0, g * g / args[2] ** 2, 0.0]))
                for lam in [1e3, 1e6, 1e9]:
                    total = sum(parts)
                    matrix = total + lam * np.eye(3)
                    inverse = np.linalg.inv(matrix)
                    # The isotropic part cancels in the trace contraction. Evaluate
                    # inv(M)-I/(lambda+mu) via its resolvent identity to avoid
                    # subtracting nearly equal traces at large lambda.
                    mu = np.trace(total) / 3
                    anisotropic_inverse = -(total - mu * np.eye(3)) @ inverse / (lam + mu)
                    for u, bracket in enumerate(
                        (np.asarray(bracket_x)[row], np.asarray(bracket_y)[row], bracket_z)
                    ):
                        poly = (
                            bracket
                            * np.sqrt(lam)
                            * (plane_a[u][row] * lam + plane_b[u][row])
                            / (lam**3 + aa[row] * lam**2 + bb[row] * lam + cc[row]) ** 1.5
                        )
                        tensor = (
                            np.sqrt(lam)
                            * (
                                np.trace(parts[u]) * np.trace(anisotropic_inverse)
                                - 3 * np.trace(parts[u] @ anisotropic_inverse)
                            )
                            / np.sqrt(np.linalg.det(matrix))
                        )
                        coefficient_error = max(
                            coefficient_error, float(abs(poly - tensor) / max(abs(tensor), 1e-100))
                        )
                        np.testing.assert_allclose(poly, tensor, rtol=2e-12, atol=1e-25)
            rec["scaled_integrand_max_relative_error"] = coefficient_error
        if not optics.get("varying", False) and optics.get("dy", 0) == 0 and optics.get("dpy", 0) == 0:
            nag = np.array(tw.get_ibs_growth_rates(formalism="Nagaitsev", **params).as_tuple())
            # Nagaitsev uses left-endpoint averaging; B&M linearly interpolates the rate.
            nag_oracle = oracle(tw, params=params, left_average=True)
            np.testing.assert_allclose(nag, nag_oracle, rtol=3e-12, atol=2e-17)
            rec["nagaitsev"] = nag.tolist()
            rec["nagaitsev_left_average_oracle"] = nag_oracle.tolist()
        print("CASE", rec)
        print(f"RATES {label} actual={actual.tolist()} expected={expected.tolist()}")
        print(f"ORBIT {label} max_abs_error={orbit_error:.17g}")
        return rec, tw

    ```

## `_madx_io.py` { #_madx_io }

[Download _madx_io.py](_madx_io.py)

??? example "Source: _madx_io.py"

    ```python
    """Independent MAD-X parsing, PTC tracking and MAD-NG 4D and 6D tracking for deterministic interchange reproducers.

    MAD-X and MAD-NG run in children inheriting CPU affinity and thread limits. Their
    scratch files live beside this helper and are removed when the service exits.
    """

    from importlib.metadata import version
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import _prebuilt  # noqa: F401 -- Set numerical threads before cpymad imports NumPy.
    from cpymad.madx import Madx


    def consume(source, sequence="testseq"):
        with (
            TemporaryDirectory(prefix="madx-io-", dir=Path(__file__).resolve().parent) as scratch,
            Madx(stdout=False, cwd=scratch) as mad,
        ):
            try:
                mad.input(source)
                mad.beam(particle="proton", pc=3)
                mad.use(sequence=sequence)
                seq = mad.sequence[sequence]
                rows = [
                    (
                        e.name,
                        float(e.at),
                        float(e.l),
                        float(e.k1) if e.base_type.name == "quadrupole" else 0.0,
                    )
                    for e in seq.elements
                    if e.name not in (sequence + "$start", sequence + "$end")
                ]
                result = {
                    "status": "accepted",
                    "length": float(seq.length),
                    "elements": rows,
                    "expanded_names": [e.name for e in seq.expanded_elements],
                }
                print(f"CONSUMER cpymad={version('cpymad')} MAD-X={mad.version}")
                return result
            except RuntimeError as exc:
                return {"status": "RuntimeError", "error": str(exc)}


    def ptc_track(source, initial, pc, sequence="testseq"):
        """Parse `source`, read back each magnet's (name, L, KNL, KSL) and track one
        ray (x, px, y, py) through PTC (model 2, method 6, 20 steps, 4D, not exact)."""
        x, px, y, py = initial
        with (
            TemporaryDirectory(prefix="madx-io-", dir=Path(__file__).resolve().parent) as scratch,
            Madx(stdout=False, cwd=scratch) as mad,
        ):
            mad.input(source)
            mad.beam(particle="proton", pc=pc)
            mad.use(sequence=sequence)
            magnets = [
                (e.name, float(e.l), [float(v) for v in e.knl], [float(v) for v in e.ksl])
                for e in mad.sequence[sequence].elements
                if e.base_type.name not in ("marker", "drift")
            ]
            mad.input(
                "ptc_create_universe; "
                "ptc_create_layout, model=2, method=6, nst=20, exact=false; "
                f"ptc_start, x={x:.17g}, px={px:.17g}, y={y:.17g}, py={py:.17g}; "
                "ptc_track, icase=4, onetable=true, dump=false; ptc_track_end; ptc_end;"
            )
            table = mad.table.trackone
            end = [float(table.x[-1]), float(table.px[-1]), float(table.y[-1]), float(table.py[-1])]
            print(f"CONSUMER cpymad={version('cpymad')} MAD-X={mad.version} PTC s_end={float(table.s[-1])!r}")
            return magnets, end


    def madng_track(line, initial):
        """Load `line` into MAD-NG through the public `Line.to_madng` and track one ray
        (x, px, y, py) with t = pt = 0 (method 6, 20 slices per element). Returns the
        exit (x, px, y, py) and the length of the sequence MAD-NG built."""
        x, px, y, py = initial
        with TemporaryDirectory(prefix="madng-io-", dir=Path(__file__).resolve().parent) as scratch:
            mng = line.to_madng(sequence_name="seq", temp_fname=str(Path(scratch) / "line"),
                                stdout=str(Path(scratch) / "stdout.txt"))
            mng.send(f"""
                local _, mflw = track {{ sequence=seq, method=6, nslice=20, save=false,
                    X0={{ x={x:.17g}, px={px:.17g}, y={y:.17g}, py={py:.17g}, t=0, pt=0 }} }}
                py:send(MAD.vector {{ mflw[1].x, mflw[1].px, mflw[1].y, mflw[1].py, seq.l }})
                py:send(MAD.env.version)
            """)
            values = [float(v) for v in mng.recv().ravel()]
            ng_version = mng.recv()
            mng.close()
        print(f"CONSUMER pymadng={version('pymadng')} MAD-NG={ng_version} sequence length={values[4]!r}")
        return values[:4], values[4]


    def madng_track_6d(line, initial):
        """Like `madng_track`, for a full ray (x, px, y, py, t, pt). MAD-NG's pt is the
        energy deviation over the reference p0c, so an energy gain is p0c * pt. Returns
        the exit (x, px, y, py, t, pt) and the length of the sequence MAD-NG built."""
        x, px, y, py, t, pt = initial
        with TemporaryDirectory(prefix="madng-io-", dir=Path(__file__).resolve().parent) as scratch:
            mng = line.to_madng(sequence_name="seq", temp_fname=str(Path(scratch) / "line"),
                                stdout=str(Path(scratch) / "stdout.txt"))
            mng.send(f"""
                local _, mflw = track {{ sequence=seq, method=6, nslice=20, save=false,
                    X0={{ x={x:.17g}, px={px:.17g}, y={y:.17g}, py={py:.17g}, t={t:.17g}, pt={pt:.17g} }} }}
                py:send(MAD.vector {{ mflw[1].x, mflw[1].px, mflw[1].y, mflw[1].py, mflw[1].t, mflw[1].pt, seq.l }})
                py:send(MAD.env.version)
            """)
            values = [float(v) for v in mng.recv().ravel()]
            ng_version = mng.recv()
            mng.close()
        print(f"CONSUMER pymadng={version('pymadng')} MAD-NG={ng_version} sequence length={values[6]!r}")
        return values[:6], values[6]

    ```

## `_matching.py` { #_matching }

[Download _matching.py](_matching.py)

??? example "Source: _matching.py"

    ```python
    """Tracking-free matching setup shared by the step-limit and knob reproducers.

    `ReadKnobs` is an action that returns the current values of named line variables, so targets can be
    set on the variables themselves. `optimizer` builds a public `Line.match()` (not yet solved) on an
    empty line without a tracking kernel, with given starting values, goals, maximum steps, optional
    weights and variable order.
    """

    import xdeps as xd
    from _prebuilt import xt


    class ReadKnobs(xd.Action):
        def __init__(self, line, names):
            self.line = line
            self.names = names

        def run(self):
            return {name: self.line[name] for name in self.names}


    def optimizer(initial, goals, maxima, weights=None, order=None):
        line = xt.Line(elements=[])
        line.build_tracker(compile=False)  # Actions read variables; no tracking kernel is needed.
        for name, value in initial.items():
            line[name] = value
        names = tuple(initial)
        action = ReadKnobs(line, names)
        opt = line.match(
            solve=False,
            verbose=False,
            show_call_counter=False,
            vary=[
                xt.Vary(
                    name,
                    step=1e-4,
                    max_step=maxima[name],
                    weight=1.0 if weights is None else weights[name],
                )
                for name in (names if order is None else order)
            ],
            targets=[xt.Target(name, goals[name], tol=1e-11, action=action) for name in names],
        )
        return line, opt

    ```

## `_prebuilt.py` { #_prebuilt }

[Download _prebuilt.py](_prebuilt.py)

??? example "Source: _prebuilt.py"

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

## `_reference_momentum.py` { #_reference_momentum }

[Download _reference_momentum.py](_reference_momentum.py)

??? example "Source: _reference_momentum.py"

    ```python
    """Analytic reference-change fixtures for transfer, covariance and W checks.

    The reference species is a proton, with canonical momenta normalized to local
    p0c. Conventional updates preserve zeta/beta0, rather than absolute time.
    """

    from _prebuilt import CTX, np, xt

    P0 = 650e6
    MASS = xt.PROTON_MASS_EV
    W0 = np.eye(6)
    for j, beta, alpha in ((0, 2.7, -0.4), (2, 1.6, 0.2), (4, 3.2, 0.0)):
        W0[j : j + 2, j : j + 2] = [[np.sqrt(beta), 0], [-alpha / np.sqrt(beta), 1 / np.sqrt(beta)]]
    NEMITT = np.array([1.7e-6, 0.9e-6])
    # Canonical geometric emittance is epsilon_n * mass0 / p0c in eV units.
    SIGMA0 = W0[:4, :4] @ np.diag(np.repeat(NEMITT * MASS / P0, 2)) @ W0[:4, :4].T


    def cases():
        """Yield native changes and their exact canonical linear transport."""
        energy = np.hypot(P0, MASS)
        for kind in ("absolute", "increment", "segment", "accelerating_segment", "cavity"):
            for ratio in (1.0, 1.8, 0.8):
                if kind == "cavity" and ratio != 1:
                    continue
                p = xt.Particles(_context=CTX, p0c=P0, mass0=MASS, zeta=0.013)
                if kind == "absolute":
                    element = xt.ReferenceEnergyChange(p0c=P0 * ratio)
                elif kind == "increment":
                    element = xt.ReferenceEnergyIncrease(Delta_p0c=P0 * (ratio - 1))
                elif kind in ("segment", "accelerating_segment"):
                    gain = np.hypot(P0 * ratio, MASS) - energy
                    element = xt.LineSegmentMap(
                        energy_ref_increment=gain,
                        energy_increment=gain if kind == "accelerating_segment" else 0,
                    )
                else:
                    element = xt.Cavity(voltage=100e6, frequency=0, phase=np.pi / 2)
                line = xt.Line(elements={"change": element, "end": xt.Marker()})
                line.particle_ref = p.copy()
                line.build_tracker(_context=CTX)
                t = (P0 * ratio / np.hypot(P0 * ratio, MASS)) / (P0 / energy)
                # Pure rebasing conserves physical energy and transverse momentum.
                expected = np.diag([1.0, 1 / ratio, 1.0, 1 / ratio, t, 1 / (ratio * t)])
                if kind in ("segment", "accelerating_segment"):
                    # LineSegmentMap smooths the transverse normalization equally
                    # between position and momentum, unlike a standalone rebase.
                    expected[:4, :4] = np.eye(4) / np.sqrt(ratio)
                yield kind, ratio, t, line, p, expected


    def twiss_modes(kind):
        # Ordinary cavities suppress the physical energy kick in 4D Twiss.
        return ("6d",) if kind == "cavity" else ("4d", "6d")


    def twiss(line, p, mode):
        init = xt.TwissInit(particle_on_co=p.copy(), W_matrix=W0.copy(), element_name="change")
        return line.twiss(init=init, method=mode, chrom=False)

    ```

## `_spin_ring.py` { #_spin_ring }

[Download _spin_ring.py](_spin_ring.py)

??? example "Source: _spin_ring.py"

    ```python
    """Deterministic spin-Twiss rings and direct one-turn transport.

    The default uses the serial prebuilt-kernel setup of the other reproducers. Explicitly setting
    XSUITE_FORCE_KERNEL_COMPILATION=1 uses the common source-compilation setup,
    so the same decks can check a complete Xtrack head source build.
    """

    import os

    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NPROC", "PJRT_NPROC",
    ):
        os.environ[name] = "1"

    import numpy as np
    import xobjects as xo
    import xtrack as xt
    from _common import report

    _MODE = os.environ.get("XSUITE_FORCE_KERNEL_COMPILATION", "0")
    if _MODE == "1":
        from _common import prepare

        CTX = xo.ContextCpu(omp_num_threads=0)

        def setup():
            prepare()
            xo.settings.force_kernel_compilation = True
            print("KERNEL_MODE forced source compilation; CPU serial, binary64")

        def kernels():
            assert xo.settings.force_kernel_compilation
            print("KERNEL_MODE forced source compilation completed")

    elif _MODE == "0":
        from _prebuilt import CTX, identity, kernels

        def setup():
            identity()

    else:
        raise ValueError("XSUITE_FORCE_KERNEL_COMPILATION must be 0 or 1")

    __all__ = ["G", "components", "kernels", "matrix", "np", "report", "ring", "setup", "track_spin"]
    G = 0.00115965218128


    def ring(tilt=0.0, *, rf=False):
        # Opposite frame rotations conjugate the map: the vertical invariant axis
        # becomes (-sin(tilt), cos(tilt), 0), without changing orbital or spin tunes.
        gamma, nbend = 200.0, 32
        elements = [xt.Rotation(rot_s_rad=tilt)]
        for index in range(nbend):
            elements.extend([
                xt.Multipole(knl=[0, (-1) ** index * 0.12]),
                xt.Bend(
                    length=1.0, angle=2 * np.pi / nbend, k0_from_h=True,
                    model="bend-kick-bend", num_multipole_kicks=1,
                    edge_entry_active=False, edge_exit_active=False,
                ),
            ])
        if rf:
            from scipy.constants import c

            beta = np.sqrt(1 - 1 / gamma**2)
            elements.append(xt.Cavity(
                voltage=2e4, frequency=20 * beta * c / nbend, phase=np.pi,
            ))
        elements.append(xt.Rotation(rot_s_rad=-tilt))
        line = xt.Line(elements=elements)
        line.particle_ref = xt.Particles(
            _context=CTX, mass0=xt.ELECTRON_MASS_EV, q0=-1, gamma0=gamma,
            anomalous_magnetic_moment=G,
        )
        line.build_tracker(_context=CTX)
        return line


    def components(particle):
        return np.array([particle.spin_x[0], particle.spin_y[0], particle.spin_z[0]])


    def track_spin(line, spin):
        line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False
        particle = line.particle_ref.copy()
        particle.spin_x, particle.spin_y, particle.spin_z = spin
        line.track(particle)
        assert particle.state[0] == 1
        return components(particle)


    def matrix(line, amplitude=1.0):
        return np.column_stack([
            track_spin(line, basis * amplitude) / amplitude for basis in np.eye(3)
        ])

    ```

## `_wedge_geometry.py` { #_wedge_geometry }

[Download _wedge_geometry.py](_wedge_geometry.py)

??? example "Source: _wedge_geometry.py"

    ```python
    """High-precision local forward circle-plane intersection for dipole-wedge checks."""

    import mpmath as mp


    def geometry(theta, k, initial):
        theta, k = mp.mpf(theta), mp.mpf(k)
        x, px, y, py, zeta = map(mp.mpf, initial)
        pz2 = 1 - px * px - py * py
        assert pz2 >= 0, "The stored particle must have real longitudinal momentum"
        pz = mp.sqrt(pz2)
        ct, st = mp.cos(theta), mp.sin(theta)
        u, v = px * ct + pz * st, pz * ct - px * st
        assert v > 0, "Select the local forward intersection"
        if k == 0:
            flight = x * st / v
            xx, zz = x + px * flight, pz * flight
            expected = [xx * ct + zz * st, u, y + py * flight, py, zeta - flight]
            return expected, (pz2, flight, zz * ct - xx * st, v)
        # The Lorentz orbit meets Z*cos(theta)-X*sin(theta)=0.
        # With t=tan(phi/2), (E+U)*t^2+2*V*t+(E-U)=0.
        # Form E-U directly and rationalize the small root to retain weak-field motion.
        e_minus_u = -k * x * st
        e_plus_u = 2 * u + e_minus_u
        radical = v * v - e_plus_u * e_minus_u
        assert radical > 0
        t = -e_minus_u / (v + mp.sqrt(radical))
        phi = 2 * mp.atan(t)
        c, s = mp.cos(phi), mp.sin(phi)
        px_lab, pz_lab = px * c - pz * s, pz * c + px * s
        xx = x + (-2 * pz * mp.sin(phi / 2) ** 2 + px * s) / k
        zz = (2 * px * mp.sin(phi / 2) ** 2 + pz * s) / k
        flight = phi / k
        # chi=1, delta=0, rvv=1 and zero reference length: Delta zeta=-flight.
        expected = [xx * ct + zz * st, px_lab * ct + pz_lab * st, y + py * flight, py, zeta - flight]
        return expected, (pz2, flight, zz * ct - xx * st, pz_lab * ct - px_lab * st)

    ```
