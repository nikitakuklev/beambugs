---
title: "COL-19: B&M applies an extra beta factor to momentum dispersion"
hide:
  - toc
---

# COL-19 — B&M applies an extra beta factor to momentum dispersion

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xfields main (`3f06c87`) |
| Source | [`BjorkenMtingwaIBS._a`, _analytical.py:898–921](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/ibs/_analytical.py#L898-L921)<br>[`_constants`](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/ibs/_analytical.py#L1418-L1458)<br>[segment map](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/linesegmentmap.h#L44-L81)<br>[off-momentum differences](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/chromatic_functions.py#L124-L135)<br>[manual's collision matrices](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xfields.tex#L2637-L2700) |

`BjorkenMtingwaIBS`, the Bjorken–Mtingwa model of intrabeam-scattering growth rates, multiplies the dispersions and their slopes in both planes by the relativistic reference speed β₀. Xtrack supplies them as derivatives with respect to the relative momentum deviation δ, which the manual's collision matrices use unscaled; integrating the collision matrix with Carlson elliptic integrals gives the correct rates and agrees with Xfields' Nagaitsev model at zero vertical dispersion. For protons at β₀ = 0.45 with beta function βₓ = 8 m, dispersion Dₓ = 1.7 m and slope Dₓ′ = 0.11, the longitudinal cooling rate is −2.890 × 10⁻³ s⁻¹ instead of −2.596 × 10⁻³ s⁻¹, 11.36% too large in magnitude; at β₀ = 0.75 the error is 22.43%.

??? example "Reproducer: COL-19.py"

    [Download COL-19.py](COL-19.py)

    ```python
    """COL-19: B&M applies an extra beta factor to momentum dispersion

    `BjorkenMtingwaIBS`, the Bjorken–Mtingwa model of intrabeam-scattering growth rates, multiplies the
    dispersions and their slopes in both planes by the relativistic reference speed β₀. Xtrack supplies
    them as derivatives with respect to the relative momentum deviation δ, which the manual's collision
    matrices use unscaled; integrating the collision matrix with Carlson elliptic integrals gives the
    correct rates and agrees with Xfields' Nagaitsev model at zero vertical dispersion. For protons at
    β₀ = 0.45 with beta function βₓ = 8 m, dispersion Dₓ = 1.7 m and slope Dₓ′ = 0.11, the longitudinal
    cooling rate is −2.890 × 10⁻³ s⁻¹ instead of −2.596 × 10⁻³ s⁻¹, 11.36% too large in magnitude; at
    β₀ = 0.75 the error is 22.43%.

    The shared helper `_ibs.py` builds each periodic 59 m segment from `LineSegmentMap` elements,
    confirms with tracked off-momentum closed orbits that the Twiss dispersion is dx/dδ, and computes
    the collision-matrix rates independently of Xfields at the same Coulomb logarithm. The script runs
    six cases (β₀ = 0.45, 0.75 and 0.999, plus reversed, vertical, varying and very small dispersion)
    and checks that Xfields' B&M rates equal that calculation with the dispersions multiplied by β₀ and
    differ from it without the factor. With very small dispersion the factor no longer matters, so there
    the rates must also match the unscaled calculation. It requires the installed prebuilt kernels
    (compilation is forbidden).
    """

    from _ibs import PARAMS, run_case
    from _prebuilt import identity, kernels, np

    identity()
    print("ORACLE L=Lx+Ly+Lz; phi=Dprime+alpha*D/beta; H=D^2/beta+beta*phi^2")
    print("Rest-frame momentum delta=gamma*u_parallel; L uses gamma*D_delta")
    print("Eigenvalue/Carlson integral: J_i=2*(sum(R)-3*R_i)/(3*sqrt(det(L)))")
    print("Amplitude rates C/2*trace(L_u*J); shared Coulomb log held fixed")

    for label, beta, optics in [
        ("low", 0.45, {}),
        ("mid", 0.75, dict(dx=-1.7, dpx=-0.11)),
        ("high", 0.999, {}),
        ("vertical", 0.45, dict(dy=0.5, dpy=-0.04)),
        ("varying", 0.75, dict(varying=True)),
        ("tiny", 0.45, dict(dx=1e-7, dpx=0)),
    ]:
        record, _ = run_case(label, beta, params=PARAMS, **optics)
        np.testing.assert_allclose(
            record["actual"], record["beta_scaled_oracle"], rtol=3e-8, atol=2e-13
        )
        if label == "tiny":
            np.testing.assert_allclose(record["actual"], record["oracle"], rtol=1e-9, atol=2e-13)
        else:
            assert np.max(abs(np.array(record["relative_error"]))) > 1e-4

    kernels()
    print("DEFECT REPRODUCED")

    ```

??? example "Helper: _ibs.py"

    [Download _ibs.py](_ibs.py)

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
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | xfields 0.27.5 (`223b4fa134a1`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | xfields 0.27.5 (`223b4fa134a1`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |

