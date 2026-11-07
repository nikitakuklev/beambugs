---
title: "Elegant shared helpers"
---

# Elegant shared helpers

These modules support the [Elegant reproducers](../index.md).
Each entry links all the helpers it needs, including those imported by another helper.
Save the linked helpers in the same directory as the downloaded reproducer scripts.

## `_ccbend.py` { #_ccbend }

[Download _ccbend.py](_ccbend.py)

??? example "Source: _ccbend.py"

    ```python
    """CCBEND fixtures and an independent rigid-magnet ray tracer for forward/backward probes.

    The native runner tracks fixed rays through one CCBEND in a fresh process and returns
    (x, xprime, y, yprime, delta) with delta = p/p0 - 1, the optimizer results printed with
    VERBOSE=1 and the log. The oracle places a straight-pole, rectangular-face magnet as a
    rigid body in laboratory coordinates, intersects rays with its two faces and integrates
    the exact Cartesian equations of motion between them with SciPy's DOP853, independently
    of elegant's splitting integrator, coordinate rotations and alignment transforms.
    """

    import os

    for _name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[_name] = "1"

    import re

    import _elegant as native
    import numpy as np
    from scipy.integrate import solve_ivp

    native.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-probe native deadline.

    P0 = 1500.0  # reference momentum, units of m c
    L = 0.6  # arc length
    ANGLE = 0.04
    K1 = 0.2
    # On-axis, and off-axis rays at 3% above and 2% below reference momentum.
    RAYS = np.array(
        [
            [0, 0, 0, 0, 0],
            [5e-4, 2e-4, -3e-4, 1e-4, 0.03],
            [-4e-4, -1e-4, 2e-4, -2e-4, -0.02],
        ],
        dtype=float,
    )
    DEFAULT = None  # parameter value that leaves the element default in place


    def lattice(**params):
        """One CCBEND. Unless overridden, optimization, reference correction, edge kicks and
        radiation are off; FRINGEMODEL=0 with EDGE_ORDER=0 applies no edge kick
        (verticalRbendFringe in ccbend.c) but keeps model 0's centering objective."""
        values = dict(
            L=L,
            ANGLE=ANGLE,
            K1=K1,
            N_SLICES=64,
            INTEGRATION_ORDER=6,
            OPTIMIZE_FSE=0,
            OPTIMIZE_DX=0,
            REFERENCE_CORRECTION=0,
            FRINGEMODEL=0,
            EDGE_ORDER=0,
            SYNCH_RAD=0,
            ISR=0,
            VERBOSE=1,
        )
        for key, value in params.items():
            if value is DEFAULT:
                values.pop(key)
            else:
                values[key] = value
        body = ",".join(
            f"{k}={v}" if isinstance(v, int) else f"{k}={v:.17g}" for k, v in values.items()
        )
        return f"B: CCBEND,{body}\nBL: LINE=(B)\n"


    def track(rays, *, reverse=False, mode="serial", **params):
        """Track rays through one CCBEND, natively backward when reverse is set.

        mode is "serial" or "mpi" (three-rank Pelegant, which needs at least three rays).
        Returns (coordinates, fits): fits lists the (FSE, dx) offsets that the element's
        optimizer reports."""
        q = np.array(rays, dtype=float, ndmin=2)
        beam = dict(zip(("x", "xp", "y", "yp"), q[:, :4].T, strict=True))
        beam.update(t=np.zeros(len(q)), p=P0 * (1 + q[:, 4]), particleID=np.arange(1, len(q) + 1))
        commands = (
            "&run_setup lattice=lattice.lte, use_beamline=BL, "
            f"p_central={P0:.17g}, back_tracking={int(reverse)}, default_order=1, output=out.sdds &end\n"
            "&run_control n_steps=1 &end\n&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
        )
        result = native.run(
            {"lattice.lte": lattice(**params), "run.ele": commands}, mode, {"beam.sdds": beam}
        )
        data = native.particles(result["out.sdds"])
        if not np.array_equal(data["particleID"], beam["particleID"]):
            raise RuntimeError("Particle IDs changed or particles were lost")
        out = np.column_stack([data[n] for n in ("x", "xp", "y", "yp", "p")])
        out[:, 4] = out[:, 4] / P0 - 1
        fits = [
            (float(f), float(d))
            for f, d in re.findall(r"optimized: FSE=\s*(\S+), dx=\s*(\S+), accuracy", result["log"])
        ]
        return out, fits


    def roundtrip(forward_params, backward_params=None):
        """Forward pass, then native backtracking of its output; returns (forward, backward, fits)."""
        fwd, fits_f = track(RAYS, **forward_params)
        back, fits_b = track(
            fwd, reverse=True, **(forward_params if backward_params is None else backward_params)
        )
        return fwd, back, (fits_f, fits_b)


    def error(a, b):
        """Largest transverse difference (positions in m, slopes in rad)."""
        return float(np.max(np.abs(np.asarray(a)[:, :4] - np.asarray(b)[:, :4])))


    def close(a, b, atol, label):
        value = error(a, b)
        if not np.isfinite(value) or value > atol:
            raise RuntimeError(f"{label}: error={value:.17g}, tolerance={atol:g}")
        return value


    def _rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


    def _ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


    def _rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


    class Magnet:
        """A rigid straight-pole magnet with rectangular faces, placed in laboratory coordinates.

        Laboratory frame: origin at the nominal entrance point, z along the incoming reference,
        x horizontal. A bend of signed angle theta deflects the reference toward -x for theta > 0.
        The magnet axis is the chord, rotated from z by -theta/2 about y, so the reference enters
        at theta/2 to it; the chord length is l = 2 (L/theta) sin(theta/2). The outgoing
        reference frame has its origin at the chord end and is rotated by -theta about y.

        Errors follow elegant's exact centre-referenced convention (MALIGN_METHOD=4): in the
        nominal magnet frame the magnet is rotated about its centre by R = Rx(pitch) Ry(yaw)
        Rz(roll), the rotation matrix of malign_mat.c, then displaced by d = (dx, dy, dz).
        A positive-angle bend fixes the signs; a negative-angle bend uses the same laboratory
        axes, as elegant does for DX and DY.

        Field, normalized by the reference rigidity, in magnet coordinates (x, y, z):
        By = (1 + fse) (theta/L + k1 x), Bx = (1 + fse) k1 y. Along the axis z a positive
        particle with momentum p (units of p0) obeys dpx/dz = -By, dpy/dz = Bx,
        dx/dz = px/pz, dy/dz = py/pz with pz = sqrt(|p|^2 - px^2 - py^2).
        """

        def __init__(
            self, angle=ANGLE, k1=K1, fse=0.0, d=(0.0, 0.0, 0.0), pitch=0.0, yaw=0.0, roll=0.0
        ):
            self.b0 = (1 + fse) * angle / L
            self.k1 = (1 + fse) * k1
            self.chord = 2 * (L / angle) * np.sin(angle / 2)
            nominal = _ry(-angle / 2)
            centre = np.array([0, 0, self.chord / 2])
            rot = _rx(pitch) @ _ry(yaw) @ _rz(roll)
            self.axes = nominal @ rot  # columns: magnet x, y, z in the laboratory
            self.origin = nominal @ (np.asarray(d, dtype=float) + centre - rot @ centre)
            self.exit_axes = _ry(-angle)
            self.exit_origin = nominal @ np.array([0, 0, self.chord])

        def _rhs(self, _z, s, momentum):
            x, y, px, py = s
            pz = np.sqrt(momentum**2 - px * px - py * py)
            return [px / pz, py / pz, -(self.b0 + self.k1 * x), self.k1 * y]

        def _body(self, r, p, z0, z1):
            """Carry a laboratory ray on the face z_magnet = z0 to the face z_magnet = z1."""
            local_r, local_p = self.axes.T @ (r - self.origin), self.axes.T @ p
            momentum = float(np.linalg.norm(p))
            sol = solve_ivp(
                self._rhs,
                (z0, z1),
                [local_r[0], local_r[1], local_p[0], local_p[1]],
                method="DOP853",
                rtol=2.3e-14,
                atol=1e-18,
                args=(momentum,),
            )
            if not sol.success:
                raise RuntimeError(sol.message)
            x, y, px, py = sol.y[:, -1]
            pz = np.sqrt(momentum**2 - px * px - py * py)
            return self.origin + self.axes @ [x, y, z1], self.axes @ [px, py, pz]

        @staticmethod
        def _to_plane(r, p, origin, normal):
            return r + p * (np.dot(origin - r, normal) / np.dot(p, normal))

        @staticmethod
        def _ray(q, origin, axes):
            x, xp, y, yp, delta = q
            direction = axes @ np.array([xp, yp, 1.0])
            return origin + axes @ np.array([x, y, 0.0]), (1 + delta) * direction / np.linalg.norm(
                direction
            )

        @staticmethod
        def _coords(r, p, origin, axes):
            local_r, local_p = axes.T @ (r - origin), axes.T @ p
            return np.array(
                [
                    local_r[0],
                    local_p[0] / local_p[2],
                    local_r[1],
                    local_p[1] / local_p[2],
                    np.linalg.norm(p) - 1,
                ]
            )

        def forward(self, rays):
            out = []
            for q in np.atleast_2d(rays):
                r, p = self._ray(q, np.zeros(3), np.eye(3))
                r = self._to_plane(r, p, self.origin, self.axes[:, 2])
                r, p = self._body(r, p, 0.0, self.chord)
                r = self._to_plane(r, p, self.exit_origin, self.exit_axes[:, 2])
                out.append(self._coords(r, p, self.exit_origin, self.exit_axes))
            return np.array(out)

        def inverse(self, rays):
            """Exact inverse of forward: retrace from the outgoing frame to the incoming frame."""
            out = []
            for q in np.atleast_2d(rays):
                r, p = self._ray(q, self.exit_origin, self.exit_axes)
                r = self._to_plane(r, p, self.origin + self.chord * self.axes[:, 2], self.axes[:, 2])
                r, p = self._body(r, p, self.chord, 0.0)
                r = self._to_plane(r, p, np.zeros(3), np.array([0.0, 0.0, 1.0]))
                out.append(self._coords(r, p, np.zeros(3), np.eye(3)))
            return np.array(out)

    ```

## `_csbend_fse.py` { #_csbend_fse }

[Download _csbend_fse.py](_csbend_fse.py)

??? example "Source: _csbend_fse.py"

    ```python
    """Shared case for the CSBEND FSE_CORRECTION reproducers: an asymmetric pure dipole with soft edges.

    `track` runs one on-axis electron through the bend with optional twiss_output (which computes the
    element matrix) and returns its exit coordinates, physical path and the FSE value that Elegant
    printed when it fitted the correction (None when it did not fit).
    """

    from __future__ import annotations

    import re
    import sys

    import numpy as np
    from _elegant import particles, run

    C = 299792458.0  # speed of light (m/s)
    P0 = 6000.0  # reference momentum in m_e c
    LENGTH = 0.75
    # Unequal edge angles make the edge-induced exit slope nonzero, so the correction has work to do.
    BEND = (
        f"CSBEND,L={LENGTH!r},ANGLE=0.09,E1=0.02,E2=-0.015,HGAP=0.025,FINT=0.5,EDGE_ORDER=1,"
        "N_SLICES=80,INTEGRATION_ORDER=6,SYNCH_RAD=0,ISR=0"
    )
    N = 12  # identical particles, so that every MPI worker tracks some


    def track(mode: str, model: int, params: dict, twiss: bool) -> tuple[dict, float | None]:
        """Exit coordinates (x, xp, y, yp, path) of the on-axis electron and the fitted FSE, if any."""
        extra = "".join(f",{k}={v!r}" for k, v in params.items())
        lattice = f"B: {BEND},EDGE1_EFFECTS={model},EDGE2_EFFECTS={model}{extra}\nEND: MARK\nBL: LINE=(B,END)\n"
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds &end\n"
            + ("&twiss_output filename=out.twi, matched=0, beta_x=2, beta_y=3 &end\n" if twiss else "")
            + "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": {"p": np.full(N, P0)}}
        )
        out = particles(result["out.sdds"])
        if len(out["p"]) != N or any(np.ptp(out[c]) > 0 for c in ("x", "xp", "y", "yp", "t", "p")):
            sys.exit(f"{mode}: particles lost or not identical: invalid")
        p = out["p"][0]
        coords = {c: float(out[c][0]) for c in ("x", "xp", "y", "yp")}
        coords["path"] = float(out["t"][0] * C * p / np.sqrt(1 + p * p))
        fits = re.findall(r"FSE optimized to (\S+)", result["log"])
        return coords, (float(fits[-1]) if fits else None)

    ```

## `_csr.py` { #_csr }

[Download _csr.py](_csr.py)

??? example "Source: _csr.py"

    ```python
    """Shared setup for the CSRCSBEND and CSRDRIFT reproducers: Elegant's physical constants,
    deterministic electron beams, one-run tracking with the wake files Elegant exports, the analytic
    steady-state CSR wake of a Gaussian bunch, and a serial Valgrind runner for the memory defects.

    Lengths are in m, momenta p in units of m_e c, and the longitudinal coordinate z = c t of a
    particle grows toward the tail of the bunch, as in Elegant's wake output column `s`.
    """

    from __future__ import annotations

    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np
    from _elegant import RunFailure, particles, run, write_beam
    from scipy.integrate import quad
    from scipy.special import ndtri

    # SDDS constants.h, the values Elegant computes with: the comparisons below are then not limited
    # by revisions of the CODATA constants.
    C = 2.99792458e8
    E_CHARGE = 1.60217733e-19
    M_E = 9.1093897e-31
    EPSILON_0 = 1e7 / (4 * np.pi * C**2)
    R_E = E_CHARGE**2 / (4 * np.pi * EPSILON_0 * M_E * C**2)
    TIMEOUT = 600  # seconds per Valgrind run


    def gaussian_beam(count: int, sigma_ct: float, p0: float, chirp: float = 0.0, x0: float = 0.0):
        """Gaussian quantiles in z = c t with rms sigma_ct, all at p0 (1 + chirp z) and x = x0.

        Quantiles (i + 1/2)/count place the particles deterministically and symmetrically about z = 0.
        """
        z = sigma_ct * ndtri((np.arange(count) + 0.5) / count) if count > 1 else np.zeros(1)
        return {
            "x": np.full(count, x0),
            "t": z / C,
            "p": p0 * (1 + chirp * z),
            "particleID": np.arange(1, count + 1),
        }


    def commands(p0: float, steps: int = 1, passes: int = 1, seed: int = 0) -> str:
        """Command file: track beam.sdds through BL, final coordinates in out.sdds."""
        random = f",random_number_seed={seed}" if seed else ""
        return (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0!r},output=out.sdds{random} &end\n"
            f"&run_control n_steps={steps},n_passes={passes} &end\n"
            "&sdds_beam input=beam.sdds,reuse_bunch=1 &end\n"
            "&track &end\n"
        )


    def track(elements: str, line: str, beam, p0: float, outputs=("out.sdds",), **kw):
        """Track `beam` through BL: LINE=(line) defined after `elements`; returns run()'s dict."""
        lattice = elements.rstrip("\n") + f"\nBL: LINE=({line})\n"
        return run(
            {"lattice.lte": lattice, "run.ele": commands(p0, **kw)},
            beams={"beam.sdds": beam},
            outputs=outputs,
        )


    def status(elements: str, line: str, beam, p0: float, **kw) -> int:
        """Exit status of a plain serial run (0 on success): the memory defects end in a signal."""
        try:
            track(elements, line, beam, p0, outputs=(), **kw)
        except RunFailure as failure:
            return failure.status
        return 0


    def page(sdds, index: int = 0) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Columns and parameters of one page of an SDDS file."""
        columns = {c.name: np.asarray(sdds[index, c.name]) for c in sdds.columns}
        parameters = {p.name: p.data[index] for p in sdds.parameters}
        return columns, parameters


    def final(result: dict) -> dict[str, np.ndarray]:
        """Final particle coordinates of a run, ordered by particle ID."""
        return particles(result["out.sdds"])


    def steady_state_wake(z: np.ndarray, sigma: float, k: float) -> np.ndarray:
        """Steady-state CSR energy change of a Gaussian bunch, per particle, in units of m_e c^2.

        Saldin, Schneidmiller and Yurkov (NIM A 398, 1997) and Stupakov and Emma (LCLS-TN-01-12, 2001):
            Delta gamma(z) = k int_0^inf lambda'(z + u) u^(-1/3) du,
        lambda the line density normalized to 1 and k = 2 N r_e L / (3 rho^2)^(1/3) for N electrons
        in a bend of length L and radius rho. With z = a sigma and u = sigma t^3 the integrable
        endpoint singularity disappears:
            Delta gamma = k sigma^(-4/3) int_0^inf 3 t lambda_1'(a + t^3) dt,
        lambda_1(a) = exp(-a^2/2)/sqrt(2 pi), lambda_1'(a) = -a lambda_1(a); the integrand vanishes
        beyond a + t^3 = 12.
        """

        def one(a: float) -> float:
            upper = max(0.0, 12.0 - a) ** (1 / 3)
            integrand = lambda t: -3 * t * (a + t**3) * np.exp(-0.5 * (a + t**3) ** 2)
            return quad(integrand, 0, upper, epsabs=1e-13, epsrel=1e-12, limit=200)[0]

        values = np.array([one(a) for a in np.asarray(z) / sigma])
        return k * sigma ** (-4 / 3) * values / np.sqrt(2 * np.pi)


    def valgrind(elements: str, line: str, beam, p0: float, **kw) -> tuple[int, str]:
        """Serial Elegant under `valgrind --track-origins=yes`: (exit status, Valgrind and run log)."""
        if shutil.which("valgrind") is None or shutil.which("elegant") is None:
            sys.exit("valgrind and elegant must be on PATH.")
        if "RPN_DEFNS" not in os.environ:
            sys.exit("Set RPN_DEFNS to the defns.rpn file of SDDS (see the elegant installation).")
        env = dict(os.environ, OMP_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="elegant-repro-") as tmp:
            work = Path(tmp)
            (work / "lattice.lte").write_text(elements.rstrip("\n") + f"\nBL: LINE=({line})\n")
            (work / "run.ele").write_text(commands(p0, **kw))
            write_beam(work / "beam.sdds", beam)
            try:
                done = subprocess.run(
                    ["valgrind", "--error-limit=no", "--track-origins=yes", "elegant", "run.ele"],
                    cwd=work,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                sys.exit(f"valgrind elegant exceeded {TIMEOUT} s")
        return done.returncode, done.stdout + done.stderr

    ```

## `_cubic_optics.py` { #_cubic_optics }

[Download _cubic_optics.py](_cubic_optics.py)

??? example "Source: _cubic_optics.py"

    ```python
    """Oracle for the higher-order chromaticities of a cell of cubic-order lens maps.

    Elegant's third-order maps keep each element's 2x2 transverse matrix to second order in the
    relative momentum offset delta (the cubic map). For such truncated maps the tune derivatives
    d^2 nu/d delta^2 and d^3 nu/d delta^3 follow from Cauchy's formula on a circle in the complex delta
    plane, independently of Elegant's trace-derivative formulas: if f is analytic, then
    f^(n)(0)/n! is the mean of f(z)/z^n over the circle |z| = RADIUS.
    """

    from __future__ import annotations

    import math

    import numpy as np

    POINTS, RADIUS = 64, 0.05
    Z = RADIUS * np.exp(2j * np.pi * np.arange(POINTS) / POINTS)


    def taylor(function, order: int = 2) -> list[np.ndarray]:
        """Taylor coefficients c_0 ... c_order of a matrix function of delta, analytic near 0."""
        values = np.array([function(z) for z in Z])
        return [np.mean(values / Z[:, None, None] ** n, axis=0) for n in range(order + 1)]


    def quad_matrix(k, length):
        """2x2 hard-edge quadrupole matrix, analytic in k (k < 0 defocuses); cos z and sin(z)/z are even."""
        z = np.sqrt(k + 0j) * length
        return np.array([[np.cos(z), np.sin(z) / z * length], [-z * np.sin(z) / length, np.cos(z)]])


    def drift_matrix(length):
        return np.array([[1.0, length], [0.0, 1.0]])


    def tune_derivatives(elements: list[list[np.ndarray]], periods: int = 1) -> tuple[float, float]:
        """(d^2 nu/d delta^2, d^3 nu/d delta^3) in cycles, for a cell repeated `periods` times.

        elements: Taylor coefficient lists in beam order. The tune is
        nu = periods * arccos(trace/2) / (2 pi), the trace taken of the cell polynomial map.
        """
        trace = np.empty(POINTS, dtype=complex)
        for i, z in enumerate(Z):
            cell = np.eye(2)
            for coefficients in elements:
                cell = sum(c * z**n for n, c in enumerate(coefficients)) @ cell
            trace[i] = np.trace(cell)
        tune = periods * np.arccos(trace / 2) / (2 * np.pi)
        second, third = (float(math.factorial(n) * np.mean(tune / Z**n).real) for n in (2, 3))
        return second, third


    def fodo_tune_derivatives(plane: str, thin: bool, periods: int = 1) -> tuple[float, float]:
        """Cell D QD D QF (quadrupole strengths 0.6 and -0.54 m^-2, or integrated for thin lenses).

        Thick lenses are 0.5 m long; the focusing sign flips in y. Drifts are 1 m. The gradient scales
        as 1/(1 + delta) (rigidity).
        """
        sign = 1 if plane == "x" else -1
        drift = taylor(lambda z: drift_matrix(1.0))
        elements = []
        for strength in (0.6, -0.54):
            k = sign * strength
            if thin:
                lens = taylor(lambda z, k=k: np.array([[1.0, 0.0], [-k / (1 + z), 1.0]]))
            else:
                lens = taylor(lambda z, k=k: quad_matrix(k / (1 + z), 0.5))
            elements += [lens, drift]
        return tune_derivatives(elements, periods)

    ```

## `_cwiggler.py` { #_cwiggler }

[Download _cwiggler.py](_cwiggler.py)

??? example "Source: _cwiggler.py"

    ```python
    """Shared fixed-particle CWIGGLER fixtures with checked survival and physical constants."""

    import _elegant as native
    import numpy as np
    from _diagnostics import run

    C = 299792458.0
    # CWIGGLER uses gwig.h's electron constants, which differ from SDDS's set.
    FIELD_RIGIDITY = 1.602176462e-19 / (9.10938188e-31 * C)
    native.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-native-process deadline.


    def close(observed, expected, atol, label):
        """Refuse nonfinite values and discrepancies beyond a fixed absolute tolerance."""
        error = float(np.max(np.abs(np.asarray(observed) - expected)))
        if not np.isfinite(error) or error > atol:
            raise RuntimeError(f"{label}: error={error:.17g}, tolerance={atol:g}")


    def track(lattice, p0, beam):
        """Track explicit (x, xp, y, yp, t, p) rows in one fresh serial process."""
        rays = np.asarray(beam, dtype=float)
        if rays.ndim != 2 or rays.shape[1] != 6 or not np.isfinite(rays).all():
            raise ValueError("Expected finite six-coordinate particle rows")
        ids = np.arange(1, len(rays) + 1)
        columns = {name: rays[:, i] for i, name in enumerate(("x", "xp", "y", "yp", "t", "p"))}
        columns["particleID"] = ids
        commands = (
            "&run_setup lattice=lattice.lte,use_beamline=BL,"
            f"p_central={p0:.17g},default_order=2,output=out.sdds,"
            "random_number_seed=271828 &end\n"
            "&run_control n_steps=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            "serial",
            {"beam.sdds": columns},
            ("out.sdds",),
        )
        data = native.particles(result["out.sdds"])
        if not np.array_equal(data["particleID"], ids):
            raise RuntimeError("Particles were lost or their IDs changed")
        return data

    ```

## `_diagnostics.py` { #_diagnostics }

[Download _diagnostics.py](_diagnostics.py)

??? example "Source: _diagnostics.py"

    ```python
    """Shared execution and finite-value checks for HISTOGRAM and WATCH output probes."""

    import math
    import os

    for _name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NPROC", "PJRT_NPROC",
    ):
        os.environ[_name] = "1"

    from _elegant import run as native_run


    def run(files, mode, beams, outputs):
        """Keep serial execution on one inherited core; MPI inherits the launcher's affinity."""
        affinity = os.sched_getaffinity(0)
        try:
            if mode == "serial":
                os.sched_setaffinity(0, {min(affinity)})
            return native_run(files, mode, beams, outputs=outputs)
        finally:
            os.sched_setaffinity(0, affinity)


    def scalar(data, name, *, column=False):
        """Read exactly one finite value, refusing malformed or missing native output."""
        if data.n_pages != 1:
            raise RuntimeError("Expected exactly one SDDS page")
        if column and len(data[0, name]) != 1:
            raise RuntimeError(f"Expected one {name} row")
        value = float(data[0, name][0] if column else data[name][0])
        if not math.isfinite(value):
            raise RuntimeError(f"Expected one finite {name} value")
        return value


    def conversion_defect(observed, expected, faulty, *, rtol):
        """Accept the physical result or the specific faulty conversion; reject other results."""
        if math.isclose(observed, expected, rel_tol=rtol, abs_tol=0):
            return False
        if not math.isclose(observed, faulty, rel_tol=rtol, abs_tol=0):
            raise RuntimeError(f"Unexpected conversion: {observed}, expected {expected} or {faulty}")
        return True

    ```

## `_elegant.py` { #_elegant }

[Download _elegant.py](_elegant.py)

??? example "Source: _elegant.py"

    ```python
    """Shared runner for the Elegant reproducers: write inputs, run elegant or Pelegant, read SDDS output.

    `elegant` must be on PATH, and `Pelegant` with `mpirun` for MPI runs. RPN_DEFNS must name the
    `defns.rpn` file shipped with SDDS; elegant refuses to start without it. MPI runs default to three ranks
    (rank 0 coordinates, two workers track), oversubscribed so that they also run on fewer cores.
    Each run uses a fresh temporary directory and one numerical thread.
    """

    from __future__ import annotations

    import argparse
    import contextlib
    import io
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    from typing import Any

    import numpy as np
    import pysdds

    MPI_RANKS = 3
    TIMEOUT = 600  # seconds per native run


    class RunFailure(RuntimeError):
        """A native run that exited with a nonzero status; `log` is its full stdout and stderr."""

        def __init__(self, program: str, status: int, log: str):
            super().__init__(f"{program} exited with status {status}:\n{log[-3000:]}")
            self.program, self.status, self.log = program, status, log


    def modes(mpi_only: bool = False) -> tuple[str, ...]:
        """Execution modes from the command line: serial, plus 3-rank Pelegant with --mpi.

        An MPI-only defect always runs Pelegant; --mpi is then implied.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--mpi", action="store_true", help="also run 3-rank Pelegant")
        args = parser.parse_args()
        if mpi_only:
            return ("mpi",)
        return ("serial", "mpi") if args.mpi else ("serial",)


    def write_beam(path: Path, beam: Any) -> None:
        """Binary SDDS particle file; Pelegant reads only binary input.

        `beam` is one page or a list of pages. In a page, array values are columns (integer arrays
        such as particleID become SDDS long) and scalar values are parameters (for example
        IDSlotsPerBunch); all pages need the same keys. Columns x, xp, y, yp, t, p default to zero;
        p is in units of m c of the species.
        """
        pages: list[dict[str, Any]] = [beam] if isinstance(beam, dict) else list(beam)
        arrays = [{k: np.asarray(v) for k, v in page.items() if np.ndim(v) == 1} for page in pages]
        counts = [len(next(iter(a.values()))) for a in arrays]
        units = {"x": "m", "xp": "", "y": "m", "yp": "", "t": "s", "p": "m$be$nc"}
        sdds = pysdds.SDDSFile(add_data_nm=True)
        sdds.mode = "binary"
        sdds.n_pages = len(pages)
        for name in dict.fromkeys([*units, *arrays[0]]):
            data = [a.get(name, np.zeros(n)) for a, n in zip(arrays, counts, strict=True)]
            if np.issubdtype(data[0].dtype, np.integer):
                sdds.add_column(name, "long", data=[d.astype(np.int32) for d in data])
            else:
                data = [d.astype(np.float64) for d in data]
                sdds.add_column(name, "double", units=units.get(name, ""), data=data)
        for name, value in pages[0].items():
            if np.ndim(value) != 0:
                continue
            values = [page[name] for page in pages]
            # pysdds checks the exact NumPy scalar type, despite its Python-scalar annotation.
            if isinstance(value, (int, np.integer)):
                data = [np.int32(v) for v in values]
                sdds.add_parameter(name, "long", data=data)  # ty: ignore[invalid-argument-type]
            else:
                sdds.add_parameter(name, "double", data=[np.float64(v) for v in values])
        # pysdds 0.6.0 prints every parameter value while validating; keep the output clean.
        with contextlib.redirect_stdout(io.StringIO()):
            pysdds.write(sdds, path, overwrite=True)


    def run(
        files: dict[str, str],
        mode: str = "serial",
        beams: dict[str, Any] | None = None,
        command: str = "run.ele",
        outputs: tuple[str, ...] = ("out.sdds",),
        *,
        ranks: int = MPI_RANKS,
    ) -> dict:
        """Run one case and return {output name: pysdds file} plus "log" (stdout and stderr).

        files: text inputs by name (lattice, command file, tables); beams: particle pages by file
        name (see write_beam), written as binary SDDS. ranks: Pelegant rank count, including rank 0.
        A nonzero exit status raises RunFailure; a timeout raises RuntimeError.
        """
        if "RPN_DEFNS" not in os.environ:
            sys.exit("Set RPN_DEFNS to the defns.rpn file of SDDS (see the elegant installation).")
        program = "elegant" if mode == "serial" else "Pelegant"
        if shutil.which(program) is None or (mode == "mpi" and shutil.which("mpirun") is None):
            sys.exit(f"{program}{' and mpirun' if mode == 'mpi' else ''} must be on PATH.")
        argv = [program, command]
        if mode == "mpi":
            argv = [
                "mpirun",
                "--oversubscribe",
                "--bind-to",
                "none",
                "-x",
                "RPN_DEFNS",
                "-x",
                "OMP_NUM_THREADS",
                "-n",
                str(ranks),
                *argv,
            ]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="elegant-repro-") as tmp:
            work = Path(tmp)
            for name, text in files.items():
                (work / name).write_text(text)
            for name, columns in (beams or {}).items():
                write_beam(work / name, columns)
            try:
                done = subprocess.run(
                    argv,
                    cwd=work,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"{program} exceeded {TIMEOUT} s") from None
            log = done.stdout + done.stderr
            if done.returncode:
                raise RunFailure(program, done.returncode, log)
            result: dict = {"log": log}
            for name in outputs:
                result[name] = pysdds.read(work / name)
        return result


    def particles(sdds, page: int = 0) -> dict[str, np.ndarray]:
        """Columns of one page, rows ordered by particleID when present (MPI output order varies).

        Raises RuntimeError when a coordinate column (x, xp, y, yp, t, p) holds a nonfinite value: such
        output is an invalid run, not evidence for a numerical discrepancy.
        """
        names = [c.name for c in sdds.columns]
        data = {n: np.asarray(sdds[page, n]) for n in names}
        for n in ("x", "xp", "y", "yp", "t", "p"):
            if n in data and not np.all(np.isfinite(data[n])):
                raise RuntimeError(f"Elegant output column {n} holds nonfinite values")
        if "particleID" in data:
            order = np.argsort(data["particleID"], kind="stable")
            data = {n: v[order] for n, v in data.items()}
        return data


    def verdict(defect: bool) -> None:
        print("DEFECT REPRODUCED" if defect else "NOT REPRODUCED")

    ```

## `_harmonic_rf.py` { #_harmonic_rf }

[Download _harmonic_rf.py](_harmonic_rf.py)

??? example "Source: _harmonic_rf.py"

    ```python
    """Zero-time SHRFDF fixtures with physical transverse-momentum reconstruction."""

    from _slicing import track

    # isort: split
    import numpy as np


    def kick(mode, phase, rays):
        """Return Px/(mc) after one isolated harmonic; input longitudinal coordinates are zero."""
        lattice = (
            f"S:SHRFDF,PERIOD_LENGTH=.03,PERIOD_PHASE={np.pi / 2:.17g},"
            f"V{mode}=240000,PHASE{mode}={phase:.17g}\nBL:LINE=(S)\n"
        )
        raw = track("harmonic", lattice, rays, p0=3200)["raw"]
        return raw[:, 5] * raw[:, 1] / np.sqrt(1 + raw[:, 1] ** 2 + raw[:, 3] ** 2)

    ```

## `_ibs_ring.py` { #_ibs_ring }

[Download _ibs_ring.py](_ibs_ring.py)

??? example "Source: _ibs_ring.py"

    ```python
    """Shared setup for the smooth-IBS reproducers: a closed ring of identical sector bends with one IBSCATTER.

    The ring has 16 identical SBENs (radius 10 m, K1 = -0.003 m^-2) and an IBSCATTER at its end. An
    SBEN of radius R focuses with k_x = 1/R^2 + K1 and k_y = -K1, so a ring of identical bends has
    constant matched beta = 1/sqrt(k), alpha = 0, and horizontal dispersion eta = (1/R)/k_x, eta' = 0.
    The beam is 256 particles with exactly the requested second moments, matched to this optics.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import particles, run

    C = 299792458.0  # SDDS constants.h
    P0 = 800.0  # reference momentum, units of m_e c
    RADIUS, K1 = 10.0, -0.003  # m, 1/m^2
    KX, KY = 1 / RADIUS**2 + K1, -K1
    BETA_X, BETA_Y = 1 / np.sqrt(KX), 1 / np.sqrt(KY)  # m
    ETA_X = (1 / RADIUS) / KX  # m
    EX, EY, SIGMA_DELTA, SIGMA_Z = 2e-9, 3e-10, 1e-5, 0.004  # m, m, 1, m
    COUNT = 256
    # Largest applied fractional change of any plane's variance or spread is 2% at this FACTOR, which
    # keeps the square-root scale factors of the smooth update near 1.
    FACTOR = 0.09465332535958268


    def beam(mean_delta):
        """Matched ring beam with mean momentum offset `mean_delta`; dispersive offset eta*delta in x.

        Orthonormalized normal samples, scaled by sqrt(COUNT), have zero mean and unit covariance, so the
        sample emittances, momentum spread and length equal the requested ones.
        """
        rng = np.random.default_rng(832)
        s = rng.normal(size=(COUNT, 6))
        s -= s.mean(axis=0)
        s = np.linalg.qr(s)[0] * np.sqrt(COUNT)
        delta = mean_delta + SIGMA_DELTA * s[:, 5]
        return {
            "x": np.sqrt(EX * BETA_X) * s[:, 0] + ETA_X * delta,
            "xp": np.sqrt(EX / BETA_X) * s[:, 1],
            "y": np.sqrt(EY * BETA_Y) * s[:, 2],
            "yp": np.sqrt(EY / BETA_Y) * s[:, 3],
            "t": SIGMA_Z / C * s[:, 4],
            "p": P0 * (1 + delta),
        }


    def track(mode, columns, factor, planes=(1, 1, 1)):
        """Track `columns` once around the ring; `planes` are the DO_X, DO_Y, DO_Z switches."""
        lattice = (
            "Q: CHARGE,TOTAL=1e-9\n"
            f"B: SBEN,L={2 * np.pi * RADIUS / 16!r},ANGLE={2 * np.pi / 16!r},K1={K1!r},E1=0,E2=0\n"
            "I: IBSCATTER,ISRING=1,BUNCHED_BEAM_MODE=0,SMOOTH=1,FORCE_MATCHED_TWISS=1,"
            f"FACTOR={factor!r},DO_X={planes[0]},DO_Y={planes[1]},DO_Z={planes[2]}\n"
            "BL: LINE=(Q,16*B,I)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
            "output=out.sdds, default_order=2 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&twiss_output matched=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})
        out = particles(result["out.sdds"])
        if len(out["p"]) != COUNT:
            raise SystemExit(f"{mode}: {COUNT - len(out['p'])} particles lost")
        return out


    def spreads(out):
        """Variances of x, xp, y, yp and rms momentum deviation of the tracked beam."""
        return np.array(
            [np.var(out["x"]), np.var(out["xp"]), np.var(out["y"]), np.var(out["yp"])]
            + [np.std(out["p"])]
        )

    ```

## `_loss_geometry.py` { #_loss_geometry }

[Download _loss_geometry.py](_loss_geometry.py)

??? example "Source: _loss_geometry.py"

    ```python
    """Fixed-ray loss fixtures and independent Cartesian geometry for loss diagnostics."""

    import math
    import os
    import re
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    for _name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[_name] = "1"

    import _elegant as native
    import numpy as np
    import pysdds

    native.TIMEOUT = 290  # ty: ignore[invalid-assignment]


    def close(actual, expected, atol=3e-14):
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise RuntimeError(f"Invalid comparison: {actual}, {expected}")
        if np.any(np.abs(a - b) > atol):
            raise RuntimeError(f"Observed {actual}; expected {expected}, tolerance {atol}")


    def fault(actual, expected, faulty, atol=3e-14):
        if np.all(np.isfinite(actual)) and np.all(np.abs(np.asarray(actual) - expected) <= atol):
            return False
        close(actual, faulty, atol)
        return True


    def world(x, z, heading=0, origin=(0.41, -0.26)):
        return [
            origin[0] + x * math.cos(heading) + z * math.sin(heading),
            origin[1] - x * math.sin(heading) + z * math.cos(heading),
        ]


    def circle(s, x0, xp0, *, length=0.9, angle=0.24):
        """Circle/radial-plane intersection, returning Cartesian x/z, tangent and local x."""
        rho = length / angle
        alpha0 = math.atan(xp0)
        cx, cz = x0 - rho * math.cos(alpha0), rho * math.sin(alpha0)
        phi = s / rho
        wx, wz = cx + rho, cz
        dot = wx * math.cos(phi) + wz * math.sin(phi)
        radius = dot + math.copysign(math.sqrt(rho * rho - wx * wx - wz * wz + dot * dot), rho)
        x, z = radius * math.cos(phi) - rho, radius * math.sin(phi)
        alpha = math.atan2(-(z - cz) / rho, (x - cx) / rho)
        return x, z, alpha, radius - rho


    def projection(lost, *, heading, origin, length=1.0, angle=0.15):
        s, x, xp = (float(lost[k][0]) for k in ("s", "x", "xp"))
        rho, phi = length / angle, angle * s / length
        # The local x lies along the radial normal to the reference circle.
        xz = world(
            rho * (math.cos(phi) - 1) + x * math.cos(phi), (rho + x) * math.sin(phi), heading, origin
        )
        return [*xz, heading - phi + math.atan(xp)]


    def same_physics(first, second):
        for a, b in zip(first, second, strict=True):
            for name in ("x", "xp", "y", "yp", "s", "t", "p", "particleID"):
                if name in a or name in b:
                    if name not in a or name not in b or not np.array_equal(a[name], b[name]):
                        raise RuntimeError(f"Global-output switch changed {name}")


    def ids(lost, out, expected_lost=(1,), expected_out=(2,)):
        close(lost["particleID"], expected_lost, 0)
        close(out["particleID"], expected_out, 0)


    def shown(values):
        return "(" + ", ".join(f"{v:.12g}" for v in values) + ")"


    def track(
        lattice,
        rays,
        *,
        heading=0,
        origin=(0.41, -0.26),
        p0=1700,
        global_coords=True,
        files=None,
        commands="",
        memcheck=False,
    ):
        """Fresh native process; optional Memcheck reports only the identified loss-output error."""
        beam = {k: np.asarray(rays, dtype=float)[:, j] for j, k in enumerate(("x", "xp", "y", "yp"))}
        beam.update(p=np.full(len(rays), p0), particleID=np.arange(1, len(rays) + 1))
        inputs = {
            "line.lte": lattice,
            "run.ele": (
                "&run_setup lattice=line.lte,use_beamline=BL,output=out.sdds,losses=lost.sdds,"
                f"p_central={p0},losses_include_global_coordinates={int(global_coords)},"
                "random_number_seed=78231 &end\n"
                f"&floor_coordinates filename=floor.sdds,X0={origin[0]},Z0={origin[1]},"
                f"theta0={heading:.17g} &end\n" + commands + "&run_control n_steps=1,n_passes=1 &end\n"
                "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
            ),
            **(files or {}),
        }
        if not memcheck:
            result = native.run(inputs, beams={"beam.sdds": beam}, outputs=("lost.sdds", "out.sdds"))
            return native.particles(result["lost.sdds"]), native.particles(result["out.sdds"]), 0

        executable = shutil.which("elegant")
        if not executable or not shutil.which("valgrind") or "RPN_DEFNS" not in os.environ:
            raise RuntimeError("Put elegant and valgrind on PATH and set RPN_DEFNS")
        with tempfile.TemporaryDirectory(prefix="elegant-loss-memcheck-") as tmp:
            work = Path(tmp)
            for name, contents in inputs.items():
                (work / name).write_text(contents)
            native.write_beam(work / "beam.sdds", beam)
            done = subprocess.run(
                [
                    "valgrind",
                    "--vgdb=no",
                    "--track-origins=yes",
                    "--error-exitcode=97",
                    "--log-file=memcheck.log",
                    executable,
                    "run.ele",
                ],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=290,
                check=False,
            )
            log = (work / "memcheck.log").read_text()
            match = re.search(r"ERROR SUMMARY: (\d+) errors from (\d+) contexts", log)
            if match is None:
                raise RuntimeError(f"Missing Memcheck summary:\n{log}")
            errors, contexts = map(int, match.groups())
            if (done.returncode, errors, contexts) not in ((0, 0, 0), (97, 1, 1)):
                raise RuntimeError(
                    f"Unexpected native/Memcheck failure:\n{done.stdout}{done.stderr}\n{log}"
                )
            if errors and (
                "Syscall param write(buf) points to uninitialised byte(s)" not in log
                or "dump_lost_particles" not in log
            ):
                raise RuntimeError(f"Unexpected Memcheck context:\n{log}")
            lost, out = (
                native.particles(pysdds.read(work / name)) for name in ("lost.sdds", "out.sdds")
            )
            return lost, out, errors


    def obstruction(xmin=-1, xmax=1, zmin=0.2, zmax=0.5):
        return (
            "SDDS1\n&parameter name=ZCenter,type=double,units=m,fixed_value=0 &end\n"
            "&parameter name=XCenter,type=double,units=m,fixed_value=0 &end\n"
            "&parameter name=Superperiodicity,type=long,fixed_value=1 &end\n"
            "&column name=Z,type=double,units=m &end\n&column name=X,type=double,units=m &end\n"
            f"&data mode=ascii &end\n5\n{zmin} {xmin}\n{zmax} {xmin}\n{zmax} {xmax}\n"
            f"{zmin} {xmax}\n{zmin} {xmin}\n"
        )

    ```

## `_lsc.py` { #_lsc }

[Download _lsc.py](_lsc.py)

??? example "Source: _lsc.py"

    ```python
    """Shared setup for the longitudinal space-charge reproducers: Elegant's physical constants,
    deterministic round electron beams, one-run tracking, Elegant's periodic FFT kick rebuilt in NumPy,
    and the open-boundary (isolated-bunch) kick of the same uniform-disk model.

    Lengths are in m, momenta p = beta*gamma in units of m_e c, energies in eV, and z = c t grows
    toward the tail of the bunch. Every model here uses p, as Elegant's impedance does (its Bessel
    argument is k r_b / p), so the comparisons isolate the boundary condition and the reference
    momentum rather than the p-versus-gamma choice.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import particles, run
    from scipy.integrate import quad
    from scipy.special import erf, k1, ndtri

    # SDDS constants.h, the values Elegant computes with.
    C = 2.99792458e8
    EPSILON_0 = 1e7 / (4 * np.pi * C**2)
    Z0 = 4e-7 * np.pi * C
    ME_EV = 0.51099906e6
    RADIUS_FACTOR = 1.7  # Elegant's default: r_b = 1.7 (sigma_x + sigma_y)/2


    def round_beam(count: int, sigma_ct: float, sigma_r: float, p: float) -> dict:
        """Gaussian quantiles in z = c t with rms sigma_ct, all at momentum p, zero slopes.

        Transverse positions lie on a golden-angle circle rescaled to rms sigma_r exactly in x and y,
        so Elegant's effective radius is RADIUS_FACTOR * sigma_r. Quantiles (i + 1/2)/count make the
        bunch symmetric about z = 0.
        """
        z = sigma_ct * ndtri((np.arange(count) + 0.5) / count)
        phi = np.arange(count) * np.pi * (np.sqrt(5) - 1)
        x, y = np.cos(phi), np.sin(phi)
        x = (x - x.mean()) / x.std() * sigma_r
        y = (y - y.mean()) / y.std() * sigma_r
        return {
            "x": x,
            "y": y,
            "t": z / C,
            "p": np.full(count, p),
            "particleID": np.arange(1, count + 1),
        }


    def track(elements: str, line: str, beam: dict, p0: float) -> dict[str, np.ndarray]:
        """Serial run of `beam` through BL: LINE=(line); final coordinates ordered by particle ID."""
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        lattice = elements.rstrip("\n") + f"\nBL: LINE=({line})\n"
        result = run({"lattice.lte": lattice, "run.ele": commands}, beams={"beam.sdds": beam})
        out = particles(result["out.sdds"])
        if len(out["p"]) != len(beam["p"]):
            raise SystemExit("particles were lost")
        return out


    def energy(p: np.ndarray | float) -> np.ndarray:
        """Total energy in eV of electrons with momentum p (m_e c)."""
        return ME_EV * np.sqrt(1 + np.asarray(p) ** 2)


    def periodic_kick(z: np.ndarray, p: float, rb: float, charge: float, length: float, bins: int):
        """Elegant's LSC energy kick (eV) at each particle, rebuilt in NumPy, and the grid period (m).

        As in lsc.c: bin width h = (z_max - z_min)/(bins - 3), centers at z_min + i h, nearest-center
        deposition, voltage V = IFFT[-i Z(k) FFT(counts)] q_m c/h over `bins` points with
        Z(k) = Z0 length/(pi k rb^2) [1 - a K1(a)], a = k rb/p, k = 2 pi f/c, zero at DC and Nyquist,
        and linear interpolation between bin centers; the kick is -V. A discrete transform of length
        `bins` is periodic, so this is the field of a train of copies spaced bins*h apart.
        """
        lower = z.min()
        h = (z.max() - lower) / (bins - 3)
        counts = np.bincount(np.floor((z - lower) / h + 0.5).astype(int), minlength=bins)
        k = 2 * np.pi * np.fft.rfftfreq(bins, h)
        impedance = np.zeros(len(k))
        a = k[1:] * rb / p
        impedance[1:] = Z0 * length / (np.pi * rb**2 * k[1:]) * (1 - a * k1(a))
        impedance[-1] = 0
        voltage = np.fft.irfft(-1j * impedance * np.fft.rfft(counts), bins) * charge / len(z) * C / h
        return -np.interp(z, lower + h * np.arange(bins), voltage), bins * h


    def open_shape(q: float, big_r: float) -> float:
        """Dimensionless on-axis field h(q, R) of an isolated Gaussian bunch of uniform disks.

        A disk of radius r_b carrying charge dQ, seen in its rest frame from a distance gamma*u on
        the axis, has the longitudinal field dQ/(2 pi eps0 r_b^2) [sign(u) - gamma u /
        sqrt(gamma^2 u^2 + r_b^2)]; E_z is unchanged by the boost. Summing over a Gaussian line
        density of rms sigma, with q = z/sigma and R = r_b/(gamma sigma):
            h(q, R) = erf(q/sqrt 2) - int phi(y) (q - y)/sqrt((q - y)^2 + R^2) dy,
        phi the unit normal density. Its Fourier transform is the impedance Z(k) above, with no
        periodic images.
        """
        integrand = lambda y: (
            np.exp(-y * y / 2) / np.sqrt(2 * np.pi) * (q - y) / np.sqrt((q - y) ** 2 + big_r**2)
        )
        smooth = quad(integrand, -14, 14, points=[q], epsabs=1e-14, epsrel=1e-13, limit=400)[0]
        return float(erf(q / np.sqrt(2)) - smooth)


    def open_kick(q: float, sigma: float, p: float, rb: float, charge: float, length: float) -> float:
        """Energy kick (eV) of an electron at z = q sigma from an isolated Gaussian bunch of total
        charge `charge` over `length`: -charge length/(2 pi eps0 rb^2) h(q, rb/(p sigma))."""
        return -charge * length / (2 * np.pi * EPSILON_0 * rb**2) * open_shape(q, rb / (p * sigma))


    def at(z: np.ndarray, kick: np.ndarray, z0: float) -> float:
        """Kick at z0, interpolated between the sorted particle positions (quantile beams)."""
        order = np.argsort(z)
        return float(np.interp(z0, z[order], kick[order]))

    ```

## `_matr.py` { #_matr }

[Download _matr.py](_matr.py)

??? example "Source: _matr.py"

    ```python
    """Prescribed-map helpers for Elegant's MATR element, which reads a map from a text file.

    `matrix_text` writes the file in the layout MATR expects: the constant vector C, the rows of the
    first-order matrix R, the second-order coefficients T and the third-order coefficients Q. Elegant
    stores the symmetric coefficients once, with the indices ordered j >= k (>= l), and each entry
    holds the total coefficient of the monomial x_j x_k (x_l) in coordinate i; coordinates are
    (x, x', y, y', s, delta), indexed from zero here.
    """

    from __future__ import annotations

    import numpy as np

    TUNES = (0.17, 0.23)  # horizontal and vertical betatron tunes of the rotation helper


    def rotations(tunes: tuple[float, float] = TUNES) -> np.ndarray:
        """6x6 matrix: rotation by 2 pi nu in (x, x') and (y, y') with beta = 1 m and alpha = 0."""
        matrix = np.eye(6)
        for j, tune in zip((0, 2), tunes, strict=True):
            c, s = np.cos(2 * np.pi * tune), np.sin(2 * np.pi * tune)
            matrix[j : j + 2, j : j + 2] = [[c, s], [-s, c]]
        return matrix


    def matrix_text(
        c: np.ndarray, r: np.ndarray, t: np.ndarray | None = None, q: np.ndarray | None = None
    ) -> str:
        """Contents of a MATR map file; t has shape (6, 6, 6) and q (6, 6, 6, 6), zero when omitted."""
        t = np.zeros((6, 6, 6)) if t is None else t
        q = np.zeros((6, 6, 6, 6)) if q is None else q
        lines = ["C: " + " ".join(f"{v:.17g}" for v in c)]
        lines += [f"R{i + 1}: " + " ".join(f"{v:.17g}" for v in r[i]) for i in range(6)]
        for i in range(6):
            for j in range(6):
                values = " ".join(f"{t[i, j, k]:.17g}" for k in range(j + 1))
                lines.append(f"T{i + 1}{j + 1}: {values}")
        for i in range(6):
            for j in range(6):
                for k in range(j + 1):
                    values = " ".join(f"{q[i, j, k, l]:.17g}" for l in range(k + 1))
                    lines.append(f"Q{i + 1}{j + 1}{k + 1}: {values}")
        return "\n".join(lines) + "\n"

    ```

## `_matter.py` { #_matter }

[Download _matter.py](_matter.py)

??? example "Source: _matter.py"

    ```python
    """Shared setup for the MATTER reproducers: a thin helium target tracked by Elegant's single-scattering model.

    One MATTER element of length 0.01 m filled with helium (Z = 2, A = 4, density 0.02813 kg/m^3),
    traversed by electrons of momentum 1000 m_e c. PLIMIT bounds the scattering probability of one
    section, and the element is divided into the number of sections that keeps it below the bound.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import particles, run

    P0 = 1000.0  # reference and beam momentum, units of m_e c
    LENGTH = 0.01  # m
    RHO, Z, A = 0.0281327461321535, 2, 4  # kg/m^3 (dilute helium), atomic number, mass number

    # SDDS constants.h and the MATTER source, as Elegant uses them.
    RE, AMU = 2.81794092e-15, 1.6605e-27  # classical electron radius (m), atomic mass unit (kg)
    BS_Y0 = 1e-8  # lower cutoff of the recoil and bremsstrahlung energy-transfer samplers


    # Mean nuclear-bremsstrahlung events per electron: 4 L/(3 X0) [-ln y0 - (1 - y0) + 3/8 (1 - y0^2)],
    # with the helium radiation length X0 = 94.32 g/cm^2 (PDG atomic and nuclear properties of helium
    # gas) divided by the density, converted from cm to m. Bremsstrahlung is rare in this thin target.
    X0 = 94.32 / (RHO * 1e-3) * 1e-2  # m
    BREMS_PER_ELECTRON = 4 * LENGTH / (3 * X0) * (-np.log(BS_Y0) - (1 - BS_Y0) + 3 / 8 * (1 - BS_Y0**2))


    def track(mode, element, seed, columns):
        """Track `columns` (beam arrays with p in m_e c) through `MATTER` + element text; return columns.

        `element` holds the MATTER parameters after the fixed Z, A and RHO. Rows are returned ordered by
        particleID, which the beam must carry (MPI output order varies).
        """
        lattice = f"M: MATTER,Z={Z},A={A},RHO={RHO!r},{element}\nBL: LINE=(M)\n"
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
            f"output=out.sdds, random_number_seed={seed} &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})
        # particles() rejects nonfinite coordinates; the output must also hold exactly the input particles.
        out = particles(result["out.sdds"])
        if len(out["p"]) != len(columns["p"]):
            raise SystemExit(f"{mode}: {len(columns['p']) - len(out['p'])} particles lost")
        if not np.array_equal(out["particleID"], np.sort(columns["particleID"])):
            raise SystemExit(f"{mode}: the output particleIDs differ from the input ones")
        return out


    def beam(count, xp=0.0, yp=0.0):
        """`count` electrons on axis at momentum P0 with the given slopes, numbered from 1."""
        return {
            "x": np.zeros(count),
            "xp": np.full(count, xp),
            "y": np.zeros(count),
            "yp": np.zeros(count) + yp,
            "t": np.zeros(count),
            "p": np.full(count, P0),
            "particleID": np.arange(1, count + 1),
        }


    def gamma(p):
        """Lorentz factor of momentum p in m_e c."""
        return np.sqrt(1 + np.asarray(p) ** 2)

    ```

## `_matter_model.py` { #_matter_model }

[Download _matter_model.py](_matter_model.py)

??? example "Source: _matter_model.py"

    ```python
    """Shared setup for the MATTER energy-loss reproducers: any material, momentum and options.

    `track` runs one MATTER element on a beam of electrons that all start on axis at one momentum.
    `model` evaluates the event rates and slice count with the formulas and constants of the MATTER
    source, so a reproducer can predict the counts its own model implies. `gamma_loss` recovers the
    energy loss of each electron from the stored momenta without cancellation.
    """

    from __future__ import annotations

    from dataclasses import dataclass

    import numpy as np
    from _elegant import particles, run

    # SDDS constants.h and the MATTER source (matter.c), as Elegant uses them.
    PI = 3.141592653589793
    RE = 2.81794092e-15  # classical electron radius (m)
    E_MKS, H_MKS, C_MKS = 1.60217733e-19, 6.6260755e-34, 2.99792458e8
    EPSILON0 = 1e7 / (4 * PI * C_MKS**2)
    K_BOLTZMANN = 1.380658e-23  # J/K
    N_AVOGADRO = 6.02214129e23
    AMU_SOURCE = 1.6605e-27  # matter.c's AMU macro (kg), used in the single-scattering rates
    M_PROTON = 1.6726485e-27  # matter.c's mp_mks (kg)
    U_KG = 1e-3 / N_AVOGADRO  # atomic mass unit implied by radiationLength's N_A/A per gram
    ALPHA_FS = E_MKS**2 / (4 * PI * EPSILON0 * H_MKS / (2 * PI) * C_MKS)  # radiationLength's alpha
    Y0 = 1e-8  # lower cutoff of the bremsstrahlung and recoil fractional-transfer samplers
    # Mean photons per radiation length above Y0: (4/3)[-ln y0 - (1 - y0) + 3/8 (1 - y0^2)] (matter.c:125)
    BREMS_PER_X0 = 4 / 3 * (-np.log(Y0) - (1 - Y0) + 3 / 8 * (1 - Y0**2))


    def radiation_length(Z: int, A: float, rho: float) -> float:
        """Radiation length in m by matter.c's radiationLength (Tsai, PDG): Z, A in AMU, rho in kg/m^3."""
        a2 = (ALPHA_FS * Z) ** 2
        f = a2 * (1 / (1 + a2) + 0.20206 + a2 * (-0.0369 + a2 * (0.0083 - a2 * 0.002)))
        light = {1: (5.31, 6.144), 2: (4.79, 5.621), 3: (4.74, 5.805), 4: (4.71, 5.924)}
        lrad, lradp = light.get(Z, (np.log(184.15 * Z ** (-1 / 3)), np.log(1194 * Z ** (-2 / 3))))
        per_atom = 4 * ALPHA_FS * RE**2 * (Z * Z * (lrad - f) + Z * lradp)  # m^2
        return 1 / (per_atom * rho * N_AVOGADRO / (A * 1e-3))


    @dataclass(frozen=True)
    class Rates:
        """Mean events per electron over the whole element, and the slice count, as matter.c sets them."""

        elastic: float  # probScatter
        brems: float  # probBSScatter
        recoil: float  # probERScatter
        sections: int  # sections0


    def model(Z: int, A: float, rho: float, length: float, p0: float, plimit: float = 0.05) -> Rates:
        """Rates of matter.c:118-130 for reference momentum p0 (m_e c) and density rho (kg/m^3)."""
        beta = p0 / np.sqrt(p0**2 + 1)
        k1 = 4 * Z * (Z + 1) * (RE / (beta * p0)) ** 2
        k2 = (Z ** (1 / 3) / 137.036 / p0) ** 2
        n_atom = rho / (AMU_SOURCE * A)
        elastic = n_atom * length * k1 * PI**3 / (k2**2 + k2 * PI**2)
        brems = length / radiation_length(Z, A, rho) * BREMS_PER_X0
        recoil = length * n_atom * 2 * PI * Z * RE**2 / p0 * (1 / Y0 - 1)
        return Rates(elastic, brems, recoil, int(elastic / plimit + 1))


    def beam(count: int, p: float) -> dict[str, np.ndarray]:
        """`count` electrons on axis with zero slopes at momentum p (m_e c), numbered from 1."""
        zeros = np.zeros(count)
        return {"x": zeros, "xp": zeros, "y": zeros, "yp": zeros, "t": zeros, "p": np.full(count, p),
                "particleID": np.arange(1, count + 1)}  # fmt: skip


    def track(mode: str, element: str, p0: float, seed: int, count: int) -> dict[str, np.ndarray]:
        """Track `count` electrons at p0 through `M: MATTER,<element>`; return the columns by particleID.

        The output must hold every input particle (none may stop in the material).
        """
        lattice = f"M: MATTER,{element}\nBL: LINE=(M)\n"
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, "
            f"output=out.sdds, random_number_seed={seed} &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        columns = beam(count, p0)
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})
        out = particles(result["out.sdds"])
        if not np.array_equal(out["particleID"], columns["particleID"]):
            raise SystemExit(f"{mode}: the output particles differ from the input ones (losses?)")
        return out


    def gamma_loss(p_in: float, p_out: np.ndarray) -> np.ndarray:
        """gamma_in - gamma_out from momenta in m_e c: (p_in - p_out)(p_in + p_out)/(gamma_in + gamma_out)."""
        g_in, g_out = np.sqrt(1 + p_in**2), np.sqrt(1 + p_out**2)
        return (p_in - p_out) * (p_in + p_out) / (g_in + g_out)

    ```

## `_misalign.py` { #_misalign }

[Download _misalign.py](_misalign.py)

??? example "Source: _misalign.py"

    ```python
    """Straight-magnet tracking, export and independent frame/offset maps."""

    import math
    import re
    import subprocess
    import tempfile
    from pathlib import Path

    from _diagnostics import run

    # isort: split
    # _diagnostics sets numerical thread limits before loading _elegant and NumPy.
    import _elegant as native
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.linalg import expm

    P0 = 2300.0
    native.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-native-process deadline.


    def close(observed, expected, tolerance):
        np.testing.assert_allclose(observed, expected, rtol=0, atol=tolerance)


    def fault(observed, expected, faulty, tolerance):
        """Recognize the physical answer or this defect; unrelated results raise."""
        if np.allclose(observed, expected, rtol=0, atol=tolerance):
            return False
        close(observed, faulty, tolerance)
        return True


    def lattice(kind, length, strength, tilt, dx=0.0, dy=0.0, method=0):
        order = 1 if kind in ("QUAD", "KQUAD") else 2
        extra = ""
        if kind.startswith("K"):
            extra = ",N_SLICES=64,INTEGRATION_ORDER=6,SYNCH_RAD=0,ISR=0"
        if order == 1:
            extra += f",MALIGN_METHOD={method},EDGE1_EFFECTS=0,EDGE2_EFFECTS=0"
        return (
            f"Q: {kind},L={length:.17g},K{order}={strength:.17g},TILT={tilt:.17g},"
            f"DX={dx:.17g},DY={dy:.17g}{extra}\nBL: LINE=(Q)\n"
        )


    def track(source, rays):
        count = len(rays)
        beam = {name: rays[:, i] for i, name in enumerate(("x", "xp", "y", "yp"))}
        beam.update(p=np.full(count, P0), particleID=np.arange(1, count + 1))
        commands = (
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0},"
            "output=out.sdds,default_order=2,random_number_seed=104729 &end\n"
            "&run_control n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
        )
        result = run(
            {"lattice.lte": source, "run.ele": commands},
            "serial",
            {"beam.sdds": beam},
            ("out.sdds",),
        )
        data = native.particles(result["out.sdds"])
        if not np.array_equal(data["particleID"], beam["particleID"]):
            raise RuntimeError("Particles were lost or particle IDs changed")
        return np.column_stack([data[name] for name in ("x", "xp", "y", "yp")])


    def export(source, fmt, *, ender=None):
        with tempfile.TemporaryDirectory(prefix="elegant-misalignment-repro-") as tmp:
            work = Path(tmp)
            (work / "input.lte").write_text(source + "\nUSE,BL\n")
            args = ["elegantto", "input.lte", "output.txt", f"-{fmt}"]
            if ender is not None:
                (work / "ender.txt").write_text(ender)
                args.append("-ender=ender.txt")
            result = subprocess.run(
                args, cwd=work, capture_output=True, text=True, check=False, timeout=290
            )
            log = result.stdout + result.stderr
            if result.returncode or re.search(r"warning|error|unsupported", log, re.IGNORECASE):
                raise RuntimeError(f"elegantto -{fmt} failed or issued a diagnostic:\n{log}")
            return (work / "output.txt").read_text()


    def coefficient(text, name):
        matches = re.findall(rf"\b{name}\s*=\s*([+\-0-9.eE]+)", text, re.IGNORECASE)
        if len(matches) != 1 or not math.isfinite(float(matches[0])):
            raise RuntimeError(f"Expected one finite {name}: {text}")
        return float(matches[0])


    def quad_map(k, length, tilt):
        # H=(px²+py²)/2 + k*cos(2t)*(x²-y²)/2 + k*sin(2t)*x*y.
        # The laboratory Hamiltonian Hessian gives M=exp(L*J*Hessian(H)).
        generator = np.zeros((4, 4))
        generator[0, 1] = generator[2, 3] = 1.0
        generator[1, 0] = -k * math.cos(2 * tilt)
        generator[1, 2] = generator[3, 0] = -k * math.sin(2 * tilt)
        generator[3, 2] = k * math.cos(2 * tilt)
        return expm(generator * length)


    def exact_quad_offset(k, length, dx, dy):
        # At delta=0 the normalized canonical longitudinal momentum is
        # ps=sqrt(1-px²-py²); the transverse Lorentz force uses the translated body.
        def rhs(_, u):
            x, px, y, py = u
            ps = math.sqrt(1 - px * px - py * py)
            return [px / ps, -k * (x - dx), py / ps, k * (y - dy)]

        solution = solve_ivp(
            rhs,
            (0.0, length),
            np.zeros(4),
            method="DOP853",
            rtol=2.3e-14,
            atol=2e-18,
            max_step=length / 32,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        x, px, y, py = solution.y[:, -1]
        ps = math.sqrt(1 - px * px - py * py)
        return np.array([x, px / ps, y, py / ps])

    ```

## `_momentum_matrix.py` { #_momentum_matrix }

[Download _momentum_matrix.py](_momentum_matrix.py)

??? example "Source: _momentum_matrix.py"

    ```python
    """Fresh native matrix/particle fixtures for finite-speed momentum conversions."""

    import _elegant as native
    import numpy as np
    from _diagnostics import run

    C = 299792458.0
    native.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-native-process deadline.


    def close(actual, expected, tolerance, label):
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        if not np.isfinite(error) or error > tolerance:
            raise AssertionError(f"{label}: error={error:.17g}, tolerance={tolerance:.17g}")
        return error


    def discrepancy(actual, expected, faulty, tolerance, label):
        """Accept the physical result or the identified faulty law; refuse other output."""
        if abs(actual - expected) <= tolerance:
            return False
        close(actual, faulty, tolerance, label)
        return True


    def track(lattice, p0, delta, *, s=None, order=2, twiss=False):
        """Compare particle and cumulative/individual matrix output from one fresh process."""
        momentum = p0 * (1 + np.asarray(delta, dtype=float))
        beta = momentum / np.hypot(momentum, 1)
        clock = np.zeros(len(momentum)) if s is None else np.asarray(s)
        beam = {"p": momentum, "t": clock / (C * beta), "particleID": np.arange(1, len(momentum) + 1)}
        commands = (
            "&run_setup lattice=lattice.lte,use_beamline=BL,"
            f"p_central={p0:.17g},default_order={order},output=particles.sdds,"
            "random_number_seed=271828 &end\n"
            "&run_control n_steps=1 &end\n"
            "&matrix_output SDDS_output=cumulative.sdds,SDDS_output_order=1,output_at_each_step=1 &end\n"
            "&matrix_output SDDS_output=individual.sdds,SDDS_output_order=1,individual_matrices=1,output_at_each_step=1 &end\n"
        )
        outputs = ["particles.sdds", "cumulative.sdds", "individual.sdds"]
        if twiss:
            commands += (
                "&twiss_output filename=twiss.sdds,matched=0,beta_x=1,beta_y=1,"
                "concat_order=2,local_dispersion=0,output_at_each_step=1 &end\n"
            )
            outputs.append("twiss.sdds")
        commands += "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
        result = run(
            {"lattice.lte": lattice, "run.ele": commands}, "serial", {"beam.sdds": beam}, tuple(outputs)
        )
        data = native.particles(result["particles.sdds"])
        if not np.array_equal(data["particleID"], beam["particleID"]):
            raise RuntimeError("Particles were lost or particle IDs changed")
        result["particles"] = data
        result["s"] = C * data["t"] * data["p"] / np.hypot(data["p"], 1)
        return result


    def value(result, name, *, output="cumulative.sdds", element=None):
        data = result[output]
        if data.n_pages != 1:
            raise RuntimeError("Expected exactly one SDDS page")
        index = -1
        if element is not None:
            names = [str(x) for x in data[0, "ElementName"]]
            if names.count(element) != 1:
                raise RuntimeError(f"Expected exactly one element named {element}")
            index = names.index(element)
        scalar = float(data[0, name][index])
        if not np.isfinite(scalar):
            raise RuntimeError(f"Nonfinite {name} in {output}")
        return scalar

    ```

## `_multipole.py` { #_multipole }

[Download _multipole.py](_multipole.py)

??? example "Source: _multipole.py"

    ```python
    """Shared setup for the KQUAD and KSEXT multipole-table reproducers: a 12-electron beam, inline
    multipole tables, and an independent integrator for a straight magnet with steering fields.

    The integrator follows the particles through a straight hard-edge magnet in canonical transverse
    momenta q = p_perp/p0, solving the Lorentz equations with scipy, without any Elegant code.
    """

    from __future__ import annotations

    from math import factorial

    import numpy as np
    from scipy.integrate import solve_ivp

    P0 = 45.0  # reference momentum, units of m_e c
    N = 12  # particles, enough for both workers of a 3-rank run
    RADIUS = 0.027  # m, reference radius of the multipole tables


    def beam() -> dict[str, np.ndarray]:
        """Twelve electrons with transverse offsets and slopes and momenta 43 to 49 m_e c."""
        return {
            "x": np.linspace(-0.018, 0.021, N),
            "xp": np.linspace(0.003, -0.004, N),
            "y": np.linspace(0.008, -0.011, N),
            "yp": np.linspace(-0.002, 0.003, N),
            "t": np.zeros(N),
            "p": np.linspace(43.0, 49.0, N),
        }


    def table_text(orders, normal, skew=None) -> str:
        """ASCII SDDS multipole table: parameter referenceRadius, columns order, normal, (skew)."""
        columns = "&column name=order, type=long &end\n&column name=normal, type=double &end\n"
        if skew is not None:
            columns += "&column name=skew, type=double &end\n"
        rows = "".join(
            f"{n} {a!r}" + (f" {b!r}" if skew is not None else "") + "\n"
            for n, a, b in zip(orders, normal, skew if skew is not None else normal, strict=True)
        )
        return (
            "SDDS1\n"
            '&parameter name=referenceRadius, type=double, units="m" &end\n'
            + columns
            + "&data mode=ascii &end\n"
            + f"{RADIUS!r}\n{len(orders)}\n"
            + rows
        )


    def canonical(b: dict[str, np.ndarray]) -> np.ndarray:
        """Rows x, y, qx, qy with q = (p/p0) * slope / sqrt(1 + xp^2 + yp^2)."""
        ratio = b["p"] / P0
        qs = ratio / np.sqrt(1 + b["xp"] ** 2 + b["yp"] ** 2)
        return np.stack((b["x"], b["y"], qs * b["xp"], qs * b["yp"]))


    def observable(state: np.ndarray, b: dict[str, np.ndarray]) -> np.ndarray:
        """Columns x, xp, y, yp from canonical rows x, y, qx, qy."""
        x, y, qx, qy = state
        qs = np.sqrt((b["p"] / P0) ** 2 - qx**2 - qy**2)
        return np.column_stack((x, qx / qs, y, qy / qs))


    def straight_body(
        state: np.ndarray,
        b: dict[str, np.ndarray],
        main_order: int,
        strength: float,
        length: float,
        factor: float = 0.0,
        kick: tuple[float, float] = (0.0, 0.0),
        harmonics=((2, 0.07), (4, -0.025)),
    ) -> np.ndarray:
        """Canonical rows after a straight magnet of the given length (m).

        Complex field C = By + iBx divided by the reference rigidity, with z = x + iy:
        C = K_m z^m/m! - (H - iV)/L - factor/L sum_n a_n (H - i(-1)^(n/2) V) (z/R)^n,
        where K_m is `strength`, (H, V) = `kick` the steering angles spread over the length and a_n
        the normal steering fractions at order n (zero steering and zero factor leave the main field).
        The equations of motion are dx/ds = qx/qs, dy/ds = qy/qs, dqx/ds = -Re C, dqy/ds = Im C with
        qs = sqrt((p/p0)^2 - qx^2 - qy^2).
        """
        ratio = b["p"] / P0

        def rhs(_, flat):
            x, y, qx, qy = flat.reshape(4, N)
            z = x + 1j * y
            field = strength * z**main_order / factorial(main_order)
            field -= (kick[0] - 1j * kick[1]) / length
            for n, a in harmonics:
                coefficient = -factor * a * (kick[0] - 1j * (-1) ** (n // 2) * kick[1])
                field += coefficient / length * (z / RADIUS) ** n
            qs = np.sqrt(ratio**2 - qx**2 - qy**2)
            return np.stack((qx / qs, qy / qs, -field.real, field.imag)).ravel()

        sol = solve_ivp(rhs, (0, length), state.ravel(), method="DOP853", rtol=3e-13, atol=2e-15)
        if not sol.success:
            raise RuntimeError("straight-magnet integration failed")
        return sol.y[:, -1].reshape(4, N)

    ```

## `_precision.py` { #_precision }

[Download _precision.py](_precision.py)

??? example "Source: _precision.py"

    ```python
    """Shared native fixtures and stable transport oracles for precision reproducers."""

    from __future__ import annotations

    import math
    import os

    for _name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[_name] = "1"

    import _elegant
    import mpmath as mp
    import numpy as np
    from _elegant import particles

    _elegant.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-native-process deadline.
    mp.mp.dps = 70
    MASS = 0.51099906  # MeV/c²: the executed Elegant/SDDS constant set.
    C = 299792458.0
    LENGTH = 0.4
    P0 = 3e9 / (MASS * 1e6)
    PIN = P0 * 1.01
    D = mp.mpf(PIN) / mp.mpf(P0)


    def native(element, rays, p0=P0, *, matrix=False):
        """Run one element with fixed rays, rejecting loss or malformed output."""
        keys = ("x", "xp", "y", "yp", "t", "p")
        beam = {
            key: np.array([ray.get(key, p0 if key == "p" else 0.0) for ray in rays]) for key in keys
        }
        commands = (
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0:.17g},"
            "output=out.sdds,default_order=1,random_number_seed=271828 &end\n"
        )
        if matrix:
            commands += "&matrix_output SDDS_output=matrix.sdds,individual_matrices=1 &end\n"
        commands += (
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
        )
        result = _elegant.run(
            {"lattice.lte": "E: " + element + "\nBL: LINE=(E)\n", "run.ele": commands},
            beams={"beam.sdds": beam},
            outputs=("out.sdds", "matrix.sdds") if matrix else ("out.sdds",),
        )
        data = result["out.sdds"]
        if data.n_pages != 1:
            raise RuntimeError("Expected one output page")
        out = particles(data)
        if len(out["p"]) != len(rays):
            raise RuntimeError("Unexpected particle loss")
        r12 = None
        if matrix:
            data = result["matrix.sdds"]
            row = list(data[0, "ElementName"]).index("E")
            r12 = float(data[0, "R12"][row])
            if not math.isfinite(r12):
                raise RuntimeError("Nonfinite analytic R12")
        return out, r12


    def verdict(defect):
        """Print the final verdict in the shared format of `_elegant.verdict`."""
        _elegant.verdict(defect)


    def require_close(observed, expected, tolerance):
        """Reject a failed control instead of counting it as a reproduced defect."""
        if abs(mp.mpf(float(observed)) - expected) > tolerance:
            raise RuntimeError(f"Failed control: {observed}, expected {expected}, atol={tolerance}")


    def position_line(label, observed, expected, method, *, coordinate="x"):
        """Print one native position and its independent expectation, in metres."""
        error = float(mp.mpf(float(observed)) - expected)
        print(
            f"{label}: {coordinate}={observed:.17g} expected={float(expected):.17g} m; "
            f"error={error:+.6e} m ({method})"
        )
        return abs(error)


    def corrector(kick, *, length=LENGTH, x=0.001, xp=0.0001):
        """Integrate a planar uniform field between parallel end planes."""
        xx, slope = mp.mpf(x), mp.mpf(xp)
        if not length:
            return xx, mp.tan(mp.atan(slope) + mp.mpf(kick) / D)
        u0 = slope / mp.sqrt(1 + slope * slope)
        kappa = mp.sin(mp.mpf(kick)) / (mp.mpf(length) * D)

        # The transverse direction cosine grows linearly with z; x'=u/sqrt(1-u²).
        def u(z):
            return u0 + kappa * z

        position = xx + mp.quad(lambda z: u(z) / mp.sqrt(1 - u(z) ** 2), [0, length])
        final_u = u(mp.mpf(length))
        return position, final_u / mp.sqrt(1 - final_u * final_u)


    def rf_transport(p0, voltage, phase, frequency, slope):
        """Momentum-profile transport, with the sloped ray's midpoint RF phase."""
        gi = mp.sqrt(1 + mp.mpf(p0) ** 2)
        beta = mp.mpf(p0) / gi
        # The explicit reference time is the axial midpoint arrival time.
        dt = mp.mpf(LENGTH) / (2 * C * beta) * (mp.sqrt(1 + mp.mpf(slope) ** 2) - 1)
        phi = mp.mpf(phase) * mp.pi / 180 + 2 * mp.pi * frequency * dt
        dg = mp.mpf(voltage) / (mp.mpf(MASS) * 1000000) * mp.sin(phi)
        pf = mp.sqrt((gi + dg) ** 2 - 1)
        return mp.quad(lambda u: mp.mpf(LENGTH) * p0 / (p0 + (pf - p0) * u), [0, 1])


    def rf_analytic(p0, voltage):
        """Energy-profile quadrature for the default analytic matrix at crest."""
        gi = mp.sqrt(1 + mp.mpf(p0) ** 2)
        dg = mp.mpf(voltage) / (mp.mpf(MASS) * 1000000)
        return mp.quad(lambda u: mp.mpf(LENGTH) * p0 / mp.sqrt((gi + dg * u) ** 2 - 1), [0, 1])


    def rf_srs(p0, voltage):
        """Crest solution of x_tt+x/8=0, where t=log(gamma/gamma_i)."""
        gi = mp.sqrt(1 + mp.mpf(p0) ** 2)
        dg = mp.mpf(voltage) / (mp.mpf(MASS) * 1000000)
        if not dg:
            return mp.mpf(LENGTH)
        t = mp.log1p(dg / gi)
        return mp.mpf(LENGTH) * gi / dg * mp.sqrt(8) * mp.sin(t / mp.sqrt(8))


    def bend_split(angle, slices, length, x, xp, d, order=4):
        """Exact split flows from the planar curvilinear Hamiltonian.

        H_D = -(1+h*x)*sqrt(d^2-q^2), H_K = h*x+h^2*x^2/2.
        q rotates during H_D. Rationalizing its position update avoids subtracting
        near-unity ratios; the path update follows the conserved drift invariant.
        """
        h = mp.mpf(angle) / mp.mpf(length)
        x = mp.mpf(x)
        q = d * mp.mpf(xp) / mp.sqrt(1 + mp.mpf(xp) ** 2)
        distance = mp.mpf(0)
        b = 1.25992104989487316477
        ds = [
            mp.mpf(v)
            for v in (0.5 / (2 - b), (1 - b) / (2 - b) / 2, (1 - b) / (2 - b) / 2, 0.5 / (2 - b))
        ]
        ks = [mp.mpf(v) for v in (1 / (2 - b), -b / (2 - b), 1 / (2 - b), 0.0)]
        if order == 2:
            ds = list(map(mp.mpf, (0.5, 0.5)))
            ks = list(map(mp.mpf, (1.0, 0.0)))
        elif order == 6:
            ds = list(
                map(
                    mp.mpf,
                    (
                        0.39225680523878,
                        0.5100434119184585,
                        -0.47105338540975655,
                        0.0687531682525181,
                        0.0687531682525181,
                        -0.47105338540975655,
                        0.5100434119184585,
                        0.39225680523878,
                    ),
                )
            )
            ks = list(
                map(
                    mp.mpf,
                    (
                        0.784513610477560,
                        0.235573213359357,
                        -1.17767998417887,
                        1.3151863206839063,
                        -1.17767998417887,
                        0.235573213359357,
                        0.784513610477560,
                        0.0,
                    ),
                )
            )
        elif order != 4:
            raise ValueError(order)
        step = mp.mpf(float(length / slices))
        for _ in range(slices):
            for df, kf in zip(ds, ks, strict=True):
                a = h * step * df
                longitudinal = mp.sqrt(d * d - q * q)
                sn, cs = mp.sin(a), mp.cos(a)
                lf = longitudinal * cs - q * sn
                # (1-cos a)/h is evaluated by 2 sin²(a/2)/h.
                numer = longitudinal * 2 * mp.sin(a / 2) ** 2 + q * sn
                distance += (1 / h + x) * d * sn / lf
                x += (1 / h + x) * numer / lf
                q = q * cs + longitudinal * sn
                q -= h * step * kf * (1 + h * x)
        return x, q / mp.sqrt(d * d - q * q), distance

    ```

## `_refmomentum.py` { #_refmomentum }

[Download _refmomentum.py](_refmomentum.py)

??? example "Source: _refmomentum.py"

    ```python
    """Fixed-ray tracking and mass-shell/rigidity checks for reference-momentum probes."""

    import math

    import _elegant as native
    import numpy as np
    from _diagnostics import run

    C = 299792458.0
    # Electron SI constants from the SDDS pin; use the executable's constant set.
    MASS = 9.1093897e-31
    CHARGE = 1.60217733e-19
    RAYS = np.array([1e-6, -1e-6, 0.5e-6, -0.5e-6, 0.0, 0.0])
    native.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-native-process deadline.


    def close(observed, expected, *, rtol=4e-14, atol=0.0):
        np.testing.assert_allclose(observed, expected, rtol=rtol, atol=atol)


    def track(lattice, p0, *, rays=None, mode="serial", twiss=False, watch=False, setup=""):
        """Fresh native process with six fixed particles, checked survival and ID ordering."""
        x = np.zeros(6) if rays is None else rays
        beam = {"x": x, "p": np.full(6, p0), "particleID": np.arange(1, 7)}
        commands = (
            "&run_setup lattice=lattice.lte,use_beamline=BL,"
            f"p_central={p0:.17g},output=out.sdds,default_order=1,"
            f"random_number_seed=271828{setup} &end\n"
            "&run_control n_passes=1 &end\n"
        )
        outputs = ["out.sdds"]
        if twiss:
            commands += "&twiss_output filename=optics.sdds,matched=0,beta_x=1,beta_y=1 &end\n"
            outputs.append("optics.sdds")
        if watch:
            outputs.append("reference.sdds")
        commands += "&sdds_beam input=initial.sdds &end\n&track soft_failure=0 &end\n"
        result = run(
            {"lattice.lte": lattice + "\nUSE,BL\n", "run.ele": commands},
            mode,
            {"initial.sdds": beam},
            tuple(outputs),
        )
        data = native.particles(result["out.sdds"])
        if not np.array_equal(data["particleID"], beam["particleID"]):
            raise RuntimeError("Particles were lost or particle IDs changed")
        result["particles"] = data
        return result


    def response(data, *, order=1):
        """Linear slope response or coefficient of x squared, extrapolated to zero amplitude."""
        xp, h = data["xp"], RAYS[0]
        if order == 1:
            coarse = (xp[0] - xp[1]) / (2 * h)
            fine = (xp[2] - xp[3]) / h
        else:
            coarse = (xp[0] + xp[1] - 2 * xp[4]) / (2 * h**2)
            fine = (xp[2] + xp[3] - 2 * xp[4]) / (2 * (h / 2) ** 2)
        # Symmetry isolates the derivative; Richardson extrapolation removes its O(h^2) error.
        return float((4 * fine - coarse) / 3)


    def quadrupole_slope(k, length):
        # Linear Lorentz force: x'' + k*x = 0, with k = K1*P0/p for an off-reference ray.
        if k > 0:
            root = math.sqrt(k)
            return -root * math.sin(root * length)
        if k < 0:
            root = math.sqrt(-k)
            return root * math.sinh(root * length)
        return 0.0

    ```

## `_rfca_lens.py` { #_rfca_lens }

[Download _rfca_lens.py](_rfca_lens.py)

??? example "Source: _rfca_lens.py"

    ```python
    """Transverse transfer of a static-field RFCA: tracked 2x2 map and Elegant's analytic matrix.

    The cavity is RFCA with FREQ=0 and PHASE=90, so every particle gains exactly VOLT and the field is a
    uniform accelerating gradient VOLT/L between hard edges, with no body focusing (the default model).
    Transverse momentum is then conserved inside the cavity, and only the END1_FOCUS and END2_FOCUS
    edge lenses focus. Energies are kinetic, momenta beta*gamma in units of m_e c.
    """

    from __future__ import annotations

    import sys

    import numpy as np
    from _elegant import particles, run

    # SDDS constants.h, as Elegant uses them.
    ME_MEV = 0.51099906
    LENGTH, VOLT = 0.7, 12e6  # m, V: a 17.14 MV/m gradient
    EPS = 1e-7  # m and rad: offsets and slopes that measure the linear map


    def momentum(kinetic_mev: float) -> float:
        gamma = 1 + kinetic_mev / ME_MEV
        return float(np.sqrt(gamma**2 - 1))


    def transfer(mode: str, kinetic_mev: float, kicks: int, end1: int, end2: int) -> dict:
        """Tracked map R (2x2, horizontal) and the matrix_output map M of the same element.

        Particles at x = +-EPS and at x' = +-EPS give the columns of the tracked map by central
        differences; tracking is linear in them, so the differences are exact up to rounding.
        """
        p0 = momentum(kinetic_mev)
        lattice = (
            f"R: RFCA,L={LENGTH!r},VOLT={VOLT!r},FREQ=0,PHASE=90,N_KICKS={kicks},"
            f"END1_FOCUS={end1},END2_FOCUS={end2},CHANGE_P0=0,T_REFERENCE=0\n"
            "BL: LINE=(R)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, "
            "output=out.sdds, default_order=1 &end\n"
            "&matrix_output SDDS_output=matrix.sdds, individual_matrices=1 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {
            "x": np.array([EPS, -EPS, 0.0, 0.0]),
            "xp": np.array([0.0, 0.0, EPS, -EPS]),
            "p": np.full(4, p0),
            "particleID": np.arange(1, 5),
        }
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": beam},
            outputs=("out.sdds", "matrix.sdds"),
        )
        out = particles(result["out.sdds"])
        if len(out["x"]) != 4:
            sys.exit(f"{mode}: particles were lost")
        x, xp = out["x"], out["xp"]
        tracked = np.array(
            [
                [(x[0] - x[1]) / (2 * EPS), (x[2] - x[3]) / (2 * EPS)],
                [(xp[0] - xp[1]) / (2 * EPS), (xp[2] - xp[3]) / (2 * EPS)],
            ]
        )
        matrix = result["matrix.sdds"]
        names = [str(n) for n in matrix[0, "ElementName"]]
        row = names.index("R")
        analytic = np.array([[float(matrix[0, f"R{i}{j}"][row]) for j in (1, 2)] for i in (1, 2)])
        gamma_in = 1 + kinetic_mev / ME_MEV
        gamma_out = float(np.hypot(out["p"][0], 1))
        if abs(gamma_out - gamma_in - VOLT / (1e6 * ME_MEV)) > 1e-12 * gamma_out:
            sys.exit(f"{mode}: the energy gain is not VOLT")
        return {"tracked": tracked, "matrix": analytic, "gamma_in": gamma_in, "gamma_out": gamma_out}

    ```

## `_scinsert.py` { #_scinsert }

[Download _scinsert.py](_scinsert.py)

??? example "Source: _scinsert.py"

    ```python
    """Shared setup for the insert_sceffects command reproducers: a one-segment ring, a bunch with exact
    second moments, the manual's linear space-charge kick and a runner that reports the SCMULT
    insertions.

    The ring is a 0.8 nC electron bunch (p = 10 mₑc) followed by R, a 1 m symplectic EMATRIX quarter
    turn matched at beta = 1 m in both planes. With one linear SCMULT after R, every particle's slope
    change over one turn is the manual's linear kick evaluated with the rms sizes after R.
    """

    from __future__ import annotations

    import itertools
    import re

    import numpy as np
    from _elegant import RunFailure, particles, run

    # SDDS constants.h, as Elegant uses them.
    C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
    P = 10.0  # electron momentum, units of m_e c
    GAMMA = np.hypot(1, P)
    BETA_C = C * P / GAMMA
    CHARGE = 0.8e-9  # C
    SX, SY, SZ = 1.5e-3, 0.8e-3, 0.025  # m
    LENGTH = 1.0  # m

    # R: (x, xp) -> (xp, -x) and (y, yp) -> (yp, -y), a quarter turn at beta = 1 m, over 1 m.
    LATTICE = (
        f"Q: CHARGE,TOTAL={CHARGE!r}\n"
        f"R: EMATRIX,L={LENGTH!r},ORDER=1,R12=1,R21=-1,R34=1,R43=-1,R55=1,R66=1\n"
        "RING: LINE=(Q,R)\n"
    )
    TAIL = (
        f"&run_setup lattice=lattice.lte,use_beamline=RING,p_central={P!r},default_order=1,"
        "output=out.sdds &end\n"
        '&twiss_output matched=1,filename="twiss.sdds" &end\n'
        "&run_control n_steps=1,n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )

    # Every combination of -sqrt(1.5), 0, +sqrt(1.5) in (x, xp, y, yp, z): exact unit variances,
    # zero means and no correlations, so the bunch has rms sizes SX, SY, SZ and is matched at 1 m.
    _UNIT = np.array(list(itertools.product([-np.sqrt(1.5), 0.0, np.sqrt(1.5)], repeat=5)))
    N = len(_UNIT)
    _x, _xp, _y, _yp, _z = _UNIT.T
    BEAM = {
        "x": SX * _x,
        "xp": SX * _xp,
        "y": SY * _y,
        "yp": SY * _yp,
        "t": SZ * _z / BETA_C,
        "p": np.full(N, P),
        "particleID": np.arange(1, N + 1),
    }
    # Coordinates (x, xp, y, yp) after R without space charge.
    IDEAL = np.column_stack([BEAM["xp"], -BEAM["x"], BEAM["yp"], -BEAM["y"]])


    def linear_kick() -> np.ndarray:
        """The manual's linear kick K L lambda(z) x_i/[sigma_i (sigma_x + sigma_y)] for one SCMULT after R.

        K = 2 N r_e/(beta^2 gamma^3) and lambda = exp(-z^2/(2 sigma_z^2))/(sqrt(2 pi) sigma_z); the
        sizes are those after R (the quarter turn exchanges positions and slopes, whose spreads are
        equal at beta = 1 m). Returns shape (N, 2).
        """
        k = 2 * CHARGE / QE * RE / (P * P * GAMMA)
        sigma = np.array([IDEAL[:, 0].std(), IDEAL[:, 2].std()])
        s = SZ * _z
        density = np.exp(-0.5 * ((s - s.mean()) / s.std()) ** 2) / (np.sqrt(2 * np.pi) * s.std())
        return k * LENGTH * density[:, None] * IDEAL[:, [0, 2]] / (sigma * sigma.sum())


    def spec(**options) -> str:
        """One insert_sceffects command with both transverse planes on and the given options."""
        options = {"horizontal": 1, "vertical": 1, **options}
        text = ",".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in options.items())
        return f"&insert_sceffects {text} &end\n"


    def track(commands: str, mode: str = "serial") -> dict:
        """Run the ring with the given insert_sceffects commands.

        Returns {"status", "log"} and, when the run completes, "insertions" (names of the elements
        that precede each SCMULT in the Twiss output) and "kick" (slope change per particle, shape
        (N, 2)). A run that exits nonzero (an abort or a crash) returns its status and log.
        """
        files = {
            "lattice.lte": LATTICE,
            "run.ele": "&global_settings echo_namelists=0 &end\n" + commands + TAIL,
        }
        try:
            result = run(files, mode, {"beam.sdds": BEAM}, outputs=("out.sdds", "twiss.sdds"))
        except RunFailure as failure:
            return {"status": failure.status, "log": failure.log}
        twiss = result["twiss.sdds"]
        names = [str(n) for n in twiss[0, "ElementName"]]
        types = [str(t) for t in twiss[0, "ElementType"]]
        insertions = [names[i - 1] for i, t in enumerate(types) if t == "SCMULT"]
        out = particles(result["out.sdds"])
        kick = np.column_stack([out["xp"] - IDEAL[:, 1], out["yp"] - IDEAL[:, 3]])
        return {"status": 0, "log": result["log"], "insertions": insertions, "kick": kick}


    def failure_line(log: str) -> str:
        """The last line of a failed run that names the failure, for printing."""
        lines = [
            ln.strip()
            for ln in log.splitlines()
            if re.search(r"error|signal|SIG|abort|fault", ln, re.IGNORECASE)
        ]
        return lines[-1] if lines else log.strip().splitlines()[-1]

    ```

## `_scmult_ring.py` { #_scmult_ring }

[Download _scmult_ring.py](_scmult_ring.py)

??? example "Source: _scmult_ring.py"

    ```python
    """Shared setup for the SCMULT bunch-train reproducers: a closed four-sector ring, bunch trains in
    bunched mode and the beam-moment helpers.

    The ring has four 0.8 m sector bends of angle pi/2 with K1 = -h^2/2 (h = pi/(2 * 0.8 m)), which
    focus equally in both planes, separated by 0.6 m straights; it is stable and closes geometrically.
    `insert_sceffects` places one SCMULT after a drift DS of length `length` at the start of the first
    straight, and WATCH elements record every particle before DS (entrance) and after the SCMULT
    (exit) on each pass. The SCMULT changes only the slopes.
    """

    from __future__ import annotations

    import itertools
    import re
    import sys

    import numpy as np
    from _elegant import particles, run

    # SDDS constants.h, as Elegant uses them.
    C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
    P = 17.0  # electron momentum, units of m_e c
    BETA = P / np.hypot(1, P)
    SX, SY = 0.7e-3, 1.8e-3  # m, rms sizes of the unscaled bunch
    STRAIGHT, BEND_LENGTH = 0.6, 0.8
    H = np.pi / (2 * BEND_LENGTH)  # curvature of a 90-degree sector bend
    SPACING = 2.3e-9  # s between bunches


    def optics(length: float) -> tuple[float, float, float]:
        """Matched (beta, alpha) at the start of the straight and beta at distance `length` into it.

        One cell is a straight followed by a sector bend with K1 = -h^2/2, which focuses equally in
        both planes with k = h/sqrt(2). The periodic Courant-Snyder parameters follow from the
        one-cell matrix M: sin(mu) = sqrt(1 - (tr M/2)^2), beta0 = M12/sin(mu),
        alpha0 = (M11 - M22)/(2 sin(mu)), gamma0 = -M21/sin(mu); a drift of length s transports
        beta to beta0 - 2 alpha0 s + gamma0 s^2.
        """
        k = H / np.sqrt(2)
        phase = k * BEND_LENGTH
        bend = np.array([[np.cos(phase), np.sin(phase) / k], [-k * np.sin(phase), np.cos(phase)]])
        cell = bend @ np.array([[1, STRAIGHT], [0, 1]])
        sine = np.sqrt(1 - (np.trace(cell) / 2) ** 2)
        beta0 = cell[0, 1] / sine
        alpha0 = (cell[0, 0] - cell[1, 1]) / (2 * sine)
        gamma0 = -cell[1, 0] / sine
        return beta0, alpha0, beta0 - 2 * alpha0 * length + gamma0 * length**2


    def bunch(matched: bool, times: np.ndarray) -> np.ndarray:
        """Particles (x, xp, y, yp, t) of one unscaled bunch, shape (n, 5).

        Every combination of the four transverse phases and the arrival times. matched=True places
        the phases 0, 90, 180, 270 degrees of a matched ellipse of rms sizes SX, SY at the start of
        the straight (exact matched second moments, including the within-plane position-slope
        covariance, and no coupling between the transverse planes); otherwise the particles are
        cold, at x = +-SX, y = +-SY with zero slopes.
        """
        if matched:
            a, b, t = np.array(list(itertools.product(np.arange(4), np.arange(4), times))).T
            beta0, alpha0, _ = optics(STRAIGHT)
            x = np.sqrt(2) * SX * np.cos(a * np.pi / 2)
            y = np.sqrt(2) * SY * np.cos(b * np.pi / 2)
            xp = (-alpha0 * x + np.sqrt(2) * SX * np.sin(a * np.pi / 2)) / beta0
            yp = (-alpha0 * y + np.sqrt(2) * SY * np.sin(b * np.pi / 2)) / beta0
        else:
            x, y, t = np.array(list(itertools.product([-SX, SX], [-SY, SY], times))).T
            xp, yp = np.zeros_like(x), np.zeros_like(y)
        return np.column_stack([x, xp, y, yp, t])


    def train(
        specs: tuple[tuple[float, float], ...],
        times: np.ndarray,
        *,
        matched: bool = False,
        shift: tuple[float, float] = (0.0, 0.0),
        reverse: bool = False,
    ) -> dict[str, np.ndarray | int]:
        """Bunch train as SDDS columns; specs lists (transverse scale, duration scale) per bunch.

        Bunch b holds particle IDs b*n+1 ... (b+1)*n (`IDSlotsPerBunch` = n) and arrives b * SPACING
        after the first. Rows are interleaved so that every MPI worker holds part of each bunch;
        reverse=True reverses the row order, so that bunch order does not follow file order.
        """
        base = bunch(matched, times)
        n = len(base)
        blocks = []
        for index, (scale, zscale) in enumerate(specs):
            x, xp, y, yp, t = base.T
            blocks.append(
                np.column_stack(
                    [
                        scale * x + shift[0],
                        scale * xp,
                        scale * y + shift[1],
                        scale * yp,
                        zscale * t + index * SPACING,
                    ]
                )
            )
        values = np.concatenate(blocks)
        ids = np.arange(1, len(values) + 1)
        values = values.reshape(len(specs), n, 5).transpose(1, 0, 2).reshape(-1, 5)
        ids = ids.reshape(len(specs), n).T.reshape(-1)
        if reverse:
            values, ids = values[::-1], ids[::-1]
        columns: dict[str, np.ndarray | int] = {
            key: values[:, i] for i, key in enumerate(("x", "xp", "y", "yp", "t"))
        }
        columns["p"] = np.full(len(values), P)
        columns["particleID"] = ids
        columns["IDSlotsPerBunch"] = n
        return columns


    def track(
        mode: str,
        columns: dict,
        bunches: int,
        *,
        length: float = STRAIGHT,
        charge: float = 2e-10,
        passes: int = 1,
        nonlinear: int = 0,
        factor: float = 1.0,
        duration: float = 0.0,
        interpolate: int = 0,
    ) -> tuple[list[dict], list[dict]]:
        """Run the ring for `passes` turns; return per-pass coordinates (entrance, exit) by particle ID.

        `charge` is the charge of each bunch. The SCMULT acts after DS, a drift of `length`.
        """
        lattice = (
            f"Q: CHARGE,TOTAL={bunches * charge!r}\n"
            'W0: WATCH,FILENAME="entrance.sdds",MODE="coordinates"\n'
            f"DS: DRIF,L={length!r}\n"
            'W1: WATCH,FILENAME="exit.sdds",MODE="coordinates"\n'
            f"DT: DRIF,L={STRAIGHT - length!r}\n"
            f"DA: DRIF,L={STRAIGHT!r}\n"
            f"B: SBEND,L={BEND_LENGTH!r},ANGLE={np.pi / 2!r},K1={-(H**2) / 2!r}\n"
            "BL: LINE=(Q,W0,DS,W1,DT,B,DA,B,DA,B,DA,B)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            '&insert_sceffects name="DS",skip=1,horizontal=1,vertical=1,'
            f"nonlinear={nonlinear},averaging_factor={factor!r},slice_duration={duration!r},"
            f"slice_interpolation={interpolate} &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},default_order=2,"
            "output=out.sdds &end\n"
            f"&run_control n_steps=1,n_passes={passes} &end\n"
            "&twiss_output matched=1 &end\n"
            "&sdds_beam input=beam.sdds,use_bunched_mode=1,track_pages_separately=0 &end\n"
            "&track soft_failure=0 &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": columns},
            outputs=("entrance.sdds", "exit.sdds"),
        )
        total = len(columns["particleID"])
        pages = []
        for name in ("entrance.sdds", "exit.sdds"):
            sdds = result[name]
            if sdds.n_pages != passes:
                sys.exit(f"{mode}: {name} has {sdds.n_pages} pages for {passes} passes")
            pages.append([particles(sdds, k) for k in range(passes)])
        for page in pages[0] + pages[1]:
            if not np.array_equal(page["particleID"], np.arange(1, total + 1)):
                sys.exit(f"{mode}: particles lost")
            if not all(np.isfinite(page[key]).all() for key in ("x", "xp", "y", "yp", "t")):
                sys.exit(f"{mode}: non-finite coordinates")
        if re.search(r"\b(error|fatal|nan)\b", result["log"], re.IGNORECASE):
            sys.exit(f"{mode}: diagnostics in the native log")
        return pages[0], pages[1]


    def rms(page: dict, rows: slice = slice(None)) -> np.ndarray:
        """Rms sizes (x, y, z) in m of a bunch; z = beta c times the arrival-time spread."""
        return np.array(
            [np.std(page["x"][rows]), np.std(page["y"][rows]), BETA * C * np.std(page["t"][rows])]
        )


    def fmt(values: np.ndarray) -> str:
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values) + 0.0) + "]"

    ```

## `_scring.py` { #_scring }

[Download _scring.py](_scring.py)

??? example "Source: _scring.py"

    ```python
    """Shared setup for the SCMULT space-charge reproducers: a closed four-bend ring, a gridded beam and
    the Gaussian-beam kick computed from the Poisson integral, independently of Elegant's code.

    One turn around a ring of four 1 m sector bends (angle pi/2, K1 = -0.5 m^-2), each preceded by a
    drift of length L. `insert_sceffects` places one SCMULT after the first drift, and a WATCH right
    after it records every particle's slope kick before the first bend.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import particles, run
    from scipy.integrate import quad
    from scipy.special import ndtri

    # SDDS constants.h, as Elegant uses them.
    C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
    P = 10.0  # momentum, units of m_e c (electrons)
    GAMMA = np.sqrt(1 + P**2)
    BETA = P / GAMMA
    CHARGE = 1e-9  # C
    BEND_LENGTH = 1.0  # m, four bends make the ring


    def levels(n: int, uniform: bool = False) -> np.ndarray:
        """n symmetric levels of unit rms: Gaussian quantiles, or equally spaced for a uniform beam."""
        q = (np.arange(n) + 0.5) / n
        q = q - 0.5 if uniform else ndtri(q)
        return q / np.std(q)


    def beam(
        sx: float, sy: float, times: np.ndarray, n: int = 8, copies: int = 1
    ) -> dict[str, np.ndarray]:
        """Particles on the full grid of n levels in x and y and the given arrival times, zero slopes.

        The rms sizes are exactly sx and sy, which is what SCMULT reads from the particles.
        """
        q = levels(n)
        x, y, t = (a.ravel() for a in np.meshgrid(sx * q, sy * q, times, indexing="ij"))
        x, y, t = (np.tile(a, copies) for a in (x, y, t))
        return {"x": x, "y": y, "t": t, "p": np.full(len(x), P), "particleID": np.arange(1, len(x) + 1)}


    def circumference(length: float) -> float:
        return 4 * (BEND_LENGTH + length)


    def track(
        mode: str,
        length: float,
        particles_in: dict[str, np.ndarray],
        *,
        horizontal: int = 1,
        vertical: int = 1,
        nonlinear: int = 1,
        uniform: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slope kicks (xp, yp) of every particle after SCMULT, in input order, and the effective
        integrating lengths of the two planes.

        The effective length of a drift of constant beam size is L (beta_in + beta_out) / (2 beta_out),
        from the trapezoid weighting that SCMULT applies to the matched beta functions at the entrance
        and exit of the drift; the beta functions are read from the matched Twiss output.
        """
        lattice = (
            f"Q: CHARGE,TOTAL={CHARGE!r}\n"
            f"D: DRIF,L={length!r}\n"
            'W: WATCH,FILENAME="watch.sdds",MODE="coordinates"\n'
            f"A: DRIF,L={length!r}\n"
            f"B: SBEND,L={BEND_LENGTH!r},ANGLE={np.pi / 2!r},K1=-0.5\n"
            "BL: LINE=(Q,D,W,B,A,B,A,B,A,B)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f'&insert_sceffects name="D",skip=1,horizontal={horizontal},vertical={vertical},'
            f"nonlinear={nonlinear},uniform_distribution={uniform} &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            '&twiss_output filename="twiss.sdds",matched=1 &end\n'
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": particles_in},
            outputs=("watch.sdds", "twiss.sdds"),
        )
        twiss = result["twiss.sdds"]
        names = list(twiss[0, "ElementName"])
        if not names.index("D") < names.index("MYSC") < names.index("W"):
            raise RuntimeError(f"unexpected element order {names}")
        leff = np.array(
            [
                length * (b[0] + b[names.index("D")]) / (2 * b[names.index("MYSC")])
                for b in (twiss[0, "betax"], twiss[0, "betay"])
            ]
        )
        if not (np.all(np.isfinite(leff)) and np.all(leff > 0)):
            raise RuntimeError(f"{mode}: invalid effective lengths {leff}")
        out = particles(result["watch.sdds"])
        if not np.array_equal(out["particleID"], particles_in["particleID"]):
            raise RuntimeError(f"{mode}: particles lost before the kick")
        # The initial slopes are zero, so the slopes after SCMULT are the kicks.
        return np.column_stack([out["xp"], out["yp"]]), leff


    def poisson_field(x: np.ndarray, y: np.ndarray, sx: float, sy: float) -> np.ndarray:
        """Transverse field (Fx, Fy), in 1/m, of a Gaussian charge density with rms sizes sx and sy.

        Heat-kernel solution of the 2D Poisson equation, written with positive real integrands:
          Fx = x/2 int_0^inf exp[-x^2/(2(sx^2+t)) - y^2/(2(sy^2+t))] / [(sx^2+t)^(3/2) (sy^2+t)^(1/2)] dt
        and Fy with the planes exchanged. Round beams give Gauss's law,
        F = (1 - exp[-r^2/(2 sigma^2)]) (x, y)/r^2, and the origin gives the linear force,
        Fx = x/[sx (sx + sy)].
        """
        r2 = (sy / sx) ** 2  # t = sx^2 u makes the integrals dimensionless

        def integral(a: float, b: float, plane: int) -> float:
            def integrand(u: float) -> float:
                common = np.exp(-(a**2) / (2 * (1 + u)) - b**2 / (2 * (r2 + u)))
                common /= 2 * np.sqrt((1 + u) * (r2 + u))
                return common * (a / (1 + u) if plane == 0 else b / (r2 + u)) / sx

            return quad(integrand, 0, np.inf, epsabs=0, epsrel=1e-13)[0]

        pairs, inverse = np.unique(np.column_stack([x, y]), axis=0, return_inverse=True)
        unique = np.array(
            [[integral(px / sx, py / sx, 0), integral(px / sx, py / sx, 1)] for px, py in pairs]
        )
        return unique[inverse.reshape(-1)]


    def kick(
        particles_in: dict[str, np.ndarray],
        leff: np.ndarray,
        *,
        nonlinear: bool,
        uniform: bool = False,
    ) -> np.ndarray:
        """Physical slope kick (xp, yp) of every particle, shape (n, 2).

        dx' = K L lambda F, with K = 2 r_e / (beta^2 gamma^3) and L the effective length of the plane;
        K multiplies the particle number per metre lambda, not a charge density.
        beta^2 gamma^3 = P^2 sqrt(1 + P^2). The longitudinal density lambda = N f(z), in particles per
        metre with N = CHARGE/e, is Gaussian, f = exp(-z^2/(2 sz^2))/(sqrt(2 pi) sz), or uniform over the
        interval of the same rms, f = 1/(sqrt(12) sz). The field F is the Gaussian transverse field
        (nonlinear) or its linear part near the axis.
        """
        sx, sy = np.std(particles_in["x"]), np.std(particles_in["y"])
        x = particles_in["x"] - np.mean(particles_in["x"])
        y = particles_in["y"] - np.mean(particles_in["y"])
        z = BETA * C * (particles_in["t"] - np.mean(particles_in["t"]))
        sz = np.std(z)
        shape = np.full_like(z, 1 / (np.sqrt(12) * sz)) if uniform else _gauss(z, sz)
        strength = 2 * (CHARGE / QE) * RE / (P**2 * np.sqrt(1 + P**2))
        if nonlinear:
            field = poisson_field(x, y, sx, sy)
        else:
            field = np.column_stack([x / (sx * (sx + sy)), y / (sy * (sx + sy))])
        return strength * shape[:, None] * leff[None, :] * field


    def _gauss(z: np.ndarray, sz: float) -> np.ndarray:
        return np.exp(-(z**2) / (2 * sz**2)) / (np.sqrt(2 * np.pi) * sz)


    def fmt(values: np.ndarray) -> str:
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"


    def require_finite(label: str, *arrays: np.ndarray) -> None:
        """Fail the run when an array holds a NaN or infinity; NaN would otherwise pass as a mismatch."""
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise RuntimeError(f"{label}: nonfinite values")


    def compare(
        label: str, observed: np.ndarray, expected: np.ndarray, rtol: float, atol: float = 1e-18
    ) -> bool:
        """Print the first particle's (xp, yp) and return True when any particle disagrees.

        Arrays of different shape or with a nonfinite entry make the comparison invalid: the run fails
        instead of returning a verdict.
        """
        observed, expected = np.asarray(observed), np.asarray(expected)
        if observed.shape != expected.shape:
            raise RuntimeError(f"{label}: shapes {observed.shape} and {expected.shape} differ")
        require_finite(label, observed, expected)
        print(f"{label}: observed={fmt(observed[0])} expected={fmt(expected[0])}")
        return not np.allclose(observed, expected, rtol=rtol, atol=atol)

    ```

## `_sdds.py` { #_sdds }

[Download _sdds.py](_sdds.py)

??? example "Source: _sdds.py"

    ```python
    """Shared helpers for the reproducers of SDDS library defects: run the SDDS utility `sddsprocess`
    on inline tables and compile small C programs against the SDDS libraries.

    `sddsprocess` must be on PATH, as for the Elegant reproducers; the SDDS build tree (its `lib`
    directory with `libmdbmth.a` and `libmdblib.a`) is found from the location of that executable.
    Compiling needs `gcc`.
    """

    from __future__ import annotations

    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np
    import pysdds

    TIMEOUT = 120  # seconds per native run


    def sdds_lib() -> Path:
        """The `lib/Linux-x86_64` directory of the SDDS build that provides `sddsprocess`."""
        tool = shutil.which("sddsprocess")
        if tool is None:
            sys.exit("sddsprocess must be on PATH (see the elegant installation).")
        lib = Path(tool).resolve().parents[2] / "lib" / "Linux-x86_64"
        if not (lib / "libmdbmth.a").exists():
            sys.exit(f"{lib} has no libmdbmth.a: sddsprocess is not in an SDDS build tree.")
        return lib


    def execute(argv: list[str], cwd: Path) -> str:
        """Run a native command; a nonzero exit status or a timeout raises RuntimeError."""
        done = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT, check=False
        )
        if done.returncode:
            raise RuntimeError(
                f"{argv[0]} exited with status {done.returncode}:\n{done.stderr[-2000:]}"
            )
        return done.stdout


    def c_check(source: str) -> list[list[str]]:
        """Compile a C program against the SDDS math libraries, run it, and return the stdout words.

        The program prints one result per line; each line comes back as a list of words.
        """
        lib = sdds_lib()
        with tempfile.TemporaryDirectory(prefix="sdds-repro-") as tmp:
            work = Path(tmp)
            (work / "check.c").write_text(source)
            execute(
                [
                    "gcc",
                    "-O2",
                    "check.c",
                    str(lib / "libmdbmth.a"),
                    str(lib / "libmdblib.a"),
                    "-fopenmp",
                    "-lm",
                    "-o",
                    "check",
                ],
                work,
            )
            return [line.split() for line in execute([str(work / "check")], work).splitlines()]


    def process(columns: dict[str, np.ndarray], *options: str) -> dict[str, np.ndarray]:
        """Run `sddsprocess` on a one-page table of double columns.

        Returns every parameter and column of the output page as a NumPy array (a parameter has
        one element). The input is written as binary SDDS, which carries the doubles exactly.
        """
        table = pysdds.SDDSFile(add_data_nm=True)
        table.mode = "binary"
        table.n_pages = 1
        for name, values in columns.items():
            table.add_column(name, "double", data=[np.asarray(values, dtype=np.float64)])
        with tempfile.TemporaryDirectory(prefix="sdds-repro-") as tmp:
            work = Path(tmp)
            pysdds.write(table, work / "input.sdds", overwrite=True)
            execute(["sddsprocess", "input.sdds", "output.sdds", *options], work)
            out = pysdds.read(work / "output.sdds")
            if out.n_pages != 1:
                raise RuntimeError(f"sddsprocess wrote {out.n_pages} pages")
            result = {p.name: np.atleast_1d(np.asarray(out[p.name].data[0])) for p in out.parameters}
            result.update({c.name: np.asarray(out[0, c.name]) for c in out.columns})
            return result

    ```

## `_slicing.py` { #_slicing }

[Download _slicing.py](_slicing.py)

??? example "Source: _slicing.py"

    ```python
    """Coordinate, lattice and derivative helpers for division and inverse-map probes.

    The native runner uses one fresh process per case. Its inputs use slopes and SDDS
    absolute momentum; the returned q uses (x, xprime, y, yprime, beta*c*t, delta).
    """

    import os

    for _name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[_name] = "1"

    import _elegant as native
    import numpy as np

    C = 299792458.0
    MASS_MEV = 0.51099906  # SDDS constants.h, used by the executed elegant build.
    native.TIMEOUT = 290  # ty: ignore[invalid-assignment]  # Per-probe native deadline.


    def beta(p):
        return p / np.hypot(p, 1)


    def track(
        _label,
        lattice,
        rays,
        *,
        p0=250.0,
        reverse=False,
        divisions=1,
        before="",
        matrix=False,
        mode="serial",
        default_order=1,
    ):
        """Track fixed rays, checking survival, IDs and finiteness before comparing maps."""
        q = np.array(rays, dtype=float, ndmin=2)
        p = p0 * (1 + q[:, 5])
        beam = dict(zip(("x", "xp", "y", "yp"), q[:, :4].T, strict=True))
        beam.update(t=q[:, 4] / (C * beta(p)), p=p, particleID=np.arange(1, len(q) + 1))
        commands = before + (
            "&run_setup lattice=lattice.lte, use_beamline=BL, "
            f"p_central={p0:.17g}, back_tracking={int(reverse)}, "
            f"element_divisions={divisions}, default_order={default_order}, output=out.sdds &end\n"
            "&run_control n_steps=1 &end\n"
        )
        if matrix:
            commands += "&matrix_output SDDS_output=matrix.sdds, SDDS_output_order=1, output_at_each_step=1 &end\n"
        commands += "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
        result = native.run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": beam},
            outputs=("out.sdds", "matrix.sdds") if matrix else ("out.sdds",),
        )
        data = native.particles(result["out.sdds"])
        if not np.array_equal(data["particleID"], beam["particleID"]):
            raise RuntimeError("Particle IDs changed or particles were lost")
        raw = np.column_stack([data[name] for name in ("x", "xp", "y", "yp", "t", "p")])
        if raw.shape != q.shape:
            raise RuntimeError("Unexpected output shape")
        out = raw.copy()
        out[:, 4] = C * raw[:, 4] * beta(raw[:, 5])
        out[:, 5] = raw[:, 5] / p0 - 1
        answer = {"q": out, "raw": raw}
        if matrix:
            mat = result["matrix.sdds"]
            answer["R"] = np.array([[mat[0, f"R{i}{j}"][-1] for j in range(1, 7)] for i in range(1, 7)])
            answer["C"] = np.array([mat[0, f"C{i}"][-1] for i in range(1, 7)])
            if not np.isfinite(answer["R"]).all() or not np.isfinite(answer["C"]).all():
                raise RuntimeError("Nonfinite matrix output")
        return answer


    def canonical(q):
        """Exact normalized transverse momentum, not the SDDS slopes."""
        q = np.array(q, ndmin=2)
        return (1 + q[:, 5, None]) * q[:, [1, 3]] / np.sqrt(1 + q[:, 1, None] ** 2 + q[:, 3, None] ** 2)


    def close(a, b, atol, label):
        error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        if not np.isfinite(error) or error > atol:
            raise RuntimeError(f"{label}: error={error:.17g}, tolerance={atol:g}")
        return error


    def discrepancy(actual, expected, faulty, atol):
        """Accept the physical result or the established faulty law; reject unrelated failures."""
        if np.allclose(actual, expected, rtol=0, atol=atol):
            return False
        close(actual, faulty, atol, "Unexpected discrepancy")
        return True


    def straight(kind, *, length=0.6, hk=0.00017, vk=-0.00011, slices=30, strength=0.0, order=4):
        pars = (
            f"K1={strength:.17g}"
            if kind == "QUAD"
            else (
                f"K{2 if kind == 'KSEXT' else 1}={strength:.17g},N_SLICES={slices},INTEGRATION_ORDER={order}"
            )
        )
        return f"Q: {kind},L={length:.17g},{pars},HKICK={hk:.17g},VKICK={vk:.17g}\nBL: LINE=(Q)\n"


    def bend(*, length=0.6, angle=0.045, hk=0.00017, vk=-0.00011, slices=66, expand=0):
        return (
            f"Q: CSBEND,L={length:.17g},ANGLE={angle:.17g},XKICK={hk:.17g},YKICK={vk:.17g},"
            f"N_SLICES={slices},INTEGRATION_ORDER=4,EXPAND_HAMILTONIAN={expand},"
            "EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,SYNCH_RAD=0,ISR=0\nBL: LINE=(Q)\n"
        )


    def linear_rays(h):
        rays = [np.zeros(6)]
        for j in range(4):
            for multiplier in (-2, -1, 1, 2):
                q = np.zeros(6)
                q[j] = multiplier * h
                rays.append(q)
        return np.array(rays)


    def derivative(out, h):
        # Four-point stencils cancel the cubic terms in the linear fringe derivative.
        return np.column_stack(
            [
                (
                    out[1 + 4 * j, :4]
                    - 8 * out[2 + 4 * j, :4]
                    + 8 * out[3 + 4 * j, :4]
                    - out[4 + 4 * j, :4]
                )
                / (12 * h)
                for j in range(4)
            ]
        )

    ```

## `_sole.py` { #_sole }

[Download _sole.py](_sole.py)

??? example "Source: _sole.py"

    ```python
    """Shared runner for the SOLE spin-tracking reproducers: one solenoid, one pass, spin tracking on.

    `track` sends identical copies of one particle (so that each MPI worker tracks some) through a
    lattice holding a single element and returns that particle's final coordinates and spin. Spin
    components (spx, spy, spz) are in the fixed straight element frame; p is βγ. Constants are those of
    SDDS `constants.h` and of the species table of Elegant's `change_particle` command, so that a
    different choice of constants cannot pass for an algorithm defect.
    """

    from __future__ import annotations

    import sys

    import numpy as np
    from _elegant import particles, run

    C, QE, ME = 299792458.0, 1.60217733e-19, 9.1093897e-31  # SDDS constants.h
    A_ELECTRON = 0.00115965218128  # electron and positron anomalous magnetic moment in Elegant
    A_PROTON = 1.792847
    A_MUON = 1.165921e-3
    COPIES = 6
    COORDINATES = ("x", "xp", "y", "yp", "t", "p")
    SPIN = ("spx", "spy", "spz")


    def track(
        mode: str,
        element: str,
        p0: float,
        start: dict[str, float],
        before: str = "",
        setup: str = "",
        after_setup: str = "",
        outputs: tuple[str, ...] = ("out.sdds",),
    ) -> tuple[dict[str, float], dict]:
        """Track COPIES copies of one particle through `S: <element>` and return its exit state.

        start: initial x, xp, y, yp, spx, spy, spz (missing values are zero; p is p0). before: commands
        ahead of run_setup (change_particle); setup: extra run_setup fields; after_setup: commands
        between run_control and the beam (matrix_output). Returns ({name: value}, run result).
        """
        lattice = f"S: {element}\nBL: LINE=(S)\n"
        commands = (
            before
            + f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, output=out.sdds, "
            f"spin_tracking=1, default_order=2{setup} &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            + after_setup
            + "&sdds_beam input=beam.sdds &end\n&track &end\n"
        )
        beam = {n: np.full(COPIES, float(start.get(n, 0.0))) for n in ("x", "xp", "y", "yp", *SPIN)}
        beam["p"] = np.full(COPIES, p0)
        result = run(
            {"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam}, outputs=outputs
        )
        out = particles(result["out.sdds"])
        if len(out["p"]) != COPIES:
            sys.exit(f"{mode}: {COPIES - len(out['p'])} particles lost")
        state = np.column_stack([out[n] for n in (*COORDINATES, *SPIN)])
        if not np.all(np.isfinite(state)):
            sys.exit(f"{mode}: nonfinite orbit or spin")
        if np.ptp(state, axis=0).max() != 0:
            sys.exit(f"{mode}: identical particles ended in different states")
        if abs(np.linalg.norm(state[0, 6:]) - 1) > 1e-14:
            sys.exit(f"{mode}: the spin norm is not 1")
        return {n: float(v) for n, v in zip((*COORDINATES, *SPIN), state[0], strict=True)}, result


    def axial_rotation(theta: float, spin=(1.0, 0.0, 0.0)) -> np.ndarray:
        """Spin rotated by the angle theta about +z (counterclockwise seen from +z)."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([c * spin[0] - s * spin[1], s * spin[0] + c * spin[1], spin[2]])


    def on_axis_angle(ks: float, length: float, anomaly: float) -> float:
        """Spin rotation angle about +z on the axis of a solenoid of strength KS, on momentum.

        KS = q B/(m c p0) for the signed physical charge q (Elegant's "-Bs/(B*Rho)" with its legacy
        charge sign). With the velocity along B, magnetic Thomas-BMT reduces to
        dS/dt = -(q/m)(1 + a) B/gamma z-hat x S; dt = gamma dz/(beta gamma c) turns it into
        dtheta/dz = -q (1 + a) B/(m c p) = -KS (1 + a).
        """
        return -ks * length * (1 + anomaly)


    def fmt(values) -> str:
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def compare(label: str, observed, expected, atol: float, rtol: float = 0.0) -> bool:
        """Print one case; return True when observed and expected agree within atol + rtol*|expected|."""
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bool(np.allclose(observed, expected, rtol=rtol, atol=atol))

    ```

## `_trwake.py` { #_trwake }

[Download _trwake.py](_trwake.py)

??? example "Source: _trwake.py"

    ```python
    """Shared setup for the TRWAKE reproducers: one TRWAKE element tracking 18 electron macroparticles,
    and the transverse wake kick of every particle computed as a direct causal pair sum, independently
    of Elegant's binning and convolution.

    The wake table has spacing `spacing` and starts at W(0) = 0, and every particle arrival time lies
    on a multiple of `spacing`, so each pair delay is a grid point and no wake interpolation enters.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import particles, run

    # SDDS constants.h: electron rest energy in eV.
    REST_EV = 510999.06
    N = 18  # particles
    RTOL, ATOL = 4e-13, 1e-19  # comparison of slopes


    def wake(delay: np.ndarray) -> np.ndarray:
        """Wx(t) in V/C/m for t >= 0 behind the source: 1.7e15 sin(2 pi 6 GHz t) exp(-t/70 ps)."""
        t = np.maximum(delay, 0)
        return np.where(delay >= 0, 1.7e15 * np.sin(2 * np.pi * 6e9 * t) * np.exp(-t / 70e-12), 0.0)


    def wake_table(spacing: float) -> str:
        """ASCII SDDS table of Wx and Wy = -Wx on 257 points of the given spacing."""
        grid = np.arange(257) * spacing
        rows = "".join(
            f"{float(t)!r} {float(w)!r} {float(-w)!r}\n" for t, w in zip(grid, wake(grid), strict=True)
        )
        return (
            "SDDS1\n"
            "&column name=time, units=s, type=double &end\n"
            "&column name=wx, units=V/C/m, type=double &end\n"
            "&column name=wy, units=V/C/m, type=double &end\n"
            "&data mode=ascii, no_row_counts=1 &end\n" + rows
        )


    def beam(times, x, y, momentum) -> dict[str, np.ndarray]:
        """Particles with zero slopes at the given arrival times, positions and momenta (m_e c)."""
        return {
            "x": np.broadcast_to(x, N).astype(float),
            "y": np.broadcast_to(y, N).astype(float),
            "t": np.asarray(times, dtype=float),
            "p": np.broadcast_to(momentum, N).astype(float),
            "particleID": np.arange(1, N + 1),
        }


    def track(
        mode: str,
        columns: dict[str, np.ndarray],
        *,
        shift: tuple[float, float] = (0.0, 0.0),
        drive: int = 1,
        n_bins: int = 0,
        charge: float = 0.6e-9,
        drift: float = 0.0,
        spacing: float = 2e-12,
    ) -> tuple[np.ndarray, str]:
        """Slope kicks (xp, yp), shape (N, 2), of every particle through TRWAKE, and the native log.

        drive=1 is the ordinary dipole wake (kick from the leading particles' offsets, probe
        exponent 0); drive=0 is the quadrupole wake (kick proportional to the particle's own offset).
        shift is the structure's misalignment (DX, DY); drift is a drift before the structure.
        """
        lattice = (
            f"CH: CHARGE,TOTAL={charge!r}\n"
            f"D0: DRIF,L={drift!r}\n"
            'R: TRWAKE,INPUTFILE="wake.sdds",TCOLUMN="time",WXCOLUMN="wx",WYCOLUMN="wy",'
            f"DX={shift[0]!r},DY={shift[1]!r},N_BINS={n_bins},INTERPOLATE=0,SMOOTHING=0,"
            f"X_DRIVE_EXPONENT={drive},Y_DRIVE_EXPONENT={drive},"
            f"X_PROBE_EXPONENT={1 - drive},Y_PROBE_EXPONENT={1 - drive}\n"
            "BL: LINE=(CH,D0,R)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={float(np.mean(columns['p']))!r},"
            "output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands, "wake.sdds": wake_table(spacing)},
            mode,
            {"beam.sdds": columns},
        )
        out = particles(result["out.sdds"])
        if not np.array_equal(out["particleID"], columns["particleID"]):
            raise RuntimeError(f"{mode}: particles lost")
        # The wake changes only the slopes (the drift acts on s, not on x or y).
        if not (np.array_equal(out["x"], columns["x"]) and np.array_equal(out["y"], columns["y"])):
            raise RuntimeError(f"{mode}: transverse positions changed")
        return np.column_stack([out["xp"], out["yp"]]), result["log"]


    def pair_sum(
        columns: dict[str, np.ndarray],
        *,
        shift: tuple[float, float] = (0.0, 0.0),
        drive: int = 1,
        charge: float = 0.6e-9,
        spacing: float = 2e-12,
        probe_offset: tuple[float, float] | None = None,
        binned: np.ndarray | None = None,
    ) -> np.ndarray:
        """Physical slope kicks (xp, yp), shape (N, 2), from the causal pair sum.

        With structure offset d, drive exponent a and probe exponent b (a + b = 1) and zero incoming
        slopes, particle i receives
          dx'_i = q (x_i - d)^b sum_j [(x_j - d)^a Wx(t_i - t_j)] / (E0 p_i),
        summed over the particles j that are not later than i, with q = charge / N the macroparticle
        charge and E0 the electron rest energy in eV; y is the same with Wy = -Wx. Translating beam
        and structure together leaves every (x - d) unchanged. Giving `probe_offset` replaces d in the
        probe factor by that value, which evaluates the probe at the global coordinate (d = 0).
        `binned` (boolean) restricts both sources and probes to the particles Elegant bins; the
        others receive no kick.
        """
        t, p = columns["t"], columns["p"]
        delays = t[:, None] - t[None, :]
        index = np.rint(delays / spacing).astype(int)
        if not np.allclose(delays / spacing, index, rtol=0, atol=3e-12):
            raise ValueError("pair delays are not multiples of the wake spacing")
        # Causal matrix W(t_i - t_j) taken on the table grid; the coincident pairs have W(0) = 0.
        matrix = np.where(index >= 0, wake(np.maximum(index, 0) * spacing), 0.0)
        if binned is None:
            binned = np.ones(N, dtype=bool)
        matrix = matrix * binned[None, :]
        kicks = np.zeros((N, 2))
        for plane, (name, sign) in enumerate((("x", 1.0), ("y", -1.0))):
            rel = columns[name] - shift[plane]
            probe = rel if probe_offset is None else columns[name] - probe_offset[plane]
            source = rel if drive else np.ones(N)
            voltage = sign * charge / N * (matrix @ source)
            kicks[:, plane] = voltage * (1.0 if drive else probe) / (REST_EV * p)
        kicks[~binned] = 0.0
        return kicks


    def fmt(values: np.ndarray) -> str:
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values) + 0.0) + "]"


    def compare(label: str, observed: np.ndarray, expected: np.ndarray) -> bool:
        """Print the last particle's (xp, yp); return True when any particle disagrees."""
        print(f"{label}: observed={fmt(observed[-1])} expected={fmt(expected[-1])}")
        return not np.allclose(observed, expected, rtol=RTOL, atol=ATOL)

    ```

## `_wake.py` { #_wake }

[Download _wake.py](_wake.py)

??? example "Source: _wake.py"

    ```python
    """Shared setup for the WAKE, TRWAKE and CORGPIPE reproducers: an ASCII SDDS wake table, one serial
    run of an electron beam through a lattice that reads it, and the energy change of each particle.

    Times are in s, momenta p = beta*gamma in units of m_e c and energies in eV. A positive wake loss
    is energy removed from an electron.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import RunFailure, particles, run

    ME_EV = 0.51099906e6  # SDDS constants.h, as Elegant uses it


    def table(columns: dict[str, np.ndarray], units: dict[str, str]) -> str:
        """ASCII SDDS page with the given double columns (all of one length)."""
        names = list(columns)
        head = "SDDS1\n" + "".join(
            f"&column name={name}, units={units[name]}, type=double &end\n" for name in names
        )
        rows = "".join(
            " ".join(repr(float(columns[name][i])) for name in names) + "\n"
            for i in range(len(columns[names[0]]))
        )
        return head + "&data mode=ascii, no_row_counts=1 &end\n" + rows


    def beam(times: np.ndarray, p: float, x: float = 0.0) -> dict[str, np.ndarray]:
        """Particles at the given arrival times, all at momentum p and offset x, zero slopes."""
        n = len(times)
        return {
            "x": np.full(n, x),
            "t": np.asarray(times, dtype=float),
            "p": np.full(n, p),
            "particleID": np.arange(1, n + 1),
        }


    def track(
        elements: str, line: str, particles_in: dict, p0: float, files: dict[str, str] | None = None
    ) -> tuple[dict[str, np.ndarray], str]:
        """Serial run through BL: LINE=(line); (final coordinates by particle ID, run log).

        `files` holds extra inputs such as wake tables. A nonzero exit raises RunFailure.
        """
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        inputs = {"lattice.lte": elements.rstrip("\n") + f"\nBL: LINE=({line})\n", "run.ele": commands}
        result = run({**inputs, **(files or {})}, beams={"beam.sdds": particles_in})
        out = particles(result["out.sdds"])
        if len(out["p"]) != len(particles_in["p"]):
            raise SystemExit("particles were lost")
        return out, result["log"]


    def loss(out: dict[str, np.ndarray], p0: float) -> np.ndarray:
        """Energy lost by each particle (eV), from its final momentum and the initial p0."""
        return ME_EV * (np.sqrt(1 + p0**2) - np.sqrt(1 + out["p"] ** 2))


    __all__ = ["ME_EV", "RunFailure", "beam", "loss", "table", "track"]

    ```

## `_watch_energy.py` { #_watch_energy }

[Download _watch_energy.py](_watch_energy.py)

??? example "Source: _watch_energy.py"

    ```python
    """WATCH energy fixtures and independent relativistic oracles for IO-07 and IO-08."""

    import math
    from decimal import Decimal, localcontext

    import numpy as np
    from _diagnostics import conversion_defect, run, scalar
    from _elegant import particles

    C = Decimal(299792458)
    E = Decimal("1.60217733e-19")
    ME = Decimal("9.1093897e-31")
    MP = Decimal("1.6726485e-27")
    MMU = Decimal("1.88353109e-28")
    ME_MEV = Decimal("0.51099906")
    PS = [0.75, 1.5, 2.75, 5.5] * 5
    LENGTH = Decimal("0.75")
    RTOL = 1e-14
    LATTICE = """D: EDRIFT,L=0.75
    C: WATCH,FILENAME="centroids.sdds",MODE="centroids"
    P: WATCH,FILENAME="parameters.sdds",MODE="parameters"
    S: WATCH,FILENAME="selected.sdds",MODE="centroids",START_PID=1,END_PID=7
    W: WATCH,FILENAME="coordinates.sdds",MODE="coordinates"
    BL: LINE=(D,C,P,S,W)
    """


    def energy(ps, mass, tabulated=False):
        """Mean K in MeV from E² − (pc)² = (mc²)², evaluated at 70-digit precision."""
        with localcontext() as ctx:
            ctx.prec = 70
            rest = ME_MEV if tabulated else mass * C**2 / E / Decimal(10)**6
            # SDDS p = beta*gamma, so gamma = sqrt(1+p²); average energies, not momenta.
            return float(sum(((1 + Decimal(str(p))**2).sqrt() - 1) * rest for p in ps) / len(ps))


    def check(mode, label, species, mass, charge=1, tabulated=False):
        """Check both energy modes, ID selection, momenta and exact-drift flight times."""
        commands = f"""&change_particle name={species} &end
    &run_setup lattice=lattice.lte,use_beamline=BL,p_central=2,output=out.sdds,random_number_seed=24681 &end
    &run_control n_passes=1 &end
    &sdds_beam input=beam.sdds,use_bunched_mode=1 &end
    &track &end
    """
        result = run(
            {"lattice.lte": LATTICE, "run.ele": commands}, mode,
            {"beam.sdds": {"p": np.array(PS), "particleID": np.arange(1, 21)}},
            ("centroids.sdds", "parameters.sdds", "selected.sdds", "coordinates.sdds", "out.sdds"),
        )
        defects = []
        values = []
        for name, ps in (("centroids.sdds", PS), ("parameters.sdds", PS), ("selected.sdds", PS[:7])):
            data = result[name]
            actual = scalar(data, "KAverage", column=True)
            expected = energy(ps, mass, tabulated)
            schema = next(c for c in data.columns if c.name == "KAverage")
            if schema.nm["units"] != "MeV" or schema.nm["description"] != "Mean kinetic energy":
                raise RuntimeError("The KAverage schema changed")
            p_mean = scalar(data, "pAverage", column=True)
            if not math.isclose(p_mean, math.fsum(ps) / len(ps), rel_tol=5e-15, abs_tol=0):
                raise RuntimeError("Mean-momentum control failed")
            affected = (mode == "mpi" and not tabulated) or (mode == "serial" and charge != 1)
            if affected:
                faulty = energy(ps, ME, True) if mode == "mpi" else expected / charge
                defects.append(conversion_defect(actual, expected, faulty, rtol=RTOL))
            elif not math.isclose(actual, expected, rel_tol=RTOL, abs_tol=0):
                raise RuntimeError("Species energy control failed")
            values.append(actual)
        with localcontext() as ctx:
            ctx.prec = 70
            # Zero slopes and zero initial time give t = L*gamma/(p*c).
            times = np.array([float(LENGTH * (1 + Decimal(str(p))**2).sqrt()
                                    / (Decimal(str(p)) * C)) for p in PS])
        max_time_error = 0.0
        for name in ("coordinates.sdds", "out.sdds"):
            data = particles(result[name])
            if not np.array_equal(data["particleID"], np.arange(1, 21)):
                raise RuntimeError("Particle-ID control failed")
            if not np.array_equal(data["p"], PS):
                raise RuntimeError("Particle-momentum control failed")
            if not np.allclose(data["t"], times, rtol=5e-15, atol=0):
                raise RuntimeError("Exact-drift time control failed")
            max_time_error = max(max_time_error, float(np.max(np.abs(data["t"] / times - 1))))
        print(
            f"{mode}/{label}: KAverage_MeV centroids={values[0]:.17g} parameters={values[1]:.17g} "
            f"expected={energy(PS, mass, tabulated):.17g}; IDs1-7={values[2]:.17g} "
            f"expected={energy(PS[:7], mass, tabulated):.17g} (mean((sqrt(1+p²)-1)*mc²)); "
            f"IDs,p exact; max drift relative error={max_time_error:.3g}"
        )
        return defects

    ```

## `_ztransverse.py` { #_ztransverse }

[Download _ztransverse.py](_ztransverse.py)

??? example "Source: _ztransverse.py"

    ```python
    """Shared setup for the broadband ZTRANSVERSE reproducers: decks, analytic impedance, Fourier oracle.

    The beam is twelve electron macroparticles (p = 50 m_e c, 2 nC) in two adjacent time bins of 50 ps, six particles
    each, so that the transverse source moment of each bin is six times its offset. `kicks` tracks it
    through a ZTRANSVERSE element, either the built-in broadband resonator or a table holding the
    documented impedance of the same resonator on the same frequency grid; `oracle` is the kick of the
    analytic impedance, evaluated as a finite Fourier series without Elegant's FFT.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import particles, run

    ELECTRON_EV = 0.51099906e6  # SDDS constants.h me_mev, in eV
    RS, Q, FR = 1e6, 1.0, 1e9  # shunt impedance Ohm/m, quality factor, resonant frequency Hz
    DT = 50e-12  # bin width, s
    CHARGE, COUNT, P = 2e-9, 12, 50.0
    TIMES = np.repeat([0.0, DT], COUNT // 2)  # s; group 1 in one bin, group 2 in the next


    def impedance(r):
        """Documented broadband impedance Z(r) = (-i/r) Rs / [1 + i Q (r - 1/r)], r = f/fr, Ohm/m.

        Z(0) = Rs/Q is the limit r -> 0 of the formula (the bracket tends to -iQ/r).
        """
        r = np.asarray(r, dtype=float)
        safe = np.where(r == 0, 1.0, r)
        z = (-1j / safe) * RS / (1 + 1j * Q * (safe - 1 / safe))
        return np.where(r == 0, RS / Q, z)


    def sdds_table(n_bins):
        """ASCII SDDS impedance table on the FFT frequency grid f_k = k/(N dt), k = 0..N/2.

        The loader doubles the DC row (it assumes a folded FFT of the impedance), so the table holds
        Rs/(2Q) there; it reads the real part alone at the Nyquist row.
        """
        k = np.arange(n_bins // 2 + 1)
        f = k / (n_bins * DT)
        z = impedance(f / FR)
        re = z.real.copy()
        re[0] /= 2
        head = (
            "SDDS1\n"
            "&column name=f, type=double, units=Hz &end\n"
            "&column name=re, type=double, units=Ohms/m &end\n"
            "&column name=im, type=double, units=Ohms/m &end\n"
            f"&data mode=ascii &end\n! page number 1\n{len(k)}\n"
        )
        rows = zip(f.tolist(), re.tolist(), z.imag.tolist())
        return head + "".join(f"{a!r} {b!r} {c!r}\n" for a, b, c in rows)


    def show(mode, label, observed, expected):
        """Print one case with 16 significant digits."""

        def fmt(values):
            return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"

        print(f"{mode} {label}: observed={fmt(observed)} expected={fmt(expected)}")


    def kicks(mode, x, n_bins, broadband):
        """Horizontal slope kicks of the two particle groups, from a run with offsets `x` (m)."""
        if broadband:
            element = (
                f"ZTRANSVERSE,BROAD_BAND=1,RS={RS!r},Q={Q!r},FREQ={FR!r},"
                f"N_BINS={n_bins},BIN_SIZE={DT!r},YFACTOR=0,SMOOTHING=0,INTERPOLATE=0"
            )
        else:
            element = (
                'ZTRANSVERSE,INPUTFILE="impedance.sdds",FREQCOLUMN="f",ZXREAL="re",ZXIMAG="im",'
                "YFACTOR=0,SMOOTHING=0,INTERPOLATE=0"
            )
        lattice = f"Q0: CHARGE,TOTAL={CHARGE!r}\nZ: {element}\nBL: LINE=(Q0,Z)\n"
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        files = {"lattice.lte": lattice, "run.ele": commands}
        if not broadband:
            files["impedance.sdds"] = sdds_table(n_bins)
        beam = {"x": x, "t": TIMES, "p": np.full(COUNT, P), "particleID": np.arange(1, COUNT + 1)}
        result = run(files, mode, {"beam.sdds": beam})
        if "adjust" in result["log"].lower():
            raise RuntimeError("ZTRANSVERSE changed the requested bin settings")
        out = particles(result["out.sdds"])
        if len(out["xp"]) != COUNT:
            raise RuntimeError(f"{mode}: particles lost")
        return out["xp"].reshape(2, -1).mean(axis=1)


    def oracle(x, n_bins, nyquist_re=None):
        """Kick of the analytic impedance, as slopes of the two groups.

        A real periodic signal of period T = N dt has the Green function (Fourier series)
            W[d] = [Z_0 + Re Z_{N/2} (-1)^d + 2 Re sum_{k=1}^{N/2-1} Z_k exp(2 pi i k d / N)] / T,
        with Z_k = Z(f_k), f_k = k/T, for a delay of d bins. The slope kick of a particle in bin i is
            dx' = q sum_j x_j W[b_i - b_j] / (p0 m c^2 / e),
        q the macroparticle charge in C, x_j the offsets (a source moment of the bins), p0 m c^2/e
        the beam momentum in volts. `nyquist_re` replaces Re Z_{N/2} (to evaluate a given endpoint).
        """
        period = n_bins * DT
        k = np.arange(1, n_bins // 2)
        z = impedance(np.arange(n_bins // 2 + 1) / period / FR)
        nyquist = z[-1].real if nyquist_re is None else nyquist_re
        q = CHARGE / COUNT
        source = x.reshape(2, -1).sum(axis=1)  # m, summed offset of each group
        bins = np.array([0, 1])
        delay = bins[:, None] - bins[None, :]  # b_i - b_j
        series = np.array(
            [
                z[0].real
                + nyquist * (-1.0) ** d
                + 2 * np.sum((z[k] * np.exp(2j * np.pi * k * d / n_bins)).real)
                for d in delay.ravel()
            ]
        ).reshape(2, 2)
        return q * (series @ source) / period / (P * ELECTRON_EV)

    ```
