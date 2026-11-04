---
title: "COL-06: Single-precision reciprocals in the complex error function reduce the Gaussian force accuracy"
hide:
  - toc
---

# COL-06 — Single-precision reciprocals in the complex error function reduce the Gaussian force accuracy

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on SDDS master (`a3f730a`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [reciprocals, wofz.c:191,197](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/wofz.c#L191-L197)<br>[accuracy contract, wofz.c:65](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/wofz.c#L65)<br>[double-precision statement, wofz.c:113](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/wofz.c#L113)<br>[types, f2c.h:21,26](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/include/f2c.h#L21-L26)<br>[complexErf, complex.cc:27–33](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/complex.cc#L27-L33)<br>[wa and wb, insertSCeffects.cc:581–584](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L581-L584)<br>[kicks, insertSCeffects.cc:592–602](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L592-L602)<br>[nonlinear force, scmult.tex:24–43](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/scmult.tex#L24-L43) |

Nonlinear SCMULT kicks for elliptical Gaussian beams lose accuracy when the SDDS complex error function evaluates its power-series reciprocals in single precision despite a 14-digit accuracy contract. For 1 nC of electrons at 10 mₑc with rms sizes 2 mm horizontally and 1 mm vertically, a sampled particle's horizontal kick is −3.149×10⁻⁵ rad, differing from the independent Poisson-integral value by −7.904×10⁻¹³ rad (2.510×10⁻⁸ relative). A scalar model reproduces the native kicks with single-precision reciprocals and agrees with the integral to about 10⁻¹⁵ relative with double-precision reciprocals. Accumulated effects over many turns have not been measured.

??? example "Reproducer: COL-06.py"

    [Download COL-06.py](COL-06.py)

    ```python
    """COL-06: Single-precision reciprocals in the complex error function reduce the Gaussian force accuracy

    Nonlinear SCMULT kicks for elliptical Gaussian beams lose accuracy when the SDDS complex error function evaluates its power-series reciprocals in single precision despite a 14-digit accuracy contract. For 1 nC of electrons at 10 mₑc with rms sizes 2 mm horizontally and 1 mm vertically, a sampled particle's horizontal kick is −3.149×10⁻⁵ rad, differing from the independent Poisson-integral value by −7.904×10⁻¹³ rad (2.510×10⁻⁸ relative). A scalar model reproduces the native kicks with single-precision reciprocals and agrees with the integral to about 10⁻¹⁵ relative with double-precision reciprocals. Accumulated effects over many turns have not been measured.

    The script tracks 16 electrons through one drift of a closed four-bend ring, with an SCMULT after
    the drift, and compares each slope kick with the Gaussian field from the Poisson integral. The
    cause is isolated by a scalar model of the power series of `wofz` that divides in binary32 or in
    binary64: the binary32 model must reproduce the native kicks, and the binary64 model the Poisson
    integral.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _scring import beam, compare, kick, levels, track
    from scipy.special import wofz

    SX, SY = 2e-3, 1e-3  # m, rms sizes; sx > sy, so the kick is evaluated without swapping planes
    LENGTH = 0.5  # m, drift with the SCMULT
    # Four transverse positions (x = ±σₓ, y = ±σᵧ) and two arrival times (±0.1 ns), each combination
    # twice.
    PARTICLES = beam(SX, SY, 1e-10 * levels(2), n=2, copies=2)
    RTOL_FAULT = 1e-9  # a correct kick agrees with the Poisson integral to 1e-12
    RTOL_MODEL = 1e-11  # the models agree with the native kicks, or the integral, to 1e-14


    def f32(value: float) -> float:
        """A C `(float)` cast of a decimal constant, promoted back to double."""
        return float(np.float32(value))


    def faddeeva(z: complex, single: bool) -> complex:
        """w(z) for Im z > 0, with the power-series branch of mdbmth/wofz.c reproduced operation by
        operation. Its reciprocals are `(float)1./j`: a binary32 division of the integer j, promoted to
        double afterwards. single=False divides in binary64 instead. The model covers only the
        power-series region; the native continued-fraction branch is not modelled.
        """
        xa, ya = abs(z.real), abs(z.imag)
        qrho = (xa / f32(6.3)) ** 2 + (ya / f32(4.4)) ** 2
        if qrho >= 0.085264:
            raise ValueError("argument outside the audited power-series region")
        xquad, yquad = xa * xa - ya * ya, xa * 2 * ya

        def reciprocal(j: int) -> float:
            return float(np.float32(1) / np.float32(j)) if single else 1.0 / j

        qrho = (1 - ya / f32(4.4) * f32(0.85)) * np.sqrt(qrho)
        n = int(np.floor(qrho * 72 + 6 + 0.5))  # i_dnnt: nearest integer
        j = 2 * n + 1
        xsum, ysum = reciprocal(j), 0.0
        for i in range(n, 0, -1):
            j -= 2
            xaux = (xsum * xquad - ysum * yquad) / i
            ysum = (xsum * yquad + ysum * xquad) / i
            xsum = xaux + reciprocal(j)
        u1 = (xsum * ya + ysum * xa) * -1.12837916709551257388 + 1.0
        v1 = (xsum * xa - ysum * ya) * 1.12837916709551257388
        damp = np.exp(-xquad)
        u2, v2 = damp * np.cos(yquad), -damp * np.sin(yquad)
        u, v = u1 * u2 - v1 * v2, u1 * v2 + v1 * u2
        return complex(u, -v if z.real < 0 else v)


    def model_kick(physical: np.ndarray, single: bool) -> np.ndarray:
        """Kick of the nonlinear branch of SCMULT with `faddeeva` in place of the exact function.

        The kick is proportional to Im w for x and Re w for y, with w = w(a) - C3 w(b), where
        a = (x + i|y|)/sd, b = (x sy/sx + i|y| sx/sy)/sd, sd = sqrt(2 (sx^2 - sy^2)) and
        C3 = exp(-x^2/(2 sx^2) - y^2/(2 sy^2)). The physical kick has the same form with the exact
        function, so the model kick is the physical kick times the ratio of the two w components.
        """
        x, y = PARTICLES["x"], PARTICLES["y"]
        sd = np.sqrt(2 * (SX**2 - SY**2))
        c3 = np.exp(-(x**2) / (2 * SX**2) - y**2 / (2 * SY**2))
        ratio = np.empty_like(physical)
        for i in range(len(x)):
            a = complex(x[i] / sd, abs(y[i]) / sd)
            b = complex(x[i] / sd * SY / SX, abs(y[i]) / sd * SX / SY)
            w = faddeeva(a, single) - c3[i] * faddeeva(b, single)
            exact = complex(wofz(a)) - c3[i] * complex(wofz(b))
            ratio[i] = [w.imag / exact.imag, w.real / exact.real]
        return physical * ratio


    def deviation(observed: np.ndarray, expected: np.ndarray) -> str:
        relative = np.abs(observed / expected - 1)
        return f"max relative deviation x={relative[:, 0].max():.4g}, y={relative[:, 1].max():.4g}"


    defect = False
    for mode in modes():
        observed, leff = track(mode, LENGTH, PARTICLES)
        physical = kick(PARTICLES, leff, nonlinear=True)
        # Faulty arithmetic: native kicks differ from the Poisson integral at the 1e-8 level.
        wrong = compare(f"{mode} native vs Poisson integral", observed, physical, RTOL_FAULT)
        print(f"{mode} native vs Poisson integral: {deviation(observed, physical)}")
        single = model_kick(physical, single=True)
        double = model_kick(physical, single=False)
        # Control: the binary32 model reproduces the native kicks, the binary64 model the integral.
        if compare(f"{mode} native vs binary32 model", observed, single, RTOL_MODEL):
            sys.exit("The binary32 model does not reproduce the native kicks.")
        print(f"{mode} native vs binary32 model: {deviation(observed, single)}")
        if compare(f"{mode} Poisson integral vs binary64 model", physical, double, RTOL_MODEL):
            sys.exit("The binary64 model does not reproduce the Poisson integral.")
        print(f"{mode} Poisson integral vs binary64 model: {deviation(physical, double)}")
        defect |= wrong
    print(f"comparison: rtol={RTOL_FAULT:g} (native vs integral), {RTOL_MODEL:g} (models)")
    verdict(defect)

    ```

??? example "Helper: _elegant.py"

    [Download _elegant.py](_elegant.py)

    ```python
    """Shared runner for the Elegant reproducers: write inputs, run elegant or Pelegant, read SDDS output.

    `elegant` must be on PATH, and `Pelegant` with `mpirun` for MPI runs. RPN_DEFNS must name the
    `defns.rpn` file shipped with SDDS; elegant refuses to start without it. MPI runs use three ranks
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
    ) -> dict:
        """Run one case and return {output name: pysdds file} plus "log" (stdout and stderr).

        files: text inputs by name (lattice, command file, tables); beams: particle pages by file
        name (see write_beam), written as binary SDDS. A nonzero exit status raises RunFailure; a timeout raises RuntimeError.
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
                str(MPI_RANKS),
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

??? example "Helper: _scring.py"

    [Download _scring.py](_scring.py)

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

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found (force discrepancy and the single-precision reciprocal candidate) | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Found (force discrepancy) | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | SDDS master (`a60b9bbed7cc`), Elegant 75a2e1a1a58b source, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | SDDS master (`a60b9bbed7cc`), Elegant 75a2e1a1a58b source, CPU, serial and 3-rank Pelegant |

