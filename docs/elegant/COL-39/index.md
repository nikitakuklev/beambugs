---
title: "COL-39: Stupakov-model CSRDRIFT trapezoid integration omits the first interval behind each particle"
hide:
  - toc
---

# COL-39 — Stupakov-model CSRDRIFT trapezoid integration omits the first interval behind each particle

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [Stupakov sum, csbend.c:5088–5112](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L5088-L5112)<br>[CSRDRIFT manual, csrdrift.tex:21–27](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csrdrift.tex#L21-L27) |

CSRDRIFT with USE_STUPAKOV=1 and the upstream bend’s default TRAPAZOID_INTEGRATION=1 starts its trapezoid sum one bin behind each particle and omits the interval from zero to one bin. For a 35 μm rms Gaussian bunch after a 0.6 m, 0.3 rad bend with 400 bins, adaptive integration of the same formula gives a wake that the code misses by up to 7.085% of its peak 1 mm behind the bend, 5.306% at 10 mm and 2.348% at 100 mm. Restoring that interval reduces these errors to 0.1425%, 0.09961% and 0.05389%. The non-trapezoid path uses a right-endpoint sum instead.

??? example "Reproducer: COL-39.py"

    [Download COL-39.py](COL-39.py)

    ```python
    """COL-39: Stupakov-model CSRDRIFT trapezoid integration omits the first interval behind each particle

    CSRDRIFT with USE_STUPAKOV=1 and the upstream bend’s default TRAPAZOID_INTEGRATION=1 starts its
    trapezoid sum one bin behind each particle and omits the interval from zero to one bin. For a 35
    μm rms Gaussian bunch after a 0.6 m, 0.3 rad bend with 400 bins, adaptive integration of the
    same formula gives a wake that the code misses by up to 7.085% of its peak 1 mm behind the bend,
    5.306% at 10 mm and 2.348% at 100 mm. Restoring that interval reduces these errors to 0.1425%,
    0.09961% and 0.05389%. The non-trapezoid path uses a right-endpoint sum instead.

    The script tracks 240001 electrons (Gaussian quantiles, σ_ct = 35 μm, p = 4000 mₑc, 10⁻¹⁶ C so
    the beam does not change) through a one-slice 0.6 m, 0.3 rad CSRCSBEND (400 bins,
    STEADY_STATE=1) and a CSRDRIFT with USE_STUPAKOV=1 and one kick at its middle, for drifts of
    2 mm, 20 mm and 200 mm, and reads the drift's exported wake. Expected is the Stupakov–Emma
    post-bend wake of the Gaussian bunch, integrated adaptively in the retarded angle. The script
    then adds back only the omitted first interval, integrated from the exported density
    derivative, which must remove almost all of the error.
    """

    import numpy as np
    from _csr import E_CHARGE, R_E, C, gaussian_beam, page, track
    from _elegant import verdict
    from scipy.integrate import quad
    from scipy.optimize import brentq

    P0, SIGMA, CHARGE, BINS = 4000.0, 35e-6, 1e-16, 400
    L, ANGLE = 0.6, 0.3
    RHO = L / ANGLE
    N = CHARGE / E_CHARGE
    BEND = (
        f"L={L},ANGLE={ANGLE},N_SLICES=1,INTEGRATION_ORDER=6,SG_HALFWIDTH=1,SGDERIV_HALFWIDTH=3,"
        f"SGDERIV_ORDER=3,BINS={BINS},STEADY_STATE=1"
    )
    BEAM = gaussian_beam(240001, SIGMA, P0)
    TOLERANCE = 0.005  # of the expected peak; with the first interval restored the error is < 0.15%


    def density(z):
        return np.exp(-0.5 * (z / SIGMA) ** 2) / (np.sqrt(2 * np.pi) * SIGMA)


    def density_prime(z):
        return -z / SIGMA**2 * density(z)


    def delay(phi: float, x: float) -> float:
        """Path difference u between a particle at x = d/rho behind the bend exit and the source
        point at retarded angle phi before the exit (Stupakov and Emma, LCLS-TN-01-12, eq. 18)."""
        return RHO * phi**3 * (phi + 4 * x) / (24 * (phi + x))


    def weight(phi: float, x: float) -> float:
        """(du/dphi)/(phi + 2x) = rho phi^2 (phi + 2x) / (8 (phi + x)^2): the kernel du/(phi + 2x) of
        the post-bend wake written in the retarded angle, regular at phi = 0."""
        return RHO * phi**2 * (phi + 2 * x) / (8 * (phi + x) ** 2)


    def expected_wake(z: np.ndarray, x: float, dz: float) -> np.ndarray:
        """Energy change over a drift step dz at distance x rho behind the bend exit (eqs. 19, 22):

            Delta gamma(z) = (4 N r_e dz/rho) [ -lambda(z + u_max)/(A + 2x)
                                                 + int_0^A lambda'(z + u(phi)) du/(phi + 2x) ],
        u_max = u(A); z grows toward the tail. The bunch is far shorter than the distance to the
        bend entrance, so the contribution of the field emitted before the bend (case C) vanishes.
        """
        k4 = 4 * N * R_E * dz / RHO
        u_max = delay(ANGLE, x)
        values = []
        for zz in z:
            integral = quad(
                lambda phi, zz=zz: density_prime(zz + delay(phi, x)) * weight(phi, x),
                0,
                ANGLE,
                epsabs=1e-3,
                epsrel=2e-10,
                limit=400,
            )[0]
            values.append(k4 * (integral - density(zz + u_max) / (ANGLE + 2 * x)))
        return np.array(values)


    def first_interval(lambda_prime: np.ndarray, h: float, x: float, dz: float) -> np.ndarray:
        """The omitted part: the same integral over 0 <= u <= h (phi from 0 to phi_h), with the
        exported density derivative interpolated linearly between bin i and bin i + 1."""
        k4 = 4 * N * R_E * dz / RHO
        phi_h = brentq(lambda f: delay(f, x) - h, 0, ANGLE, xtol=1e-16)
        out = np.zeros(len(lambda_prime))
        for i in range(len(lambda_prime) - 1):
            a, b = lambda_prime[i], lambda_prime[i + 1]
            out[i] = (
                k4
                * quad(
                    lambda f, a=a, b=b: (a + (b - a) * delay(f, x) / h) * weight(f, x),
                    0,
                    phi_h,
                    epsrel=1e-11,
                )[0]
            )
        return out


    defect = False
    for drift in (0.002, 0.02, 0.2):
        lattice = (
            f"Q: CHARGE,TOTAL={CHARGE!r}\nB: CSRCSBEND,{BEND}\n"
            f'D: CSRDRIFT,L={drift},USE_STUPAKOV=1,N_KICKS=1,STUPAKOV_OUTPUT="wake.sdds"'
        )
        wake, info = page(track(lattice, "Q,B,D", BEAM, P0, outputs=("wake.sdds",))["wake.sdds"])
        x = drift / 2 / RHO  # the single kick sits at the middle of the drift
        if not np.isclose(info["x"], x, rtol=1e-12):
            raise SystemExit("The kick is not at the middle of the drift.")
        expected = expected_wake(wake["s"], x, drift)
        peak = np.max(np.abs(expected))
        error = np.max(np.abs(wake["DeltaGamma"] - expected)) / peak
        h = wake["s"][1] - wake["s"][0]
        restored = wake["DeltaGamma"] + first_interval(
            wake["LinearDensityDeriv"] / (CHARGE * C**2), h, x, drift
        )
        error_restored = np.max(np.abs(restored - expected)) / peak
        print(
            f"serial kick {drift / 2:g} m behind the bend: peak observed={np.max(np.abs(wake['DeltaGamma'])):.10g} "
            f"expected={peak:.10g}; max |wake - expected| / peak={100 * error:.4g}%; "
            f"with the first interval added back {100 * error_restored:.4g}%"
        )
        if error_restored > TOLERANCE:
            raise SystemExit("Adding back the first interval does not reproduce the expected wake.")
        defect |= error > TOLERANCE
    print(f"tolerance {TOLERANCE:.1%} of the expected peak")
    verdict(defect)

    ```

??? example "Helper: _csr.py"

    [Download _csr.py](_csr.py)

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

??? example "Helper: _elegant.py"

    [Download _elegant.py](_elegant.py)

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

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source reading, not executed for this claim |
| Found (executed verification) | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

