---
title: "COL-37: CSRCSBEND with LINEARIZE=1 mixes arrival time with path length, applies ETILT twice and drops SYNCH_RAD"
hide:
  - toc
---

# COL-37 — CSRCSBEND with LINEARIZE=1 mixes arrival time with path length, applies ETILT twice and drops SYNCH_RAD

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [conversion to ct, csbend.c:3062–3064](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3062-L3064)<br>[matrix branch, csbend.c:3164–3165](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3164-L3165)<br>[conversion back, csbend.c:3833–3836](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3833-L3836)<br>[section matrix, csbend.c:2879–2888](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2879-L2888)<br>[tilt, csbend.c:2891](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2891)<br>[ETILT offset, csbend.c:2909–2910](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2909-L2910)<br>[integrator branch, csbend.c:3166–3221](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3166-L3221)<br>[parameters, track_data.c:2052–2053](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2052-L2053)<br>[CSBEND manual, csbend.tex:107](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L107)<br>[errors, csbend.tex:124–131](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L124-L131) |

CSRCSBEND with LINEARIZE=1 replaces the integrator by a linear matrix, but applies the matrix's path-length row to the arrival time c t, includes ETILT both in the matrix and in the element's misalignment transform, and skips the classical radiation requested by SYNCH_RAD=1. In a 0.8 m, 0.12 rad bend, an on-axis electron at p = 8 mₑc arrives at c t = 0.8000 m instead of the bend length over β = v/c, 0.8062 m; with ETILT = 2 mrad at 1200 mₑc, y and y′ are 2.001 and 2.002 times those of the rotated circular orbit; and the classical relative momentum loss −5.843×10⁻⁸ becomes zero. With LINEARIZE=0 all three agree with these expectations within 10⁻⁶.

??? example "Reproducer: COL-37.py"

    [Download COL-37.py](COL-37.py)

    ```python
    """COL-37: CSRCSBEND with LINEARIZE=1 mixes arrival time with path length, applies ETILT twice and drops SYNCH_RAD

    CSRCSBEND with LINEARIZE=1 replaces the integrator by a linear matrix, but applies the matrix's
    path-length row to the arrival time c t, includes ETILT both in the matrix and in the element's
    misalignment transform, and skips the classical radiation requested by SYNCH_RAD=1. In a 0.8 m, 0.12
    rad bend, an on-axis electron at p = 8 mₑc arrives at c t = 0.8000 m instead of the bend length over
    β = v/c, 0.8062 m; with ETILT = 2 mrad at 1200 mₑc, y and y′ are 2.001 and 2.002 times those of the
    rotated circular orbit; and the classical relative momentum loss −5.843×10⁻⁸ becomes zero. With
    LINEARIZE=0 all three agree with these expectations within 10⁻⁶.

    The script tracks single electrons through a 0.8 m, 0.12 rad CSRCSBEND (CSR=0) with LINEARIZE=1
    and, as the control, with LINEARIZE=0 (the symplectic integrator). It compares the arrival time of
    an on-axis electron at p = 8 mₑc with L/β; y and y′ of an on-axis electron at p = 1200 mₑc with
    ETILT = 2 mrad with the intersection of the rotated circular orbit and the design exit plane; and
    the momentum of an on-axis electron at p = 1200 mₑc with SYNCH_RAD=1 with the integrated
    classical loss.
    """

    import numpy as np
    from _csr import R_E, C, final, gaussian_beam, track
    from _elegant import verdict
    from scipy.optimize import brentq

    L, ANGLE = 0.8, 0.12
    RHO = L / ANGLE
    RTOL = 1e-6  # the LINEARIZE=0 control agrees within 2e-7 relative (integration error)


    def run_bend(p0: float, linearize: int, params: str) -> dict:
        bend = f"B: CSRCSBEND,L={L},ANGLE={ANGLE},N_SLICES=64,CSR=0,BINS=32,LINEARIZE={linearize}"
        out = final(track(f"{bend}{params}", "B", gaussian_beam(1, 0.0, p0), p0))
        return {name: float(values[0]) for name, values in out.items()}


    # Arrival time: the on-axis particle travels exactly L, so c t = L/beta, beta = p/sqrt(1+p^2).
    P_LOW = 8.0
    CT_EXPECTED = L * np.sqrt(1 + P_LOW**2) / P_LOW

    # ETILT: the bend's field is rotated by ETILT about the entrance axis, so the circular orbit of
    # radius rho lies in a plane rotated by ETILT; it ends where it crosses the design exit plane
    # (through rho (1 - cos A, 0, sin A) with normal (sin A, 0, cos A), coordinates (x, y, s)).
    ETILT = 0.002
    EXIT_POINT = RHO * np.array([1 - np.cos(ANGLE), 0.0, np.sin(ANGLE)])
    EXIT_NORMAL = np.array([np.sin(ANGLE), 0.0, np.cos(ANGLE)])


    def orbit(a: float) -> np.ndarray:
        return RHO * np.array(
            [(1 - np.cos(a)) * np.cos(ETILT), -(1 - np.cos(a)) * np.sin(ETILT), np.sin(a)]
        )


    a_exit = brentq(
        lambda a: np.dot(orbit(a) - EXIT_POINT, EXIT_NORMAL), ANGLE - 0.01, ANGLE + 0.01, xtol=1e-16
    )
    tangent = np.array(
        [np.sin(a_exit) * np.cos(ETILT), -np.sin(a_exit) * np.sin(ETILT), np.cos(a_exit)]
    )
    Y_EXPECTED = orbit(a_exit)[1]  # the y axis is perpendicular to the design bend plane
    YP_EXPECTED = tangent[1] / np.dot(tangent, EXIT_NORMAL)  # y' = dy/ds = P_y/P_s

    # Classical radiation of an on-axis electron in a uniform field: the momentum p0 (1 + delta)
    # loses d(delta)/ds = -a (1 + delta)^2 with a = 2 r_e p0^3 / (3 rho^2) (Larmor power in a fixed
    # field scales with gamma^2), so delta(L) = 1/(1 + a L) - 1.
    P_HIGH = 1200.0
    A_SR = 2 * R_E * P_HIGH**3 / (3 * RHO**2)
    DELTA_EXPECTED = 1 / (1 + A_SR * L) - 1

    defect = False
    for linearize in (0, 1):
        tag = f"serial LINEARIZE={linearize}"
        t = run_bend(P_LOW, linearize, "")
        tilt = run_bend(P_HIGH, linearize, f",ETILT={ETILT}")
        sr = run_bend(P_HIGH, linearize, ",SYNCH_RAD=1")
        cases = [
            ("p=8 on axis: c t (m)", C * t["t"], CT_EXPECTED),
            (f"ETILT={ETILT}: y (m)", tilt["y"], Y_EXPECTED),
            (f"ETILT={ETILT}: y'", tilt["yp"], YP_EXPECTED),
            ("SYNCH_RAD=1: delta", sr["p"] / P_HIGH - 1, DELTA_EXPECTED),
        ]
        for label, observed, expected in cases:
            good = np.isclose(observed, expected, rtol=RTOL, atol=0)
            print(
                f"{tag} {label}: observed={observed:.13g} expected={expected:.13g} ratio={observed / expected:.7g}"
            )
            if linearize == 0 and not good:
                raise SystemExit(f"{tag} {label}: the control disagrees with the expectation")
            defect |= linearize == 1 and not good
    print(f"comparison rtol={RTOL:g}")
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

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source reading, not executed for this claim |
| Found (executed verification) | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

