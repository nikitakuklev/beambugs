---
title: "COL-43: CSRCSBEND and CSRDRIFT apply the CSR wake half a bin toward the bunch tail"
hide:
  - toc
---

# COL-43 — CSRCSBEND and CSRDRIFT apply the CSR wake half a bin toward the bunch tail

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [lower + ih, lower + (i + 1)h) and states that "the coordinate of the bin center is
(iBin+0.5)*(*binSize) + *lower" ([binParticleCoordinate, csbend.c:4169–4170](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L4169-L4170)<br>[second binning routine, csbend.c:4225–4226](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L4225-L4226)<br>[bend output, csbend.c:3717–3721](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3717-L3721)<br>[Stupakov output, csbend.c:5328–5331](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L5328-L5331)<br>[bend kick, csbend.c:3613–3617](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3613-L3617)<br>[drift kick, csbend.c:4530–4535](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L4530-L4535)<br>[Stupakov kick, csbend.c:5149–5154](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L5149-L5154)<br>[CSRCSBEND manual, csrcsbend.tex:6–18](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csrcsbend.tex#L6-L18) |

CSRCSBEND, and CSRDRIFT in the default and Stupakov models, interpolate the CSR wake as if each value sat at its bin's lower edge, although the histogram and the wake output place it at the bin center, so every particle receives the wake half a bin toward the bunch tail. For a 10 pC, 35 μm rms Gaussian bunch in a 0.6 m, 0.3 rad bend with 400 bins, the kicks deviate from the wake at the particles by up to 1.262% of its peak, and the mean energy loss is 0.5104% above the closed-form Gaussian value, whereas the wake sampled at the particles is within 0.03107%. The bias halves with each doubling of the bin count.

??? example "Reproducer: COL-43.py"

    [Download COL-43.py](COL-43.py)

    ```python
    """COL-43: CSRCSBEND and CSRDRIFT apply the CSR wake half a bin toward the bunch tail

    CSRCSBEND, and CSRDRIFT in the default and Stupakov models, interpolate the CSR wake as if each value sat at its bin's lower edge, although the histogram and the wake output place it at the bin center, so every particle receives the wake half a bin toward the bunch tail. For a 10 pC, 35 μm rms Gaussian bunch in a 0.6 m, 0.3 rad bend with 400 bins, the kicks deviate from the wake at the particles by up to 1.262% of its peak, and the mean energy loss is 0.5104% above the closed-form Gaussian value, whereas the wake sampled at the particles is within 0.03107%. The bias halves with each doubling of the bin count.

    The script tracks 240001 electrons (Gaussian quantiles, σ_ct = 35 μm, p = 4000 mₑc, 10 pC)
    through a one-slice 0.6 m, 0.3 rad CSRCSBEND (STEADY_STATE=1) with 200, 400 and 800 bins and
    rebuilds every particle's kick from the exported wake, placing wake value i at the bin center the
    output declares and, alternatively, at the bin's lower edge. It compares the mean energy change
    with the closed-form Gaussian mean, which the wake sampled at bin centers matches. It repeats the
    rebuild for a following 20 mm CSRDRIFT with one kick, in the default model (the bend's wake) and
    with USE_STUPAKOV=1, from the coordinates a WATCH records at the bend exit.
    """

    import numpy as np
    from _csr import E_CHARGE, R_E, C, final, gaussian_beam, page, track
    from _elegant import particles, verdict
    from scipy.special import gamma

    P0, SIGMA, CHARGE = 4000.0, 35e-6, 1e-11
    L, ANGLE = 0.6, 0.3
    RHO = L / ANGLE
    DRIFT = 0.02
    BEND = (
        f"L={L},ANGLE={ANGLE},N_SLICES=1,INTEGRATION_ORDER=6,SG_HALFWIDTH=1,SGDERIV_HALFWIDTH=3,"
        "SGDERIV_ORDER=3,STEADY_STATE=1"
    )
    BEAM = gaussian_beam(240001, SIGMA, P0)
    K = 2 * (CHARGE / E_CHARGE) * R_E * L / (3 * RHO**2) ** (1 / 3)  # see _csr.steady_state_wake
    # Mean steady-state change of gamma over a Gaussian bunch: averaging k int lambda'(z+u) u^(-1/3) du
    # over lambda gives -k Gamma(5/6) / (2^(4/3) sqrt(pi) sigma^(4/3)) (density autocorrelation).
    MEAN = -K * gamma(5 / 6) / (2 ** (4 / 3) * np.sqrt(np.pi) * SIGMA ** (4 / 3))
    ATOL = 1e-11  # m_e c; the rebuild with the right placement reproduces every kick within ~1e-12


    def reconstruct(position: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linear interpolation of the wake at `position` (in bins from the grid's lower end), with
        value i placed at the declared bin center i + 1/2 and, alternatively, at the lower edge i."""
        index = np.arange(len(values))
        return np.interp(position, index + 0.5, values), np.interp(position, index, values)


    def compare(label: str, observed: np.ndarray, position: np.ndarray, values: np.ndarray) -> bool:
        center, edge = reconstruct(position, values)
        peak = np.max(np.abs(values))
        e_center, e_edge = np.max(np.abs(observed - center)), np.max(np.abs(observed - edge))
        print(
            f"serial {label}: max |kick - wake at the particle| observed={e_center:.4g} m_e c "
            f"({e_center / peak:.4%} of the peak) expected=0; with value i at the lower edge {e_edge:.3g} m_e c"
        )
        if e_edge > ATOL:
            raise SystemExit(f"{label}: the lower-edge rebuild does not reproduce the kicks.")
        return e_center > ATOL


    defect = False
    for bins in (200, 400, 800):
        lattice = f'Q: CHARGE,TOTAL={CHARGE!r}\nB: CSRCSBEND,{BEND},BINS={bins},OUTPUT_FILE="wake.sdds"'
        result = track(lattice, "Q,B", BEAM, P0, outputs=("out.sdds", "wake.sdds"))
        out, (wake, info) = final(result), page(result["wake.sdds"])
        # The kick is Delta gamma(ct)/p0 (1 + x/rho) in relative momentum (csbend.c); dividing out
        # the path factor leaves Delta gamma at the particle. The output s is ct measured from the
        # middle of the histogram, which for this symmetric bunch is the middle of its ct range.
        kick = (out["p"] - P0) / (1 + out["x"] / RHO)
        ct = C * out["t"]
        s = ct - (ct.max() + ct.min()) / 2
        h = info["BinSize"]
        position = (s - (wake["s"][0] - h / 2)) / h
        defect |= compare(f"bend, {bins} bins", kick, position, wake["DeltaGamma"])
        center, _ = reconstruct(position, wake["DeltaGamma"])
        print(
            f"serial bend, {bins} bins: mean Delta gamma observed={np.mean(kick):.10g} "
            f"expected={MEAN:.10g} (ratio {np.mean(kick) / MEAN:.7f}); wake at the particle "
            f"{np.mean(center):.10g} (ratio {np.mean(center) / MEAN:.7f})"
        )

    # Drifts: one kick at the middle of a 20 mm CSRDRIFT after the 400-bin bend.
    for model, extra, output in (
        ("default", "", "wake.sdds"),
        ("Stupakov", ',USE_STUPAKOV=1,STUPAKOV_OUTPUT="drift.sdds"', "drift.sdds"),
    ):
        lattice = (
            f'Q: CHARGE,TOTAL={CHARGE!r}\nB: CSRCSBEND,{BEND},BINS=400,OUTPUT_FILE="wake.sdds"\n'
            f'W: WATCH,FILENAME="exit.sdds"\nD: CSRDRIFT,L={DRIFT},N_KICKS=1{extra}'
        )
        result = track(lattice, "Q,B,W,D", BEAM, P0, outputs=("out.sdds", "exit.sdds", output))
        exit_beam, out = particles(result["exit.sdds"]), final(result)
        wake, info = page(result[output])
        beta = exit_beam["p"] / np.sqrt(1 + exit_beam["p"] ** 2)
        ct_exit = C * exit_beam["t"]
        # ct at the kick: half the drift travelled along the particle's direction at its speed.
        ct = ct_exit + DRIFT / 2 * np.sqrt(1 + exit_beam["xp"] ** 2 + exit_beam["yp"] ** 2) / beta
        if model == "default":
            # The bend's wake, frozen, scaled by the drift step over the bend length; its grid is the
            # bend-exit histogram (range 1.2 times the ct spread, centered) moving with the light
            # front, so its lower end has advanced by the distance travelled.
            h = info["BinSize"]
            lower = (ct_exit.max() + ct_exit.min()) / 2 - 0.6 * np.ptp(ct_exit) + DRIFT / 2
            values = wake["DeltaGamma"] * DRIFT / L
        else:
            # The Stupakov model rebins the bunch at the kick over 1.2 times its ct spread.
            h = 1.2 * np.ptp(ct) / len(wake["s"])
            lower = (ct.max() + ct.min()) / 2 - 0.6 * np.ptp(ct)
            values = wake["DeltaGamma"]
            if not np.allclose(np.diff(wake["s"]), h, rtol=1e-9):
                raise SystemExit("The Stupakov grid spacing differs from the rebinned range.")
        defect |= compare(f"{model} CSRDRIFT", out["p"] - exit_beam["p"], (ct - lower) / h, values)
    print(f"rebuild tolerance {ATOL:g} m_e c")
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
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed (bend and both drift kick paths; an implementation defect, not an established binning convention) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

