---
title: "COL-36: CSRCSBEND computes CSR with the nominal bend radius, ignoring the field error FSE"
hide:
  - toc
---

# COL-36 — CSRCSBEND computes CSR with the nominal bend radius, ignoring the field error FSE

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [actual radius, csbend.c:2846–2852](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2846-L2852)<br>[CSR constant, csbend.c:3132–3133](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3132-L3133)<br>[slippage, csbend.c:3437–3438](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3437-L3438)<br>[IGF argument, csbend.c:3456–3458](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3456-L3458)<br>[drift hand-off, csbend.c:3961](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3961)<br>[CSBEND manual, errors, csbend.tex:124–133](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L124-L133)<br>[CSRCSBEND manual, csrcsbend.tex:25–28](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csrcsbend.tex#L25-L28) |

CSRCSBEND computes the CSR strength, the transient slippage length and the argument of the integrated Green function (IGF=1) from the nominal radius ρ = L/ANGLE, although FSE, the documented field-strength error, bends the orbit on the radius ρ/(1 + FSE). Steady-state CSR scales as the field curvature to the power 2/3, so the wake should grow by (1 + FSE)^(2/3); in a 0.6 m, 0.3 rad bend whose FSE = 0 wake matches the analytic Gaussian wake within 0.1470%, the wake is instead identical for FSE = 0, 0.02 and 0.1. At FSE = 0.02 the wake should be 1.329% stronger and the slippage length 2.339 mm instead of 2.250 mm; the IGF wake is also unchanged.

??? example "Reproducer: COL-36.py"

    [Download COL-36.py](COL-36.py)

    ```python
    """COL-36: CSRCSBEND computes CSR with the nominal bend radius, ignoring the field error FSE

    CSRCSBEND computes the CSR strength, the transient slippage length and the argument of the integrated Green function (IGF=1) from the nominal radius ρ = L/ANGLE, although FSE, the documented field-strength error, bends the orbit on the radius ρ/(1 + FSE). Steady-state CSR scales as the field curvature to the power 2/3, so the wake should grow by (1 + FSE)^(2/3); in a 0.6 m, 0.3 rad bend whose FSE = 0 wake matches the analytic Gaussian wake within 0.1470%, the wake is instead identical for FSE = 0, 0.02 and 0.1. At FSE = 0.02 the wake should be 1.329% stronger and the slippage length 2.339 mm instead of 2.250 mm; the IGF wake is also unchanged.

    The script tracks 240001 electrons (Gaussian quantiles, σ_ct = 35 μm, p = 4000 mₑc, 10⁻¹⁶ C)
    through a one-slice 0.6 m, 0.3 rad CSRCSBEND (400 bins) with FSE = 0, 0.02 and 0.1 and reads the
    exported wake. With STEADY_STATE=1 the wake must scale as the field curvature to the power 2/3,
    (1 + FSE)^(2/3), and the FSE = 0 control must match the analytic Gaussian wake. With
    STEADY_STATE=0 the reported slippage must follow the circle of the actual field radius to the
    fixed exit plane. With IGF=1 at p = 50 mₑc the script reports whether the wake changes with FSE.
    """

    import numpy as np
    from _csr import E_CHARGE, R_E, gaussian_beam, page, steady_state_wake, track
    from _elegant import verdict
    from scipy.optimize import brentq

    P0, SIGMA, CHARGE = 4000.0, 35e-6, 1e-16
    L, ANGLE = 0.6, 0.3
    RHO = L / ANGLE  # nominal radius; FSE changes the field, not the survey
    BEND = f"L={L},ANGLE={ANGLE},N_SLICES=1,INTEGRATION_ORDER=6,BINS=400,SG_HALFWIDTH=1,SGDERIV_HALFWIDTH=3,SGDERIV_ORDER=3"
    FSES = (0.0, 0.02, 0.1)
    K = 2 * (CHARGE / E_CHARGE) * R_E * L / (3 * RHO**2) ** (1 / 3)  # see _csr.steady_state_wake


    def wake(fse: float, mode: str, p0: float = P0, count: int = 240001):
        lattice = f'Q: CHARGE,TOTAL={CHARGE!r}\nB: CSRCSBEND,{BEND},{mode},FSE={fse!r},OUTPUT_FILE="wake.sdds"'
        beam = gaussian_beam(count, SIGMA, p0)
        return page(track(lattice, "Q,B", beam, p0, outputs=("wake.sdds",))["wake.sdds"])


    def slippage(fse: float) -> float:
        """Small-angle slippage rho_B psi^3/24 on the orbit of field radius rho_B = rho/(1 + FSE).

        The orbit starts on the reference axis and ends where the circle of radius rho_B crosses the
        design exit plane (through rho (1 - cos A, sin A) with normal (sin A, cos A)); psi is the
        angle it has turned through there. This is the slippage law the code applies to the
        nominal orbit, evaluated on the orbit the field produces.
        """
        rho_b = RHO / (1 + fse)
        point, normal = (
            RHO * np.array([1 - np.cos(ANGLE), np.sin(ANGLE)]),
            np.array([np.sin(ANGLE), np.cos(ANGLE)]),
        )
        psi = brentq(
            lambda a: np.dot(rho_b * np.array([1 - np.cos(a), np.sin(a)]) - point, normal),
            1e-9,
            1.0,
            xtol=1e-16,
        )
        return rho_b * psi**3 / 24


    defect = False
    steady = {fse: wake(fse, "STEADY_STATE=1") for fse in FSES}
    reference = steady[0.0][0]["DeltaGamma"]
    analytic = steady_state_wake(steady[0.0][0]["s"], SIGMA, K)
    control = np.max(np.abs(reference - analytic)) / np.max(np.abs(analytic))
    print(f"serial FSE=0 steady-state wake: max |wake - analytic| / peak = {control:.4%} (control)")
    if control > 0.003:
        raise SystemExit("The FSE=0 control disagrees with the analytic wake.")
    for fse in FSES[1:]:
        columns, _ = steady[fse]
        # Steady-state CSR scales as rho^(-2/3) (k above); the field radius is rho/(1 + FSE).
        ratio = np.sum(columns["DeltaGamma"] * reference) / np.sum(reference**2)  # least-squares scale
        expected = (1 + fse) ** (2 / 3)
        print(
            f"serial FSE={fse} steady-state wake / FSE=0 wake: observed={ratio:.10f} expected={expected:.10f}"
        )
        defect |= not np.isclose(ratio, expected, rtol=1e-4)
    for fse in FSES:
        _, info = wake(fse, "STEADY_STATE=0")
        expected = slippage(fse)
        print(
            f"serial FSE={fse} transient slippage length (m): observed={info['SlippageLength']:.10g} expected={expected:.10g}"
        )
        if fse == 0 and not np.isclose(info["SlippageLength"], expected, rtol=1e-6):
            raise SystemExit("The FSE=0 slippage disagrees with rho angle^3/24.")
        defect |= fse != 0 and not np.isclose(info["SlippageLength"], expected, rtol=1e-4)
    # IGF: the integrated Green function depends on the curvature through mu = 3 gamma^3 z/(2 rho);
    # the script only reports whether its wake responds to FSE.
    igf = {
        fse: wake(fse, "STEADY_STATE=1,IGF=1", p0=50.0, count=40001)[0]["DeltaGamma"] for fse in FSES
    }
    for fse in FSES[1:]:
        same = np.array_equal(igf[fse], igf[0.0])
        print(f"serial IGF=1, p=50: FSE={fse} wake identical to FSE=0: {same}")
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
| Confirmed (nominal radius in the CSR strength, the bend slippage and the IGF argument; not the proposed ANGLE/FSE equivalence) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

