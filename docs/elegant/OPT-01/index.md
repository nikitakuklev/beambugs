---
title: "OPT-01: Unmatched alpha derivatives report phase derivatives"
hide:
  - toc
---

# OPT-01 — Unmatched alpha derivatives report phase derivatives

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`computeChromaticAlphaValue`, nonperiodic branch, src/chrom.c:1192–1200](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/chrom.c#L1192-L1200)<br>[`computeChromaticityValue`, src/chrom.c:1166–1167](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/chrom.c#L1166-L1167)<br>[`computeChromaticities`, assigning `dalphax` and `dalphay`, src/chrom.c:867–895](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/chrom.c#L867-L895)<br>[`dalphax/dp` and `dalphay/dp` are the chromatic derivatives of the alpha functions, dα/dδ, manual, doc/elegant.tex:8234–8235](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8234-L8235) |

Elegant's `twiss_output` with `matched=0` propagates user-given entrance optics through a non-periodic beamline and writes the momentum derivatives of the Twiss alpha functions, `dalphax/dp` and `dalphay/dp`. These columns instead hold the derivatives of the betatron phase, 2π times the chromaticities. For a 0.5 m quadrupole with K1 = 0.6 m⁻² followed by a 1 m drift, entering with βx = 3 m, αx = 0.4, βy = 2 m and αy = −0.3, the exact hard-edge map gives dαx/dδ = 0.004523 and dαy/dδ = 1.549, where δ is the relative momentum offset; Elegant writes −0.3591 and 0.07649. The tune and beta derivatives are correct.

??? example "Reproducer: OPT-01.py"

    [Download OPT-01.py](OPT-01.py)

    ```python
    """OPT-01: Unmatched alpha derivatives report phase derivatives

    Elegant's `twiss_output` with `matched=0` propagates user-given entrance optics through a
    non-periodic beamline and writes the momentum derivatives of the Twiss alpha functions, `dalphax/dp`
    and `dalphay/dp`. These columns instead hold the derivatives of the betatron phase, 2π times the
    chromaticities. For a 0.5 m quadrupole with K1 = 0.6 m⁻² followed by a 1 m drift, entering with βx =
    3 m, αx = 0.4, βy = 2 m and αy = −0.3, the exact hard-edge map gives dαx/dδ = 0.004523 and dαy/dδ =
    1.549, where δ is the relative momentum offset; Elegant writes −0.3591 and 0.07649. The tune and
    beta derivatives are correct.

    The script transports fixed entrance optics through the quadrupole and drift in serial Elegant (and
    with `--mpi` in 3-rank Pelegant), and compares the written alpha derivatives with a complex-step
    derivative of the exact hard-edge map with respect to the momentum offset. The written tune and beta
    derivatives are the control: they must agree with the same oracle, and the alpha derivatives are
    shown to equal 2π times the tune derivatives.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    K1, QUAD_L, DRIFT_L = 0.6, 0.5, 1.0  # m^-2, m, m
    ENTRANCE = {"x": (3.0, 0.4), "y": (2.0, -0.3)}  # (beta, alpha) at the entrance
    RTOL, ATOL = 1e-12, 1e-14
    STEP = 1e-30  # complex step: derivative = Im f(delta + i*STEP)/STEP, exact to rounding

    LATTICE = f"Q: QUAD,L={QUAD_L!r},K1={K1!r}\nD: DRIF,L={DRIFT_L!r}\nBL: LINE=(Q,D)\n"
    COMMANDS = (
        "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&twiss_output filename=twiss.sdds, matched=0, "
        "beta_x=3, alpha_x=0.4, beta_y=2, alpha_y=-0.3 &end\n"
        "&bunched_beam n_particles_per_bunch=12 &end\n"
        "&track &end\n"
    )


    def quad(k, length):
        """2x2 transfer matrix of a hard-edge quadrupole, analytic in k (k < 0 is defocusing)."""
        z = np.sqrt(k + 0j) * length  # cos z and sin(z)/z are even in z, so the branch is irrelevant
        return np.array([[np.cos(z), np.sin(z) / z * length], [-z * np.sin(z) / length, np.cos(z)]])


    def twiss_derivatives(plane, delta):
        """(d nu/d delta, d beta/d delta, d alpha/d delta) at the exit for fixed entrance optics.

        The gradient scales as K1/(1 + delta) (rigidity), the drift does not depend on delta. With the
        ellipse matrix B = [[beta, -alpha], [-alpha, gamma]] at the entrance, B1 = M B0 M^T gives
        beta1 = B1[0,0], alpha1 = -B1[0,1]. The phase advance is atan(M12/(M11 beta0 - M12 alpha0)).
        All three are analytic in delta, so a complex step differentiates them exactly.
        """
        beta0, alpha0 = ENTRANCE[plane]
        k = K1 if plane == "x" else -K1  # the quadrupole defocuses in y
        d = delta + 1j * STEP
        m = np.array([[1, DRIFT_L], [0, 1]]) @ quad(k / (1 + d), QUAD_L)
        b0 = np.array([[beta0, -alpha0], [-alpha0, (1 + alpha0**2) / beta0]])
        b1 = m @ b0 @ m.T
        phase = np.arctan(m[0, 1] / (m[0, 0] * beta0 - m[0, 1] * alpha0))
        return np.array([phase / (2 * np.pi), b1[0, 0], -b1[0, 1]]).imag / STEP


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected disagree."""
        bad = not np.allclose(observed, expected, rtol=RTOL, atol=ATOL)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    defect = False
    print(
        "columns: d nu/dp (1/(2 pi)), d beta/dp (m), d alpha/dp; expected from the exact hard-edge map"
    )
    for mode in modes():
        twiss = run({"lattice.lte": LATTICE, "run.ele": COMMANDS}, mode, outputs=("twiss.sdds",))[
            "twiss.sdds"
        ]
        for plane in "xy":
            observed = np.array([twiss[f"d{name}{plane}/dp"][0] for name in ("nu", "beta", "alpha")])
            expected = twiss_derivatives(plane, 0.0)
            # Control: the tune and beta derivatives agree with the oracle, so the entrance optics,
            # the K1/(1+delta) scaling and the sign conventions are right.
            if compare(f"{mode} {plane} control, nu and beta", observed[:2], expected[:2]):
                sys.exit(
                    "The tune/beta control disagrees with the oracle: the comparison is not valid."
                )
            defect |= compare(f"{mode} {plane} d alpha/dp", observed[2], expected[2])
            # The written alpha derivative equals 2 pi times the tune derivative, i.e. the phase one.
            print(f"{mode} {plane} 2*pi*d nu/dp={2 * np.pi * observed[0]:.16g}")
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
    verdict(defect)

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
| Found | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

