---
title: "COL-13: Round Gaussian BEAMBEAM produces NaNs at its center"
hide:
  - toc
---

# COL-13 — Round Gaussian BEAMBEAM produces NaNs at its center

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [radial kick, ionEffects.cc:1485](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ionEffects.cc#L1485)<br>[helper selection, ionEffects.cc:1411–1412](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ionEffects.cc#L1411-L1412)<br>[Gaussian branch, do_tracking.c:6316–6322](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L6316-L6322)<br>[BEAMBEAM manual, beambeam.tex:6–8](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/beambeam.tex#L6-L8) |

A Gaussian BEAMBEAM opposing bunch with transverse rms widths σₓ and σᵧ satisfying |σₓ − σᵧ|/σₓ < 10⁻⁹ turns a centered particle's slopes and momentum into NaN, even at zero opposing charge. The immediate output retains the invalid particles without reporting losses, and Elegant exits successfully. Symmetry and the continuous radial-force limit require zero kick at the center. For σₓ = σᵧ = 1.300 mm, off-axis particles at horizontal position x = 0.1σₓ have finite kicks agreeing with the analytic force, while a centered particle with σᵧ/σₓ = 1.02 retains zero slopes and momentum 1300 mₑc.

??? example "Reproducer: COL-13.py"

    [Download COL-13.py](COL-13.py)

    ```python
    """COL-13: Round Gaussian BEAMBEAM produces NaNs at its center

    A Gaussian BEAMBEAM opposing bunch with transverse rms widths σₓ and σᵧ satisfying
    |σₓ − σᵧ|/σₓ < 10⁻⁹ turns a centered particle's slopes and momentum into NaN, even at zero opposing
    charge. The immediate output retains the invalid particles without reporting losses, and Elegant
    exits successfully. Symmetry and the continuous radial-force limit require zero kick at the center.
    For σₓ = σᵧ = 1.300 mm, off-axis particles at horizontal position x = 0.1σₓ have finite kicks
    agreeing with the analytic force, while a centered particle with σᵧ/σₓ = 1.02 retains zero slopes
    and momentum 1300 mₑc.

    The script tracks 12 particles at x = y = 0 through one Gaussian BEAMBEAM element for opposing
    charges 0 and ±0.9 nC, once with a round bunch and once with σᵧ/σₓ = 1.02, and compares the
    resulting slopes and momentum with the zero-force limit. Controls are the elliptical bunch and
    12 identical off-axis particles at each of x/σ = 0.1 and 0.01, whose kicks are compared with a
    closed form that evaluates 1 − exp(−r²/2σ²) stably.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    # SDDS constants.h, as Elegant uses them.
    QE, RE = 1.60217733e-19, 2.81794092e-15
    P0 = 1300.0  # momentum, units of m_e c
    SIGMA = 1.3e-3  # m
    N = 12
    EPS = np.finfo(float).eps


    def track(mode, charge, ratio, x):
        """(xp, yp, p) arrays after one BEAMBEAM with sigma_y = ratio * sigma_x, particles at (x, 0)."""
        lattice = (
            f"K: BEAMBEAM,DISTRIBUTION=gaussian,CHARGE={charge!r},XSIZE={SIGMA!r},YSIZE={ratio * SIGMA!r}\n"
            "BL: LINE=(K)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0!r},output=out.sdds,"
            "losses=lost.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {"x": np.full(N, x), "p": np.full(N, P0)}
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": beam},
            outputs=("out.sdds", "lost.sdds"),
        )
        out, lost = result["out.sdds"], result["lost.sdds"]
        if len(out[0, "xp"]) != N or len(lost[0, "particleID"]) != 0:
            sys.exit(f"{mode}: particles were lost, which is not the case under test")
        return tuple(np.asarray(out[0, name]) for name in ("xp", "yp", "p"))


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in values) + "]"


    defect = False
    for mode in modes():
        # At the center the unique zero-radius limit of the kick is zero, for any charge:
        # dp ~ C1 [1 - exp(-r^2/2s^2)]/r -> C1 r/(2 s^2) -> 0.
        expected = (np.zeros(N), np.zeros(N), np.full(N, P0))
        for charge in (0.0, 0.9e-9, -0.9e-9):
            for ratio in (1.0, 1.02):
                observed = track(mode, charge, ratio, 0.0)
                ok = all(
                    np.array_equal(o, e) for o, e in zip(observed, expected, strict=True)
                )  # NaN compares unequal
                name = f"{mode} center charge={charge * 1e9:+.1f} nC sigma_y/sigma_x={ratio}"
                print(
                    f"{name}: observed xp,yp,p={fmt([o[0] for o in observed])} "
                    f"expected xp,yp,p={fmt([e[0] for e in expected])}"
                )
                if ratio == 1.0:
                    defect |= not ok
                elif not ok:
                    sys.exit("The elliptical-bunch control does not give the zero center kick.")
        # Finite-radius control: the same routine at x = f sigma. Kick of a particle at (x, 0):
        # xp = -(2 r_e Q/e)/p0 [1 - exp(-x^2/2s^2)]/x, with 1 - exp(-u) = -expm1(-u).
        charge = 0.9e-9
        for fraction in (0.1, 0.01):
            x = fraction * SIGMA
            prefactor = 2 * RE * charge / QE / P0 / x
            expected_xp = -prefactor * -np.expm1(-(fraction**2) / 2)
            # The routine's 1 - exp(...) loses digits near zero: allow one half-ulp of exp(...)
            # times the prefactor, plus 8 ulp of rounding in the remaining arithmetic.
            tolerance = abs(prefactor) * EPS / 2 + abs(expected_xp) * 8 * EPS
            xp = track(mode, charge, 1.0, x)[0]
            print(
                f"{mode} off axis x/sigma={fraction}: observed xp={fmt(xp[:1])} "
                f"expected xp={fmt([expected_xp])}"
            )
            if not np.all(np.abs(xp - expected_xp) <= tolerance):
                sys.exit("The off-axis control disagrees with the closed form.")
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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |

