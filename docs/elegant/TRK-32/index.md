---
title: "TRK-32: CCBEND with ETILT as its only error skips the entrance misalignment transform"
hide:
  - toc
---

# TRK-32 — CCBEND with ETILT as its only error skips the entrance misalignment transform

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`track_through_ccbend_impl`, ccbend.c:518–529](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L518-L529)<br>[ccbend.c:637–642](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L637-L642)<br>[parameter table, track_data.c:3818–3823](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3818-L3823)<br>[manual, ccbend.tex:89–95](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ccbend.tex#L89-L95) |

A CCBEND whose only error is the roll ETILT, with DX, DY, DZ, EPITCH and EYAW all zero, applies the misalignment transform at the exit but not at the entrance. A nonzero DX, even 10⁻¹⁶ m, restores the full transform, so the result jumps at zero displacement. For a 0.9 m, 0.12 rad bend rolled by 2 mrad, an on-axis electron leaves at y = 8.262×10⁻¹⁹ m and y′ = −1.199×10⁻⁴ instead of y = −1.079×10⁻⁴ m and y′ = −2.390×10⁻⁴, the limit from both sides of DX = 0, which the small-angle estimates for a rolled dipole confirm. A roll-only error thus gives half the vertical kick and no vertical displacement.

??? example "Reproducer: TRK-32.py"

    [Download TRK-32.py](TRK-32.py)

    ```python
    """TRK-32: CCBEND with ETILT as its only error skips the entrance misalignment transform

    A CCBEND whose only error is the roll ETILT, with DX, DY, DZ, EPITCH and EYAW all zero, applies the
    misalignment transform at the exit but not at the entrance. A nonzero DX, even 10⁻¹⁶ m, restores the
    full transform, so the result jumps at zero displacement. For a 0.9 m, 0.12 rad bend rolled by 2
    mrad, an on-axis electron leaves at y = 8.262×10⁻¹⁹ m and y′ = −1.199×10⁻⁴ instead of y =
    −1.079×10⁻⁴ m and y′ = −2.390×10⁻⁴, the limit from both sides of DX = 0, which the small-angle
    estimates for a rolled dipole confirm. A roll-only error thus gives half the vertical kick and no
    vertical displacement.

    The script tracks on-axis electrons through a CCBEND whose only error is ETILT and through the same
    magnet with an added displacement DX = ±10⁻¹⁶ m and 10⁻¹⁰ m, which cannot change a finite roll
    response. The two-sided limit at DX = 0 is the expected exit orbit; a small-angle estimate for a
    rolled dipole checks its size, and the ETILT = 0 runs are the control.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    P0 = 6000.0  # reference momentum in m_e c (3.066 GeV electrons)
    LENGTH, ANGLE, SLICES = 0.9, 0.12, 64
    N = 12  # identical on-axis particles, so that every MPI worker tracks some
    # DX = +1e-16 and -1e-16 agree to about 1e-18 m; a displacement of 1e-16 m can move the exit orbit
    # by at most about that much, so the limit is known far better than TOL.
    TOL = 1e-12
    COLS = ("x", "xp", "y", "yp")


    def fmt(values):
        return "[" + ", ".join(f"{v:.12g}" for v in np.atleast_1d(values)) + "]"


    def exit_orbit(mode, etilt, dx):
        """Exit (x, xp, y, yp) of on-axis electrons through one CCBEND with default settings."""
        lattice = (
            f"B: CCBEND,L={LENGTH!r},ANGLE={ANGLE!r},N_SLICES={SLICES},ETILT={etilt!r},DX={dx!r}\n"
            "BL: LINE=(B)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        beam = {"p": np.full(N, P0)}
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if len(out["p"]) != N:
            sys.exit(f"{mode}: particles were lost")
        q = np.array([out[c] for c in COLS])
        if np.max(np.abs(q - q[:, :1])) > 0:
            sys.exit(f"{mode}: identical particles leave with different coordinates")
        return q[:, 0]


    defect = []
    print(f"columns: {', '.join(COLS)} at the exit; L={LENGTH} m, ANGLE={ANGLE} rad, p0={P0:g} m_e c")
    for mode in modes():
        for etilt in (0.002, -0.002, 0.0):
            plus, minus = exit_orbit(mode, etilt, 1e-16), exit_orbit(mode, etilt, -1e-16)
            if np.max(np.abs(plus - minus)) > TOL:
                sys.exit(f"{mode} ETILT={etilt}: the two sides of DX=0 disagree: invalid")
            # The response is continuous in DX, so the limit DX -> 0 from both sides is the expected
            # exit orbit at DX = 0.
            expected = (plus + minus) / 2
            tiny = exit_orbit(mode, etilt, 1e-10)
            if np.max(np.abs(tiny - expected)) > 1e-9:
                sys.exit(f"{mode} ETILT={etilt}: DX=1e-10 is not continuous with the limit: invalid")
            observed = exit_orbit(mode, etilt, 0.0)
            diff = float(np.max(np.abs(observed - expected)))
            label = f"{mode} ETILT={etilt:g}"
            print(f"{label} DX=+-1e-16 (expected): {fmt(plus)} / {fmt(minus)}")
            print(f"{label} DX=1e-10: {fmt(tiny)}")
            print(
                f"{label} DX=0: observed={fmt(observed)} expected={fmt(expected)} max|diff|={diff:.4g}"
            )
            if etilt == 0.0:
                if diff > TOL:
                    sys.exit(f"{label}: the error-free control depends on DX: invalid")
                continue
            # Size check: a dipole rolled by e about its axis bends the orbit vertically by -ANGLE*e
            # spread over the magnet, so to leading order y = -L*ANGLE*e/2 and yp = -ANGLE*e.
            estimate = np.array([-LENGTH * ANGLE * etilt / 2, -ANGLE * etilt])
            if np.max(np.abs(expected[2:] / estimate - 1)) > 0.01:
                sys.exit(f"{label}: the limit disagrees with the rolled-dipole estimate: invalid")
            print(f"{label} rolled-dipole estimate (y, yp): {fmt(estimate)}")
            defect.append(diff > TOL)
    print(f"comparison: abs tolerance {TOL:g} on x, xp, y, yp")
    verdict(bool(defect) and all(defect))

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
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Found (cause in the source and comparison with CSBEND and the rolled-dipole estimate) | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

