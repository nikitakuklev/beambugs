---
title: "TRK-36: Identical CCBENDs that share a tracking-based matrix lose the path-length reference correction"
hide:
  - toc
---

# TRK-36 — Identical CCBENDs that share a tracking-based matrix lose the path-length reference correction

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`determineMatrixHigherOrder`, analyze.c:1030–1049](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/analyze.c#L1030-L1049)<br>[`track_through_ccbend_impl`, ccbend.c:287–300](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L287-L300)<br>[ccbend.c:664–673](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L664-L673)<br>[ccbend.c:204–211](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L204-L211)<br>[manual, ccbend.tex:40–43](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ccbend.tex#L40-L43)<br>[default, track_data.c:3855](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3855)<br>[ccbend.tex:113–116](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ccbend.tex#L113-L116)<br>[`share_tracking_based_matrices`, elegant.tex:4132–4133](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4132-L4133) |

When a matrix command such as twiss_output runs before tracking, identical CCBENDs reuse the first one's tracking-based matrix, the default, but not its stored path-length correction, which the default REFERENCE_CORRECTION=1 subtracts in tracking. Each later copy of an untilted CCBEND then adds its uncorrected path excess. In a line of three CCBENDs with L = 0.85 m, ANGLE = 0.16 rad and K₁ = 0.35 m⁻², the reference electron travels 1.534×10⁻⁵ m more than the nominal length, twice the 7.672×10⁻⁶ m excess of one uncorrected magnet; without twiss_output or without sharing the path is nominal. A design TILT makes each copy recompute its correction, so tilted CCBENDs are not affected.

??? example "Reproducer: TRK-36.py"

    [Download TRK-36.py](TRK-36.py)

    ```python
    """TRK-36: Identical CCBENDs that share a tracking-based matrix lose the path-length reference correction

    When a matrix command such as twiss_output runs before tracking, identical CCBENDs reuse the first
    one's tracking-based matrix, the default, but not its stored path-length correction, which the
    default REFERENCE_CORRECTION=1 subtracts in tracking. Each later copy of an untilted CCBEND then
    adds its uncorrected path excess. In a line of three CCBENDs with L = 0.85 m, ANGLE = 0.16 rad and
    K₁ = 0.35 m⁻², the reference electron travels 1.534×10⁻⁵ m more than the nominal length, twice the
    7.672×10⁻⁶ m excess of one uncorrected magnet; without twiss_output or without sharing the path is
    nominal. A design TILT makes each copy recompute its correction, so tilted CCBENDs are not affected.

    The script tracks a reference electron through three identical CCBENDs, each followed by a short
    drift, with and without a preceding twiss_output and with matrix sharing on and off. The expected
    path is the nominal length, which REFERENCE_CORRECTION=1 gives in the runs without sharing; the
    size of the error is predicted independently from one magnet tracked with REFERENCE_CORRECTION=0.
    A design TILT, which forces each copy to recompute its reference, is shown as the case where the
    defect does not occur.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    C = 299792458.0  # speed of light (m/s)
    P0 = 6000.0  # reference momentum in m_e c
    BEND = "CCBEND,L=0.85,ANGLE=0.16,K1=0.35,N_SLICES=64"
    COPIES, DRIFT = 3, 0.03
    NOMINAL = COPIES * (0.85 + DRIFT)
    TOL = 1e-12  # the corrected runs return the nominal path to rounding (observed below 5e-16 m)
    N = 12  # identical particles, so that every MPI worker tracks some


    def path_length(mode, lattice, twiss, share):
        """Path of the reference electron through the line, from its arrival time."""
        commands = (
            f"&global_settings share_tracking_based_matrices={int(share)} &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds &end\n"
            + ("&twiss_output filename=out.twi, matched=0, beta_x=2, beta_y=3 &end\n" if twiss else "")
            + "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        beam = {"p": np.full(N, P0)}
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if len(out["p"]) != N or np.ptp(out["t"]) > 0:
            sys.exit(f"{mode}: particles lost or not identical: invalid")
        p = out["p"][0]
        return out["t"][0] * C * p / np.sqrt(1 + p * p)


    def line(tilt=0.0):
        cells = ",".join(["B,D"] * COPIES)
        return f"B: {BEND},TILT={tilt!r}\nD: DRIF,L={DRIFT!r}\nEND: MARK\nBL: LINE=({cells},END)\n"


    defect = []
    for mode in modes():
        # Independent size: the uncorrected path excess of one magnet, which the correction subtracts.
        single = (
            path_length(mode, f"B: {BEND},REFERENCE_CORRECTION=0\nBL: LINE=(B)\n", False, True) - 0.85
        )
        predicted = (COPIES - 1) * single  # every copy after the first misses its subtraction
        print(
            f"{mode} one uncorrected magnet: path excess {single:.10g} m;"
            f" predicted excess with sharing {predicted:.10g} m"
        )
        for twiss, share in ((False, False), (False, True), (True, False), (True, True)):
            excess = path_length(mode, line(), twiss, share) - NOMINAL
            label = f"{mode} twiss_output={'yes' if twiss else 'no'} sharing={'on' if share else 'off'}"
            print(f"{label}: path - nominal observed={excess:.10g} m expected=0")
            if not (twiss and share):
                if abs(excess) > TOL:
                    sys.exit(f"{label}: the control does not return the nominal path: invalid")
                continue
            if abs(excess - predicted) > TOL:
                sys.exit(f"{label}: the excess is not the missing corrections: the cause differs")
            defect.append(abs(excess) > TOL)
        excess = path_length(mode, line(0.13), True, True) - NOMINAL
        print(
            f"{mode} TILT=0.13 twiss_output=yes sharing=on (copies recompute their reference):"
            f" path - nominal observed={excess:.10g} m expected=0"
        )
        if abs(excess) > TOL:
            sys.exit("The tilted case also loses the correction: the scope differs.")
    print(f"comparison: abs tolerance {TOL:g} m")
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
| Found (cause in the matrix-sharing copy and the RF energy shift) | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed (untilted CCBENDs; a nonzero design TILT forces reoptimization and masks the defect) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

