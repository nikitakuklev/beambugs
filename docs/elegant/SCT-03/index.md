---
title: "SCT-03: MPI Touschek generation crashes when tracking is disabled"
hide:
  - toc
---

# SCT-03 — MPI Touschek generation crashes when tracking is disabled

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`lostParticle` declaration, touschekScatter.c:345](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L345)<br>[tracking branch, touschekScatter.c:735–797](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L735-L797)<br>[cleanup, touschekScatter.c:886–889](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L886-L889)<br>[`free_czarray_2d`, mdblib/array.c:420–424](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdblib/array.c#L420-L424)<br>[namelist default, touschekScatter.nl:37](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.nl#L37)<br>[touschek_scatter, elegant.tex:7699](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7699)<br>[`do_track`, elegant.tex:7809–7813](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7809-L7813) |

When Pelegant, Elegant's MPI version, runs the touschek_scatter command with its default do_track = 0, which generates the scattered particles without tracking them, it crashes with a segmentation fault after generating the particles at the first TSCATTER element. Serial Elegant completes the same input. The cleanup frees the array of lost particles on the master rank, but that array exists only when tracking is enabled, and the freeing routine dereferences it. For three TSCATTER elements and 40 generated particles per element, serial Elegant writes three pages of 40 particles, and Pelegant with do_track = 1 does too, but Pelegant with do_track = 0 aborts.

??? example "Reproducer: SCT-03.py"

    [Download SCT-03.py](SCT-03.py)

    ```python
    """SCT-03: MPI Touschek generation crashes when tracking is disabled

    When Pelegant, Elegant's MPI version, runs the touschek_scatter command with its default do_track =
    0, which generates the scattered particles without tracking them, it crashes with a segmentation
    fault after generating the particles at the first TSCATTER element. Serial Elegant completes the
    same input. The cleanup frees the array of lost particles on the master rank, but that array exists
    only when tracking is enabled, and the freeing routine dereferences it. For three TSCATTER elements
    and 40 generated particles per element, serial Elegant writes three pages of 40 particles, and
    Pelegant with do_track = 1 does too, but Pelegant with do_track = 0 aborts.

    The script runs touschek_scatter on a three-element TSCATTER lattice with 40 generated particles per
    element, once in serial Elegant and twice in 3-rank Pelegant, with tracking disabled and enabled. It
    compares the pages of the scattered-particle file with the expected three pages of 40 and reports
    how a run that aborts ends. The serial run and the tracked parallel run are the controls: they show
    that the input is valid and that the crash follows from skipping tracking.
    """

    import sys

    from _elegant import RunFailure, modes, run, verdict

    PARTICLES = 40  # generated per TSCATTER site
    LATTICE = "T0: TSCATTER\nD0: DRIF,L=0.4\nT1: TSCATTER\nD1: DRIF,L=0.9\nT2: TSCATTER\nBL: LINE=(T0,D0,T1,D1,T2)\n"
    # Momentum aperture of +-0.003 at the three sites, in the table format touschek_scatter reads.
    APERTURE = """SDDS1
    &column name=s, type=double, units=m &end
    &column name=deltaPositive, type=double &end
    &column name=deltaNegative, type=double &end
    &column name=ElementName, type=string &end
    &column name=ElementType, type=string &end
    &column name=ElementOccurence, type=long &end
    &data mode=ascii, no_row_counts=1 &end
    0.0 0.003 -0.003 T0 TSCATTER 1
    0.4 0.003 -0.003 T1 TSCATTER 1
    1.3 0.003 -0.003 T2 TSCATTER 1
    """
    COMMANDS = """&run_setup lattice=lattice.lte, use_beamline=BL, p_central=800.0, random_number_seed=31317 &end
    &run_control n_passes=1 &end
    &twiss_output matched=0, beta_x=8, alpha_x=0.3, beta_y=5, alpha_y=-0.2 &end
    &touschek_scatter charge=1e-09, frequency=230609583.07692307, emit_x=2e-09, emit_y=3e-10,
     sigma_dp=0.0008, sigma_s=0.004, Momentum_Aperture="aperture.sdds", Momentum_Aperture_scale=1,
     verbosity=4, n_simulated={n}, ignored_portion=0, do_track={track}, bunch="scattered.sdds" &end
    """


    def crash_signature(log):
        """True when the log shows the established crash: SIGSEGV after the beam data is freed, before the rest."""
        return (
            "Terminated by SIGSEGV" in log
            and "Freeing beam data" in log
            and "Freeing other data" not in log
        )


    def generate(mode, track):
        """Rows per page of the scattered-particle file, or None after the established crash.

        Any other failure propagates and ends the script without a verdict.
        """
        files = {
            "lattice.lte": LATTICE,
            "aperture.sdds": APERTURE,
            "run.ele": COMMANDS.format(n=PARTICLES, track=track),
        }
        try:
            result = run(files, mode, outputs=("scattered.sdds",))
        except RunFailure as error:
            if mode == "mpi" and not track and crash_signature(error.log):
                return None
            raise
        bunch = result["scattered.sdds"]
        return [len(bunch[page, "x"]) for page in range(bunch.n_pages)]


    defect = False
    print("rows per page of the scattered-particle file, one page per TSCATTER site")
    for mode in ("serial", *modes(mpi_only=True)):
        for track in (0, 1):
            if mode == "serial" and track:
                continue
            observed = generate(mode, track)
            control = mode == "serial" or track
            label = f"{mode} do_track={track}" + (" (control)" if control else "")
            shown = (
                str(observed)
                if observed is not None
                else "[SIGSEGV after 'Freeing beam data', before 'Freeing other data']"
            )
            print(f"{label}: observed={shown} expected={[PARTICLES] * 3}")
            # A completed run must hold three pages of 40 particles; anything else is an unrelated
            # output failure. The failing run may already have written the first site's page.
            if observed is not None and observed != [PARTICLES] * 3:
                sys.exit(f"{label} completed with pages {observed}: the comparison is not valid")
            defect |= observed is None
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
| Found | `gpt-6-astra` | xhigh | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, 3-rank Pelegant |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

