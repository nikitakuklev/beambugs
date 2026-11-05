---
title: "OPT-19: tracking_matrix_cleanup=1 crashes at second order"
hide:
  - toc
---

# OPT-19 — tracking_matrix_cleanup=1 crashes at second order

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`determineMatrixHigherOrder`, src/analyze.c:1528–1531](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/analyze.c#L1528-L1531)<br>[frees Q and sets `M->Q = NULL`, src/matrix.c:439–443](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix.c#L439-L443)<br>[`remove_s_dependent_matrix_elements`, src/matrix.c:1271–1290](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix.c#L1271-L1290)<br>[`global_settings`, src/global_settings.nl:29](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/global_settings.nl#L29) |

Setting `global_settings tracking_matrix_cleanup=1`, which removes spurious path-coordinate terms from element maps that Elegant derives by tracking, makes Elegant crash with a segmentation fault when the matrix order is 2, the `run_setup` default. The clean-up writes third-order coefficients after the third-order storage of the map has been freed. A beamline of two CCBENDs, whose maps are derived by tracking, ends with "Terminated by SIGSEGV" when its second-order matrix is computed; the same input at order 3, or with the clean-up off, finishes and writes R56 = 0.05613 m.

??? example "Reproducer: OPT-19.py"

    [Download OPT-19.py](OPT-19.py)

    ```python
    """OPT-19: tracking_matrix_cleanup=1 crashes at second order

    Setting `global_settings tracking_matrix_cleanup=1`, which removes spurious path-coordinate terms
    from element maps that Elegant derives by tracking, makes Elegant crash with a segmentation fault
    when the matrix order is 2, the `run_setup` default. The clean-up writes third-order coefficients
    after the third-order storage of the map has been freed. A beamline of two CCBENDs, whose maps are
    derived by tracking, ends with "Terminated by SIGSEGV" when its second-order matrix is computed; the
    same input at order 3, or with the clean-up off, finishes and writes R56 = 0.05613 m.

    The script computes the second-order matrix of two CCBENDs, whose maps Elegant derives by tracking,
    in serial Elegant with `global_settings tracking_matrix_cleanup=1` and `default_order=2`, and
    reports whether the run ends with a segmentation fault. The controls are the same input with the
    cleanup off and the same input at order 3; both must finish and write the matrix.
    """

    from _elegant import RunFailure, run, verdict

    LATTICE = "E: DRIF,L=0.8\nB: CCBEND,L=1.1,ANGLE=0.16,K1=0.1,K2=2,N_SLICES=32\nBL: LINE=(B,E,B)\n"


    def case(cleanup, order):
        """Return (finished, note) for one run."""
        commands = (
            f"&global_settings tracking_matrix_cleanup={cleanup} &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central_mev=3000,"
            f" default_order={order} &end\n"
            f"&matrix_output SDDS_output=matrix.sdds, SDDS_output_order={order} &end\n"
        )
        try:
            result = run({"lattice.lte": LATTICE, "run.ele": commands}, outputs=("matrix.sdds",))
        except RunFailure as failure:
            crashed = "SIGSEGV" in failure.log
            return False, f"exit status {failure.status}{', Terminated by SIGSEGV' if crashed else ''}"
        r56 = float(result["matrix.sdds"][0, "R56"][-1])
        return True, f"finished, R56={r56:.10g} m"


    defect = False
    for cleanup, order, control in ((0, 2, True), (1, 3, True), (1, 2, False)):
        finished, note = case(cleanup, order)
        label = "control, " if control else ""
        print(
            f"{label}tracking_matrix_cleanup={cleanup} default_order={order}: observed={note} expected=finished"
        )
        if control and not finished:
            raise SystemExit("A control run failed: invalid comparison.")
        defect |= not finished
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

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

