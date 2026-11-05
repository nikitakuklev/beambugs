---
title: "IO-01: Pelegant fails to load a multi-page beam file when a later page leaves a worker without rows"
hide:
  - toc
---

# IO-01 — Pelegant fails to load a multi-page beam file when a later page leaves a worker without rows

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [activity flag, sdds_beam.c:886](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_beam.c#L886)<br>[activity assignment and column reads, sdds_beam.c:1092–1113](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_beam.c#L1092-L1113)<br>[empty-column check, SDDS_extract.c:886–888](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/SDDSlib/SDDS_extract.c#L886-L888)<br>[row partition, SDDS_MPI_binary.c:2243–2251](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/SDDSlib/SDDS_MPI_binary.c#L2243-L2251)<br>[track_pages_separately, doc/elegant.tex:7454–7455](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7454-L7455)<br>[unsupported cases, doc/Pelegant.tex:222–228](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/Pelegant.tex#L222-L228) |

When Pelegant, Elegant's MPI version, loads several particle-file pages with `track_pages_separately=0`, it should combine them into one beam. If a later page gives no rows to a worker that read an earlier page, the run stops with "Unable to get column--no rows left" before tracking. A five-particle file split into pages of four and one fails on three ranks (two workers), whereas serial Elegant and Pelegant with a one-plus-four split return all five particles unchanged.

??? example "Reproducer: IO-01.py"

    [Download IO-01.py](IO-01.py)

    ```python
    """IO-01: Pelegant fails to load a multi-page beam file when a later page leaves a worker without rows

    When Pelegant, Elegant's MPI version, loads several particle-file pages with `track_pages_separately=0`, it should combine them into one beam. If a later page gives no rows to a worker that read an earlier page, the run stops with "Unable to get column--no rows left" before tracking. A five-particle file split into pages of four and one fails on three ranks (two workers), whereas serial Elegant and Pelegant with a one-plus-four split return all five particles unchanged.

    The script writes a binary SDDS beam file of five particles in two pages and tracks it through a zero-length drift, so that the output must reproduce the input exactly. The page sizes 4+1 trigger the fault, and the controls 1+4 and 3+2 hold the same particles but give every page after the first at least two rows. Serial Elegant is the control for the 4+1 case.
    """

    import sys
    from itertools import pairwise

    import numpy as np
    from _elegant import RunFailure, modes, particles, run, verdict

    N = 5
    X = 1e-4 * np.arange(1, N + 1)  # m, distinct per particle so that every row is checked
    IDS = np.arange(1, N + 1)
    # A zero-length drift leaves the coordinates unchanged, so the output equals the input exactly.
    EXPECTED = {"x": X, "xp": np.zeros(N), "y": np.zeros(N), "yp": np.zeros(N), "t": np.zeros(N)} | {
        "p": np.full(N, 3.0)
    }
    LATTICE = "D: EDRIFT,L=0\nBL: LINE=(D)\n"
    COMMANDS = (
        "&run_setup lattice=lattice.lte,use_beamline=BL,p_central=3,output=out.sdds,"
        "default_order=2,load_balancing_on=0 &end\n"
        "&run_control n_steps=1,n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds,track_pages_separately=0 &end\n"
        "&track &end\n"
    )
    ERROR = "Unable to get column--no rows left"


    def pages_of(sizes):
        """The five particles split into consecutive pages of the given sizes."""
        edges = list(pairwise(np.cumsum([0, *sizes])))
        return [{"x": X[a:b], "p": np.full(len(X[a:b]), 3.0), "particleID": IDS[a:b]} for a, b in edges]


    def load(mode, pages):
        """(all six columns by particleID, "") read back from the output, or (None, error text) if the run stops."""
        try:
            result = run(
                {"lattice.lte": LATTICE, "run.ele": COMMANDS},
                mode,
                {"beam.sdds": pages_of(pages)},
            )
        except RunFailure as error:
            return None, ERROR if ERROR in error.log else error.log[-300:]
        if result["out.sdds"].n_pages != 1:
            sys.exit(f"{mode}: the output is not one page")
        return particles(result["out.sdds"]), ""


    def matches_input(found):
        """True when the output holds each input particle once with all six coordinates unchanged."""
        return (
            len(found["particleID"]) == N
            and np.array_equal(found["particleID"], IDS)
            and all(np.array_equal(found[name], EXPECTED[name]) for name in EXPECTED)
        )


    def compare(label, found, message):
        """Print the loaded (ID, x) rows or the error, next to the five rows of the input file."""
        rows = (
            "stopped: " + message
            if found is None
            else [(int(i), float(x)) for i, x in zip(found["particleID"], found["x"], strict=True)]
        )
        print(
            f"{label}: observed={rows} expected={[(int(i), float(x)) for i, x in zip(IDS, X, strict=True)]}"
        )


    defect = False
    for mode in ("serial", *modes(mpi_only=True)):
        for pages in ((4, 1), (1, 4), (3, 2)):
            found, message = load(mode, pages)
            label = f"{mode} pages {pages[0]}+{pages[1]}"
            compare(label, found, message)
            # All six coordinates of the output are compared with the input; the rows printed show x only.
            loaded = found is not None and matches_input(found)
            if mode == "serial" or pages != (4, 1):
                # Serial and the MPI controls must load all five particles unchanged.
                if not loaded:
                    sys.exit(f"{label}: control failed, the comparison is invalid ({message})")
            elif found is None:
                # Only the recognized error is the defect; any other failure is a failed run.
                if message != ERROR:
                    sys.exit(f"{label}: the run failed with another error ({message})")
                defect = True
            elif not loaded:
                sys.exit(f"{label}: the run succeeded but the output differs from the input")
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

