---
title: "DOC-10: Manual gives the tune-shift amplitude as (1 + α)q²/β and the lines_only default as 0"
hide:
  - toc
---

# DOC-10 — Manual gives the tune-shift amplitude as (1 + α)q²/β and the lines_only default as 0

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [amplitude definition in `tune_shift_with_amplitude`, doc/elegant.tex:8412–8413](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8412-L8413)<br>[in the `twiss_output` column list, doc/elegant.tex:8241–8242](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8241-L8242)<br>[the namelist listing, doc/elegant.tex:8425](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8425)<br>[`Ax = x²(1 + αx²)/βx`, src/twiss.cc:3595](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3595)<br>[`Ay`, src/twiss.cc:3610](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3610)<br>[`lines_only = 1`, src/twiss.nl:28](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.nl#L28) |

The Elegant manual defines the amplitude of `tune_shift_with_amplitude` and of the `twiss_output` columns dnux/dAx, dnux/dAy, dnuy/dAx and dnuy/dAy as A_q = (1 + α_q)q²/β_q, with α_q, β_q the Twiss functions and q = x or y, and lists the `lines_only` default as 0. Elegant uses the Courant–Snyder invariant (1 + α_q²)q²/β_q and defaults `lines_only` to 1. At a start with αx = −0.8574 and βx = 6.877 m, a particle at x = 1.999 mm has Ax = 1.008 μm, where the printed formula gives 0.08286 μm; omitting `lines_only` with a 4×4 grid tracks 7 points on two lines instead of the 16 of the documented default.

??? example "Reproducer: DOC-10.py"

    [Download DOC-10.py](DOC-10.py)

    ```python
    """DOC-10: Manual gives the tune-shift amplitude as (1 + α)q²/β and the lines_only default as 0

    The Elegant manual defines the amplitude of `tune_shift_with_amplitude` and of the `twiss_output`
    columns dnux/dAx, dnux/dAy, dnuy/dAx and dnuy/dAy as A_q = (1 + α_q)q²/β_q, with α_q, β_q the Twiss
    functions and q = x or y, and lists the `lines_only` default as 0. Elegant uses the Courant–Snyder
    invariant (1 + α_q²)q²/β_q and defaults `lines_only` to 1. At a start with αx = −0.8574 and βx =
    6.877 m, a particle at x = 1.999 mm has Ax = 1.008 μm, where the printed formula gives 0.08286 μm;
    omitting `lines_only` with a 4×4 grid tracks 7 points on two lines instead of the 16 of the
    documented default.

    The script reads both statements from `doc/elegant.tex` of the Elegant source checkout named by
    `ELEGANT_SOURCE` (required) and exits if either is missing or unrecognized. It then runs serial
    Elegant `tune_shift_with_amplitude` in tracking mode on eight FODO cells with one thin octupole,
    started where αx = −0.8574, and compares the written amplitudes Ax with x²(1 + αx²)/βx and with
    the manual's x²(1 + αx)/βx; it counts the tracked grid points with `lines_only` omitted and with
    `lines_only=0`, the documented default.
    """

    import os
    import re
    import sys
    from pathlib import Path

    import numpy as np
    from _elegant import run, verdict

    LATTICE = """F: QUAD,L=0.25,K1=1.1,ORDER=1
    G: QUAD,L=0.25,K1=-1.0,ORDER=1
    D: DRIF,L=0.7
    E: DRIF,L=0.9
    O: MULT,ORDER=3,KNL=3.0
    C: LINE=(F,D,G,E)
    A: LINE=(O,8*C)
    """
    GRID = 4

    source = os.environ.get("ELEGANT_SOURCE")
    if not source:
        sys.exit("ELEGANT_SOURCE must name the Elegant source checkout.")
    manual = Path(source) / "doc" / "elegant.tex"
    if not manual.is_file():
        sys.exit(f"{manual} does not exist.")
    text = manual.read_text()
    section = text[text.index(r"\subsection{tune\_shift\_with\_amplitude") :]
    section = section[: section.index(r"\subsection", 1)]
    formula = re.findall(r"\$A_q = \(1 \+ \\alpha_q(\^2)?\) q\^2/\\beta_q\$", section)
    default = re.findall(r"long lines_only = (\d+);", section)
    if len(formula) != 1 or len(default) != 1:
        sys.exit(
            f"The manual's amplitude formula or lines_only default is unrecognized: {formula} {default}"
        )
    manual_square = formula[0] == "^2"
    manual_default = int(default[0])
    print(
        f"manual: A_q = (1 + alpha_q{'^2' if manual_square else ''}) q^2/beta_q; lines_only default = {manual_default}"
    )


    def tswa(lines_only):
        option = "" if lines_only is None else f", lines_only={lines_only}"
        commands = (
            "&run_setup lattice=lattice.lte, use_beamline=A, p_central_mev=2400, default_order=3 &end\n"
            "&tune_shift_with_amplitude turns=4096, x0=1e-6, y0=1e-6, x1=0.002, y1=0.002,"
            f" grid_size={GRID}, order=1, tune_output=tunes.sdds{option} &end\n"
            "&twiss_output filename=twiss.sdds, matched=1, output_at_each_step=1, concat_order=3 &end\n"
            "&run_control &end\n&bunched_beam &end\n&track &end\n"
        )
        result = run(
            {"lattice.lte": LATTICE, "run.ele": commands}, outputs=("tunes.sdds", "twiss.sdds")
        )
        return result["tunes.sdds"], result["twiss.sdds"]


    defect = False
    tunes, twiss = tswa(None)
    beta, alpha = float(twiss[0, "betax"][0]), float(twiss[0, "alphax"][0])
    x = np.asarray(tunes[0, "x"], float)
    ax = np.asarray(tunes[0, "Ax"], float)
    # The Courant-Snyder invariant of a particle at (x, x' = 0) is gamma x^2 = (1 + alpha^2) x^2/beta.
    invariant = x**2 * (1 + alpha**2) / beta
    printed = x**2 * (1 + alpha**2 if manual_square else 1 + alpha) / beta
    print(f"start of ring: betax={beta:.10g} m alphax={alpha:.10g}")
    i = int(np.argmax(x))
    print(
        f"Ax at x={x[i]:.6g} m: written={ax[i]:.10g} m, (1+alpha^2) x^2/beta={invariant[i]:.10g} m,"
        f" manual formula={printed[i]:.10g} m"
    )
    if not np.allclose(ax, invariant, rtol=1e-12):
        sys.exit("The written Ax is not the Courant-Snyder invariant: unexpected implementation.")
    mismatch = not np.allclose(ax, printed, rtol=1e-12)
    print(
        f"manual amplitude formula reproduces the written Ax: observed={'no' if mismatch else 'yes'} expected=yes"
    )
    defect |= mismatch

    full = len(tswa(0)[0][0, "x"])
    lines = len(tswa(1)[0][0, "x"])
    omitted = len(x)
    expected = full if manual_default == 0 else lines
    print(
        f"grid points: lines_only=0 -> {full}, lines_only=1 -> {lines} (control: {GRID}x{GRID} grid and two lines)"
    )
    if full != GRID * GRID or lines != 2 * GRID - 1:
        sys.exit(
            "The explicit lines_only runs do not give the full grid and the two lines: invalid comparison."
        )
    print(
        f"grid points with lines_only omitted: observed={omitted} expected (manual default {manual_default})={expected}"
    )
    defect |= omitted != expected
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
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

