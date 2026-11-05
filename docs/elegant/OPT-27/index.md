---
title: "OPT-27: Tracking tune shift with amplitude retries without end when NAFF finds no tune in its window"
hide:
  - toc
---

# OPT-27 — Tracking tune shift with amplitude retries without end when NAFF finds no tune in its window

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Tags | <span class="bb-tag">extreme-regime</span> [?](../../methodology.md#tags) |
| Source | [`computeTuneShiftWithAmplitude`, src/twiss.cc:3582–3585](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3582-L3585)<br>[src/twiss.cc:3624–3629](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3624-L3629)<br>[the retry, src/twiss.cc:3666–3684](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3666-L3684)<br>[`computeTunesFromTracking`, src/twiss.cc:4104–4117](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L4104-L4117)<br>[`PerformNAFF`, fftpack/naff.c:169](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/fftpack/naff.c#L169)<br>[skips every line outside the window and stops without a frequency when none is left, fftpack/naff.c:197–208](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/fftpack/naff.c#L197-L208)<br>[src/twiss.cc:3483–3484](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3483-L3484)<br>[successive tunes are looked for in the region of the given width around the previous tune, doc/elegant.tex:8468–8473](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8468-L8473)<br>[`scaling_iterations` and the scale factors control automatic scaling of the amplitudes, doc/elegant.tex:8474–8481](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8474-L8481)<br>[the command is deprecated and it is the user's responsibility to optimize its parameters, doc/elegant.tex:8389–8402](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8389-L8402) |

In tracking mode, Elegant's `tune_shift_with_amplitude` treats a failed NAFF tune analysis as a lost particle, shrinks the amplitudes and retries, resetting its retry counter each time, so an analysis that fails at every amplitude never ends the run. A tune window `nux_roi_width` that is narrower than the FFT line spacing 1/turns and contains no line causes this. For four linear FODO cells with tune νx = 0.1444, 256 turns and `nux_roi_width` = 10⁻¹⁰, the nearest line is 1.727×10⁻⁴ from the tune; with `scaling_iterations` = 2 the run retried over 4000 times in 30 s, repeating one state. With `spread_only=1` it finishes, and the default width 0.02 always contains a line, since at least 100 turns are required.

??? example "Reproducer: OPT-27.py"

    [Download OPT-27.py](OPT-27.py)

    ```python
    """OPT-27: Tracking tune shift with amplitude retries without end when NAFF finds no tune in its window

    In tracking mode, Elegant's `tune_shift_with_amplitude` treats a failed NAFF tune analysis as a lost
    particle, shrinks the amplitudes and retries, resetting its retry counter each time, so an analysis
    that fails at every amplitude never ends the run. A tune window `nux_roi_width` that is narrower
    than the FFT line spacing 1/turns and contains no line causes this. For four linear FODO cells with
    tune νx = 0.1444, 256 turns and `nux_roi_width` = 10⁻¹⁰, the nearest line is 1.727×10⁻⁴ from the
    tune; with `scaling_iterations` = 2 the run retried over 4000 times in 30 s, repeating one state.
    With `spread_only=1` it finishes, and the default width 0.02 always contains a line, since at least
    100 turns are required.

    The script runs serial Elegant `tune_shift_with_amplitude` in tracking mode (256 turns, a 3×3 grid
    on the axes, `scaling_iterations=2`, `verbose=1`) on four linear FODO cells, whose horizontal
    tune it computes independently from the 2×2 matrices, and shows that no FFT frequency of the
    NAFF search lies within the narrow window. With the default `nux_roi_width` = 0.02 (control) and
    with the narrow width under `spread_only=1` (control) the run finishes; with the narrow width in
    coefficient mode it is killed after 30 s. The script counts the completed retries in the partial
    output and checks that the largest grid amplitude has reached its limit x0·√2, which means the
    largest amplitude x1 has been driven to zero and every further retry repeats the same failed
    analysis.
    """

    import os
    import re
    import shutil
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path

    import numpy as np
    from _elegant import verdict

    TURNS, GRID, ITERATIONS = 256, 3, 2
    X0, X1 = 1e-6, 3e-4
    HANG_TIMEOUT = 30.0  # seconds; the controls finish in well under 1 s
    QUAD_L, QUAD_K1, DRIFT_L = 0.2, 1.0, 1.0
    LATTICE = f"""QF: QUAD,L={QUAD_L},K1={QUAD_K1}
    QD: QUAD,L={QUAD_L},K1={-QUAD_K1}
    D: DRIF,L={DRIFT_L}
    C: LINE=(QF,D,QD,D)
    RING: LINE=(4*C)
    """


    def commands(roi_width, spread_only):
        # Elegant requires tune_shift_with_amplitude before twiss_output.
        return (
            "&run_setup lattice=lattice.lte, use_beamline=RING, p_central=1000, default_order=2 &end\n"
            "&run_control &end\n"
            f"&tune_shift_with_amplitude turns={TURNS}, grid_size={GRID}, order=1, lines_only=1,"
            f" x0={X0}, y0={X0}, x1={X1}, y1={X1}, nux_roi_width={roi_width},"
            f" spread_only={spread_only}, scaling_iterations={ITERATIONS}, verbose=1 &end\n"
            "&twiss_output filename=twiss.sdds, matched=1, output_at_each_step=1 &end\n"
            "&bunched_beam &end\n&track &end\n"
        )


    def run_limited(roi_width, spread_only, timeout):
        """Run elegant; return (finished, exit status or None, elapsed s, stdout so far)."""
        if "RPN_DEFNS" not in os.environ:
            sys.exit("Set RPN_DEFNS to the defns.rpn file of SDDS (see the elegant installation).")
        if shutil.which("elegant") is None:
            sys.exit("elegant must be on PATH.")
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="elegant-repro-") as tmp:
            work = Path(tmp)
            (work / "lattice.lte").write_text(LATTICE)
            (work / "run.ele").write_text(commands(roi_width, spread_only))
            with open(work / "log.txt", "w") as log:
                start = time.monotonic()
                proc = subprocess.Popen(
                    ["elegant", "run.ele"], cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                try:
                    status = proc.wait(timeout=timeout)
                    finished = True
                except subprocess.TimeoutExpired:
                    proc.kill()  # stdout still in elegant's buffer is lost: counts are lower bounds
                    proc.wait()
                    status, finished = None, False
                elapsed = time.monotonic() - start
            return finished, status, elapsed, (work / "log.txt").read_text(errors="replace")


    def thick_quad(k1, length):
        w = np.sqrt(abs(k1))
        if k1 > 0:
            c, s = np.cos(w * length), np.sin(w * length)
            return np.array([[c, s / w], [-w * s, c]])
        c, s = np.cosh(w * length), np.sinh(w * length)
        return np.array([[c, s / w], [w * s, c]])


    # Independent linear tune: product of the horizontal 2x2 matrices, cos(2 pi nu) = trace/2.
    drift = np.array([[1.0, DRIFT_L], [0.0, 1.0]])
    cell = drift @ thick_quad(-QUAD_K1, QUAD_L) @ drift @ thick_quad(QUAD_K1, QUAD_L)
    ring = np.linalg.matrix_power(cell, 4)
    nux = np.arccos(np.trace(ring) / 2) / (2 * np.pi)
    # NAFF starts from the largest FFT line at frequencies k/turns and skips every line outside
    # [nu - w/2, nu + w/2]; with no line inside it returns a failure. The linear tune does not
    # change with amplitude, so the window around it stays empty at every amplitude.
    nearest = round(nux * TURNS) / TURNS
    print(
        f"independent linear tune nux = {nux:.13f}; nearest FFT line {nearest} at distance "
        f"{abs(nearest - nux):.4g}; narrow window half-width 5e-11"
    )

    finished, status, elapsed, log = run_limited(0.02, 0, 60)
    tunes = [float(v) for v in re.findall(r"nux=([0-9.eE+-]+)", log)]
    print(
        f"control nux_roi_width=0.02: observed=finished={finished}, status={status}, "
        f"{elapsed:.2f} s, first NAFF tune {tunes[0]:.10f} expected=finished, status 0"
    )
    if not (finished and status == 0 and abs(tunes[0] - nux) < 1e-7):
        sys.exit("The default-width control did not finish with the linear tune: invalid.")
    finished, status, elapsed, log = run_limited(1e-10, 1, 60)
    print(
        f"control nux_roi_width=1e-10 spread_only=1: observed=finished={finished}, "
        f"status={status}, {elapsed:.2f} s, NAFF-failure warnings {log.count('NAFF failed')} "
        "expected=finished, status 0"
    )
    if not (finished and status == 0):
        sys.exit("The spread-only control did not finish: invalid.")

    finished, status, elapsed, log = run_limited(1e-10, 0, HANG_TIMEOUT)
    retries = log.count("All tunes computed for TSWA")
    # x on the grid is sqrt(ix (x1 - x0)^2/(n - 1) + x0^2); each retry sets x1 to a third of itself
    # (the source's (upper + 2 lower)/3 with lower = 0), so x1 underflows to 0 after about 680 retries
    # and the largest grid amplitude becomes x0 sqrt(2). From then on every retry is identical.
    amplitudes = [float(v) for v in re.findall(r"Tunes for TSWA: x=([0-9.eE+-]+)", log)]
    largest = max(amplitudes[-2 * GRID + 1 :]) if amplitudes else float("nan")
    print(
        f"nux_roi_width=1e-10 spread_only=0: observed=finished={finished}, killed after "
        f"{elapsed:.1f} s, at least {retries} completed retries, largest x in the last retry "
        f"{largest:.6e} m (x0 sqrt(2) = {X0 * np.sqrt(2):.6e}: x1 = 0) "
        f"expected=finished, with a result or an error (scaling_iterations={ITERATIONS})"
    )
    verdict(not finished and retries > 100 * ITERATIONS)

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

