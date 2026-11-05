---
title: "OPT-21: Tracking tune shift with amplitude ignores nuy_roi_width and the amplitude scaling factors"
hide:
  - toc
---

# OPT-21 — Tracking tune shift with amplitude ignores nuy_roi_width and the amplitude scaling factors

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`computeTuneShiftWithAmplitude`, peak windows, src/twiss.cc:3619–3622](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3619-L3622)<br>[the amplitude retry, src/twiss.cc:3666–3684](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3666-L3684)<br>[the namelist, src/twiss.nl:34–38](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.nl#L34-L38)<br>[`nux_roi_width`, `nuy_roi_width` are the widths of the regions of interest for the x and y tunes, doc/elegant.tex:8468–8473](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8468-L8473)<br>[a tune shift above `scale_down_limit` decreases the amplitude by `scale_down_factor`, one below `scale_up_limit` increases it by `scale_up_factor`, doc/elegant.tex:8474–8481](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8474-L8481) |

In tracking mode, Elegant's `tune_shift_with_amplitude` searches the vertical tune peak within `nux_roi_width` and never reads `nuy_roi_width`, `scale_down_factor`, `scale_up_factor` or `scale_up_limit`, all documented in the manual. When the tune change exceeds `scale_down_limit`, it sets the largest amplitude to one third of itself plus two thirds of the smallest, whatever factor was requested, and it never scales amplitudes up. For eight FODO cells with a thin octupole, changing any of these controls (vertical window 0.02 to 10⁻¹⁰, scale-up factor 2 to 5, scale-up limit 0.1 to 10⁻¹⁰, scale-down factor 2 to 10) leaves every tracked tune and fitted coefficient bit-identical; both scale-down factors reduce the largest amplitude from 1.999 mm to 0.6657 mm.

??? example "Reproducer: OPT-21.py"

    [Download OPT-21.py](OPT-21.py)

    ```python
    """OPT-21: Tracking tune shift with amplitude ignores nuy_roi_width and the amplitude scaling factors

    In tracking mode, Elegant's `tune_shift_with_amplitude` searches the vertical tune peak within
    `nux_roi_width` and never reads `nuy_roi_width`, `scale_down_factor`, `scale_up_factor` or
    `scale_up_limit`, all documented in the manual. When the tune change exceeds `scale_down_limit`, it
    sets the largest amplitude to one third of itself plus two thirds of the smallest, whatever factor
    was requested, and it never scales amplitudes up. For eight FODO cells with a thin octupole,
    changing any of these controls (vertical window 0.02 to 10⁻¹⁰, scale-up factor 2 to 5, scale-up
    limit 0.1 to 10⁻¹⁰, scale-down factor 2 to 10) leaves every tracked tune and fitted coefficient
    bit-identical; both scale-down factors reduce the largest amplitude from 1.999 mm to 0.6657 mm.

    The script runs serial Elegant `tune_shift_with_amplitude` in tracking mode (4096 turns, a 4×4 grid
    from 1 μm to 2 mm, `tune_output`) on eight FODO cells with one thin octupole, changing one
    documented control at a time: `nuy_roi_width` from 0.02 to 10⁻¹⁰, `scale_up_factor` from 2 to 5,
    `scale_up_limit` from 0.1 to 10⁻¹⁰, and, with `scale_down_limit` = 10⁻⁵, `scale_down_factor` from
    2 to 10. It compares the tracked grids (x, y, νx, νy) and the fitted coefficients bit for bit. The
    largest tune change on the grid is printed as a control: it exceeds the narrow vertical window by
    five orders of magnitude and stays far below the scale-up limit of 0.1.
    """

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
    COMMON = "turns=4096, x0=1e-6, y0=1e-6, x1=0.002, y1=0.002, grid_size=4, order=1"
    COEFFICIENTS = ["dnux/dAx", "dnux/dAy", "dnuy/dAx", "dnuy/dAy"]


    def tswa(options):
        """Tracked grid (x, y, nux, nuy) and fitted coefficients for one set of options."""
        commands = (
            "&run_setup lattice=lattice.lte, use_beamline=A, p_central_mev=2400, default_order=3 &end\n"
            f"&tune_shift_with_amplitude {COMMON}, tune_output=tunes.sdds, {options} &end\n"
            "&twiss_output filename=twiss.sdds, matched=1, output_at_each_step=1, concat_order=3 &end\n"
            "&run_control &end\n&bunched_beam &end\n&track &end\n"
        )
        result = run(
            {"lattice.lte": LATTICE, "run.ele": commands}, outputs=("tunes.sdds", "twiss.sdds")
        )
        grid = np.array(
            [np.asarray(result["tunes.sdds"][0, c], float) for c in ("x", "y", "nux", "nuy")]
        )
        coefficients = np.array([float(result["twiss.sdds"][name][0]) for name in COEFFICIENTS])
        return grid, coefficients


    def identical(a, b):
        return a[0].shape == b[0].shape and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


    defect = False
    up = "nuy_roi_width=0.02, scale_up_limit=0.1, scale_up_factor=2"
    base = tswa(up)
    spread = np.ptp(base[0][3])
    print(
        f"control: vertical tune range on the grid = {spread:.4g}, largest tune change = "
        f"{max(np.ptp(base[0][2]), spread):.4g}; amplitude range x = {base[0][0].min():.4g} to "
        f"{base[0][0].max():.10g} m"
    )
    print(f"base coefficients {', '.join(COEFFICIENTS)} (1/m): {np.array2string(base[1], precision=8)}")
    # Each pair differs in one documented control; the manual says each changes the algorithm here.
    pairs = [
        (
            "nuy_roi_width 0.02 -> 1e-10 (vertical peak search window narrower than the tune change)",
            up.replace("nuy_roi_width=0.02", "nuy_roi_width=1e-10"),
        ),
        (
            "scale_up_factor 2 -> 5 (tune change below scale_up_limit=0.1: amplitudes scaled up)",
            up.replace("scale_up_factor=2", "scale_up_factor=5"),
        ),
        (
            "scale_up_limit 0.1 -> 1e-10 (scale-up no longer triggered)",
            up.replace("scale_up_limit=0.1", "scale_up_limit=1e-10"),
        ),
    ]
    for label, options in pairs:
        same = identical(base, tswa(options))
        print(f"{label}: observed={'identical' if same else 'different'} expected=different")
        defect |= same
    down2 = tswa("scale_down_limit=1e-5, scale_down_factor=2")
    down10 = tswa("scale_down_limit=1e-5, scale_down_factor=10")
    # The manual: a tune change above scale_down_limit decreases the amplitude by scale_down_factor.
    # The source instead sets x1 = (x1 + 2 x0)/3 on every retry, whatever the factor.
    print(f"scale_down_factor=2: largest x on the grid = {down2[0][0].max():.10g} m")
    print(f"scale_down_factor=10: largest x on the grid = {down10[0][0].max():.10g} m")
    same = identical(down2, down10)
    print(
        f"scale_down_factor 2 -> 10: observed={'identical' if same else 'different'} expected=different"
    )
    defect |= same
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
| Confirmed (nuy_roi_width, scale_down_factor, scale_up_factor and scale_up_limit) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

