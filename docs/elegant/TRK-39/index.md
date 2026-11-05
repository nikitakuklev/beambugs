---
title: "TRK-39: CSBEND FSE_CORRECTION is applied only after a command computes the bend's matrix"
hide:
  - toc
---

# TRK-39 — CSBEND FSE_CORRECTION is applied only after a command computes the bend's matrix

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`compute_matrix`, compute_matrices.c:1793–1798](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1793-L1798)<br>[`track_through_csbend`, csbend.c:1111](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1111)<br>[csbend.c:1504–1506](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1504-L1506)<br>[csbend.c:6678–6700](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L6678-L6700)<br>[manual, csbend.tex:76–80](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L76-L80) |

CSBEND fits the FSE_CORRECTION strength adjustment only when a command such as twiss_output computes the bend's transfer matrix; a run that only tracks applies no correction and prints no warning. For an error-free 0.75 m, 0.09 rad bend with edge model 2 or 4, the corrected bend, obtained after twiss_output or by setting the fitted FSE by hand, has exit slope x′ = 0 and, with FSE_CORRECTION=1, the nominal path 0.75 m. Tracking alone gives x′ = 5.031×10⁻⁶ and a path 4.441×10⁻⁵ m shorter, so adding a twiss_output command changes the tracked orbit.

??? example "Reproducer: TRK-39.py"

    [Download TRK-39.py](TRK-39.py)

    ```python
    """TRK-39: CSBEND FSE_CORRECTION is applied only after a command computes the bend's matrix

    CSBEND fits the FSE_CORRECTION strength adjustment only when a command such as twiss_output computes
    the bend's transfer matrix; a run that only tracks applies no correction and prints no warning. For
    an error-free 0.75 m, 0.09 rad bend with edge model 2 or 4, the corrected bend, obtained after
    twiss_output or by setting the fitted FSE by hand, has exit slope x′ = 0 and, with FSE_CORRECTION=1,
    the nominal path 0.75 m. Tracking alone gives x′ = 5.031×10⁻⁶ and a path 4.441×10⁻⁵ m shorter, so
    adding a twiss_output command changes the tracked orbit.

    The script tracks an on-axis electron through an error-free CSBEND with FSE_CORRECTION=1 and edge
    models 2 and 4, once in a tracking-only run and once after twiss_output. The expected exit slope is
    that of the bend with the fitted FSE set by hand and refitting off, and the expected path is the
    nominal length that FSE_CORRECTION=1 imposes. The run with twiss_output is the control.
    """

    import sys

    from _csbend_fse import LENGTH, track
    from _elegant import modes, verdict

    TOL = 1e-10  # the fitted FSE is printed to 6 digits; that moves the exit slope by below 1e-12 rad

    defect = []
    for mode in modes():
        for model in (2, 4):
            with_twiss, fitted = track(mode, model, {"FSE_CORRECTION": 1}, twiss=True)
            if fitted is None:
                sys.exit(f"{mode} model {model}: twiss_output did not fit the correction: invalid")
            # Expected slope: the fitted correction applied by hand, independent of the fitting call.
            expected = track(mode, model, {"FSE_CORRECTION": 0, "FSE": fitted}, twiss=False)[0]["xp"]
            only, refit = track(mode, model, {"FSE_CORRECTION": 1}, twiss=False)
            label = f"{mode} EDGE_EFFECTS={model}"
            print(f"{label}: fitted FSE {fitted:.6g}; expected exit xp {expected:.6g}, path {LENGTH} m")
            for name, run_ in (
                ("twiss_output then track (control)", with_twiss),
                ("tracking only", only),
            ):
                print(
                    f"{label} {name}: xp={run_['xp']:.10g} path={run_['path']:.13g}"
                    f" (diff {run_['xp'] - expected:.4g} rad, {run_['path'] - LENGTH:.4g} m)"
                )
            if abs(with_twiss["xp"] - expected) > TOL or abs(with_twiss["path"] - LENGTH) > 1e-12:
                sys.exit(f"{label}: the control disagrees: invalid")
            print(f"{label} tracking only: correction fitted={refit is not None}")
            defect.append(abs(only["xp"] - expected) > TOL and refit is None)
    print("comparison: abs tolerance 1e-10 rad on xp, 1e-12 m on path")
    verdict(bool(defect) and all(defect))

    ```

??? example "Helper: _csbend_fse.py"

    [Download _csbend_fse.py](_csbend_fse.py)

    ```python
    """Shared case for the CSBEND FSE_CORRECTION reproducers: an asymmetric pure dipole with soft edges.

    `track` runs one on-axis electron through the bend with optional twiss_output (which computes the
    element matrix) and returns its exit coordinates, physical path and the FSE value that Elegant
    printed when it fitted the correction (None when it did not fit).
    """

    from __future__ import annotations

    import re
    import sys

    import numpy as np
    from _elegant import particles, run

    C = 299792458.0  # speed of light (m/s)
    P0 = 6000.0  # reference momentum in m_e c
    LENGTH = 0.75
    # Unequal edge angles make the edge-induced exit slope nonzero, so the correction has work to do.
    BEND = (
        f"CSBEND,L={LENGTH!r},ANGLE=0.09,E1=0.02,E2=-0.015,HGAP=0.025,FINT=0.5,EDGE_ORDER=1,"
        "N_SLICES=80,INTEGRATION_ORDER=6,SYNCH_RAD=0,ISR=0"
    )
    N = 12  # identical particles, so that every MPI worker tracks some


    def track(mode: str, model: int, params: dict, twiss: bool) -> tuple[dict, float | None]:
        """Exit coordinates (x, xp, y, yp, path) of the on-axis electron and the fitted FSE, if any."""
        extra = "".join(f",{k}={v!r}" for k, v in params.items())
        lattice = f"B: {BEND},EDGE1_EFFECTS={model},EDGE2_EFFECTS={model}{extra}\nEND: MARK\nBL: LINE=(B,END)\n"
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds &end\n"
            + ("&twiss_output filename=out.twi, matched=0, beta_x=2, beta_y=3 &end\n" if twiss else "")
            + "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": {"p": np.full(N, P0)}}
        )
        out = particles(result["out.sdds"])
        if len(out["p"]) != N or any(np.ptp(out[c]) > 0 for c in ("x", "xp", "y", "yp", "t", "p")):
            sys.exit(f"{mode}: particles lost or not identical: invalid")
        p = out["p"][0]
        coords = {c: float(out[c][0]) for c in ("x", "xp", "y", "yp")}
        coords["path"] = float(out["t"][0] * C * p / np.sqrt(1 + p * p))
        fits = re.findall(r"FSE optimized to (\S+)", result["log"])
        return coords, (float(fits[-1]) if fits else None)

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

