---
title: "TRK-38: CSBEND FSE_CORRECTION cancels a dipole-only strength error and a horizontal steering kick"
hide:
  - toc
---

# TRK-38 — CSBEND FSE_CORRECTION cancels a dipole-only strength error and a horizontal steering kick

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`csbend_update_fse_adjustment`, csbend.c:6678–6700](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L6678-L6700)<br>[`csbend_fse_adjustment_penalty`, csbend.c:6661–6674](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L6661-L6674)<br>[`track_through_csbend`, csbend.c:1111–1116](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1111-L1116)<br>[manual, csbend.tex:76–80](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L76-L80)<br>[parameter table, track_data.c:1961–1962](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1961-L1962)<br>[FSE_CORRECTION, track_data.c:2017](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2017) |

When a matrix calculation fits CSBEND's FSE_CORRECTION=1 or 2 for edge model 2 or 4, the calibration retains FSE_DIPOLE and XKICK and absorbs their horizontal exit-slope effects. For a 0.75 m, 0.09 rad bend, adding FSE_DIPOLE = 7×10⁻⁴ to the bend with its error-free correction held fixed gives exit slope x′ = −6.287×10⁻⁵, as the same error given as FSE does with refitting, but the refitted FSE_DIPOLE run gives |x′| < 10⁻¹⁵. XKICK = 8×10⁻⁵ should give x′ = 7.984×10⁻⁵ and also gives |x′| < 10⁻¹⁵. The vertical deflection of YKICK is kept.

??? example "Reproducer: TRK-38.py"

    [Download TRK-38.py](TRK-38.py)

    ```python
    """TRK-38: CSBEND FSE_CORRECTION cancels a dipole-only strength error and a horizontal steering kick

    When a matrix calculation fits CSBEND's FSE_CORRECTION=1 or 2 for edge model 2 or 4, the
    calibration retains FSE_DIPOLE and XKICK and absorbs their horizontal exit-slope effects. For a
    0.75 m, 0.09 rad bend, adding FSE_DIPOLE = 7×10⁻⁴ to the bend with its error-free correction
    held fixed gives exit slope x′ = −6.287×10⁻⁵, as the same error given as FSE does with
    refitting, but the refitted FSE_DIPOLE run gives |x′| < 10⁻¹⁵. XKICK = 8×10⁻⁵ should give x′ =
    7.984×10⁻⁵ and also gives |x′| < 10⁻¹⁵. The vertical deflection of YKICK is kept.

    The script fits the correction once for the error-free bend (edge models 2 and 4, twiss_output
    before tracking) and reads the fitted FSE from the log. With refitting off and that FSE held fixed,
    it adds FSE, FSE_DIPOLE, XKICK or YKICK and tracks an on-axis electron: these are the expected exit
    slopes. It then tracks the same errors with FSE_CORRECTION=1, which refits with the error present.
    FSE and YKICK are the controls; FSE_DIPOLE and XKICK show the defect.
    """

    import sys

    from _csbend_fse import track
    from _elegant import modes, verdict

    # The fitted FSE is printed with 6 significant digits (about 5e-12 absolute here), which moves the
    # fixed-correction exit slope by below 1e-12 rad; the defects are near 6e-5 rad.
    TOL = 1e-10
    ERRORS = (
        ("FSE", 7e-4, "xp"),
        ("FSE_DIPOLE", 7e-4, "xp"),
        ("XKICK", 8e-5, "xp"),
        ("YKICK", 8e-5, "yp"),
    )

    defect = []
    for mode in modes():
        for model in (2, 4):
            calibration, fitted = track(mode, model, {"FSE_CORRECTION": 1}, twiss=True)
            if fitted is None or abs(calibration["xp"]) > 1e-14:
                sys.exit(f"{mode} model {model}: the error-free correction was not fitted: invalid")
            print(
                f"{mode} EDGE_EFFECTS={model}: error-free fitted FSE {fitted:.6g}, exit xp {calibration['xp']:.3g}"
            )
            for name, value, column in ERRORS:
                # Expected: the error added to the bend whose ideal edge correction is held fixed, so the
                # correction compensates only the edges, as documented.
                fixed = {"FSE_CORRECTION": 0, "FSE": fitted + (value if name == "FSE" else 0.0)}
                if name != "FSE":
                    fixed[name] = value
                expected = track(mode, model, fixed, twiss=True)[0][column]
                observed, refit = track(mode, model, {"FSE_CORRECTION": 1, name: value}, twiss=True)
                if refit is None:
                    sys.exit(f"{mode} model {model} {name}: the correction was not refitted: invalid")
                diff = observed[column] - expected
                print(
                    f"{mode} EDGE_EFFECTS={model} {name}={value:g} exit {column}: observed={observed[column]:.10g}"
                    f" expected={expected:.10g} diff={diff:.4g}"
                )
                if name in ("FSE", "YKICK"):
                    if abs(diff) > TOL:
                        sys.exit(f"{mode} model {model} {name}: the control disagrees: invalid")
                else:
                    defect.append(abs(diff) > TOL)
    print(f"comparison: abs tolerance {TOL:g} rad")
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
| Confirmed (FSE_DIPOLE and XKICK; the claimed YKICK cancellation was refuted) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

