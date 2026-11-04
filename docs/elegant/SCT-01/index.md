---
title: "SCT-01: Pressure-region diagnostic reads an invalid sample index"
hide:
  - toc
---

# SCT-01 — Pressure-region diagnostic reads an invalid sample index

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`computeAverageGasPressures` coverage check, pressureData.c:207–218](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/physics/pressureData.c#L207-L218)<br>[interval limits, ionEffects.cc:593–616](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ionEffects.cc#L593-L616)<br>[averaging call, ionEffects.cc:633–634](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ionEffects.cc#L633-L634)<br>[default, ionEffects.nl:53](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ionEffects.nl#L53)<br>[call at the start of tracking, elegant.c:1300–1301](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L1300-L1301)<br>[spacing checks, pressureData.c:179–191](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/physics/pressureData.c#L179-L191) |

With IONEFFECTS' default pressure averaging, a lattice interval lying entirely before or after the gas-pressure samples triggers an error diagnostic that reads position index −1. Evaluating the message reads 8 bytes before the position array instead of safely reporting the coverage problem. With samples at 2 m and 3 m and an interval from 0 to 1 m, the run stops with a value that is not a sample position in the message, and a memory checker confirms an invalid read of size 8. Users whose pressure profiles miss an element's averaging interval are affected.

??? example "Reproducer: SCT-01.py"

    [Download SCT-01.py](SCT-01.py)

    ```python
    """SCT-01: Pressure-region diagnostic reads an invalid sample index

    With IONEFFECTS' default pressure averaging, a lattice interval lying entirely before or after the
    gas-pressure samples triggers an error diagnostic that reads position index −1. Evaluating the
    message reads 8 bytes before the position array instead of safely reporting the coverage problem.
    With samples at 2 m and 3 m and an interval from 0 to 1 m, the run stops with a value that is not a
    sample position in the message, and a memory checker confirms an invalid read of size 8. Users whose
    pressure profiles miss an element's averaging interval are affected.

    The script writes a one-element IONEFFECTS lattice spanning 0 to 1 m and runs serial Elegant under
    Valgrind with four pressure profiles: two that reach the interval (samples at 0, 0.5 and 1 m, and
    only at 0.2 and 0.8 m), which are controls that must complete without an invalid read, one sampled
    only beyond the interval at 2 m and 3 m, and one sampled only before it at −3 m to −1 m. The verdict
    is an invalid read in computeAverageGasPressures during a run that ends with the pressure-region
    coverage error.
    """

    import os
    import re
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from _elegant import verdict

    LATTICE = "Q: CHARGE,TOTAL=1e-9\nD: DRIF,L=1\nI: IONEFFECTS\nL: LINE=(Q,D,I)\n"
    # The single IONEFFECTS element spans s in [0, 1] m, the whole lattice.
    COMMANDS = """&run_setup lattice=lattice.lte, use_beamline=L, p_central_mev=100, random_number_seed=31317 &end
    &run_control n_steps=1, n_passes=1 &end
    &ion_effects pressure_profile=pressure.sdds, ion_properties=ions.sdds, macro_ions=10,
     ion_span[0]=0.01, ion_span[1]=0.01, field_calculation_method=gaussian, verbosity=11 &end
    &bunched_beam n_particles_per_bunch=10, emit_x=1e-9, emit_y=1e-9, beta_x=1, beta_y=1,
     sigma_s=1e-3, sigma_dp=1e-4 &end
    &track &end
    """
    IONS = """SDDS1
    &column name=IonName, type=string &end
    &column name=Mass, units=AMU, type=double &end
    &column name=ChargeState, type=long &end
    &column name=SourceName, type=string &end
    &column name=CrossSection, units=Mb, type=double &end
    &data mode=ascii, no_row_counts=1 &end
    H2+ 2.016 1 H2 1.0
    """
    PRESSURE = """SDDS1
    &parameter name=Temperature, units=K, type=double &end
    &column name=s, units=m, type=double &end
    &column name=H2, units=Torr, type=double &end
    &data mode=ascii, no_row_counts=1 &end
    293
    {rows}"""
    # Pressure samples that must be uniformly spaced. The first two profiles reach the interval and
    # are controls: the second has samples inside the interval without spanning it, which passes.
    CASES = (
        ("profile 0 to 1 m covers the element (control)", (0.0, 0.5, 1.0), True),
        ("profile 0.2 to 0.8 m lies inside the element (control)", (0.2, 0.8), True),
        ("profile 2 to 3 m lies beyond the element", (2.0, 3.0), False),
        ("profile -3 to -1 m lies before the element", (-3.0, -2.0, -1.0), False),
    )
    # computeAverageGasPressures prints "(iStart=>s[iStart], iEnd=>s[iEnd])" when either index is -1;
    # the failure it reports is recognized by its opening words, whatever follows.
    COVERAGE_ERROR = "Failed to find indices corresponding to pressure region"
    # One Valgrind error block: a header line, then the call stack ("at"/"by" lines).
    INVALID_READ = re.compile(r"Invalid read of size (\d+)\n((?:==\d+==\s+(?:at|by) .*\n)+)")
    TIMEOUT = 600  # seconds per Valgrind run


    def inputs(samples):
        rows = "".join(f"{s!r} 1e-9\n" for s in samples)
        return {
            "lattice.lte": LATTICE,
            "run.ele": COMMANDS,
            "ions.sdds": IONS,
            "pressure.sdds": PRESSURE.format(rows=rows),
        }


    def checked_run(samples):
        """Serial Elegant under Valgrind: (exit status, log, invalid reads inside computeAverageGasPressures)."""
        if shutil.which("valgrind") is None or shutil.which("elegant") is None:
            sys.exit("valgrind and elegant must be on PATH.")
        if "RPN_DEFNS" not in os.environ:
            sys.exit("Set RPN_DEFNS to the defns.rpn file of SDDS (see the elegant installation).")
        env = dict(os.environ, OMP_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="elegant-repro-") as tmp:
            for name, text in inputs(samples).items():
                (Path(tmp) / name).write_text(text)
            try:
                done = subprocess.run(
                    ["valgrind", "-q", "--error-limit=no", "elegant", "run.ele"],
                    cwd=tmp,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                sys.exit(f"valgrind elegant exceeded {TIMEOUT} s")
        log = done.stdout + done.stderr
        reads = [
            (size, stack.splitlines()[0].split(": ", 1)[1])
            for size, stack in INVALID_READ.findall(log)
            if "computeAverageGasPressures" in stack
        ]
        return done.returncode, log, reads


    defect = False
    print(
        "invalid reads in computeAverageGasPressures for the pressure region [0, 1] m, serial elegant"
    )
    for label, samples, control in CASES:
        status, log, reads = checked_run(samples)
        covered = COVERAGE_ERROR in log
        if control:
            if status or covered or reads:
                sys.exit(
                    f"serial {label}: the run fails or reads outside the array: not a valid control"
                )
            print(f"serial {label}: observed=run completes, no invalid read expected=same")
            continue
        # An uncovered profile may only fail with the intended coverage error.
        if status != 0 and not covered:
            sys.exit(f"serial {label}: exit status {status} without the coverage error:\n{log[-2000:]}")
        shown = f"invalid read of size {reads[0][0]} at {reads[0][1]}" if reads else "no invalid read"
        print(f"serial {label}: observed={shown} expected=a coverage error with no invalid read")
        defect |= bool(reads)
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
| Confirmed (source reading) | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source reading, not executed for this claim |
| Confirmed (source reading) | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source reading, not executed for this claim |

