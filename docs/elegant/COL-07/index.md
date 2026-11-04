---
title: "COL-07: WAKE cache substitutes the first requested column pair"
hide:
  - toc
---

# COL-07 — WAKE cache substitutes the first requested column pair

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [record, wake.c:366–370](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L366-L370)<br>[cache lookup, wake.c:423–425](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L423-L425)<br>[cache reuse, wake.c:473–475](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L473-L475)<br>[column retrieval, wake.c:442–459](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L442-L459)<br>[parameter table, track_data.c:2530–2531](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2530-L2531)<br>[grid spacing, wake.c:487](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L487)<br>[wake manual, wake.tex:1–12](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/wake.tex#L1-L12) |

When two WAKE elements in one Elegant run select different wake columns (WCOLUMN) or time columns (TCOLUMN) from the same table file, the second element silently reuses the first element's columns, because the table is cached by file name alone. For a table with kernels A and B = 3A, the earliest particles of a 12-electron, 2 nC bunch should lose the sum of both kernels, 8000 eV; the shared file gives 4000 eV (A applied twice) or 12000 eV (B applied twice), depending on element order, which is an error of -50% or +50%. A byte-identical copy of the table under another file name gives the correct result. Anyone who selects several kernels from one wake file is affected.

??? example "Reproducer: COL-07.py"

    [Download COL-07.py](COL-07.py)

    ```python
    """COL-07: WAKE cache substitutes the first requested column pair

    When two WAKE elements in one Elegant run select different wake columns (WCOLUMN) or time columns (TCOLUMN) from the same table file, the second element silently reuses the first element's columns, because the table is cached by file name alone. For a table with kernels A and B = 3A, the earliest particles of a 12-electron, 2 nC bunch should lose the sum of both kernels, 8000 eV; the shared file gives 4000 eV (A applied twice) or 12000 eV (B applied twice), depending on element order, which is an error of -50% or +50%. A byte-identical copy of the table under another file name gives the correct result. Anyone who selects several kernels from one wake file is affected.

    The script tracks twelve electrons (four arrival times, three particles each) through two
    zero-length WAKE elements that read a table holding two wake columns, B = 3A, and two time
    columns, and measures each group's energy loss. It compares the loss with the causal particle-pair
    sum of the two requested kernels. The control gives the second element a byte-identical copy of
    the table under another file name, which restores the requested kernel in every case.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    ELECTRON_EV = 0.51099906e6  # SDDS constants.h me_mev, in eV
    CHARGE, COUNT, P = 2e-9, 12, 10.0
    GROUP_TIMES = np.array([0, 8, 24, 48]) * 1e-12  # s, three particles per group
    TIMES = np.repeat(GROUP_TIMES, 3)
    ROWS = 100
    K = np.arange(ROWS)
    TABLE = {
        "t": K * 1e-12,  # s
        "t2": 2 * K * 1e-12,
        "A": 4e12 * np.exp(-K / 17),  # V/C
        "B": 3 * 4e12 * np.exp(-K / 17),
    }
    UNITS = {"t": "s", "t2": "s", "A": "V/C", "B": "V/C"}
    ATOL_CONTROL = 1e-7  # eV; floor from subtracting gamma values near 10.05
    MIN_ERROR = 100.0  # eV; the defect shifts the losses by thousands of eV


    def sdds_table():
        """The wake table as an ASCII SDDS page."""
        head = "SDDS1\n" + "".join(
            f"&column name={name}, type=double, units={UNITS[name]} &end\n" for name in TABLE
        )
        rows = "".join(
            " ".join(repr(float(TABLE[name][i])) for name in TABLE) + "\n" for i in range(ROWS)
        )
        return head + f"&data mode=ascii &end\n! page number 1\n{ROWS}\n" + rows


    def expected_loss(first, second):
        """Energy lost by each group, eV, if the elements apply their requested kernels.

        The wake energy loss of a particle is q sum_j W(t_i - t_j) over the particles at or ahead of
        it (t_j <= t_i), with q the macroparticle charge in C and W in V/C, giving eV. The kernel of
        an element is the table column W(tcolumn): zero-length elements leave arrival times
        unchanged, so the losses of the two elements add. Delays fall on table nodes.
        """
        q = CHARGE / COUNT
        loss = np.zeros(len(GROUP_TIMES))
        for tcol, wcol in (first, second):
            for i, ti in enumerate(GROUP_TIMES):
                for tj in GROUP_TIMES[GROUP_TIMES <= ti]:
                    loss[i] += 3 * q * np.interp(ti - tj, TABLE[tcol], TABLE[wcol])
        return loss


    def tracked_loss(mode, first, second, separate):
        """Energy lost by each group in eV, from the momentum change of the tracked beam."""
        names = ("wake.sdds", "wake2.sdds" if separate else "wake.sdds")
        lattice = f"Q: CHARGE,TOTAL={CHARGE!r}\n"
        for n, ((tcol, wcol), file) in enumerate(zip((first, second), names)):
            lattice += (
                f'W{n}: WAKE,INPUTFILE="{file}",TCOLUMN="{tcol}",WCOLUMN="{wcol}",'
                "N_BINS=0,SMOOTHING=0,INTERPOLATE=0\n"
            )
        lattice += "BL: LINE=(Q,W0,W1)\n"
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        files = {"lattice.lte": lattice, "run.ele": commands, "wake.sdds": sdds_table()}
        if separate:
            files["wake2.sdds"] = sdds_table()  # same bytes, other file name
        beam = {"t": TIMES, "p": np.full(COUNT, P), "particleID": np.arange(1, COUNT + 1)}
        out = particles(run(files, mode, {"beam.sdds": beam})["out.sdds"])
        if not np.array_equal(out["particleID"], np.arange(1, COUNT + 1)):
            sys.exit(f"{mode}: particles lost")
        if not np.allclose(out["t"], TIMES, rtol=0, atol=1e-21):
            sys.exit(f"{mode}: arrival times changed")
        gain = np.sqrt(1 + out["p"] ** 2) - np.sqrt(1 + P**2)
        return -ELECTRON_EV * gain.reshape(-1, 3).mean(axis=1)


    def compare(mode, label, observed, expected):
        """Print one case and return its largest deviation from the pair sum, eV.

        Nonfinite or mismatched arrays are a failed run: NaN would otherwise slip past the thresholds.
        """
        if observed.shape != expected.shape or not (
            np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))
        ):
            sys.exit(f"{mode} {label}: invalid observed or expected losses")
        error = np.max(np.abs(observed - expected))
        print(
            f"{mode} {label}: observed={np.array2string(observed, separator=', ', precision=10)} "
            f"expected={np.array2string(expected, separator=', ', precision=10)} "
            f"max deviation={error:.10g} eV"
        )
        return error


    CASES = (
        ("W columns A then B", ("t", "A"), ("t", "B")),
        ("W columns B then A", ("t", "B"), ("t", "A")),
        ("T columns t then t2, W column A", ("t", "A"), ("t2", "A")),
    )
    defect = False
    print("energy loss in eV of the four groups (arrival times 0, 8, 24, 48 ps)")
    for mode in modes():
        for label, first, second in CASES:
            expected = expected_loss(first, second)
            control = compare(
                mode,
                f"{label}, separate files",
                tracked_loss(mode, first, second, separate=True),
                expected,
            )
            if control > ATOL_CONTROL:
                sys.exit(f"{mode}: the separate-file control is off by {control:g} eV")
            error = compare(
                mode,
                f"{label}, shared file",
                tracked_loss(mode, first, second, separate=False),
                expected,
            )
            defect |= error > MIN_ERROR
    print(
        f"control tolerance {ATOL_CONTROL:g} eV; defect if a shared-file loss is off by > {MIN_ERROR:g} eV"
    )
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
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

