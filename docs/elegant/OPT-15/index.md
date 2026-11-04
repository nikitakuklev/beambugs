---
title: "OPT-15: Individual matrix output keeps nonlinear coefficients of preceding elements"
hide:
  - toc
---

# OPT-15 — Individual matrix output keeps nonlinear coefficients of preceding elements

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`null_matrices`, matrix.c:183–200](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix.c#L183-L200)<br>[`initialize_matrices`, matrix.c:130–177](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix.c#L130-L177)<br>[`run_matrix_output`, matrix_output.c:422–429](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix_output.c#L422-L429)<br>[`SDDS_set_matrices`, matrix_output.c:733–752](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix_output.c#L733-L752)<br>[`individual_matrices` writes the individual on-trajectory matrices of the elements, elegant.tex:5362–5363](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5362-L5363) |

Elegant's `matrix_output` with `individual_matrices=1` should write the map of each element on its own trajectory, but the second- and third-order coefficients of an element carry leftover values from earlier elements. A 1 m drift after a 0.4 m quadrupole with K1 = 0.7 m⁻² is written with T161 = 0.3243, T262 = 0.05496, U1661 = -0.3204 and U2662 = -0.05457, while a drift does not depend on the relative momentum offset δ and all four are 0. Putting the drift first or switching the quadrupole off gives zeros. Users who read nonlinear coefficients of individual elements from this output are affected.

??? example "Reproducer: OPT-15.py"

    [Download OPT-15.py](OPT-15.py)

    ```python
    """OPT-15: Individual matrix output keeps nonlinear coefficients of preceding elements

    Elegant's `matrix_output` with `individual_matrices=1` should write the map of each element on its
    own trajectory, but the second- and third-order coefficients of an element carry leftover values
    from earlier elements. A 1 m drift after a 0.4 m quadrupole with K1 = 0.7 m⁻² is written with T161 =
    0.3243, T262 = 0.05496, U1661 = -0.3204 and U2662 = -0.05457, while a drift does not depend on the
    relative momentum offset δ and all four are 0. Putting the drift first or switching the quadrupole
    off gives zeros. Users who read nonlinear coefficients of individual elements from this output are
    affected.

    The script writes the individual-element matrices of a quadrupole followed by a drift to an SDDS
    file, in serial Elegant (and with `--mpi` in 3-rank Pelegant), and compares the drift's
    coefficients with the exact drift map. The controls are the drift placed before the quadrupole and
    a quadrupole of zero strength; both give exact zeros. Second-order output, without the third-order
    coefficients, shows the same error.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    DRIFT = 1.0  # m
    QUAD = 0.4  # m
    ATOL = 1e-15  # the exact drift coefficients are zero; the control values are exactly zero


    def commands(order):
        return (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central=200, default_order={order}"
            " &end\n"
            "&run_control n_steps=1 &end\n"
            f"&matrix_output SDDS_output=mat.sdds, SDDS_output_order={order}, individual_matrices=1"
            " &end\n"
            "&bunched_beam n_particles_per_bunch=12 &end\n"
            "&track &end\n"
        )


    def drift_row(mode, k1, quad_first, order):
        """R11, R12, T161, T262 (and U1661, U2662 at third order) written for the drift."""
        names = ["R11", "R12", "T161", "T262"] + (["U1661", "U2662"] if order == 3 else [])
        sequence = "Q,D" if quad_first else "D,Q"
        lattice = (
            f"Q: QUAD,L={QUAD!r},K1={k1!r},ORDER={order}\n"
            f"D: DRIF,L={DRIFT!r},ORDER={order}\n"
            f"BL: LINE=({sequence})\n"
        )
        result = run({"lattice.lte": lattice, "run.ele": commands(order)}, mode, outputs=("mat.sdds",))
        matrices = result["mat.sdds"]
        row = list(matrices[0, "ElementName"]).index("D")
        return np.array([float(matrices[0, name][row]) for name in names])


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected disagree."""
        bad = not np.allclose(observed, expected, rtol=0, atol=ATOL)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    defect = False
    print("columns: drift R11, R12, T161, T262, U1661, U2662 (second order: first four)")
    for mode in modes():
        for order in (3, 2):
            # A drift is x1 = x + L x', x'1 = x' for every momentum offset, so R11 = R12 = 1 (L = 1 m).
            # The transverse drift relations have no nonlinear coefficients, so T161, T262, U1661 and
            # U2662 are zero; T161 and T262 would be the momentum derivatives of R11 and R22 (the
            # coefficients of x delta and x' delta).
            expected = np.array([1, DRIFT, 0, 0, 0, 0][: 6 if order == 3 else 4])
            for label, k1, quad_first in (
                ("control, drift before quadrupole", 0.7, False),
                ("control, K1=0", 0.0, True),
            ):
                if compare(
                    f"{mode} order={order} {label}", drift_row(mode, k1, quad_first, order), expected
                ):
                    sys.exit("A control has nonzero coefficients: the comparison is not valid.")
            observed = drift_row(mode, 0.7, True, order)
            # The target row must be the drift (finite, R11 = 1, R12 = L) before the verdict, which
            # then rests on the nonlinear coefficients alone.
            if not np.isfinite(observed).all() or np.any(np.abs(observed[:2] - expected[:2]) > ATOL):
                sys.exit("The drift row is not finite with R11 = 1 and R12 = L: invalid comparison.")
            compare(f"{mode} order={order} K1=0.7, drift after quadrupole", observed, expected)
            defect |= bool(np.any(np.abs(observed[2:] - expected[2:]) > ATOL))
    print(f"comparison: absolute tolerance {ATOL:g}")
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
| Found | `gpt-6.1-sol` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

