---
title: "DOC-01: Manual gives the third-order momentum compaction term as δ² instead of δ³"
hide:
  - toc
---

# DOC-01 — Manual gives the third-order momentum compaction term as δ² instead of δ³

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [manual text for `alphac`, `alphac2` and `alphac3`, doc/elegant.tex:8254–8257](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8254-L8257)<br>[orbit series definition, twiss.cc:253–260](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L253-L260)<br>[`alpha3` accumulation, twiss.cc:2398–2409](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L2398-L2409) |

The Elegant manual gives the path length as s = s₀ + αc L δ + αc2 L δ² + αc3 L δ², with δ the relative momentum offset, for the Twiss outputs `alphac`, `alphac2` and `alphac3`. The last power should be 3: `alphac3` is the coefficient of δ³. For a prescribed one-turn map of length 10 m with path 10 m + 2 m·δ² + 7 m·δ³, Elegant writes αc = 0, αc2 = 0.2000 and αc3 = 0.7000, and tracking gives 10.01 m at δ = −0.1 and 10.03 m at δ = +0.1; the printed equation predicts 10.09 m at both.

??? example "Reproducer: DOC-01.py"

    [Download DOC-01.py](DOC-01.py)

    ```python
    """DOC-01: Manual gives the third-order momentum compaction term as δ² instead of δ³

    The Elegant manual gives the path length as s = s₀ + αc L δ + αc2 L δ² + αc3 L δ², with δ the
    relative momentum offset, for the Twiss outputs `alphac`, `alphac2` and `alphac3`. The last power
    should be 3: `alphac3` is the coefficient of δ³. For a prescribed one-turn map of length 10 m with
    path 10 m + 2 m·δ² + 7 m·δ³, Elegant writes αc = 0, αc2 = 0.2000 and αc3 = 0.7000, and tracking
    gives 10.01 m at δ = −0.1 and 10.03 m at δ = +0.1; the printed equation predicts 10.09 m at both.

    The script reads the compaction equation from `doc/elegant.tex` of the Elegant source checkout named
    by `ELEGANT_SOURCE` and exits if it is missing or unrecognized. It prescribes the one-turn map with
    zero transverse orbit and dispersion, reads `alphac`, `alphac2` and `alphac3` from `twiss_output`,
    and tracks particles at six momentum offsets to obtain the path length from their arrival times. It
    compares the tracked path with the manual's equation and with the equation using δ³. A map without
    the cubic term, for which the two equations coincide, is the control.
    """

    import os
    import re
    import sys
    from pathlib import Path

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from _matr import matrix_text, rotations

    C = 299792458.0  # m/s, c_mks of SDDS
    P0 = 1000.0  # reference momentum, units of m_e c
    LENGTH = 10.0  # m
    T566 = 2.0  # m: path = LENGTH + T566 delta^2 + Q5666 delta^3
    DELTAS = np.tile([-0.1, -0.05, -0.02, 0.02, 0.05, 0.1], 2)  # twice, so each MPI worker tracks some
    ATOL = 2e-13  # m, rounding of the tracked path and of the written coefficients

    # The manual's path-length equation, s = s_o + alphac L delta + alphac2 L delta^2 + alphac3 L delta^N;
    # the exponents of the alphac2 and alphac3 terms are captured.
    MANUAL_EQUATION = re.compile(
        r"s\s*=\s*s_o\s*\+\s*\\alpha_c\s+L\s*\\delta\s*"
        r"\+\s*\\alpha_\{c2\}\s+L\s*\\delta\^\{?(\d+)\}?\s*"
        r"\+\s*\\alpha_\{c3\}\s+L\s*\\delta\^\{?(\d+)\}?"
    )

    COMMANDS = """\
    &run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3,
     output=out.sdds &end
    &run_control n_steps=1, n_passes=1 &end
    &twiss_output filename=twiss.sdds, concat_order=3 &end
    &sdds_beam input=beam.sdds &end
    &track soft_failure=0 &end
    """


    def manual_exponent():
        """Power of delta that the manual's path-length equation gives the alphac3 term."""
        source = os.environ.get("ELEGANT_SOURCE")
        if not source:
            sys.exit("ELEGANT_SOURCE must name the Elegant source checkout.")
        manual = Path(source) / "doc" / "elegant.tex"
        if not manual.is_file():
            sys.exit(f"{manual} does not exist.")
        found = MANUAL_EQUATION.findall(manual.read_text())
        if len(found) != 1 or found[0][0] != "2" or found[0][1] not in ("2", "3"):
            sys.exit(f"The manual's compaction equation is missing or unrecognized: {found}")
        return int(found[0][1])


    def case(mode, a3):
        """Written (alphac, alphac2, alphac3) and the tracked path of each particle."""
        # Pure longitudinal map: s_out = s_in + 10 + 2 delta^2 + 10 a3 delta^3, transverse rotations.
        t = np.zeros((6, 6, 6))
        t[4, 5, 5] = T566
        q = np.zeros((6, 6, 6, 6))
        q[4, 5, 5, 5] = LENGTH * a3
        constants = np.array([0, 0, 0, 0, LENGTH, 0.0])
        beam = {
            "p": P0 * (1 + DELTAS),
            "particleID": np.arange(1, len(DELTAS) + 1),
        }
        result = run(
            {
                "lattice.lte": "E: MATR,L=10,FILENAME=MAP.TXT,ORDER=3\nBL: LINE=(E)\n",
                "run.ele": COMMANDS,
                "MAP.TXT": matrix_text(constants, rotations(), t, q),
            },
            mode,
            {"beam.sdds": beam},
            outputs=("twiss.sdds", "out.sdds"),
        )
        twiss = result["twiss.sdds"]
        out = particles(result["out.sdds"])
        # Path length from the arrival time and the output velocity: s = beta c t.
        path = out["p"] / np.sqrt(1 + out["p"] ** 2) * C * out["t"]
        return np.array([float(twiss[name][0]) for name in ("alphac", "alphac2", "alphac3")]), path


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected disagree."""
        bad = not np.allclose(observed, expected, rtol=0, atol=ATOL)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    manual_power = manual_exponent()
    print(f"manual: the alphac3 term of the path length is alphac3 L delta^{manual_power}")
    defect = False
    print("columns: path length at delta = -0.1, +0.1 (m); expected from the printed equation")
    for mode in modes():
        for a3 in (0.0, 0.7):
            alphac, path = case(mode, a3)
            # The map is exactly s = L + T566 delta^2 + L a3 delta^3, so the coefficients of
            # (s - L)/L in delta, delta^2 and delta^3 are 0, T566/L and a3.
            if compare(f"{mode} a3={a3:g} alphac, alphac2, alphac3", alphac, [0, T566 / LENGTH, a3]):
                sys.exit("The written coefficients differ from the prescribed map: invalid comparison.")
            exact = LENGTH + T566 * DELTAS**2 + LENGTH * a3 * DELTAS**3
            if compare(f"{mode} a3={a3:g} tracked path, max error", np.max(np.abs(path - exact)), 0.0):
                sys.exit("The tracked path differs from the prescribed map: invalid comparison.")
            # The manual's equation, s = L (1 + alphac delta + alphac2 delta^2 + alphac3 delta^N)
            # with the power N read from the manual, and the same with the cubic power.
            printed = LENGTH * (
                1 + alphac[0] * DELTAS + alphac[1] * DELTAS**2 + alphac[2] * DELTAS**manual_power
            )
            cubic = LENGTH * (1 + alphac[0] * DELTAS + alphac[1] * DELTAS**2 + alphac[2] * DELTAS**3)
            ends = [0, 5]  # delta = -0.1 and +0.1
            if compare(
                f"{mode} a3={a3:g} delta^3 equation, max deviation", np.max(abs(path - cubic)), 0
            ):
                sys.exit("The cubic equation disagrees with tracking: invalid comparison.")
            bad = compare(f"{mode} a3={a3:g} path at delta=-0.1,+0.1", path[ends], printed[ends])
            if a3 == 0.0 and bad:
                sys.exit("The printed equation fails without a cubic term: invalid comparison.")
            defect |= bad
            print(
                f"{mode} a3={a3:g} largest deviation of the manual equation: {max(abs(path - printed)):.4g} m"
            )
    print(f"comparison: absolute tolerance {ATOL:g} m")
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

??? example "Helper: _matr.py"

    [Download _matr.py](_matr.py)

    ```python
    """Prescribed-map helpers for Elegant's MATR element, which reads a map from a text file.

    `matrix_text` writes the file in the layout MATR expects: the constant vector C, the rows of the
    first-order matrix R, the second-order coefficients T and the third-order coefficients Q. Elegant
    stores the symmetric coefficients once, with the indices ordered j >= k (>= l), and each entry
    holds the total coefficient of the monomial x_j x_k (x_l) in coordinate i; coordinates are
    (x, x', y, y', s, delta), indexed from zero here.
    """

    from __future__ import annotations

    import numpy as np

    TUNES = (0.17, 0.23)  # horizontal and vertical betatron tunes of the rotation helper


    def rotations(tunes: tuple[float, float] = TUNES) -> np.ndarray:
        """6x6 matrix: rotation by 2 pi nu in (x, x') and (y, y') with beta = 1 m and alpha = 0."""
        matrix = np.eye(6)
        for j, tune in zip((0, 2), tunes, strict=True):
            c, s = np.cos(2 * np.pi * tune), np.sin(2 * np.pi * tune)
            matrix[j : j + 2, j : j + 2] = [[c, s], [-s, c]]
        return matrix


    def matrix_text(
        c: np.ndarray, r: np.ndarray, t: np.ndarray | None = None, q: np.ndarray | None = None
    ) -> str:
        """Contents of a MATR map file; t has shape (6, 6, 6) and q (6, 6, 6, 6), zero when omitted."""
        t = np.zeros((6, 6, 6)) if t is None else t
        q = np.zeros((6, 6, 6, 6)) if q is None else q
        lines = ["C: " + " ".join(f"{v:.17g}" for v in c)]
        lines += [f"R{i + 1}: " + " ".join(f"{v:.17g}" for v in r[i]) for i in range(6)]
        for i in range(6):
            for j in range(6):
                values = " ".join(f"{t[i, j, k]:.17g}" for k in range(j + 1))
                lines.append(f"T{i + 1}{j + 1}: {values}")
        for i in range(6):
            for j in range(6):
                for k in range(j + 1):
                    values = " ".join(f"{q[i, j, k, l]:.17g}" for l in range(k + 1))
                    lines.append(f"Q{i + 1}{j + 1}{k + 1}: {values}")
        return "\n".join(lines) + "\n"

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Found | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

