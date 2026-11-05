---
title: "IO-02: SDDS Gill–Miller integral of a three-sample table is zero and leaves a result unset"
hide:
  - toc
---

# IO-02 — SDDS Gill–Miller integral of a three-sample table is zero and leaves a result unset

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on SDDS master (`a3f730a`) |
| Source | [contract, GillMillerIntegration.c:34–40](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/GillMillerIntegration.c#L34-L40)<br>[gmintegral, SDDSaps/doc/sddsprocess.tex:320–322](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/SDDSaps/doc/sddsprocess.tex#L320-L322)<br>[setup, GillMillerIntegration.c:54–60](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/GillMillerIntegration.c#L54-L60)<br>[loop, GillMillerIntegration.c:64](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/GillMillerIntegration.c#L64)<br>[final slot, GillMillerIntegration.c:145–149](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/GillMillerIntegration.c#L145-L149)<br>[wrapper, SDDSaps/SDDSaps.c:4165–4171](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/SDDSaps/SDDSaps.c#L4165-L4171)<br>[caller, SDDSaps.c:1849–1851](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/SDDSaps/SDDSaps.c#L1849-L1851) |

When `sddsprocess` computes a `gmintegral` from three samples, or a program calls the SDDS routine `GillMillerIntegration` with three samples, the final integral silently becomes zero. For f = 1 at x = 0, 1, 2 the correct integral is 2; for f = x at x = 0.2, 0.7, 2.3 it is 2.625, yet the utility returns 0 for both without warning. Direct library calls also leave the middle cumulative result unwritten, while four-sample constant and nonuniform cubic controls agree with exact antiderivatives to double precision roundoff.

??? example "Reproducer: IO-02.py"

    [Download IO-02.py](IO-02.py)

    ```python
    """IO-02: SDDS Gill–Miller integral of a three-sample table is zero and leaves a result unset

    When `sddsprocess` computes a `gmintegral` from three samples, or a program calls the SDDS routine `GillMillerIntegration` with three samples, the final integral silently becomes zero. For f = 1 at x = 0, 1, 2 the correct integral is 2; for f = x at x = 0.2, 0.7, 2.3 it is 2.625, yet the utility returns 0 for both without warning. Direct library calls also leave the middle cumulative result unwritten, while four-sample constant and nonuniform cubic controls agree with exact antiderivatives to double precision roundoff.

    The script integrates constant and linear three-sample tables with `sddsprocess` and calls the library on uniform constant three- and four-sample tables prefilled with a sentinel. It compares with the exact antiderivative; the controls are the trapezoid rule of `sddsprocess` and four-sample constant and cubic tables. It needs `sddsprocess` on PATH and `gcc`.
    """

    import sys

    import numpy as np
    from _elegant import verdict
    from _sdds import c_check, process

    SENTINEL = -12345.0
    RTOL = 1e-14  # four-sample controls agree with the antiderivative to double roundoff
    C_SOURCE = r"""
    #include <stdio.h>
    extern int GillMillerIntegration(double *, double *, double *, double *, long);
    int main(void) {
      double x[] = {0, 1, 2, 3}, f[] = {1, 1, 1, 1};
      for (long n = 3; n <= 4; n++) {
        double integral[4] = {-12345, -12345, -12345, -12345};
        double error[4] = {-12345, -12345, -12345, -12345};
        int status = GillMillerIntegration(integral, error, f, x, n);
        printf("%ld %d", n, status);
        for (long j = 0; j < n; j++) printf(" %.17g", integral[j]);
        for (long j = 0; j < n; j++) printf(" %.17g", error[j]);
        printf("\n");
      }
      return 0;
    }
    """


    def fmt(values):
        return "[" + ", ".join(f"{v:.17g}" for v in np.ravel(values)) + "]"


    def compare(label, observed, expected):
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")


    defect = False

    # Library routine: f = 1 on x = 0, 1, 2, (3). The cumulative integral from x[0] is x - x[0].
    for line in c_check(C_SOURCE):
        n, status = int(line[0]), int(line[1])
        integral = np.array(line[2 : 2 + n], dtype=float)
        error = np.array(line[2 + n :], dtype=float)
        expected = np.arange(n, dtype=float)
        compare(f"API n={n} status={status} integral", integral, expected)
        print(f"API n={n} error estimates: {fmt(error)} (unwritten entries keep {SENTINEL:g})")
        if status != 0:
            sys.exit("the routine reported failure: the comparison is invalid")
        if n == 4:
            if not np.allclose(integral, expected, rtol=RTOL, atol=1e-15):
                sys.exit("the four-sample control failed: the comparison is invalid")
        else:
            # Success status, wrong final integral, and the middle entries never written.
            defect |= not np.isclose(integral[-1], expected[-1], rtol=RTOL, atol=0)
            defect |= bool(integral[1] == SENTINEL and error[1] == SENTINEL)

    # Utility: exact integral of x**power from the first to the last sample.
    for label, x, power in (
        ("constant-three", [0.0, 1.0, 2.0], 0),
        ("linear-three-nonuniform", [0.2, 0.7, 2.3], 1),
        ("constant-four", [0.0, 1.0, 2.0, 3.0], 0),
        ("cubic-four-nonuniform", [0.2, 0.7, 1.4, 2.3], 3),
    ):
        x = np.array(x)
        out = process(
            {"x": x, "f": x**power},
            "-process=f,gmintegral,GM,functionOf=x",
            "-process=f,integral,Trap,functionOf=x",
        )
        expected = (x[-1] ** (power + 1) - x[0] ** (power + 1)) / (power + 1)
        compare(f"sddsprocess {label} GM", out["GM"], expected)
        compare(f"sddsprocess {label} trapezoid", out["Trap"], expected)
        if len(x) == 3:
            defect |= not np.isclose(out["GM"][0], expected, rtol=RTOL, atol=0)
        elif not np.isclose(out["GM"][0], expected, rtol=RTOL, atol=0):
            sys.exit("the four-sample control failed: the comparison is invalid")
        if power == 0 or len(x) == 3:
            # The trapezoid rule is exact for these tables, so it validates the expected value.
            if not np.isclose(out["Trap"][0], expected, rtol=RTOL, atol=0):
                sys.exit("the trapezoid control failed: the expected value is invalid")
    print(f"comparison: rtol={RTOL:g}")
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

??? example "Helper: _sdds.py"

    [Download _sdds.py](_sdds.py)

    ```python
    """Shared helpers for the reproducers of SDDS library defects: run the SDDS utility `sddsprocess`
    on inline tables and compile small C programs against the SDDS libraries.

    `sddsprocess` must be on PATH, as for the Elegant reproducers; the SDDS build tree (its `lib`
    directory with `libmdbmth.a` and `libmdblib.a`) is found from the location of that executable.
    Compiling needs `gcc`.
    """

    from __future__ import annotations

    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np
    import pysdds

    TIMEOUT = 120  # seconds per native run


    def sdds_lib() -> Path:
        """The `lib/Linux-x86_64` directory of the SDDS build that provides `sddsprocess`."""
        tool = shutil.which("sddsprocess")
        if tool is None:
            sys.exit("sddsprocess must be on PATH (see the elegant installation).")
        lib = Path(tool).resolve().parents[2] / "lib" / "Linux-x86_64"
        if not (lib / "libmdbmth.a").exists():
            sys.exit(f"{lib} has no libmdbmth.a: sddsprocess is not in an SDDS build tree.")
        return lib


    def execute(argv: list[str], cwd: Path) -> str:
        """Run a native command; a nonzero exit status or a timeout raises RuntimeError."""
        done = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT, check=False
        )
        if done.returncode:
            raise RuntimeError(
                f"{argv[0]} exited with status {done.returncode}:\n{done.stderr[-2000:]}"
            )
        return done.stdout


    def c_check(source: str) -> list[list[str]]:
        """Compile a C program against the SDDS math libraries, run it, and return the stdout words.

        The program prints one result per line; each line comes back as a list of words.
        """
        lib = sdds_lib()
        with tempfile.TemporaryDirectory(prefix="sdds-repro-") as tmp:
            work = Path(tmp)
            (work / "check.c").write_text(source)
            execute(
                [
                    "gcc",
                    "-O2",
                    "check.c",
                    str(lib / "libmdbmth.a"),
                    str(lib / "libmdblib.a"),
                    "-fopenmp",
                    "-lm",
                    "-o",
                    "check",
                ],
                work,
            )
            return [line.split() for line in execute([str(work / "check")], work).splitlines()]


    def process(columns: dict[str, np.ndarray], *options: str) -> dict[str, np.ndarray]:
        """Run `sddsprocess` on a one-page table of double columns.

        Returns every parameter and column of the output page as a NumPy array (a parameter has
        one element). The input is written as binary SDDS, which carries the doubles exactly.
        """
        table = pysdds.SDDSFile(add_data_nm=True)
        table.mode = "binary"
        table.n_pages = 1
        for name, values in columns.items():
            table.add_column(name, "double", data=[np.asarray(values, dtype=np.float64)])
        with tempfile.TemporaryDirectory(prefix="sdds-repro-") as tmp:
            work = Path(tmp)
            pysdds.write(table, work / "input.sdds", overwrite=True)
            execute(["sddsprocess", "input.sdds", "output.sdds", *options], work)
            out = pysdds.read(work / "output.sdds")
            if out.n_pages != 1:
                raise RuntimeError(f"sddsprocess wrote {out.n_pages} pages")
            result = {p.name: np.atleast_1d(np.asarray(out[p.name].data[0])) for p in out.parameters}
            result.update({c.name: np.asarray(out[0, c.name]) for c in out.columns})
            return result

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | SDDS master (`a60b9bbed7cc`), source with Elegant 75a2e1a1a58b, CPU, serial utility and native library API |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | SDDS master (`a60b9bbed7cc`), source with Elegant 75a2e1a1a58b, CPU, serial utility and native library API |

