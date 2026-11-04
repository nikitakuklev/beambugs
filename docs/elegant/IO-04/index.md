---
title: "IO-04: SDDS moment accumulation discards the earlier batches"
hide:
  - toc
---

# IO-04 — SDDS moment accumulation discards the earlier batches

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on SDDS master (`a3f730a`) |
| Source | [`sum = partial_sum`, moments.c:365–367](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/moments.c#L365-L367)<br>[weighted branch, moments.c:426–430](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/moments.c#L426-L430)<br>[wrappers, moments.c:324–327 and 382–385](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/moments.c#L324-L327)<br>[prototypes, include/mdb.h:886–893](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/include/mdb.h#L886-L893) |

The SDDS routines `accumulateMoments` and `accumulateWeightedMoments` return the mean, rms and standard deviation of data supplied in batches (first call with reset = 1, later calls with reset = 0), but in their default single-thread form they do not combine the batches. For the values 2, 4, 8, 10 supplied as [2, 4] then [8, 10], the mean is 6, the rms √46 = 6.782 and the sample standard deviation √(40/3) = 3.651, as one call over all four values gives. The unweighted routine returns a mean of 4.5 (the last batch's sum over the total count) and the unit-weight routine returns 9 (the last batch's mean). Programs that feed these routines data in batches are affected.

??? example "Reproducer: IO-04.py"

    [Download IO-04.py](IO-04.py)

    ```python
    """IO-04: SDDS moment accumulation discards the earlier batches

    The SDDS routines `accumulateMoments` and `accumulateWeightedMoments` return the mean, rms and standard deviation of data supplied in batches (first call with reset = 1, later calls with reset = 0), but in their default single-thread form they do not combine the batches. For the values 2, 4, 8, 10 supplied as [2, 4] then [8, 10], the mean is 6, the rms √46 = 6.782 and the sample standard deviation √(40/3) = 3.651, as one call over all four values gives. The unweighted routine returns a mean of 4.5 (the last batch's sum over the total count) and the unit-weight routine returns 9 (the last batch's mean). Programs that feed these routines data in batches are affected.

    The script compiles a small C program that calls the routines on the whole data and on the two batches, with unit weights and with weights 1, 2, 3, 4, and compares the combined mean, rms and standard deviation with closed forms; the one-call results and `computeMoments` are the controls. It needs `sddsprocess` on PATH, to locate the SDDS libraries, and `gcc`.
    """

    import sys

    import numpy as np
    from _elegant import verdict
    from _sdds import c_check

    RTOL = 1e-14
    C_SOURCE = r"""
    #include <stdio.h>
    extern long computeMoments(double *, double *, double *, double *, double *, long);
    extern long accumulateMoments(double *, double *, double *, double *, long, long);
    extern long accumulateWeightedMoments(double *, double *, double *, double *, double *, long, long);

    static void report(const char *name, double m, double r, double s) {
      printf("%s %.17g %.17g %.17g\n", name, m, r, s);
    }

    int main(void) {
      double x[] = {2, 4, 8, 10}, unit[] = {1, 1, 1, 1}, w[] = {1, 2, 3, 4};
      double mean, rms, sd, mad;
      if (!computeMoments(&mean, &rms, &sd, &mad, x, 4)) return 1;
      report("computeMoments", mean, rms, sd);
      if (!accumulateMoments(&mean, &rms, &sd, x, 4, 1)) return 2;
      report("unweighted-whole", mean, rms, sd);
      if (!accumulateMoments(&mean, &rms, &sd, x, 2, 1)) return 3;
      if (!accumulateMoments(&mean, &rms, &sd, x + 2, 2, 0)) return 4;
      report("unweighted-split", mean, rms, sd);
      if (!accumulateWeightedMoments(&mean, &rms, &sd, x, unit, 4, 1)) return 5;
      report("unit-weight-whole", mean, rms, sd);
      if (!accumulateWeightedMoments(&mean, &rms, &sd, x, unit, 2, 1)) return 6;
      if (!accumulateWeightedMoments(&mean, &rms, &sd, x + 2, unit + 2, 2, 0)) return 7;
      report("unit-weight-split", mean, rms, sd);
      if (!accumulateWeightedMoments(&mean, &rms, &sd, x, w, 4, 1)) return 8;
      report("weighted-whole", mean, rms, sd);
      if (!accumulateWeightedMoments(&mean, &rms, &sd, x, w, 2, 1)) return 9;
      if (!accumulateWeightedMoments(&mean, &rms, &sd, x + 2, w + 2, 2, 0)) return 10;
      report("weighted-split", mean, rms, sd);
      return 0;
    }
    """


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in values) + "] (mean, rms, sample sd)"


    def moments(x, w):
        """Mean, rms and sample standard deviation of x with weights w, as the routines define them.

        mean = sum(w x)/sum(w), rms = sqrt(sum(w x^2)/sum(w)), and the standard deviation is
        sqrt((rms^2 - mean^2) n/(n - 1)) with n the number of values, not the sum of weights.
        """
        mean = np.sum(w * x) / np.sum(w)
        rms = np.sqrt(np.sum(w * x**2) / np.sum(w))
        return np.array([mean, rms, np.sqrt((rms**2 - mean**2) * len(x) / (len(x) - 1))])


    x = np.array([2.0, 4.0, 8.0, 10.0])
    weights = {"unit": np.ones(4), "weighted": np.array([1.0, 2.0, 3.0, 4.0])}
    expected = {
        "computeMoments": moments(x, weights["unit"]),
        "unweighted-whole": moments(x, weights["unit"]),
        "unweighted-split": moments(x, weights["unit"]),
        "unit-weight-whole": moments(x, weights["unit"]),
        "unit-weight-split": moments(x, weights["unit"]),
        "weighted-whole": moments(x, weights["weighted"]),
        "weighted-split": moments(x, weights["weighted"]),
    }
    # Closed forms for the unit-weight data 2, 4, 8, 10: mean 6, rms sqrt(46), sd sqrt(40/3).
    if not np.allclose(
        expected["computeMoments"], [6, np.sqrt(46), np.sqrt(40 / 3)], rtol=RTOL, atol=0
    ):
        sys.exit("the closed forms disagree with the moment definitions")
    defect = False
    for name, *words in c_check(C_SOURCE):
        observed = np.array(words, dtype=float)
        print(f"{name}: observed={fmt(observed)} expected={fmt(expected[name])}")
        agrees = np.allclose(observed, expected[name], rtol=RTOL, atol=0)
        if name.endswith("whole") or name == "computeMoments":
            if not agrees:
                sys.exit(f"{name}: a one-call control failed: the comparison is invalid")
        else:
            defect |= not agrees
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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | SDDS master (`a60b9bbed7cc`), source with Elegant 75a2e1a1a58b, CPU, serial native library API, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | SDDS master (`a60b9bbed7cc`), source with Elegant 75a2e1a1a58b, CPU, serial native library API, one thread |

