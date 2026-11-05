---
title: "IO-03: SDDS complete elliptic integral E(k) returns π/4 at the endpoints k = ±1"
hide:
  - toc
---

# IO-03 — SDDS complete elliptic integral E(k) returns π/4 at the endpoints k = ±1

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on SDDS master (`a3f730a`) |
| Source | [`K * (1 - sum / 2)`, elliptic.c:72–98](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/elliptic.c#L72-L98)<br>[`KE_cei`, elliptic.c:100–130](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/elliptic.c#L100-L130)<br>[rpn_cei2, rpns/code/math.c:744–756](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/rpns/code/math.c#L744-L756)<br>[cei1 and cei2, SDDSaps/doc/rpn.tex:110–113](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/SDDSaps/doc/rpn.tex#L110-L113) |

Evaluating the complete elliptic integral of the second kind, E(k) = ∫₀^{π/2} √(1 − k² sin²θ) dθ, with `sddsprocess` (rpn `cei2`) or the SDDS routines `E_cei` and `KE_cei` returns π/4 = 0.7854 at k = ±1. At those endpoints the integrand is cos θ and E = 1, so the result is 21.46% too small, without a utility warning. At the six tested interior moduli, results agree with an independent implementation within 2×10⁻¹⁵; at k = 0.8, E = 1.276.

??? example "Reproducer: IO-03.py"

    [Download IO-03.py](IO-03.py)

    ```python
    """IO-03: SDDS complete elliptic integral E(k) returns π/4 at the endpoints k = ±1

    Evaluating the complete elliptic integral of the second kind, E(k) = ∫₀^{π/2} √(1 − k² sin²θ) dθ, with `sddsprocess` (rpn `cei2`) or the SDDS routines `E_cei` and `KE_cei` returns π/4 = 0.7854 at k = ±1. At those endpoints the integrand is cos θ and E = 1, so the result is 21.46% too small, without a utility warning. At the six tested interior moduli, results agree with an independent implementation within 2×10⁻¹⁵; at k = 0.8, E = 1.276.

    The script evaluates E at k = −1, 0, 0.25, 0.8, 0.99, 1 − 2⁻³⁰, the largest double below 1, and 1 with the library routines `E_cei` and `KE_cei` in a small C program and with `sddsprocess`. It compares with SciPy's Cephes `ellipe`, an independent implementation, and with the exact value 1 at the endpoints. The interior moduli are the control. It needs `sddsprocess` on PATH and `gcc`.
    """

    import sys

    import numpy as np
    from _elegant import verdict
    from _sdds import c_check, process
    from scipy.special import ellipe

    MODULI = np.array([-1.0, 0.0, 0.25, 0.8, 0.99, 1 - 2.0**-30, np.nextafter(1.0, 0.0), 1.0])
    ATOL = 5e-15  # interior agreement with Cephes; the largest difference seen is 1.1e-15
    C_SOURCE = r"""
    #include <math.h>
    #include <stdio.h>
    extern double E_cei(double);
    extern double *KE_cei(double, double *);
    int main(void) {
      const double moduli[] = {MODULI};
      for (unsigned i = 0; i < sizeof(moduli) / sizeof(moduli[0]); ++i) {
        double values[2];
        KE_cei(moduli[i], values);
        printf("%.17g %.17g\n", E_cei(moduli[i]), values[1]);
      }
      return 0;
    }
    """.replace("MODULI", ", ".join(repr(float(k)) for k in MODULI))


    # E(k) = int_0^{pi/2} sqrt(1 - k^2 sin^2 t) dt is the Cephes ellipe at parameter m = k^2, whose
    # value at m = 1 is exactly 1 (the integrand is cos t).
    expected = ellipe(MODULI**2)
    api = np.array(c_check(C_SOURCE), dtype=float)
    results = {
        "E_cei": api[:, 0],
        "KE_cei.E": api[:, 1],
        "sddsprocess cei2": process({"k": MODULI}, "-define=column,E,k cei2")["E"],
    }
    # Wrong-shaped or nonfinite output is an invalid run: NaN interior values would pass an
    # "error > ATOL" test, and infinite endpoint values would satisfy the defect threshold.
    for name, observed in results.items():
        if observed.shape != MODULI.shape or not np.all(np.isfinite(observed)):
            sys.exit(
                f"{name}: the output is not {MODULI.size} finite values: the comparison is invalid"
            )
    if not np.all(np.isfinite(expected)):
        sys.exit("the Cephes expected values are not finite")
    for i, k in enumerate(MODULI):
        values = " ".join(f"{name}={observed[i]:.17g}" for name, observed in results.items())
        print(f"k={k:.17g}: observed: {values}; expected={expected[i]:.17g}")
    endpoint = np.abs(MODULI) == 1
    defect = True
    for name, observed in results.items():
        if not np.allclose(observed[~endpoint], expected[~endpoint], rtol=0, atol=ATOL):
            sys.exit(f"{name}: an interior control failed: the comparison is invalid")
        defect &= bool(np.all(np.abs(observed[endpoint] - 1) > 0.2))
    print(f"comparison: interior atol={ATOL:g}; endpoints exact value 1")
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

