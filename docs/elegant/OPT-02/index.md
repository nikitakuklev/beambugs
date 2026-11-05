---
title: "OPT-02: Cubic compaction omits momentum–transverse quadratic terms"
hide:
  - toc
---

# OPT-02 — Cubic compaction omits momentum–transverse quadratic terms

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`compute_twiss_parameters`, `alpha3` accumulation, src/twiss.cc:2398–2411](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L2398-L2411)<br>[`alpha2` accumulation, src/twiss.cc:2384–2386](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L2384-L2386)<br>[`alphac3` defined as the third-order momentum compaction, manual, doc/elegant.tex:8254–8258](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8254-L8258) |

Elegant's `twiss_output` computes the third-order momentum compaction αc3 of a periodic cell but leaves out the path-length terms in which the momentum offset δ multiplies a transverse coordinate that is itself driven by δ². Take a 10 m cell with betatron tunes 0.17 and 0.23, a second-order term T166 = 0.2 m (horizontal shift ∝ δ²) and a path term T561 = 0.3 (path length ∝ xδ). The closed orbit is x = 0.1 m δ², so the path length is 10 m + 0.03 m δ³ and αc3 = 0.003. Elegant writes the correct second-order dispersion 0.1 m but αc3 = 0.

??? example "Reproducer: OPT-02.py"

    [Download OPT-02.py](OPT-02.py)

    ```python
    """OPT-02: Cubic compaction omits momentum–transverse quadratic terms

    Elegant's `twiss_output` computes the third-order momentum compaction αc3 of a periodic cell but
    leaves out the path-length terms in which the momentum offset δ multiplies a transverse coordinate
    that is itself driven by δ². Take a 10 m cell with betatron tunes 0.17 and 0.23, a second-order term
    T166 = 0.2 m (horizontal shift ∝ δ²) and a path term T561 = 0.3 (path length ∝ xδ). The closed orbit
    is x = 0.1 m δ², so the path length is 10 m + 0.03 m δ³ and αc3 = 0.003. Elegant writes the correct
    second-order dispersion 0.1 m but αc3 = 0.

    The script writes a 10 m periodic EMATRIX cell with stable horizontal and vertical rotations
    (tunes 0.17 and 0.23), `T166 = 0.2` and `T561 = 0.3`, and runs `twiss_output` at third order in
    serial Elegant (and with `--mpi` in 3-rank Pelegant). It compares `etax2`, `alphac`, `alphac2` and
    `alphac3` with the exact closed orbit x(δ) = η₂δ², where (I - R)η₂ = (T166, T266). A control cell
    with first-order dispersion and `T511` instead of `T561` produces the same kind of path term from
    two transverse coordinates, which Elegant includes in `alphac2`.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    LENGTH = 10.0  # m, both C5 and L of the cell
    TUNES = (0.17, 0.23)  # horizontal and vertical betatron tunes of the cell, cycles
    RTOL, ATOL = 3e-13, 1e-15


    def rotation(tune):
        c, s = np.cos(2 * np.pi * tune), np.sin(2 * np.pi * tune)
        return np.array([[c, s], [-s, c]])


    R = np.eye(6)
    R[:2, :2], R[2:4, 2:4] = rotation(TUNES[0]), rotation(TUNES[1])


    def lattice(extra):
        elements = ",".join(f"R{i + 1}{j + 1}={float(R[i, j])!r}" for i in range(6) for j in range(6))
        return f"E: EMATRIX,L={LENGTH!r},C5={LENGTH!r},{elements}{extra}\nBL: LINE=(E)\n"


    COMMANDS = (
        "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&twiss_output filename=twiss.sdds, matched=1 &end\n"
        "&bunched_beam n_particles_per_bunch=12 &end\n"
        "&track &end\n"
    )


    def twiss(mode, extra):
        out = run({"lattice.lte": lattice(extra), "run.ele": COMMANDS}, mode, outputs=("twiss.sdds",))
        return out["twiss.sdds"]


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def compare(label, observed, expected, controls=0):
        """Print one case; return True when the entries after the first `controls` disagree.

        All values must be finite and the first `controls` entries must agree, otherwise the run is
        invalid and exits nonzero before any verdict.
        """
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        if not (np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))):
            sys.exit(f"{label}: nonfinite value: invalid run")
        if not np.allclose(observed[:controls], expected[:controls], rtol=RTOL, atol=ATOL):
            sys.exit(f"{label}: the first {controls} entries disagree with the oracle: invalid run")
        return not np.allclose(observed[controls:], expected[controls:], rtol=RTOL, atol=ATOL)


    def orbit_coefficient(source):
        """Closed-orbit coefficient c of x = c*delta^n: (I - R) c = source, horizontal component."""
        return np.linalg.solve(np.eye(2) - R[:2, :2], source)[0]


    defect = False
    print("columns: etax (m), etax2 (m), alphac, alphac2, alphac3")
    for mode in modes():
        # Defect case: the quadratic terms are T166 (x shift 0.2 delta^2) and T561 (path 0.3 x delta).
        # The closed orbit is x = eta2 delta^2 with (I - R) eta2 = (0.2, 0), so eta2 = 0.1 m, and the
        # path per turn is C5 + 0.3 x delta = 10 m + 0.3*eta2*delta^3, hence alphac3 = 0.3*eta2/L.
        # No first- or second-order path term exists, so alphac = alphac2 = 0.
        t = twiss(mode, ",T166=0.2,T561=0.3")
        eta2 = orbit_coefficient([0.2, 0.0])
        observed = [t[0, "etax"][-1], t["etax2"][0], t["alphac"][0], t["alphac2"][0], t["alphac3"][0]]
        expected = [0.0, eta2, 0.0, 0.0, 0.3 * eta2 / LENGTH]
        # etax, etax2, alphac and alphac2 are mandatory controls; only alphac3 carries the defect flag.
        defect |= compare(f"{mode} T166=0.2, T561=0.3", observed, expected, controls=4)
        # Control: R16 = 0.2 gives first-order dispersion eta1 = (I - R)^-1 (0.2, 0); the path term
        # T511 x^2 = 0.3 eta1^2 delta^2 gives alphac2 = 0.3*eta1^2/L, a transverse-transverse
        # product that Elegant does include.
        t = twiss(mode, ",R16=0.2,T511=0.3")
        eta1 = orbit_coefficient([0.2, 0.0])
        observed = [t[0, "etax"][-1], t["etax2"][0], t["alphac"][0], t["alphac2"][0], t["alphac3"][0]]
        expected = [eta1, 0.0, 0.0, 0.3 * eta1**2 / LENGTH, 0.0]
        if compare(f"{mode} control, R16=0.2, T511=0.3", observed, expected):
            sys.exit("The T511 control disagrees with the oracle: the comparison is not valid.")
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
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
| Found | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

