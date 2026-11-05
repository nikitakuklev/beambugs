---
title: "OPT-25: moments_output crashes on a one-element beamline"
hide:
  - toc
---

# OPT-25 — moments_output crashes on a one-element beamline

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`dumpBeamMoments`, starting row, src/moments.c:270–282](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L270-L282)<br>[`moments_output` computes periodic or propagates non-periodic beam moments, doc/elegant.tex:5549](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5549)<br>[with `matched=0` the given values are propagated, doc/elegant.tex:5606–5607](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5606-L5607)<br>[the starting ellipses are given in all three planes, doc/elegant.tex:5632–5633](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5632-L5633) |

Elegant's `moments_output` crashes with a segmentation fault on a beamline of one element when it writes moments along the beamline (the default `final_values_only=0`): writing the starting row reads the Coulomb logarithm of the element after the first, which does not exist. The manual supports propagating given starting moments through a non-periodic beamline, and a single 1.7 m drift entered with Twiss βx = 3 m, αx = 0.4 and emittance εx = 2 nm is valid input. With `final_values_only=1`, or with a zero-length marker appended, Elegant finishes and writes the drift-transported beam-matrix elements σ11 = 5.515×10⁻⁹ m² and σ12 = 5.147×10⁻¹⁰ m.

??? example "Reproducer: OPT-25.py"

    [Download OPT-25.py](OPT-25.py)

    ```python
    """OPT-25: moments_output crashes on a one-element beamline

    Elegant's `moments_output` crashes with a segmentation fault on a beamline of one element when it
    writes moments along the beamline (the default `final_values_only=0`): writing the starting row
    reads the Coulomb logarithm of the element after the first, which does not exist. The manual
    supports propagating given starting moments through a non-periodic beamline, and a single 1.7 m
    drift entered with Twiss βx = 3 m, αx = 0.4 and emittance εx = 2 nm is valid input. With
    `final_values_only=1`, or with a zero-length marker appended, Elegant finishes and writes the
    drift-transported beam-matrix elements σ11 = 5.515×10⁻⁹ m² and σ12 = 5.147×10⁻¹⁰ m.

    The script runs serial Elegant `moments_output` with `matched=0` and given starting ellipses on a
    beamline holding one 1.7 m drift, with the default `final_values_only=0`, and reports whether the
    run ends with a segmentation fault. The controls are the same input with `final_values_only=1` and
    the drift followed by a zero-length marker with the default output; both must finish, and their
    final horizontal covariance must equal the drift transport D Σ Dᵀ of the starting ellipse.
    """

    import sys

    import numpy as np
    from _elegant import RunFailure, run, verdict

    L = 1.7
    EMIT, BETA, ALPHA = 2e-9, 3.0, 0.4
    LATTICE = f"D: DRIF,L={L}\nM: MARK\nONE: LINE=(D)\nTWO: LINE=(D,M)\n"
    # Starting horizontal covariance from (emittance, beta, alpha), drift-transported:
    # s11 = e (beta - 2 alpha L + gamma L^2), s12 = e (-alpha + gamma L).
    GAMMA = (1 + ALPHA**2) / BETA
    EXPECTED = [EMIT * (BETA - 2 * ALPHA * L + GAMMA * L**2), EMIT * (-ALPHA + GAMMA * L)]


    def case(line, final_only):
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline={line}, p_central_mev=2400, default_order=2 &end\n"
            "&moments_output filename=moments.sdds, matched=0, equilibrium=0, radiation=0,"
            f" final_values_only={final_only}, emit_x={EMIT!r}, beta_x={BETA!r}, alpha_x={ALPHA!r},"
            " emit_y=1e-9, beta_y=2, emit_z=1e-6, beta_z=4 &end\n"
        )
        try:
            result = run({"lattice.lte": LATTICE, "run.ele": commands}, outputs=("moments.sdds",))
        except RunFailure as failure:
            crashed = "SIGSEGV" in failure.log
            return None, f"exit status {failure.status}{', Terminated by SIGSEGV' if crashed else ''}"
        moments = result["moments.sdds"]
        final = [float(moments[0, "s1"][-1]) ** 2, float(moments[0, "s12"][-1])]
        return final, f"finished, final s11, s12 = [{final[0]:.10g}, {final[1]:.10g}]"


    defect = False
    print(f"expected final s11, s12 = [{EXPECTED[0]:.10g}, {EXPECTED[1]:.10g}] (m^2, m)")
    for line, final_only, control in (("ONE", 1, True), ("TWO", 0, True), ("ONE", 0, False)):
        final, note = case(line, final_only)
        label = f"{'control, ' if control else ''}beamline {line} final_values_only={final_only}"
        print(f"{label}: observed={note} expected=finished")
        if control and (final is None or not np.allclose(final, EXPECTED, rtol=1e-12)):
            sys.exit("A control run failed or disagrees with the drift transport: invalid comparison.")
        defect |= final is None
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
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

