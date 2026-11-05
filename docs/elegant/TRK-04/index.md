---
title: "TRK-04: KQUAD fringes ignore the fractional strength error FSE that scales the body"
hide:
  - toc
---

# TRK-04 — KQUAD fringes ignore the fractional strength error FSE that scales the body

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [body strength, multipole.c:791–795](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L791-L795)<br>[entrance fringe, multipole.c:1131–1132](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1131-L1132)<br>[exit fringe, multipole.c:1224–1225](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1224-L1225)<br>[nonlinear fringe coefficient, fringe.c:110](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/fringe.c#L110)<br>[linear fringe matrices, fringe.c:53–54](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/fringe.c#L53-L54)<br>[K1 scaling, quad_matrix.c:40](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/quad_matrix.c#L40)<br>[entrance, quad_matrix.c:364](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/quad_matrix.c#L364)<br>[exit, quad_matrix.c:374](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/quad_matrix.c#L374)<br>[FSE definition, track_data.c:1262](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1262)<br>[fringe definition, kquad.tex:84–95](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/kquad.tex#L84-L95) |

KQUAD scales its body's normalized gradient by 1 + FSE, where FSE is the fractional strength error, but both nonlinear fringe kicks keep the unscaled gradient K₁. A uniform field error leaves the fringe profile normalized to the central gradient unchanged, so K₁ = 5 m⁻² with FSE = 0.2 must track like K₁ = 6 m⁻² with FSE = 0, and FSE = −1 must be a plain drift. For a 0.4 m magnet and electron momentum 2000 mₑc, the exit slopes differ by up to 419.4 nrad, and at FSE = −1 the exit position is displaced by up to 54.35 nm.

??? example "Reproducer: TRK-04.py"

    [Download TRK-04.py](TRK-04.py)

    ```python
    """TRK-04: KQUAD fringes ignore the fractional strength error FSE that scales the body

    KQUAD scales its body's normalized gradient by 1 + FSE, where FSE is the fractional strength error,
    but both nonlinear fringe kicks keep the unscaled gradient K₁. A uniform field error leaves the
    fringe profile normalized to the central gradient unchanged, so K₁ = 5 m⁻² with FSE = 0.2 must track
    like K₁ = 6 m⁻² with FSE = 0, and FSE = −1 must be a plain drift. For a 0.4 m magnet and electron
    momentum 2000 mₑc, the exit slopes differ by up to 419.4 nrad, and at FSE = −1 the exit position is
    displaced by up to 54.35 nm.

    The script tracks twelve particles through a KQUAD with cubic fringes at both ends, specified by an
    explicit K1 so that the pole-tip conversion does not enter. It compares K1 = 5 with FSE = 0.2
    against K1 = 6, and K1 = 5 with FSE = -1 against the drift x + L xp, y + L yp. Controls compare
    the finite-strength pair with both fringes off, and K1 = 0 with the analytic drift.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    P0 = 2000.0  # electron momentum, units of m_e c
    LENGTH = 0.4
    # Twelve particles inside the bore: (x, xp, y, yp) scaled by 0.5 ... 1.
    SCALE = np.linspace(0.5, 1.0, 12)
    BEAM = {
        "x": 0.006 * SCALE,
        "xp": 0.001 * SCALE,
        "y": -0.004 * SCALE,
        "yp": -0.0005 * SCALE,
        "p": np.full(12, P0),
        "particleID": np.arange(1, 13),
    }
    ATOL = 1e-14  # the controls agree to 2e-15


    def track(mode, k1, fse, edges):
        """Exit (x, xp, y, yp) of the twelve particles through a KQUAD with the given K1 and FSE."""
        flags = f"EDGE1_EFFECTS={edges},EDGE2_EFFECTS={edges},EDGE1_LINEAR=0,EDGE2_LINEAR=0"
        lattice = f"E: KQUAD,L={LENGTH!r},N_SLICES=64,K1={k1!r},FSE={fse!r},{flags}\nBL: LINE=(E)\n"
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
            "output=out.sdds, default_order=2 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": BEAM})
        out = particles(result["out.sdds"])
        if len(out["x"]) != 12:
            sys.exit(f"{mode}: {12 - len(out['x'])} particles lost")
        return np.column_stack([out["x"], out["xp"], out["y"], out["yp"]])


    def drift():
        # Without a field a particle moves on a straight line and keeps its slopes.
        return np.column_stack(
            [
                BEAM["x"] + LENGTH * BEAM["xp"],
                BEAM["xp"],
                BEAM["y"] + LENGTH * BEAM["yp"],
                BEAM["yp"],
            ]
        )


    def compare(label, observed, expected):
        """Print one case (last particle, largest difference per column); True when they disagree."""
        bad = not np.allclose(observed, expected, rtol=0, atol=ATOL)
        last = lambda a: "[" + ", ".join(f"{v:.16g}" for v in a[-1]) + "]"
        print(
            f"{label}: observed={last(observed)} expected={last(expected)} "
            f"max|observed-expected|=[{', '.join(f'{v:.4g}' for v in np.abs(observed - expected).max(axis=0))}]"
        )
        return bad


    defect = False
    print("columns: x (m), xp, y (m), yp of the last particle")
    for mode in modes():
        # Controls: without fringes the two descriptions agree, and K1 = 0 is a drift.
        if compare(
            f"{mode} control, no fringes, K1=5 FSE=0.2 vs K1=6",
            track(mode, 5.0, 0.2, 0),
            track(mode, 6.0, 0.0, 0),
        ):
            sys.exit("The bodies of equivalent descriptions differ: the comparison is not valid.")
        if compare(f"{mode} control, K1=0 vs drift", track(mode, 0.0, 0.0, 2), drift()):
            sys.exit("A zero-gradient magnet is not a drift: the comparison is not valid.")
        # The field of (K1, FSE) = (5, 0.2) is that of (6, 0), with the same normalised fringe profile.
        defect |= compare(
            f"{mode} K1=5 FSE=0.2 vs K1=6 FSE=0", track(mode, 5.0, 0.2, 2), track(mode, 6.0, 0.0, 2)
        )
        # FSE = -1 removes the whole field, fringes included.
        defect |= compare(f"{mode} K1=5 FSE=-1 vs drift", track(mode, 5.0, -1.0, 2), drift())
    print(f"comparison: atol={ATOL:g}")
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
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

