---
title: "TRK-35: CSBEND with EXPAND_HAMILTONIAN=1 leaves the curvature term out of the path length"
hide:
  - toc
---

# TRK-35 — CSBEND with EXPAND_HAMILTONIAN=1 leaves the curvature term out of the path length

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`integrate_csbend_ordn_expanded`, csbend.c:2159–2167](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2159-L2167)<br>[csbend.c:2253](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2253)<br>[parameter table, track_data.c:1977](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1977) |

A CSBEND with EXPAND_HAMILTONIAN=1 accumulates the path length without the leading curvature term hx, where h is the curvature and x the horizontal offset, so the bend body adds no first-order path dependence on x, the slope x′ or the momentum deviation δ. For a 1.2 m, 0.13 rad sector bend, the path-length matrix elements R51, R52 and R56 are 9.992×10⁻¹¹, 7.772×10⁻¹¹ m and −1.835×10⁻⁸ m instead of the linear sector-bend values 0.1296, 0.07789 m and 3.377×10⁻³ m, which the exact Hamiltonian reproduces. The dispersion R16 = 0.07789 m is kept, so a ring of such bends loses the momentum compaction of its bends.

??? example "Reproducer: TRK-35.py"

    [Download TRK-35.py](TRK-35.py)

    ```python
    """TRK-35: CSBEND with EXPAND_HAMILTONIAN=1 leaves the curvature term out of the path length

    A CSBEND with EXPAND_HAMILTONIAN=1 accumulates the path length without the leading curvature term
    hx, where h is the curvature and x the horizontal offset, so the bend body adds no first-order path
    dependence on x, the slope x′ or the momentum deviation δ. For a 1.2 m, 0.13 rad sector bend, the
    path-length matrix elements R51, R52 and R56 are 9.992×10⁻¹¹, 7.772×10⁻¹¹ m and −1.835×10⁻⁸ m
    instead of the linear sector-bend values 0.1296, 0.07789 m and 3.377×10⁻³ m, which the exact
    Hamiltonian reproduces. The dispersion R16 = 0.07789 m is kept, so a ring of such bends loses the
    momentum compaction of its bends.

    The script tracks a reference electron and pairs of electrons offset by ±10⁻⁵ in x, x′ and δ
    through a sector CSBEND without edges or radiation, for the exact and the expanded Hamiltonian, and
    forms the path-length row R51, R52, R56 and the dispersion R16 by central differences. It compares
    them with the linear sector-bend transfer matrix; the exact-Hamiltonian run is the control.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    C = 299792458.0  # speed of light (m/s)
    P0 = 6000.0  # reference momentum in m_e c
    LENGTH, ANGLE, SLICES = 1.2, 0.13, 128
    STEP = 1e-5
    # The exact-Hamiltonian row agrees with the linear matrix to about 1.4e-10; second-order terms of a
    # central difference with step 1e-5 are below 1e-10.
    TOL = 1e-8


    def fmt(values):
        return "[" + ", ".join(f"{v:.12g}" for v in np.atleast_1d(values)) + "]"


    def response(mode, expand):
        """(R51, R52, R56, R16) by central differences in x, xp and delta."""
        lattice = (
            f"B: CSBEND,L={LENGTH!r},ANGLE={ANGLE!r},N_SLICES={SLICES},INTEGRATION_ORDER=6,"
            f"EXPAND_HAMILTONIAN={expand},EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,SYNCH_RAD=0,ISR=0\n"
            "BL: LINE=(B)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        # Rows: reference, +x, -x, +xp, -xp, +delta, -delta (two copies so every MPI worker tracks some).
        offsets = np.zeros((7, 3))
        for j in range(3):
            offsets[1 + 2 * j, j], offsets[2 + 2 * j, j] = STEP, -STEP
        offsets = np.vstack([offsets, offsets])
        beam = {
            "x": offsets[:, 0],
            "xp": offsets[:, 1],
            "p": P0 * (1 + offsets[:, 2]),
            "particleID": np.arange(1, len(offsets) + 1),
        }
        files = {"lattice.lte": lattice, "run.ele": commands}
        out = particles(run(files, mode, {"beam.sdds": beam})["out.sdds"])
        if len(out["p"]) != len(offsets):
            sys.exit(f"{mode}: particles were lost")
        # Elegant writes the arrival time t = path/(beta c); convert back with each particle's speed.
        path = out["t"] * C * out["p"] / np.sqrt(1 + out["p"] ** 2)
        x = out["x"]
        rows = [(path[1 + 2 * j] - path[2 + 2 * j]) / (2 * STEP) for j in range(3)]
        r16 = (x[5] - x[6]) / (2 * STEP)
        return np.array([*rows, r16])


    # Linear sector bend of radius rho and angle theta: x(s) = R11 x0 + R12 x0' + rho(1 - cos(s/rho)) delta,
    # and the physical path grows at d(path)/ds = 1 + x/rho to first order, so
    # R51 = sin(theta), R52 = rho (1 - cos(theta)), R56 = rho (theta - sin(theta)), R16 = rho (1 - cos(theta)).
    rho = LENGTH / ANGLE
    expected = np.array(
        [
            np.sin(ANGLE),
            rho * (1 - np.cos(ANGLE)),
            rho * (ANGLE - np.sin(ANGLE)),
            rho * (1 - np.cos(ANGLE)),
        ]
    )

    defect = []
    print(
        f"columns: R51, R52, R56 (m), R16 (m); L={LENGTH} m, ANGLE={ANGLE} rad, {SLICES} slices, order 6"
    )
    for mode in modes():
        exact = response(mode, 0)
        print(
            f"{mode} exact Hamiltonian (control): observed={fmt(exact)} expected={fmt(expected)}"
            f" max|diff|={np.max(np.abs(exact - expected)):.3g}"
        )
        if np.max(np.abs(exact - expected)) > TOL:
            sys.exit("The exact-Hamiltonian control disagrees with the linear matrix: invalid.")
        expanded = response(mode, 1)
        print(
            f"{mode} EXPAND_HAMILTONIAN=1: observed={fmt(expanded)} expected={fmt(expected)}"
            f" max|diff|={np.max(np.abs(expanded - expected)):.3g}"
        )
        if abs(expanded[3] - expected[3]) > TOL:
            sys.exit("The expanded Hamiltonian loses the dispersion too: the cause differs.")
        defect.append(bool(np.max(np.abs(expanded[:3] - expected[:3])) > TOL))
    print(f"comparison: abs tolerance {TOL:g}")
    verdict(all(defect))

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
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

