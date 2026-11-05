---
title: "TRK-37: CSBEND edge model 5 adds a constant path offset and drops the y² path term when K3 is small"
hide:
  - toc
---

# TRK-37 — CSBEND edge model 5 adds a constant path offset and drops the y² path term when K3 is small

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`curvedDipoleFringe`, csbend.c:6830–6839](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L6830-L6839)<br>[6833](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L6833)<br>[6896](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L6896)<br>[6967](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L6967)<br>[7008](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L7008)<br>[csbend.c:7026](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L7026)<br>[manual, csbend.tex:96–101](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L96-L101)<br>[`curvedDipoleFringeCalc`, curvedDipoleFringeCalc.c:461–505](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/elegantTools/curvedDipoleFringeCalc.c#L461-L505) |

A CSBEND edge with model 5 and fringe integral |K₃/(1+δ)| < 10⁻⁵, where δ is the momentum deviation, evaluates its path-length formula with a misplaced parenthesis. Each such edge adds the constant −K₂/(2(1+δ)²) to every particle's path and drops the term −K₂y²/(2(1+δ)²), the K₃ → 0 limit of the exact formula used for larger K₃. A pure dipole's soft edge has K₃ = 0: for curvature 0.1 m⁻¹ and a 2 cm fringe, K₂ = 10⁻⁴ and each edge shortens the path by 5.000×10⁻⁵ m, and K₂ = 0.006 gives 3.000×10⁻³ m.

??? example "Reproducer: TRK-37.py"

    [Download TRK-37.py](TRK-37.py)

    ```python
    """TRK-37: CSBEND edge model 5 adds a constant path offset and drops the y² path term when K3 is small

    A CSBEND edge with model 5 and fringe integral |K₃/(1+δ)| < 10⁻⁵, where δ is the momentum deviation,
    evaluates its path-length formula with a misplaced parenthesis. Each such edge adds the constant
    −K₂/(2(1+δ)²) to every particle's path and drops the term −K₂y²/(2(1+δ)²), the K₃ → 0 limit of the
    exact formula used for larger K₃. A pure dipole's soft edge has K₃ = 0: for curvature 0.1 m⁻¹ and a
    2 cm fringe, K₂ = 10⁻⁴ and each edge shortens the path by 5.000×10⁻⁵ m, and K₂ = 0.006 gives
    3.000×10⁻³ m.

    The script tracks electrons on axis and at y = ±1.5 mm through a CSBEND with edge model 5 on one
    edge, for both EDGE_ORDER settings, with the fringe integral K3 just below and just above the
    10⁻⁵ switch between the small-argument and exact formulas. The expected path is the K3 → 0 limit of
    the exact formula: no on-axis offset and a y² term −K₂y²/(2(1+δ)²). The cases above the switch are
    the control; a weak fringe with K₂ = 10⁻⁴ shows the realistic size.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    C = 299792458.0  # speed of light (m/s)
    P0 = 6000.0  # reference momentum in m_e c
    LENGTH, ANGLE, EDGE_ANGLE, Y0 = 0.7, 0.07, 0.035, 1.5e-3
    # Branch continuity at the 1e-5 switch is about 4e-14 m in the y^2 term; path rounding is ~1e-16 m.
    TOL = 1e-12


    def paths(mode, edge, order, k2, k3, edge_angle):
        """Path minus L of electrons at y = 0, +Y0, -Y0, with model 5 on one edge only."""
        lattice = (
            f"B: CSBEND,L={LENGTH!r},ANGLE={ANGLE!r},E{edge}={edge_angle!r},N_SLICES=64,INTEGRATION_ORDER=6,"
            f"EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,EDGE{edge}_EFFECTS=5,EDGE_ORDER={order},"
            f"FRINGE{edge}K2={k2!r},FRINGE{edge}K3={k3!r},SYNCH_RAD=0,ISR=0\n"
            "BL: LINE=(B)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        y = np.tile([0.0, Y0, -Y0], 4)  # four copies so that every MPI worker tracks some
        beam = {"y": y, "p": np.full(len(y), P0), "particleID": np.arange(1, len(y) + 1)}
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if len(out["p"]) != len(y):
            sys.exit(f"{mode}: particles were lost")
        p = out["p"]
        path = out["t"] * C * p / np.sqrt(1 + p * p) - LENGTH
        return path[0], (path[1] + path[2]) / 2 - path[0]


    defect = []
    print(
        f"L={LENGTH} m, ANGLE={ANGLE} rad, p0={P0:g} m_e c, delta=0; y2 term is the path change at y=+-{Y0} m"
    )
    for mode in modes():
        cases = [(2, o, 0.006, k3, EDGE_ANGLE) for o in (1, 2) for k3 in (0.0, 9.9e-6, 1.01e-5, 2e-5)]
        cases += [(1, o, 0.006, k3, EDGE_ANGLE) for o in (1, 2) for k3 in (0.0, 1.01e-5)]
        # A pure dipole whose field falls off as (1 + tanh(u/w))/2 has, by the fringe-integral definitions
        # of curvedDipoleFringeCalc, K2 = h^2 w/2 and K3 = 0; h = 0.1 /m and w = 2 cm give K2 = 1e-4.
        cases += [(2, 1, 1e-4, k3, 0.0) for k3 in (0.0, 1.01e-5)]
        for edge, order, k2, k3, edge_angle in cases:
            on_axis, y2 = paths(mode, edge, order, k2, k3, edge_angle)
            # Exact branch: dtau gets 0.5*(focY + 0.5*focX*((exp(2m) - 1)/m - 2))*y^2/(1+delta) with
            # m = K3/(1+delta) and focY = K2/(1+delta); as m -> 0 the bracket tends to focY, and the
            # path is -dtau, so the path changes by -K2*y^2/(2(1+delta)^2) and has no constant term.
            # The body of a pure dipole adds no path change for a vertical offset.
            expected_y2 = -0.5 * k2 * Y0 * Y0
            label = (
                f"{mode} {'entrance' if edge == 1 else 'exit'} EDGE_ORDER={order} K2={k2:g} K3={k3:g}"
            )
            print(
                f"{label}: on-axis path - L observed={on_axis:.6g} expected=0;"
                f" y2 term observed={y2:.6g} expected={expected_y2:.6g}"
            )
            bad = abs(on_axis) > TOL or (edge == 2 and abs(y2 - expected_y2) > TOL)
            if k3 > 1e-5:
                if bad:
                    sys.exit(f"{label}: the exact-branch control disagrees: invalid")
            else:
                if abs(on_axis + 0.5 * k2) > TOL:
                    sys.exit(f"{label}: the offset is not -K2/2: the cause differs")
                defect.append(bad)
    print(f"comparison: abs tolerance {TOL:g} m")
    verdict(bool(defect) and all(defect))

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

