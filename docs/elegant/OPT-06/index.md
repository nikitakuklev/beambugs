---
title: "OPT-06: Analytic QUAD cubic chromatic coefficients are wrong"
hide:
  - toc
---

# OPT-06 — Analytic QUAD cubic chromatic coefficients are wrong

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`quadrupole_matrix`, focusing-branch δ² coefficients of both transverse blocks, quad_matrix.c:162–199](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/quad_matrix.c#L162-L199)<br>[the corresponding coefficients, quad_matrix.c:259–296](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/quad_matrix.c#L259-L296) |

Elegant's aligned, hard-edge QUAD writes incorrect third-order matrix coefficients for transverse transport proportional to δ², where δ is the relative momentum offset. For length L = 0.5 m and strength K₁ = 0.6 m⁻², exact transport has R₁₁(δ) = cos(u/√(1+δ)), with u = √(K₁) L, giving U₁₆₆₁ = −0.07222; Elegant writes −0.07245, with relative errors across both planes of 0.1853% to 0.6598%. In a one-period FODO cell using cubic lens maps, d²νₓ/dδ² and d³νₓ/dδ³ are 0.1300 and −0.07925 instead of 0.1268 and −0.08637, where νₓ is the horizontal tune. Users of third-order quadrupole optics and higher-order chromaticities are affected.

??? example "Reproducer: OPT-06.py"

    [Download OPT-06.py](OPT-06.py)

    ```python
    """OPT-06: Analytic QUAD cubic chromatic coefficients are wrong

    Elegant's aligned, hard-edge QUAD writes incorrect third-order matrix coefficients for transverse
    transport proportional to δ², where δ is the relative momentum offset. For length L = 0.5 m and
    strength K₁ = 0.6 m⁻², exact transport has R₁₁(δ) = cos(u/√(1+δ)), with u = √(K₁) L, giving U₁₆₆₁ =
    −0.07222; Elegant writes −0.07245, with relative errors across both planes of 0.1853% to 0.6598%. In
    a one-period FODO cell using cubic lens maps, d²νₓ/dδ² and d³νₓ/dδ³ are 0.1300 and −0.07925 instead
    of 0.1268 and −0.08637, where νₓ is the horizontal tune. Users of third-order quadrupole optics and
    higher-order chromaticities are affected.

    The script builds `QUAD,L=0.5,K1=0.6` with a third-order map in serial Elegant (and with `--mpi` in
    3-rank Pelegant) and compares the transverse 2x2 blocks of the written map, as Taylor series in δ,
    with the exact hard-edge transport. The same quadrupole as a tracking-derived `KQUAD` matrix, and
    the zeroth- and first-order coefficients of the analytic matrix, are the controls. The one-period
    FODO cell then gives the horizontal and vertical chromaticities, compared with the same exact maps
    truncated at second order in δ.
    """

    import sys

    import numpy as np
    from _cubic_optics import fodo_tune_derivatives, quad_matrix, taylor
    from _elegant import modes, run, verdict

    K1, LENGTH = 0.6, 0.5
    CONTROL_RTOL = 1e-12  # analytic R and T coefficients agree with the oracle to rounding
    TRACKING_RTOL, TRACKING_ATOL = 1e-5, 1e-6  # tracking-derived matrix: finite-difference accuracy
    RTOL, ATOL = 1e-9, 1e-12

    COMMANDS = (
        "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&matrix_output SDDS_output=map.sdds, SDDS_output_order=3 &end\n"
        "&bunched_beam n_particles_per_bunch=12 &end\n"
        "&track &end\n"
    )
    FODO_COMMANDS = (
        "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&twiss_output filename=twiss.sdds, concat_order=3, higher_order_chromaticity=1,\n"
        " higher_order_chromaticity_points=11, higher_order_chromaticity_range=0.01 &end\n"
        "&bunched_beam n_particles_per_bunch=12 &end\n"
        "&track &end\n"
    )
    FODO_LATTICE = "QF: QUAD,L=0.5,K1=0.6\nQD: QUAD,L=0.5,K1=-0.54\nD: DRIF,L=1\nBL: LINE=(QF,D,QD,D)\n"
    QUAD = f"QUAD,L={LENGTH!r},K1={K1!r}"
    KQUAD = f"KQUAD,L={LENGTH!r},K1={K1!r},N_SLICES=128,INTEGRATION_ORDER=4,TRACKING_MATRIX=3"


    def map_columns(mode, element):
        out = run(
            {"lattice.lte": f"E: {element}\nBL: LINE=(E)\n", "run.ele": COMMANDS},
            mode,
            outputs=("map.sdds",),
        )
        exported = out["map.sdds"]
        return lambda name: exported[0, name][-1]  # last row: the map through the element


    def coefficients(read, plane):
        """Elegant's R, T (x delta) and U (x delta^2) entries of one transverse block, as arrays [3][2][2].

        Entry (i, j) of order n is R_ij, T_i6j or U_i66j with 1-based indices offset by the plane.
        """
        first = 1 if plane == "x" else 3
        names = ("R{i}{j}", "T{i}6{j}", "U{i}66{j}")
        return np.array(
            [
                [[read(name.format(i=first + a, j=first + b)) for b in range(2)] for a in range(2)]
                for name in names
            ]
        )


    def exact(plane):
        """Taylor coefficients c_0, c_1, c_2 in delta of the exact hard-edge block.

        The gradient scales as K1/(1 + delta) (rigidity); the quadrupole defocuses in y. With
        u = sqrt(K1) L the x block has R11(delta) = cos(u/sqrt(1 + delta)), whose delta^2 coefficient
        is -(3 u sin u + u^2 cos u)/8. All entries are analytic in delta, so Cauchy's formula on a
        circle in the complex delta plane gives the coefficients to rounding.
        """
        sign = 1 if plane == "x" else -1
        return np.array([c.real for c in taylor(lambda z: quad_matrix(sign * K1 / (1 + z), LENGTH))])


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.ravel(values)) + "]"


    def compare(label, observed, expected, rtol=RTOL, atol=ATOL):
        """Print one case; return True when observed and expected disagree."""
        observed, expected = np.asarray(observed), np.asarray(expected)
        bad = not np.allclose(observed, expected, rtol=rtol, atol=atol)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    defect = False
    print("columns per block: R_ab, T_a6b, U_a66b with a, b over the plane's two coordinates")
    for mode in modes():
        analytic, tracked = map_columns(mode, QUAD), map_columns(mode, KQUAD)
        for plane in "xy":
            expected = exact(plane)
            got, control = coefficients(analytic, plane), coefficients(tracked, plane)
            # Control 1: orders 0 and 1 of the analytic matrix are right.
            if compare(
                f"{mode} {plane} control, R and T of QUAD", got[:2], expected[:2], CONTROL_RTOL, 1e-14
            ):
                sys.exit("The R/T control disagrees with the oracle: the comparison is not valid.")
            # Control 2: the tracking-derived KQUAD matrix reproduces the oracle at order 2.
            if compare(
                f"{mode} {plane} control, U of KQUAD",
                control[2],
                expected[2],
                TRACKING_RTOL,
                TRACKING_ATOL,
            ):
                sys.exit("The KQUAD control disagrees with the oracle: the comparison is not valid.")
            first = 1 if plane == "x" else 3
            names = [f"U{first + a}66{first + b}" for a in range(2) for b in range(2)]
            print(f"{mode} {plane} U entries: {names}")
            defect |= compare(f"{mode} {plane} U of QUAD", got[2], expected[2])

        # Consequence: higher chromaticities of the one-period FODO cell (cubic maps, analytic QUAD).
        twiss = run(
            {"lattice.lte": FODO_LATTICE, "run.ele": FODO_COMMANDS}, mode, outputs=("twiss.sdds",)
        )["twiss.sdds"]
        for plane in "xy":
            observed = [twiss[f"dnu{plane}/dp{order}"][0] for order in (2, 3)]
            expected = fodo_tune_derivatives(plane, thin=False)
            defect |= compare(
                f"{mode} FODO dnu{plane}/dp2, dnu{plane}/dp3", observed, expected, 1e-9, 1e-12
            )
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
    verdict(defect)

    ```

??? example "Helper: _cubic_optics.py"

    [Download _cubic_optics.py](_cubic_optics.py)

    ```python
    """Oracle for the higher-order chromaticities of a cell of cubic-order lens maps.

    Elegant's third-order maps keep each element's 2x2 transverse matrix to second order in the
    relative momentum offset delta (the cubic map). For such truncated maps the tune derivatives
    d^2 nu/d delta^2 and d^3 nu/d delta^3 follow from Cauchy's formula on a circle in the complex delta
    plane, independently of Elegant's trace-derivative formulas: if f is analytic, then
    f^(n)(0)/n! is the mean of f(z)/z^n over the circle |z| = RADIUS.
    """

    from __future__ import annotations

    import math

    import numpy as np

    POINTS, RADIUS = 64, 0.05
    Z = RADIUS * np.exp(2j * np.pi * np.arange(POINTS) / POINTS)


    def taylor(function, order: int = 2) -> list[np.ndarray]:
        """Taylor coefficients c_0 ... c_order of a matrix function of delta, analytic near 0."""
        values = np.array([function(z) for z in Z])
        return [np.mean(values / Z[:, None, None] ** n, axis=0) for n in range(order + 1)]


    def quad_matrix(k, length):
        """2x2 hard-edge quadrupole matrix, analytic in k (k < 0 defocuses); cos z and sin(z)/z are even."""
        z = np.sqrt(k + 0j) * length
        return np.array([[np.cos(z), np.sin(z) / z * length], [-z * np.sin(z) / length, np.cos(z)]])


    def drift_matrix(length):
        return np.array([[1.0, length], [0.0, 1.0]])


    def tune_derivatives(elements: list[list[np.ndarray]], periods: int = 1) -> tuple[float, float]:
        """(d^2 nu/d delta^2, d^3 nu/d delta^3) in cycles, for a cell repeated `periods` times.

        elements: Taylor coefficient lists in beam order. The tune is
        nu = periods * arccos(trace/2) / (2 pi), the trace taken of the cell polynomial map.
        """
        trace = np.empty(POINTS, dtype=complex)
        for i, z in enumerate(Z):
            cell = np.eye(2)
            for coefficients in elements:
                cell = sum(c * z**n for n, c in enumerate(coefficients)) @ cell
            trace[i] = np.trace(cell)
        tune = periods * np.arccos(trace / 2) / (2 * np.pi)
        second, third = (float(math.factorial(n) * np.mean(tune / Z**n).real) for n in (2, 3))
        return second, third


    def fodo_tune_derivatives(plane: str, thin: bool, periods: int = 1) -> tuple[float, float]:
        """Cell D QD D QF (quadrupole strengths 0.6 and -0.54 m^-2, or integrated for thin lenses).

        Thick lenses are 0.5 m long; the focusing sign flips in y. Drifts are 1 m. The gradient scales
        as 1/(1 + delta) (rigidity).
        """
        sign = 1 if plane == "x" else -1
        drift = taylor(lambda z: drift_matrix(1.0))
        elements = []
        for strength in (0.6, -0.54):
            k = sign * strength
            if thin:
                lens = taylor(lambda z, k=k: np.array([[1.0, 0.0], [-k / (1 + z), 1.0]]))
            else:
                lens = taylor(lambda z, k=k: quad_matrix(k / (1 + z), 0.5))
            elements += [lens, drift]
        return tune_derivatives(elements, periods)

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
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

