---
title: "OPT-07: Higher chromaticities mix cell and repeated-period phases"
hide:
  - toc
---

# OPT-07 — Higher chromaticities mix cell and repeated-period phases

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`computeHigherOrderChromaticities`, one-cell map construction and trace, chrom.c:928–942](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/chrom.c#L928-L942)<br>[trace fit and chromaticity formulas, chrom.c:949–956](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/chrom.c#L949-L956)<br>[`propagate_twiss_parameters` has already multiplied by `n_periods`, twiss.cc:806–807](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L806-L807)<br>[multiplies the resulting `chrom2` and `chrom3` by `n_periods` again, twiss.cc:2296–2299](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L2296-L2299)<br>[the same scaling, twiss.cc:2344–2347](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L2344-L2347)<br>[`dnux/dp2` and `dnux/dp3` as d²ν/dδ² and d³ν/dδ³, doc/elegant.tex:8228–8231](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8228-L8231)<br>[`n_periods` field, src/twiss.nl:77](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.nl#L77) |

With `higher_order_chromaticity=1` and `n_periods=2`, Elegant's `twiss_output` converts single-cell trace derivatives using the repeated lattice's phase, producing wrong second and third chromaticities. On a continuous phase branch, a stable cell's tune ν and every momentum derivative must scale with the period count. For 0.5 m `KQUAD`s with K1 = 0.6 and −0.54 m⁻² and 1 m drifts, one period gives d²νₓ/dδ² = 0.1268 and d³νₓ/dδ³ = −0.08637, where δ is the relative momentum offset. Two periods require 0.2537 and −0.1727, but the output gives 0.1865 and −0.3286; the tune and first chromaticity scale correctly. Users computing higher chromaticities of repeated cells are affected.

??? example "Reproducer: OPT-07.py"

    [Download OPT-07.py](OPT-07.py)

    ```python
    """OPT-07: Higher chromaticities mix cell and repeated-period phases

    With `higher_order_chromaticity=1` and `n_periods=2`, Elegant's `twiss_output` converts single-cell
    trace derivatives using the repeated lattice's phase, producing wrong second and third
    chromaticities. On a continuous phase branch, a stable cell's tune ν and every momentum derivative
    must scale with the period count. For 0.5 m `KQUAD`s with K1 = 0.6 and −0.54 m⁻² and 1 m drifts, one
    period gives d²νₓ/dδ² = 0.1268 and d³νₓ/dδ³ = −0.08637, where δ is the relative momentum offset.
    Two periods require 0.2537 and −0.1727, but the output gives 0.1865 and −0.3286; the tune and first
    chromaticity scale correctly. Users computing higher chromaticities of repeated cells are affected.

    The script computes the tune and its first three momentum derivatives of a FODO cell of
    tracking-derived `KQUAD` quadrupoles with `n_periods` 1 and 2 in serial Elegant (and with `--mpi` in
    3-rank Pelegant). It compares the two-period values with the exact scaling by the period count, and
    the one-period values with the exact hard-edge maps truncated at second order in δ. The tune and the
    first chromaticity are the controls, since they scale correctly.
    """

    import sys

    import numpy as np
    from _cubic_optics import fodo_tune_derivatives
    from _elegant import modes, run, verdict

    RTOL, ATOL = 1e-9, 1e-12  # exact scaling by the period count
    ORACLE_RTOL = 1e-5  # one-period values against the oracle: finite-difference accuracy of Elegant

    LATTICE = (
        "QF: KQUAD,N_SLICES=128,INTEGRATION_ORDER=4,TRACKING_MATRIX=3,L=0.5,K1=0.6,TILT=0\n"
        "QD: KQUAD,N_SLICES=128,INTEGRATION_ORDER=4,TRACKING_MATRIX=3,L=0.5,K1=-0.54,TILT=0\n"
        "D: DRIF,L=1\nBL: LINE=(QF,D,QD,D)\n"
    )
    NAMES = ("nu{}", "dnu{}/dp", "dnu{}/dp2", "dnu{}/dp3")  # tune and its derivatives with respect to δ


    def commands(periods):
        return (
            "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            f"&twiss_output filename=twiss.sdds, concat_order=3, n_periods={periods},\n"
            " higher_order_chromaticity=1, higher_order_chromaticity_points=11,\n"
            " higher_order_chromaticity_range=0.01 &end\n"
            "&bunched_beam n_particles_per_bunch=12 &end\n"
            "&track &end\n"
        )


    def derivatives(mode, periods, plane):
        twiss = run(
            {"lattice.lte": LATTICE, "run.ele": commands(periods)}, mode, outputs=("twiss.sdds",)
        )["twiss.sdds"]
        return np.array([twiss[name.format(plane)][0] for name in NAMES])


    def fmt(values):
        return "[" + ", ".join(f"{v:.15g}" for v in np.ravel(values)) + "]"


    def compare(label, observed, expected, rtol=RTOL, atol=ATOL):
        """Print one case; return True when observed and expected disagree."""
        observed, expected = np.asarray(observed), np.asarray(expected)
        if not (np.isfinite(observed).all() and np.isfinite(expected).all()):
            sys.exit(f"{label}: nonfinite value, the comparison is not valid.")
        bad = not np.allclose(observed, expected, rtol=rtol, atol=atol)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    defect = False
    print("columns: nu, dnu/dp, dnu/dp2, dnu/dp3 (cycles); expected = n_periods x one-period value")
    for mode in modes():
        for plane in "xy":
            one, two = derivatives(mode, 1, plane), derivatives(mode, 2, plane)
            # Control: the one-period second and third derivatives agree with the exact hard-edge
            # maps truncated at second order in delta, so the one-period calculation is sound.
            oracle = fodo_tune_derivatives(plane, thin=False)
            if compare(
                f"{mode} {plane} control, one period vs exact maps", one[2:], oracle, ORACLE_RTOL, 1e-6
            ):
                sys.exit("The one-period values disagree with the oracle: the comparison is not valid.")
            # The period count multiplies the tune and every momentum derivative of a stable cell on a
            # fixed phase branch. Control: tune and first chromaticity do scale.
            if compare(f"{mode} {plane} control, nu and dnu/dp, two periods", two[:2], 2 * one[:2]):
                sys.exit("The tune/chromaticity control does not scale: the comparison is not valid.")
            defect |= compare(f"{mode} {plane} two periods, dnu/dp2 and dnu/dp3", two[2:], 2 * one[2:])
            # The same invariant against the oracle with n_periods = 2.
            compare(
                f"{mode} {plane} two periods vs exact maps",
                two[2:],
                fodo_tune_derivatives(plane, thin=False, periods=2),
                ORACLE_RTOL,
                1e-6,
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

