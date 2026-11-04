---
title: "OPT-05: Zero-length MULT quadrupole cubic matrix contains infinities"
hide:
  - toc
---

# OPT-05 — Zero-length MULT quadrupole cubic matrix contains infinities

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`mult_matrix`, zero-length substitution, compute_matrices.c:3249–3250](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L3249-L3250)<br>[calls `quadrupole_matrix(KnL / length, length, …)`, compute_matrices.c:3262–3263](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L3262-L3263)<br>[`quadrupole_matrix`, horizontal-plane terms divided by `cos(2kl) - 1`, quad_matrix.c:160 and 172](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/quad_matrix.c#L160-L172)<br>[vertical-plane terms divided by `1 - cosh(2kl)`, quad_matrix.c:178 and 190](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/quad_matrix.c#L178-L190)<br>[MULT parameters (`L` defaults to 0, `KNL` is the integrated strength, `MATRIX_ORDER`), track_data.c:505–517](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L505-L517)<br>[the strength definition K_nL, doc/mult.tex:5–15](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/mult.tex#L5-L15) |

The third-order transfer map of a zero-length quadrupole, MULT with L=0, ORDER=1 and integrated strength KNL = 0.6 m⁻¹, contains infinite cubic coefficients. Concatenation contaminates 336 map coefficients with NaN, including R₂₁, which should be −0.6000 m⁻¹; finite second-order and short-length controls isolate the singular thin-lens construction. For a thin-lens FODO cell using cubic lens maps, horizontal tune νₓ has d²νₓ/dδ² and d³νₓ/dδ³ equal to NaN instead of 0.2030 and −0.06786, where δ is the relative momentum offset. Users requesting third-order optics for zero-length MULT quadrupoles are affected.

??? example "Reproducer: OPT-05.py"

    [Download OPT-05.py](OPT-05.py)

    ```python
    """OPT-05: Zero-length MULT quadrupole cubic matrix contains infinities

    The third-order transfer map of a zero-length quadrupole, MULT with L=0, ORDER=1 and integrated
    strength KNL = 0.6 m⁻¹, contains infinite cubic coefficients. Concatenation contaminates 336 map
    coefficients with NaN, including R₂₁, which should be −0.6000 m⁻¹; finite second-order and
    short-length controls isolate the singular thin-lens construction. For a thin-lens FODO cell using
    cubic lens maps, horizontal tune νₓ has d²νₓ/dδ² and d³νₓ/dδ³ equal to NaN instead of 0.2030 and
    −0.06786, where δ is the relative momentum offset. Users requesting third-order optics for
    zero-length MULT quadrupoles are affected.

    The script builds a thin-lens quadrupole `MULT,L=0,ORDER=1,KNL=0.6` in serial Elegant (and with
    `--mpi` in 3-rank Pelegant) with a third-order map, reads the element's raw matrix from the matrix
    printout and the exported concatenated map, and compares them with the ideal thin-lens kick. The
    same lens in a thin-lens FODO cell gives the higher-order chromaticities, which are compared with
    the derivatives of the tune of the cubic lens maps (truncated at δ²). Both control maps (a
    second-order map of the same lens and a third-order map of a 1 μm long lens) must be finite, and
    their R₂₁ coefficients must match the zero-length and finite-length hard-edge expectations,
    respectively.
    """

    import re
    import sys

    import numpy as np
    from _cubic_optics import fodo_tune_derivatives
    from _elegant import modes, run, verdict

    KNL = 0.6  # integrated strength, m^-1
    RTOL, ATOL = 1e-6, 1e-12


    def single_lens(length, matrix_order):
        lattice = f"E: MULT,L={length!r},ORDER=1,KNL={KNL!r}{matrix_order}\nBL: LINE=(E)\n"
        # The element matrix is printed to standard output (the "individual matrices" printout),
        # because the exported SDDS map holds only the concatenated product.
        commands = (
            "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&matrix_output SDDS_output=map.sdds, SDDS_output_order=3, printout=/dev/stdout,\n"
            " printout_order=3, individual_matrices=1 &end\n"
            "&bunched_beam n_particles_per_bunch=12 &end\n"
            "&track &end\n"
        )
        return {"lattice.lte": lattice, "run.ele": commands}


    def raw_cubic(log):
        """Cubic coefficients U_ijkl of the element's own matrix, from its printout.

        A line `Qijk:` lists U_ijk1 ... U_ijkk. The element section ends where the effective
        (concatenated) matrix section begins.
        """
        start = log.index("E#1 pInput")
        section = log[start : log.index("Effective element matrix", start)]
        cubic = {}
        for label, values in re.findall(
            r"^Q(\d{3}):((?:[ \t]+\S+)+)[ \t]*$", section, flags=re.MULTILINE
        ):
            for l, value in enumerate(values.split(), start=1):
                cubic[f"U{label}{l}"] = float(value)
        return cubic


    def run_lens(mode, length, matrix_order=""):
        out = run(single_lens(length, matrix_order), mode, outputs=("map.sdds",))
        exported = out["map.sdds"]
        row = len(exported[0, "ElementName"]) - 1  # last row: the concatenated map through the lens
        return {c.name: exported[0, c.name][row] for c in exported.columns}, raw_cubic(out["log"])


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.atleast_1d(values)) + "]"


    def compare(label, observed, expected, rtol=RTOL, atol=ATOL):
        """Print one case; return True when observed and expected disagree (NaN counts as disagreeing)."""
        observed, expected = np.atleast_1d(observed), np.atleast_1d(expected)
        bad = not np.allclose(observed, expected, rtol=rtol, atol=atol)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    def ideal_lens_terms():
        """Thin-lens kick to first order in the transverse coordinates.

        x' -> x' - KNL x/(1 + delta), y' -> y' + KNL y/(1 + delta); x is unchanged. An exact
        finite-angle canonical-momentum kick is not an additive slope kick; the coefficients below
        are the transverse-linear chromatic terms. The cubic map keeps the delta dependence through
        delta^2, 1/(1 + delta) = 1 - delta + delta^2, so the xp row has R21 = -KNL, T261 = +KNL
        (x delta) and U2661 = -KNL (x delta^2). The position row has no momentum dependence:
        R11 = 1, U1441 = 0.
        """
        return {"R11": 1.0, "R21": -KNL, "R43": KNL, "T261": KNL, "U2661": -KNL, "U1441": 0.0}


    def thick_lens_r21(length):
        """Exact hard-edge quadrupole R21 = -k sin(k L) with k^2 = KNL/L, for the finite-length control."""
        k = np.sqrt(KNL / length)
        return -k * np.sin(k * length)


    FODO_LATTICE = (
        "QF: MULT,ORDER=1,KNL=0.6\nQD: MULT,ORDER=1,KNL=-0.54\nD: DRIF,L=1\nBL: LINE=(QF,D,QD,D)\n"
    )
    FODO_COMMANDS = (
        "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&twiss_output filename=twiss.sdds, concat_order=3, higher_order_chromaticity=1,\n"
        " higher_order_chromaticity_points=11, higher_order_chromaticity_range=0.01 &end\n"
        "&bunched_beam n_particles_per_bunch=12 &end\n"
        "&track &end\n"
    )

    defect = False
    for mode in modes():
        # Controls first: a second-order map of the zero-length lens, and a third-order map of a
        # 1 micron lens of the same integrated strength, must be finite, and their R21 must match the
        # zero-length and finite-length hard-edge expectations, respectively.
        exported, raw = run_lens(mode, 0, ",MATRIX_ORDER=2")
        finite = all(np.isfinite(v) for v in exported.values() if isinstance(v, (float, np.floating)))
        if not finite or compare(f"{mode} control order 2, R21", exported["R21"], -KNL):
            sys.exit("The second-order control is not finite and exact: the comparison is not valid.")
        short = 1e-6
        exported, raw = run_lens(mode, short)
        finite = all(np.isfinite(v) for v in exported.values() if isinstance(v, (float, np.floating)))
        if not finite or compare(
            f"{mode} control L=1e-6 order 3, R21", exported["R21"], thick_lens_r21(short)
        ):
            sys.exit("The short-lens control is not finite and exact: the comparison is not valid.")

        # The defect: the zero-length lens at third order.
        exported, raw = run_lens(mode, 0)
        ideal = ideal_lens_terms()
        names = list(ideal)
        observed = [exported[n] for n in names]
        print(f"{mode} third-order map {names}")
        defect |= compare(f"{mode} thin lens, exported map", observed, [ideal[n] for n in names])
        bad_columns = sorted(
            n for n, v in exported.items() if isinstance(v, (float, np.floating)) and not np.isfinite(v)
        )
        print(
            f"{mode} thin lens, nonfinite columns of the exported map: observed={len(bad_columns)} expected=0"
        )
        defect |= bool(bad_columns)
        bad_raw = {n: v for n, v in raw.items() if not np.isfinite(v)}
        print(
            f"{mode} thin lens, nonfinite raw cubic coefficients of the element: "
            f"observed={len(bad_raw)} {bad_raw} expected=0"
        )
        defect |= bool(bad_raw)
        print(f"{mode} thin lens, raw U1441: observed={raw['U1441']} expected=0.0")

        # Consequence: the higher-order chromaticities of a thin-lens FODO cell.
        twiss = run(
            {"lattice.lte": FODO_LATTICE, "run.ele": FODO_COMMANDS}, mode, outputs=("twiss.sdds",)
        )["twiss.sdds"]
        for plane in "xy":
            observed = [twiss[f"dnu{plane}/dp{order}"][0] for order in (2, 3)]
            defect |= compare(
                f"{mode} FODO dnu{plane}/dp2, dnu{plane}/dp3",
                observed,
                fodo_tune_derivatives(plane, thin=True),
                1e-9,
                1e-12,
            )
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}; chromaticities rtol=1e-9, atol=1e-12")
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

