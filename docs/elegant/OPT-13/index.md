---
title: "OPT-13: Quick second chromaticity translates a supplied closed orbit twice"
hide:
  - toc
---

# OPT-13 — Quick second chromaticity translates a supplied closed orbit twice

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`compute_periodic_twiss`, twiss.cc:173–190](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L173-L190)<br>[`compute_twiss_parameters`, twiss.cc:2289–2293](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L2289-L2293)<br>[`computeHigherOrderChromaticities`, chrom.c:928–947](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/chrom.c#L928-L947)<br>[`accumulate_matrices`, compute_matrices.c:214–294](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L214-L294)<br>[`concat_matrices`, concat_mat.c:31–132](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/concat_mat.c#L31-L132)<br>[Twiss parameters on a closed orbit, elegant.tex:8084–8085](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8084-L8085)<br>[quick mode as a quicker way to the second-order chromaticity, elegant.tex:8147–8148](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8147-L8148)<br>[`MATR` map file format, matr.tex:1–20](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/matr.tex#L1-L20)<br>[the `T_MATR` case, compute_matrices.c:1576–1601](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1576-L1601) |

With `quick_higher_order_chromaticity=1`, Elegant shifts a one-turn map by the closed orbit after it has already been expanded about that orbit, corrupting the second chromaticity d²ν/dδ², where ν is tune and δ is relative momentum offset. For a prescribed cubic map with a closed orbit 10 mm off axis and a momentum-independent Jacobian there, the exact horizontal derivative is 0; quick mode gives −0.08750 and full mode −3.198×10⁻¹⁰. In this map, halving the displacement reduces the error fourfold, while an on-axis orbit or removal of the cubic term leaves only fitting noise.

??? example "Reproducer: OPT-13.py"

    [Download OPT-13.py](OPT-13.py)

    ```python
    """OPT-13: Quick second chromaticity translates a supplied closed orbit twice

    With `quick_higher_order_chromaticity=1`, Elegant shifts a one-turn map by the closed orbit after it
    has already been expanded about that orbit, corrupting the second chromaticity d²ν/dδ², where ν is
    tune and δ is relative momentum offset. For a prescribed cubic map with a closed orbit 10 mm off axis
    and a momentum-independent Jacobian there, the exact horizontal derivative is 0; quick mode gives
    −0.08750 and full mode −3.198×10⁻¹⁰. In this map, halving the displacement reduces the error
    fourfold, while an on-axis orbit or removal of the cubic term leaves only fitting noise.

    The script prescribes a stable cubic one-turn map with a fixed point at x = b for every δ and a
    Jacobian there that is the same rotation for every δ, so the tunes do not depend on δ. It runs
    `twiss_output` on the closed orbit in quick and in full mode, and compares the written d²νx/dδ²
    with 0 and, for quick mode, with the value the double translation predicts. Tracked particle pairs
    confirm the fixed point and the Jacobian; full mode, zero displacement and zero cubic term are the
    controls.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from _matr import TUNES, matrix_text, rotations

    P0 = 1000.0  # reference momentum, units of m_e c
    LENGTH = 10.0  # m
    BETA_X = 1.0  # m, horizontal beta function of the rotation A (alpha = 0), in slope coordinates
    DELTAS = (-0.01, 0.0, 0.01)
    PROBE = 1e-5  # half-step of the Jacobian probes
    COPIES = 2  # copies of each particle, so that each MPI worker tracks some
    ATOL = 2e-8  # trace-fit floor of the full calculation is 3.2e-10
    PREDICTION_ATOL = 1e-8

    COMMANDS = """\
    &run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=3,
     output=out.sdds &end
    &run_control n_steps=1, n_passes=1 &end
    &twiss_output filename=twiss.sdds, output_at_each_step=1, concat_order=3,
     higher_order_chromaticity=1, higher_order_chromaticity_points=11,
     higher_order_chromaticity_range=0.01, quick_higher_order_chromaticity={quick} &end
    &closed_orbit output=orbit.sdds, output_monitors_only=0, closed_orbit_accuracy=1e-14,
     closed_orbit_accuracy_requirement=1e-12, iteration_fraction=1 &end
    &sdds_beam input=beam.sdds &end
    &track soft_failure=0 &end
    """


    def prescribed_map(b, a):
        """MATR file of F(u, v, delta) = (b, 0) + A (u - b, v + a (u - b)^2 delta) in the x plane.

        A is the rotation by 2 pi nu_x (betatron tune 0.17) in (x, x') with beta_x = BETA_X = 1 m and
        alpha_x = 0, a = 100 1/m^2 gives the kick a (u - b)^2 delta in the slope v, and the y plane is
        the rotation with tune 0.23. Expanding (u - b)^2 = u^2 - 2 b u + b^2 gives the C, R, T and Q coefficients below. At
        u = b the map returns (b, 0) for every delta, and its Jacobian there is exactly A for every
        delta, so the tunes do not depend on delta.
        """
        r = rotations()
        c, s = r[0, 0], r[0, 1]
        constants = np.array([(1 - c) * b, s * b, 0, 0, LENGTH, 0.0])
        t = np.zeros((6, 6, 6))
        q = np.zeros((6, 6, 6, 6))
        for i, factor in ((0, s), (1, c)):  # rows of A acting on the kick a (u - b)^2 delta
            r[i, 5] = a * b * b * factor
            t[i, 5, 0] = -2 * a * b * factor
            q[i, 5, 0, 0] = a * factor
        return matrix_text(constants, r, t, q), rotations()[:2, :2]


    def probe_beam(b):
        """Four particles per delta at +-PROBE in x and in xp about the fixed point (b, 0)."""
        rows = []
        for delta in DELTAS:
            for dx, dxp in ((PROBE, 0), (-PROBE, 0), (0, PROBE), (0, -PROBE)):
                rows.append((b + dx, dxp, P0 * (1 + delta)))
        x, xp, p = np.repeat(np.array(rows), COPIES, axis=0).T
        return {"x": x, "xp": xp, "p": p, "particleID": np.arange(1, len(x) + 1)}


    def case(mode, b, a, quick):
        """(written d2nu_x/ddelta2, closed-orbit x, largest tracked Jacobian error)."""
        text, jacobian = prescribed_map(b, a)
        # Elegant upper-cases names in the lattice, so the map file must have an upper-case name.
        result = run(
            {
                "lattice.lte": "E: MATR,L=10,FILENAME=MAP.TXT,ORDER=3\nBL: LINE=(E)\n",
                "run.ele": COMMANDS.format(quick=quick),
                "MAP.TXT": text,
            },
            mode,
            {"beam.sdds": probe_beam(b)},
            outputs=("twiss.sdds", "orbit.sdds", "out.sdds"),
        )
        orbit = result["orbit.sdds"]
        if int(orbit["failed"][0]):
            sys.exit(f"{mode}: the closed-orbit search failed")
        out = particles(result["out.sdds"])
        if not (np.array_equal(out["x"], np.repeat(out["x"][::COPIES], COPIES))):
            sys.exit("Identical particles left the element differently.")
        # Symmetric differences about the fixed point give the transverse Jacobian at each delta.
        x, xp = (out[name][::COPIES].reshape(len(DELTAS), 4) for name in ("x", "xp"))
        tracked = np.empty((len(DELTAS), 2, 2))
        for row, column in enumerate((x, xp)):
            tracked[:, row, 0] = (column[:, 0] - column[:, 1]) / (2 * PROBE)  # d/dx
            tracked[:, row, 1] = (column[:, 2] - column[:, 3]) / (2 * PROBE)  # d/dxp
        return (
            float(result["twiss.sdds"]["dnux/dp2"][0]),
            float(orbit[0, "x"][0]),
            float(np.max(np.abs(tracked - jacobian))),
        )


    def compare(label, observed, expected, atol):
        """Print one case; return True when observed and expected disagree."""
        if not (np.isfinite(observed) and np.isfinite(expected)):
            sys.exit(f"{label}: nonfinite value, the comparison is not valid.")
        bad = abs(observed - expected) > atol
        print(f"{label}: observed={observed:.16g} expected={expected:.16g}")
        return bad


    defect = False
    for mode in modes():
        for b, a, quick in (
            (0.01, 100.0, 0),  # control: full calculation
            (0.0, 100.0, 1),  # control: closed orbit at the origin
            (0.01, 0.0, 1),  # control: no cubic term
            (0.005, 100.0, 1),  # half the displacement
            (0.01, 100.0, 1),
        ):
            label = f"{mode} b={b:g} a={a:g} {'quick' if quick else 'full'}"
            written, orbit_x, jacobian_error = case(mode, b, a, quick)
            if not np.isfinite([written, orbit_x, jacobian_error]).all():
                sys.exit(f"{label}: nonfinite written derivative, orbit or Jacobian error.")
            # The fixed point (b, 0) and the constant Jacobian are checked by tracking, independently of
            # the Twiss calculation.
            print(
                f"{label}: closed orbit x={orbit_x:.16g}, tracked Jacobian error={jacobian_error:.3g}"
            )
            if abs(orbit_x - b) > 1e-12 or jacobian_error > 3e-11:
                sys.exit("The prescribed map is not realized by tracking: invalid comparison.")
            # Exact answer: the tunes do not depend on delta, so d2nu/ddelta2 = 0. The double
            # translation evaluates the Jacobian at u = 2b. With chi = a beta_x b (the kick in the
            # slope coordinate, normalized by beta_x), trace = 2 cos(mu) + 2 chi sin(mu) delta
            # + O(delta^2) with mu = 2 pi nu_x; differentiating nu = arccos(trace / 2) / (2 pi) twice
            # gives d2nu/ddelta2 = -chi^2 cot(mu) / (2 pi).
            mu = 2 * np.pi * TUNES[0]
            chi = a * BETA_X * b
            shifted = -(chi**2) / np.tan(mu) / (2 * np.pi) if quick else 0.0
            bad = compare(f"{label} d2nu_x/ddelta2", written, 0.0, ATOL)
            if not (quick and b and a):
                if bad:
                    sys.exit("A control has nonzero second chromaticity: invalid comparison.")
            else:
                defect |= bad
                print(f"{label}: double translation predicts {shifted:.16g}")
                # A value that agrees with the exact zero is not this defect; only a nonzero value
                # must follow the double translation.
                if bad and abs(written - shifted) > PREDICTION_ATOL:
                    sys.exit("The written value does not follow the double translation.")
    print(f"comparison: absolute tolerance {ATOL:g}")
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

??? example "Helper: _matr.py"

    [Download _matr.py](_matr.py)

    ```python
    """Prescribed-map helpers for Elegant's MATR element, which reads a map from a text file.

    `matrix_text` writes the file in the layout MATR expects: the constant vector C, the rows of the
    first-order matrix R, the second-order coefficients T and the third-order coefficients Q. Elegant
    stores the symmetric coefficients once, with the indices ordered j >= k (>= l), and each entry
    holds the total coefficient of the monomial x_j x_k (x_l) in coordinate i; coordinates are
    (x, x', y, y', s, delta), indexed from zero here.
    """

    from __future__ import annotations

    import numpy as np

    TUNES = (0.17, 0.23)  # horizontal and vertical betatron tunes of the rotation helper


    def rotations(tunes: tuple[float, float] = TUNES) -> np.ndarray:
        """6x6 matrix: rotation by 2 pi nu in (x, x') and (y, y') with beta = 1 m and alpha = 0."""
        matrix = np.eye(6)
        for j, tune in zip((0, 2), tunes, strict=True):
            c, s = np.cos(2 * np.pi * tune), np.sin(2 * np.pi * tune)
            matrix[j : j + 2, j : j + 2] = [[c, s], [-s, c]]
        return matrix


    def matrix_text(
        c: np.ndarray, r: np.ndarray, t: np.ndarray | None = None, q: np.ndarray | None = None
    ) -> str:
        """Contents of a MATR map file; t has shape (6, 6, 6) and q (6, 6, 6, 6), zero when omitted."""
        t = np.zeros((6, 6, 6)) if t is None else t
        q = np.zeros((6, 6, 6, 6)) if q is None else q
        lines = ["C: " + " ".join(f"{v:.17g}" for v in c)]
        lines += [f"R{i + 1}: " + " ".join(f"{v:.17g}" for v in r[i]) for i in range(6)]
        for i in range(6):
            for j in range(6):
                values = " ".join(f"{t[i, j, k]:.17g}" for k in range(j + 1))
                lines.append(f"T{i + 1}{j + 1}: {values}")
        for i in range(6):
            for j in range(6):
                for k in range(j + 1):
                    values = " ".join(f"{q[i, j, k, l]:.17g}" for l in range(k + 1))
                    lines.append(f"Q{i + 1}{j + 1}{k + 1}: {values}")
        return "\n".join(lines) + "\n"

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

