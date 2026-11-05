---
title: "OPT-14: Slice normalized emittances omit the mean momentum offset"
hide:
  - toc
---

# OPT-14 — Slice normalized emittances omit the mean momentum offset

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P1 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`performSliceAnalysis`, whole-beam values, sliceAnalysis.c:543–546](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sliceAnalysis.c#L543-L546)<br>[the per-slice values, sliceAnalysis.c:594–601](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sliceAnalysis.c#L594-L601)<br>[`exact_normalized_emittance` states that the approximate normalized emittance is ε⟨βγ⟩, elegant.tex:4143–4144](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4143-L4144)<br>[slice analysis always uses that approximate computation, elegant.tex:7572–7573](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7572-L7573) |

Elegant's `slice_analysis` normalizes each longitudinal slice's geometric emittance with the reference momentum instead of the slice's mean momentum, while the whole-beam normalization includes the mean momentum offset. For three monoenergetic slices at 900 mₑc with geometric emittance 2.400×10⁻⁷ m, the physical normalized emittance is 2.160×10⁻⁴ m in both planes. Reference momenta of 450 and 1350 mₑc produce 1.080×10⁻⁴ and 3.240×10⁻⁴ m; even references 1% below or above the slice momentum give −1% or +1% errors, and the dispersion-corrected values behave identically.

??? example "Reproducer: OPT-14.py"

    [Download OPT-14.py](OPT-14.py)

    ```python
    """OPT-14: Slice normalized emittances omit the mean momentum offset

    Elegant's `slice_analysis` normalizes each longitudinal slice's geometric emittance with the
    reference momentum instead of the slice's mean momentum, while the whole-beam normalization includes
    the mean momentum offset. For three monoenergetic slices at 900 mₑc with geometric emittance
    2.400×10⁻⁷ m, the physical normalized emittance is 2.160×10⁻⁴ m in both planes. Reference momenta of
    450 and 1350 mₑc produce 1.080×10⁻⁴ and 3.240×10⁻⁴ m; even references 1% below or above the slice
    momentum give −1% or +1% errors, and the dispersion-corrected values behave identically.

    The script tracks 24 particles through a zero-length drift in serial Elegant and runs
    `slice_analysis` with three slices, for reference momenta of 450, 891, 900, 909 and 1350 mₑc. The
    particles are symmetric phase-space quartets whose covariances vanish exactly, so the geometric
    emittance and the normalized emittance follow from the construction without any estimator noise.
    The whole-beam emittance of the same run and the slice values at the reference 900 mₑc, which
    equals the beam momentum, are the controls.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    P = 900.0  # momentum of every particle, units of m_e c
    A, B = 8e-4, 3e-4  # m, rad: the phase-space quartet is x = +-A, xp = +-B in all four combinations
    REFERENCES = (450.0, 891.0, 900.0, 909.0, 1350.0)
    RTOL = 1e-12

    # Three slices in time of eight particles: two copies of the quartet each. In y the quartet is
    # scaled to (2 x, xp / 2), which has the same emittance. The mean momentum offset is the same in
    # every slice, so the beam is monoenergetic and has no dispersion.
    BEAM = {
        "x": np.tile([A, A, -A, -A], 6),
        "xp": np.tile([B, -B, B, -B], 6),
        "t": np.repeat([0.0, 2e-10, 4e-10], 8),
        "p": np.full(24, P),
    }
    BEAM["y"] = 2 * BEAM["x"]
    BEAM["yp"] = BEAM["xp"] / 2

    LATTICE = "K: EDRIFT,L=0\nBL: LINE=(K)\n"


    def commands(reference):
        return (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={reference!r}, "
            "output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&slice_analysis output=slices.sdds, n_slices=3, final_values_only=0 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )


    def final(slices, name):
        """Value of a column on the last pass of the run (the only pass here)."""
        return float(slices[0, name][-1])


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected disagree."""
        bad = not np.allclose(observed, expected, rtol=RTOL, atol=0)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    defect = False
    print("columns: enx, eny, ecnx, ecny (m) of slice 1; the slices are identical by construction")
    for mode in modes():
        for reference in REFERENCES:
            result = run(
                {"lattice.lte": LATTICE, "run.ele": commands(reference)},
                mode,
                {"beam.sdds": BEAM},
                outputs=("slices.sdds", "out.sdds"),
            )
            slices = result["slices.sdds"]
            # Within a quartet the means and the x-xp covariance vanish exactly, so the geometric
            # emittance is sqrt(<x^2><xp^2>) = A*B = 2.4e-7 m (the y quartet is the same product). The
            # beam is monoenergetic, so the dispersion correction vanishes and the normalized emittance
            # of the physical momentum P is <beta gamma> * A*B = P*A*B = 2.16e-4 m in both planes.
            expected = P * A * B
            whole = [final(slices, f"{prefix}{plane}") for prefix in ("en", "ecn") for plane in "xy"]
            if compare(f"{mode} reference={reference:g} whole beam (control)", whole, [expected] * 4):
                sys.exit("The whole-beam control disagrees with the quartet value: invalid comparison.")
            columns = [f"{prefix}{plane}" for prefix in ("en", "ecn") for plane in "xy"]
            per_slice = np.array(
                [
                    [final(slices, f"{name}Slice{number:02d}") for name in columns]
                    for number in (1, 2, 3)
                ]
            )
            if not np.isfinite(per_slice).all():
                sys.exit(f"{mode} reference={reference:g}: nonfinite slice emittance.")
            counts = [int(final(slices, f"particlesSlice{number:02d}")) for number in (1, 2, 3)]
            if counts != [8, 8, 8] or not np.array_equal(per_slice, np.tile(per_slice[0], (3, 1))):
                sys.exit(f"{mode}: slices differ or are not populated as designed: {counts}")
            mismatch = compare(f"{mode} reference={reference:g} slice 1", per_slice[0], [expected] * 4)
            if reference == P:
                # Control: with the reference equal to the beam momentum, every slice, plane and
                # emittance (twelve values) must equal the physical value.
                if not np.allclose(per_slice, expected, rtol=RTOL, atol=0):
                    sys.exit("The reference = beam momentum control disagrees: invalid comparison.")
            else:
                defect |= mismatch
            print(
                f"{mode} reference={reference:g}: slice/expected={per_slice[0, 0] / expected:.16g},"
                f" reference/beam momentum={reference / P:.16g}"
            )
    print(f"comparison: rtol={RTOL:g}")
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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |

