---
title: "SCT-08: Smooth IBS shifts the beam centroid when there is dispersion and a momentum offset"
hide:
  - toc
---

# SCT-08 — Smooth IBS shifts the beam centroid when there is dispersion and a momentum offset

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [smooth update, ibs_tracking.c:513–515](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibs_tracking.c#L513-L515)<br>[centroid and scaling origin, ibs_tracking.c:665–691](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibs_tracking.c#L665-L691) |

With the smooth method (SMOOTH=1), IBSCATTER measures the beam centroid c and scales the transverse coordinates about c + ημ, where η is the dispersion and μ the beam's mean fractional momentum offset δ. The centroid already contains the dispersive displacement ημ, so the origin is displaced twice and the centroid moves by (1−a)ημ, where a is the amplitude growth factor, instead of staying fixed. In a closed ring with η = 14.29 m and μ = 1×10⁻⁴, an amplitude growth of 0.8501% moves the mean horizontal position by −12.14 µm instead of 0; with μ = 0 it stays put. Users of smooth IBS with dispersion and an off-momentum beam are affected.

??? example "Reproducer: SCT-08.py"

    [Download SCT-08.py](SCT-08.py)

    ```python
    """SCT-08: Smooth IBS shifts the beam centroid when there is dispersion and a momentum offset

    With the smooth method (SMOOTH=1), IBSCATTER measures the beam centroid c and scales the transverse
    coordinates about c + ημ, where η is the dispersion and μ the beam's mean fractional momentum offset
    δ. The centroid already contains the dispersive displacement ημ, so the origin is displaced twice
    and the centroid moves by (1−a)ημ, where a is the amplitude growth factor, instead of staying fixed.
    In a closed ring with η = 14.29 m and μ = 1×10⁻⁴, an amplitude growth of 0.8501% moves the mean
    horizontal position by −12.14 µm instead of 0; with μ = 0 it stays put. Users of smooth IBS with
    dispersion and an off-momentum beam are affected.

    The script tracks a 256-particle beam once around a ring of sixteen identical sector bends, with
    horizontal dispersion η = 14.29 m, ending in an IBSCATTER with the smooth method. It compares the
    mean horizontal position with that of a run with FACTOR=0, once with a mean momentum offset μ =
    1×10⁻⁴ and once with μ = 0 as the control. Centered emittance growth leaves the centroid where it
    is, so the expected shift is zero; the script also evaluates the shift (1−a)ημ that the displaced
    scaling origin predicts.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _ibs_ring import ETA_X, FACTOR, P0, beam, track

    ATOL = 1e-15  # m; positions are about 1.4e-3 m, so rounding is about 3e-19 m
    FORMULA_RTOL, FORMULA_ATOL = 1e-9, 2e-18  # shift against the displaced-origin prediction


    def fmt(values):
        return "[" + ", ".join(f"{v:.12g}" for v in np.ravel(values)) + "]"


    def compare(label, observed, expected, rtol=0.0, atol=ATOL):
        bad = not np.allclose(observed, expected, rtol=rtol, atol=atol)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    defect = False
    print("mean x shift in m caused by the IBS update (FACTOR minus FACTOR=0)")
    for mode in modes():
        for label, mean_delta in (("mean delta 0 control", 0.0), ("mean delta 1e-4", 1e-4)):
            columns = beam(mean_delta)
            reference = track(mode, columns, 0.0)
            out = track(mode, columns, FACTOR)
            shift = np.mean(out["x"]) - np.mean(reference["x"])
            # The mean momentum offset after the update must be unchanged, or the shift is not the
            # dispersive displacement of this offset.
            mu = np.mean(out["p"]) / P0 - 1
            if abs(mu - (np.mean(reference["p"]) / P0 - 1)) > 1e-12:
                sys.exit(f"{mode}: the update changed the mean momentum offset")
            # Coordinates scale by a about the origin c + eta mu, where c is the measured centroid,
            # so the centroid moves by (1 - a) eta mu, with a**2 the variance ratio of x.
            a = np.sqrt(np.var(out["x"]) / np.var(reference["x"]))
            bad = compare(f"{mode} {label}", shift, 0.0)
            if compare(
                f"{mode} {label}, displaced-origin prediction",
                shift,
                (1 - a) * ETA_X * mu + 0.0,
                FORMULA_RTOL,
                FORMULA_ATOL,
            ):
                sys.exit(f"{mode}: the shift does not follow (1-a)*eta*mu")
            if mean_delta == 0.0 and bad:
                sys.exit(f"{mode}: the control moved the centroid: the comparison is not valid")
            defect |= bad
    print(f"comparison: atol={ATOL:g} m")
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

??? example "Helper: _ibs_ring.py"

    [Download _ibs_ring.py](_ibs_ring.py)

    ```python
    """Shared setup for the smooth-IBS reproducers: a closed ring of identical sector bends with one IBSCATTER.

    The ring has 16 identical SBENs (radius 10 m, K1 = -0.003 m^-2) and an IBSCATTER at its end. An
    SBEN of radius R focuses with k_x = 1/R^2 + K1 and k_y = -K1, so a ring of identical bends has
    constant matched beta = 1/sqrt(k), alpha = 0, and horizontal dispersion eta = (1/R)/k_x, eta' = 0.
    The beam is 256 particles with exactly the requested second moments, matched to this optics.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import particles, run

    C = 299792458.0  # SDDS constants.h
    P0 = 800.0  # reference momentum, units of m_e c
    RADIUS, K1 = 10.0, -0.003  # m, 1/m^2
    KX, KY = 1 / RADIUS**2 + K1, -K1
    BETA_X, BETA_Y = 1 / np.sqrt(KX), 1 / np.sqrt(KY)  # m
    ETA_X = (1 / RADIUS) / KX  # m
    EX, EY, SIGMA_DELTA, SIGMA_Z = 2e-9, 3e-10, 1e-5, 0.004  # m, m, 1, m
    COUNT = 256
    # Largest applied fractional change of any plane's variance or spread is 2% at this FACTOR, which
    # keeps the square-root scale factors of the smooth update near 1.
    FACTOR = 0.09465332535958268


    def beam(mean_delta):
        """Matched ring beam with mean momentum offset `mean_delta`; dispersive offset eta*delta in x.

        Orthonormalized normal samples, scaled by sqrt(COUNT), have zero mean and unit covariance, so the
        sample emittances, momentum spread and length equal the requested ones.
        """
        rng = np.random.default_rng(832)
        s = rng.normal(size=(COUNT, 6))
        s -= s.mean(axis=0)
        s = np.linalg.qr(s)[0] * np.sqrt(COUNT)
        delta = mean_delta + SIGMA_DELTA * s[:, 5]
        return {
            "x": np.sqrt(EX * BETA_X) * s[:, 0] + ETA_X * delta,
            "xp": np.sqrt(EX / BETA_X) * s[:, 1],
            "y": np.sqrt(EY * BETA_Y) * s[:, 2],
            "yp": np.sqrt(EY / BETA_Y) * s[:, 3],
            "t": SIGMA_Z / C * s[:, 4],
            "p": P0 * (1 + delta),
        }


    def track(mode, columns, factor, planes=(1, 1, 1)):
        """Track `columns` once around the ring; `planes` are the DO_X, DO_Y, DO_Z switches."""
        lattice = (
            "Q: CHARGE,TOTAL=1e-9\n"
            f"B: SBEN,L={2 * np.pi * RADIUS / 16!r},ANGLE={2 * np.pi / 16!r},K1={K1!r},E1=0,E2=0\n"
            "I: IBSCATTER,ISRING=1,BUNCHED_BEAM_MODE=0,SMOOTH=1,FORCE_MATCHED_TWISS=1,"
            f"FACTOR={factor!r},DO_X={planes[0]},DO_Y={planes[1]},DO_Z={planes[2]}\n"
            "BL: LINE=(Q,16*B,I)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
            "output=out.sdds, default_order=2 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&twiss_output matched=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})
        out = particles(result["out.sdds"])
        if len(out["p"]) != COUNT:
            raise SystemExit(f"{mode}: {COUNT - len(out['p'])} particles lost")
        return out


    def spreads(out):
        """Variances of x, xp, y, yp and rms momentum deviation of the tracked beam."""
        return np.array(
            [np.var(out["x"]), np.var(out["xp"]), np.var(out["y"]), np.var(out["yp"])]
            + [np.std(out["p"])]
        )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

