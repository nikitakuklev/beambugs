---
title: "SCT-07: Smooth IBS ignores the plane switches DO_X, DO_Y and DO_Z"
hide:
  - toc
---

# SCT-07 — Smooth IBS ignores the plane switches DO_X, DO_Y and DO_Z

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [random-method plane gates, ibs_tracking.c:479–489](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibs_tracking.c#L479-L489)<br>[smooth update, ibs_tracking.c:513–516](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibs_tracking.c#L513-L516)<br>[IBSCATTER parameters, track_data.c:2512–2515](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2512-L2515) |

With the smooth method (SMOOTH=1), an IBSCATTER element applies the intrabeam-scattering growth to all three planes whatever the plane switches DO_X, DO_Y and DO_Z say. With all three switches at 0 the beam should leave the element unchanged, as it does for FACTOR=0, and with one switch at 1 only that plane should change. In a closed ring of 16 identical bends and 256 particles, every switch setting gives the same output as the fully enabled element: the horizontal variance grows by 1.707% and the rms momentum deviation by 2.000%, where the all-off case should show 0%.

??? example "Reproducer: SCT-07.py"

    [Download SCT-07.py](SCT-07.py)

    ```python
    """SCT-07: Smooth IBS ignores the plane switches DO_X, DO_Y and DO_Z

    With the smooth method (SMOOTH=1), an IBSCATTER element applies the intrabeam-scattering growth to
    all three planes whatever the plane switches DO_X, DO_Y and DO_Z say. With all three switches at 0
    the beam should leave the element unchanged, as it does for FACTOR=0, and with one switch at 1 only
    that plane should change. In a closed ring of 16 identical bends and 256 particles, every switch
    setting gives the same output as the fully enabled element: the horizontal variance grows by 1.707%
    and the rms momentum deviation by 2.000%, where the all-off case should show 0%.

    The script tracks a 256-particle beam once around a ring of sixteen identical sector bends, with
    constant matched beta functions, ending in an IBSCATTER with the smooth method and a growth-rate
    factor that makes the largest applied change 2%. It compares the beam's variances and momentum
    spread with those of a run with FACTOR=0, for all planes switched off and for each plane alone. A
    plane that is off must leave its spread equal to the FACTOR=0 value, and a plane that is on must
    change it exactly as in the run with all three planes on, which is the control.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _ibs_ring import FACTOR, beam, spreads, track

    RTOL = 1e-12  # relative; serial runs agree exactly and MPI reductions differ at about 1e-16
    MEAN_DELTA = 1e-4


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.ravel(values)) + "]"


    def compare(label, observed, expected):
        bad = not np.allclose(observed, expected, rtol=RTOL, atol=0)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    def checked(spread, positive=False):
        """The spreads of a run; exits when they are not finite (and, for a reference, positive)."""
        if not np.all(np.isfinite(spread)) or (positive and not np.all(spread > 0)):
            sys.exit(f"invalid spreads {fmt(spread)}: the comparison is not valid")
        return spread


    columns = beam(MEAN_DELTA)
    defect = False
    print("spread ratios to the FACTOR=0 run: var(x), var(x'), var(y), var(y'), rms(p)")
    for mode in modes():
        reference = checked(spreads(track(mode, columns, 0.0)), positive=True)
        full = checked(spreads(track(mode, columns, FACTOR))) / reference
        # The control must show a visible update in x and in p, or the comparison is empty.
        if full[0] < 1.01 or full[4] < 1.01:
            sys.exit(f"{mode}: the all-planes update is not visible: the comparison is not valid")
        for label, planes in (
            ("all planes off", (0, 0, 0)),
            ("x only", (1, 0, 0)),
            ("y only", (0, 1, 0)),
            ("z only", (0, 0, 1)),
        ):
            observed = checked(spreads(track(mode, columns, FACTOR, planes))) / reference
            # The smooth update of a plane does not depend on the other planes' switches, so a plane
            # that is on changes its spread as in the all-planes run, and a plane that is off keeps
            # the FACTOR=0 value (ratio 1). Columns: x and x' follow DO_X, y and y' DO_Y, rms(p) DO_Z.
            on = np.array([planes[0], planes[0], planes[1], planes[1], planes[2]], dtype=bool)
            defect |= compare(f"{mode} {label}", observed, np.where(on, full, 1.0))
    print(f"comparison: rtol={RTOL:g}")
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
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

