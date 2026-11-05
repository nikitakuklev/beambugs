---
title: "COL-02: Pelegant falsely warns that fast-orbit-feedback BPM readings are not reduced across ranks"
hide:
  - toc
---

# COL-02 — Pelegant falsely warns that fast-orbit-feedback BPM readings are not reduced across ranks

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [warning, fast_orbit_feedback.c:1716–1719](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/fast_orbit_feedback.c#L1716-L1719)<br>[monitor hook, do_tracking.c:1349–1362](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L1349-L1362)<br>[storeBPMReading, do_tracking.c:4386–4469](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L4386-L4469)<br>[distributed sums, compute_centroids.c:805–837](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_centroids.c#L805-L837)<br>[FAST_ORBIT_FEEDBACK, elegant.c:1858–1891](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L1858-L1891)<br>[fast_orbit_feedback manual, elegant.tex:3323–3324](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L3323-L3324) |

When a user runs the fast orbit feedback command under MPI (Pelegant), Elegant warns that beam-position-monitor (BPM) readings are not reduced across ranks, which would leave each rank's monitor reading only its own particles. With 24 particles split between two tracking ranks whose local means differ, the monitor instead reads the whole beam's mean, as products of lens and drift matrices predict, for one and three turns per readout and feedback gain 0 or 0.2500: 3.474×10⁻⁴ m horizontally and 3.150×10⁻³ m vertically after one turn. The warning is therefore false for these readings, although the manual says Pelegant results are identical to serial for a deterministic beam.

??? example "Reproducer: COL-02.py"

    [Download COL-02.py](COL-02.py)

    ```python
    """COL-02: Pelegant falsely warns that fast-orbit-feedback BPM readings are not reduced across ranks

    When a user runs the fast orbit feedback command under MPI (Pelegant), Elegant warns that
    beam-position-monitor (BPM) readings are not reduced across ranks, which would leave each rank's
    monitor reading only its own particles. With 24 particles split between two tracking ranks whose
    local means differ, the monitor instead reads the whole beam's mean, as products of lens and drift
    matrices predict, for one and three turns per readout and feedback gain 0 or 0.2500: 3.474×10⁻⁴ m
    horizontally and 3.150×10⁻³ m vertically after one turn. The warning is therefore false for these
    readings, although the manual says Pelegant
    results are identical to serial for a deterministic beam.

    The script tracks 24 particles through a stable alternating-gradient cell with two correctors and
    one monitor, and runs `fast_orbit_feedback` for one and for three turns per readout, with
    integral gain 0 and 0.25. The first half of the rows sits at one position and the second half at
    another, so each of the two tracking ranks of the three-rank run holds a different local mean. The
    expected monitor reading is the absolute offset of the whole beam's mean, propagated by
    independently multiplied lens and drift matrices. Serial runs are the control.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    P = 100.0  # electron momentum, units of m_e c
    N = 24  # particles
    # Rows 0-11 and 12-23 sit at different positions, so the two ranks' local means differ.
    BEAM = {
        "x": np.repeat([0.001, -0.003], N // 2),
        "xp": np.repeat([0.0006, -0.0002], N // 2),
        "y": np.repeat([-0.002, 0.006], N // 2),
        "yp": np.full(N, -0.0003),
        "p": np.full(N, P),
    }
    LENS = 0.1  # m, length of each quadrupole
    RTOL, ATOL = 3e-13, 1e-17
    WARNING = "BPM readings are not reduced across ranks"


    def quadrupole(k):
        """Transfer matrix of a quadrupole of length LENS with strength k: q'' = -k q."""
        root = np.sqrt(abs(k))
        angle = LENS * root
        c = np.cos(angle) if k > 0 else np.cosh(angle)
        s = np.sin(angle) / root if k > 0 else np.sinh(angle) / root
        return np.array([[c, s], [-k * s, c]])


    def expected_readings(turns):
        """|position| of the mean of the beam at the monitor after `turns` turns, per plane.

        One turn is drift (1 m), F (K1 = 10), drift, G (K1 = -10), read at the monitor. Elegant's K1 > 0
        focuses in x and defocuses in y, so the y matrix has the lens signs exchanged.
        """
        drift = np.array([[1.0, 1.0], [0.0, 1.0]])
        mx = quadrupole(-10) @ drift @ quadrupole(10) @ drift
        my = quadrupole(10) @ drift @ quadrupole(-10) @ drift
        x = np.linalg.matrix_power(mx, turns) @ [BEAM["x"].mean(), BEAM["xp"].mean()]
        y = np.linalg.matrix_power(my, turns) @ [BEAM["y"].mean(), BEAM["yp"].mean()]
        return np.abs([x[0], y[0]])


    def readings(mode, gain, turns):
        """The monitor readings (xBpmRms, yBpmRms) of the first readout and whether the warning shows."""
        lattice = (
            "H: HKICK\nV: VKICK\nD: DRIF,L=1\n"
            f"F: QUAD,L={LENS},K1=10\nG: QUAD,L={LENS},K1=-10\nM: MONI\n"
            "BL: LINE=(H,V,D,F,D,G,M)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},default_order=1 &end\n"
            "&run_control n_steps=1 &end\n"
            "&twiss_output matched=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            f"&fast_orbit_feedback passes_between_updates={turns},steering_Ki={gain!r},"
            "fixed_length_matrix=1,output=fofb.sdds,verbosity=0 &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": BEAM},
            outputs=("fofb.sdds",),
        )
        table = result["fofb.sdds"]
        observed = np.array([table[0, "xBpmRms"][0], table[0, "yBpmRms"][0]])
        return observed, WARNING in result["log"]


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"


    defect = True  # until a reading disagrees with the whole-beam mean, or the warning is absent
    warned = False
    for mode in ("serial", *modes(mpi_only=True)):
        for gain, turns in ((0.0, 1), (0.25, 1), (0.25, 3)):
            observed, warning = readings(mode, gain, turns)
            expected = expected_readings(turns)
            agrees = bool(np.allclose(observed, expected, rtol=RTOL, atol=ATOL))
            print(
                f"{mode} gain={gain}, {turns} turn(s) per readout: monitor readings "
                f"observed={fmt(observed)} expected={fmt(expected)}; warning printed: {warning}"
            )
            if mode == "serial" and (not agrees or warning):
                sys.exit(f"{mode}: the serial control disagrees with the matrices or warns")
            if mode == "mpi":
                warned |= warning
                # A missing reduction would leave each rank with its own local mean.
                defect &= agrees
    defect &= warned
    print(f'warning text checked: "{WARNING}"')
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g} m")
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
| Found | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source reading, not executed for this claim |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial control and 3-rank Pelegant |

