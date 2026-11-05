---
title: "COL-23: TRWAKE quadrupole probe coordinates omit element misalignment"
hide:
  - toc
---

# COL-23 — TRWAKE quadrupole probe coordinates omit element misalignment

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [tracking caller, do_tracking.c:2483–2486](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2483-L2486)<br>[trwake.c:154–156](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L154-L156)<br>[binTransverseTimeDistribution, trwake.c:649–660](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L649-L660)<br>[gather call, trwake.c:251–255](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L251-L255)<br>[`applyTransverseWakeKicks`, trwake.c:328–331](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L328-L331)<br>[trwake.c:151–152](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L151-L152)<br>[quadrupole wake, trwake.tex:53–58](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/trwake.tex#L53-L58)<br>[misalignment parameters, track_data.c:2604–2605](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2604-L2605) |

TRWAKE quadrupole wakes (probe exponents 1, drive exponents 0) use each particle's global transverse coordinate even when the structure has misalignments DX and DY. The kick must use displacement from the structure, so translating beam and structure together must preserve the kicks, and a beam on the structure's axis must receive none. For a 0.6000 nC electron bunch represented by 18 macroparticles, a common shift of (3.100, −1.700) mm raises the trailing particle's horizontal slope kick from 1.498×10⁻⁵ to 6.660×10⁻⁵, and an on-axis beam receives 5.161×10⁻⁵ instead of zero. Dipole wakes, with probe factor 1, pass the translation controls.

??? example "Reproducer: COL-23.py"

    [Download COL-23.py](COL-23.py)

    ```python
    """COL-23: TRWAKE quadrupole probe coordinates omit element misalignment

    TRWAKE quadrupole wakes (probe exponents 1, drive exponents 0) use each particle's global transverse
    coordinate even when the structure has misalignments DX and DY. The kick must use displacement from
    the structure, so translating beam and structure together must preserve the kicks, and a beam on the
    structure's axis must receive none. For a 0.6000 nC electron bunch represented by 18 macroparticles,
    a common shift of (3.100, −1.700) mm raises the trailing particle's horizontal slope kick from
    1.498×10⁻⁵ to 6.660×10⁻⁵, and an on-axis beam receives 5.161×10⁻⁵ instead of zero. Dipole wakes,
    with probe factor 1, pass the translation controls.

    The script tracks the same 18 electron macroparticles through a TRWAKE element with a damped-sine
    wake on a 2 ps grid, for the quadrupole wake centered and translated, a beam on the translated
    axis, and the dipole wake translated as the control. It compares every slope with a direct causal
    pair sum over the structure-relative coordinates, and shows that evaluating the probe at the global
    coordinate reproduces the observed slopes.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _trwake import ATOL, RTOL, N, beam, compare, pair_sum, track

    # Six arrival-time groups of three particles; offsets and momenta vary across the bunch.
    TIMES = np.repeat(np.arange(-25, 26, 10), 3) * 1e-12  # s
    RELATIVE = np.column_stack([np.linspace(-0.8, 0.9, N), np.linspace(0.5, -0.7, N)]) * 1e-3  # m
    MOMENTUM = np.linspace(31, 49, N)  # m_e c
    SHIFT = (3.1e-3, -1.7e-3)  # m, common translation of beam and structure

    # (label, shift, quadrupole wake, beam on the structure axis)
    CASES = (
        ("quadrupole, centered", (0.0, 0.0), True, False),
        ("quadrupole, translated (3.1, -1.7) mm", SHIFT, True, False),
        ("quadrupole, translated (-4.3, 2.6) mm", (-4.3e-3, 2.6e-3), True, False),
        ("quadrupole, beam on the translated axis", SHIFT, True, True),
        ("dipole control, translated (3.1, -1.7) mm", SHIFT, False, False),
    )

    defect = False
    for mode in modes():
        for label, shift, quadrupole, on_axis in CASES:
            position = np.broadcast_to(shift, (N, 2)).copy()
            if not on_axis:
                position += RELATIVE
            columns = beam(TIMES, position[:, 0], position[:, 1], MOMENTUM)
            drive = 0 if quadrupole else 1
            observed, _ = track(mode, columns, shift=shift, drive=drive)
            # Expected: the pair sum over coordinates relative to the structure.
            expected = pair_sum(columns, shift=shift, drive=drive)
            # Cause: the probe factor taken in the global frame instead.
            global_probe = pair_sum(columns, shift=shift, drive=drive, probe_offset=(0.0, 0.0))
            bad = compare(f"{mode} {label}", observed, expected)
            translated = shift != (0.0, 0.0)
            if translated and quadrupole:
                if bad:
                    # A mismatch must be explained by the global-frame probe to count as the defect.
                    if not np.allclose(observed, global_probe, rtol=RTOL, atol=ATOL):
                        sys.exit(f"{mode} {label}: kick follows neither the pair sum nor global probe")
                else:
                    print(f"{mode} {label}: kick agrees with the structure-relative pair sum")
                defect |= bad
            elif bad:
                sys.exit(
                    f"{mode} {label}: control disagrees with the pair sum; the comparison is invalid"
                )
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g} on the slopes of all {N} particles")
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

??? example "Helper: _trwake.py"

    [Download _trwake.py](_trwake.py)

    ```python
    """Shared setup for the TRWAKE reproducers: one TRWAKE element tracking 18 electron macroparticles,
    and the transverse wake kick of every particle computed as a direct causal pair sum, independently
    of Elegant's binning and convolution.

    The wake table has spacing `spacing` and starts at W(0) = 0, and every particle arrival time lies
    on a multiple of `spacing`, so each pair delay is a grid point and no wake interpolation enters.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import particles, run

    # SDDS constants.h: electron rest energy in eV.
    REST_EV = 510999.06
    N = 18  # particles
    RTOL, ATOL = 4e-13, 1e-19  # comparison of slopes


    def wake(delay: np.ndarray) -> np.ndarray:
        """Wx(t) in V/C/m for t >= 0 behind the source: 1.7e15 sin(2 pi 6 GHz t) exp(-t/70 ps)."""
        t = np.maximum(delay, 0)
        return np.where(delay >= 0, 1.7e15 * np.sin(2 * np.pi * 6e9 * t) * np.exp(-t / 70e-12), 0.0)


    def wake_table(spacing: float) -> str:
        """ASCII SDDS table of Wx and Wy = -Wx on 257 points of the given spacing."""
        grid = np.arange(257) * spacing
        rows = "".join(
            f"{float(t)!r} {float(w)!r} {float(-w)!r}\n" for t, w in zip(grid, wake(grid), strict=True)
        )
        return (
            "SDDS1\n"
            "&column name=time, units=s, type=double &end\n"
            "&column name=wx, units=V/C/m, type=double &end\n"
            "&column name=wy, units=V/C/m, type=double &end\n"
            "&data mode=ascii, no_row_counts=1 &end\n" + rows
        )


    def beam(times, x, y, momentum) -> dict[str, np.ndarray]:
        """Particles with zero slopes at the given arrival times, positions and momenta (m_e c)."""
        return {
            "x": np.broadcast_to(x, N).astype(float),
            "y": np.broadcast_to(y, N).astype(float),
            "t": np.asarray(times, dtype=float),
            "p": np.broadcast_to(momentum, N).astype(float),
            "particleID": np.arange(1, N + 1),
        }


    def track(
        mode: str,
        columns: dict[str, np.ndarray],
        *,
        shift: tuple[float, float] = (0.0, 0.0),
        drive: int = 1,
        n_bins: int = 0,
        charge: float = 0.6e-9,
        drift: float = 0.0,
        spacing: float = 2e-12,
    ) -> tuple[np.ndarray, str]:
        """Slope kicks (xp, yp), shape (N, 2), of every particle through TRWAKE, and the native log.

        drive=1 is the ordinary dipole wake (kick from the leading particles' offsets, probe
        exponent 0); drive=0 is the quadrupole wake (kick proportional to the particle's own offset).
        shift is the structure's misalignment (DX, DY); drift is a drift before the structure.
        """
        lattice = (
            f"CH: CHARGE,TOTAL={charge!r}\n"
            f"D0: DRIF,L={drift!r}\n"
            'R: TRWAKE,INPUTFILE="wake.sdds",TCOLUMN="time",WXCOLUMN="wx",WYCOLUMN="wy",'
            f"DX={shift[0]!r},DY={shift[1]!r},N_BINS={n_bins},INTERPOLATE=0,SMOOTHING=0,"
            f"X_DRIVE_EXPONENT={drive},Y_DRIVE_EXPONENT={drive},"
            f"X_PROBE_EXPONENT={1 - drive},Y_PROBE_EXPONENT={1 - drive}\n"
            "BL: LINE=(CH,D0,R)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={float(np.mean(columns['p']))!r},"
            "output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands, "wake.sdds": wake_table(spacing)},
            mode,
            {"beam.sdds": columns},
        )
        out = particles(result["out.sdds"])
        if not np.array_equal(out["particleID"], columns["particleID"]):
            raise RuntimeError(f"{mode}: particles lost")
        # The wake changes only the slopes (the drift acts on s, not on x or y).
        if not (np.array_equal(out["x"], columns["x"]) and np.array_equal(out["y"], columns["y"])):
            raise RuntimeError(f"{mode}: transverse positions changed")
        return np.column_stack([out["xp"], out["yp"]]), result["log"]


    def pair_sum(
        columns: dict[str, np.ndarray],
        *,
        shift: tuple[float, float] = (0.0, 0.0),
        drive: int = 1,
        charge: float = 0.6e-9,
        spacing: float = 2e-12,
        probe_offset: tuple[float, float] | None = None,
        binned: np.ndarray | None = None,
    ) -> np.ndarray:
        """Physical slope kicks (xp, yp), shape (N, 2), from the causal pair sum.

        With structure offset d, drive exponent a and probe exponent b (a + b = 1) and zero incoming
        slopes, particle i receives
          dx'_i = q (x_i - d)^b sum_j [(x_j - d)^a Wx(t_i - t_j)] / (E0 p_i),
        summed over the particles j that are not later than i, with q = charge / N the macroparticle
        charge and E0 the electron rest energy in eV; y is the same with Wy = -Wx. Translating beam
        and structure together leaves every (x - d) unchanged. Giving `probe_offset` replaces d in the
        probe factor by that value, which evaluates the probe at the global coordinate (d = 0).
        `binned` (boolean) restricts both sources and probes to the particles Elegant bins; the
        others receive no kick.
        """
        t, p = columns["t"], columns["p"]
        delays = t[:, None] - t[None, :]
        index = np.rint(delays / spacing).astype(int)
        if not np.allclose(delays / spacing, index, rtol=0, atol=3e-12):
            raise ValueError("pair delays are not multiples of the wake spacing")
        # Causal matrix W(t_i - t_j) taken on the table grid; the coincident pairs have W(0) = 0.
        matrix = np.where(index >= 0, wake(np.maximum(index, 0) * spacing), 0.0)
        if binned is None:
            binned = np.ones(N, dtype=bool)
        matrix = matrix * binned[None, :]
        kicks = np.zeros((N, 2))
        for plane, (name, sign) in enumerate((("x", 1.0), ("y", -1.0))):
            rel = columns[name] - shift[plane]
            probe = rel if probe_offset is None else columns[name] - probe_offset[plane]
            source = rel if drive else np.ones(N)
            voltage = sign * charge / N * (matrix @ source)
            kicks[:, plane] = voltage * (1.0 if drive else probe) / (REST_EV * p)
        kicks[~binned] = 0.0
        return kicks


    def fmt(values: np.ndarray) -> str:
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values) + 0.0) + "]"


    def compare(label: str, observed: np.ndarray, expected: np.ndarray) -> bool:
        """Print the last particle's (xp, yp); return True when any particle disagrees."""
        print(f"{label}: observed={fmt(observed[-1])} expected={fmt(expected[-1])}")
        return not np.allclose(observed, expected, rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

