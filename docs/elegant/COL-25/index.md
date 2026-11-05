---
title: "COL-25: MPI TRWAKE adds fixed-bin histograms whose time origins differ between ranks"
hide:
  - toc
---

# COL-25 — MPI TRWAKE adds fixed-bin histograms whose time origins differ between ranks

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [local bunch mean, trwake.c:102–109](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L102-L109)<br>[global extrema, trwake.c:118–121](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L118-L121)<br>[fixed-bin origin, trwake.c:132–134](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L132-L134)<br>[automatic bins, trwake.c:135–138](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L135-L138)<br>[histogram sum, trwake.c:206–213](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L206-L213)<br>[local bunch counts, lrwake.c:149–166](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L149-L166)<br>[wake definition, trwake.tex:1–28](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/trwake.tex#L1-L28)<br>[bunched mode, trwake.tex:59–62](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/trwake.tex#L59-L62)<br>[setup, trwake.c:380–381](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L380-L381)<br>[TRWAKE caller, do_tracking.c:2483–2486](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2483-L2486)<br>[CHARGE, do_tracking.c:1441–1457](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L1441-L1457) |

With multiple bunches and a fixed TRWAKE bin count (N_BINS > 0) under MPI (Pelegant), each tracking rank centers its histogram on its own bunch mean, so unequal rank means make summed bins represent different physical times. The correct kick is a causal sum within each bunch; serial tracking, MPI with equal rank means and automatic bins reproduce it. For two 80.00 pC bunches whose early and late halves occupy different ranks, the first particle of a late half receives no horizontal slope kick instead of 6.720×10⁻⁷, and the last particle's kick is 60.00% too small.

??? example "Reproducer: COL-25.py"

    [Download COL-25.py](COL-25.py)

    ```python
    """COL-25: MPI TRWAKE adds fixed-bin histograms whose time origins differ between ranks

    With multiple bunches and a fixed TRWAKE bin count (N_BINS > 0) under MPI (Pelegant), each tracking
    rank centers its histogram on its own bunch mean, so unequal rank means make summed bins represent
    different physical times. The correct kick is a causal sum within each bunch; serial tracking, MPI
    with equal rank means and automatic bins reproduce it. For two 80.00 pC bunches whose early and late
    halves occupy different ranks, the first particle of a late half receives no horizontal slope kick
    instead of 6.720×10⁻⁷, and the last particle's kick is 60.00% too small.

    The script tracks two 24-particle bunches, 3 ns apart, through one dipole TRWAKE with a linear
    causal wake and 64 fixed bins. Every particle has the same transverse offsets, so the correct kick
    is a direct causal pair sum over the particles of the same bunch, written without any binning. In
    one input row order both tracking ranks see the same time distribution; in the other, one rank holds
    the early half and the other the late half of each bunch, so their mean times differ. A prediction
    that replaces each time by its offset from its rank's mean explains the result; serial runs, equal
    rank means and automatic bins are passing controls, while halving the bin width leaves the
    erroneous result unchanged.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    REST_EV = 510999.06  # SDDS constants.h: electron rest energy, eV
    P = 41.0  # electron momentum, units of m_e c
    SLOPE = 4e25  # V/C/m per s; Wx(tau) = SLOPE tau for tau >= 0, Wy = 0.7 Wx, W(0) = 0
    CHARGE = 80e-12  # C per bunch
    X0, Y0 = 1.1e-3, -0.6e-3  # m, the same for every particle
    # Offsets from the bunch center (ps), four particles at each; bunch centers 2 ns and 5 ns.
    OFFSETS = np.repeat([-10, -6, -2, 2, 6, 10], 4) * 1e-12
    N = len(OFFSETS)  # particles per bunch
    CENTERS = (2e-9, 5e-9)
    BINS = 64
    RTOL, ATOL = 2e-12, 1e-19


    def wake_table(spacing):
        """ASCII SDDS table of Wx, Wy on 257 points of the given spacing."""
        rows = "".join(
            f"{tau!r} {SLOPE * tau!r} {0.7 * SLOPE * tau!r}\n"
            for tau in (float(n) * spacing for n in range(257))
        )
        return (
            "SDDS1\n"
            "&column name=time, units=s, type=double &end\n"
            "&column name=wx, units=V/C/m, type=double &end\n"
            "&column name=wy, units=V/C/m, type=double &end\n"
            "&data mode=ascii, no_row_counts=1 &end\n" + rows
        )


    def kicks(mode, layout, spacing, bins):
        """Slope kicks (xp, yp) of the 2N particles, ordered by particle ID."""
        t = np.concatenate([OFFSETS + c for c in CENTERS])
        ids = np.arange(1, 2 * N + 1)
        if layout == "skew":
            # One rank gets the early half of each bunch, the other the late half.
            order = np.r_[0:12, 24:36, 12:24, 36:48]
        else:
            # Both ranks get two of the four particles at each time of each bunch.
            half = np.arange(N).reshape(6, 4)[:, :2].ravel()
            other = np.arange(N).reshape(6, 4)[:, 2:].ravel()
            order = np.r_[half, half + N, other, other + N]
        beam = {
            "x": np.full(2 * N, X0)[order],
            "y": np.full(2 * N, Y0)[order],
            "t": t[order],
            "p": np.full(2 * N, P),
            "particleID": ids[order],
            "IDSlotsPerBunch": N,
        }
        lattice = (
            f"Q: CHARGE,TOTAL={2 * CHARGE!r}\n"
            'W: TRWAKE,INPUTFILE="wake.sdds",TCOLUMN="time",WXCOLUMN="wx",WYCOLUMN="wy",'
            f"N_BINS={bins},BUNCHED_BEAM_MODE=1,SMOOTHING=0,INTERPOLATE=0,"
            "X_DRIVE_EXPONENT=1,Y_DRIVE_EXPONENT=1,X_PROBE_EXPONENT=0,Y_PROBE_EXPONENT=0\n"
            "BL: LINE=(Q,W)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds,use_bunched_mode=1,track_pages_separately=0 &end\n"
            "&track &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands, "wake.sdds": wake_table(spacing)},
            mode,
            {"beam.sdds": beam},
        )
        out = particles(result["out.sdds"])
        if not np.array_equal(out["particleID"], ids):
            sys.exit(f"{mode}: particles lost")
        # The wake changes only the slopes, which start at zero.
        return np.column_stack([out["xp"], out["yp"]])


    def pair_sum(times):
        """Slope kicks (xp, yp) of one bunch from a direct causal pair sum over its particles.

        Particle i receives dq'_i = (Q/N) sum_j q_j W_q(t_i - t_j)/(E0 p) for q = x, y, summed over the
        particles j of the same bunch behind which it arrives (W(0) = 0), with Q/N the macroparticle
        charge, q_j the source's offset and E0 the electron rest energy in eV. The times lie on the
        wake table's grid, so no interpolation enters.
        """
        delay = np.maximum(times[:, None] - times[None, :], 0.0)
        voltage = (CHARGE / N) * SLOPE * delay.sum(axis=1)  # V/m, times the source offset
        return np.column_stack([X0 * voltage, 0.7 * Y0 * voltage]) / (REST_EV * P)


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values) + 0.0) + "]"


    def compare(label, observed, expected):
        """Print the kicks of particles 13 (first of the late half) and 24 of the first bunch."""
        for index in (12, 23):
            print(
                f"{label}, particle {index + 1}: "
                f"observed={fmt(observed[index])} expected={fmt(expected[index])}"
            )
        return bool(np.allclose(observed, expected, rtol=RTOL, atol=ATOL))


    defect = False
    for mode in ("serial", *modes(mpi_only=True)):
        for layout, spacing, bins in (
            ("balanced", 2e-12, BINS),
            ("skew", 2e-12, BINS),
            ("skew", 1e-12, BINS),
            ("skew", 2e-12, 0),
        ):
            expected = np.vstack([pair_sum(OFFSETS)] * 2)
            auto = "automatic bins" if bins == 0 else f"{bins} bins of {spacing * 1e12:g} ps"
            label = f"{mode} {layout} rows, {auto}"
            observed = kicks(mode, layout, spacing, bins)
            agrees = compare(label, observed, expected)
            if not (mode == "mpi" and layout == "skew" and bins):
                if not agrees:
                    sys.exit(f"{label}: control disagrees with the pair sum; the comparison is invalid")
                continue
            # Cause: the early-half rank centers its grid 6 ps before the bunch center and the
            # late-half rank 6 ps after it, so the summed histogram pairs times that differ by 12 ps.
            # Each particle then behaves as if it arrived at its offset from its own rank's mean time.
            shifted = OFFSETS - np.where(np.arange(N) < N // 2, -6e-12, 6e-12)
            local = np.vstack([pair_sum(shifted)] * 2)
            if not np.allclose(observed, local, rtol=RTOL, atol=ATOL) and not agrees:
                sys.exit(f"{label}: kicks follow neither the pair sum nor the rank-local origins")
            print(f"{label}, particle 13: rank-local-origin prediction={fmt(local[12])}")
            print(f"{label}: particle 24 kick is {1 - observed[23, 0] / expected[23, 0]:.4%} too small")
            defect |= not agrees
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
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

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial control and 3-rank Pelegant |

