---
title: "COL-24: TRWAKE fixed bins use an absent or rank-local time origin"
hide:
  - toc
---

# COL-24 — TRWAKE fixed bins use an absent or rank-local time origin

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [tmean to zero, trwake.c:40](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L40)<br>[lrwake.c:117–165](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L117-L165)<br>[single-bucket branch, trwake.c:80–85](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L80-L85)<br>[multiple-bucket branch, trwake.c:102–109](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L102-L109)<br>[fixed-bin branch, trwake.c:132–134](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L132-L134)<br>[Deposition, trwake.c:641–648](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L641-L648)<br>[gather, trwake.c:316](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L316)<br>[warning, trwake.c:164–168](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L164-L168)<br>[convolution, wake.c:514](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L514)<br>[wake time, trwake.tex:1–3](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/trwake.tex#L1-L3)<br>[recommends autoscaling, trwake.tex:26–28](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/trwake.tex#L26-L28)<br>[N_BINS, track_data.c:2599](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2599)<br>[local bunch mean, trwake.c:102–109](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L102-L109)<br>[global extrema, trwake.c:118–121](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L118-L121)<br>[fixed-bin origin, trwake.c:132–134](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L132-L134)<br>[automatic bins, trwake.c:135–138](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L135-L138)<br>[histogram sum, trwake.c:206–213](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L206-L213)<br>[local bunch counts, lrwake.c:149–166](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L149-L166)<br>[wake definition, trwake.tex:1–28](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/trwake.tex#L1-L28)<br>[bunched mode, trwake.tex:59–62](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/trwake.tex#L59-L62)<br>[setup, trwake.c:380–381](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L380-L381)<br>[TRWAKE caller, do_tracking.c:2483–2486](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2483-L2486)<br>[CHARGE, do_tracking.c:1441–1457](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L1441-L1457) |

With a fixed TRWAKE bin count, a single bunch uses a grid centered on time zero, while multiple bunches under MPI use separate grids centered on each worker's local mean. Particles can fall outside the grid, or summed bins can represent different physical times; a stationary wake instead depends on particle time differences within each bunch. A 0.6000 nC bunch shifted by 0.7000 ns receives zero trailing horizontal kick instead of 1.302×10⁻⁵ rad. In a two-bunch MPI case with unequal worker means, a trailing kick is 60.00% too small. Causal pair sums and automatic-bin controls establish the expected kicks.

??? example "Reproducer: COL-24.py"

    [Download COL-24.py](COL-24.py)

    ```python
    """COL-24: TRWAKE fixed bins use an absent or rank-local time origin

    With a fixed TRWAKE bin count, a single bunch uses a grid centered on time zero, while multiple
    bunches under MPI use separate grids centered on each worker's local mean. Particles can fall
    outside the grid, or summed bins can represent different physical times; a stationary wake
    instead depends on particle time differences within each bunch. A 0.6000 nC bunch shifted by
    0.7000 ns receives zero trailing horizontal kick instead of 1.302×10⁻⁵ rad. In a two-bunch MPI
    case with unequal worker means, a trailing kick is 60.00% too small. Causal pair sums and
    automatic-bin controls establish the expected kicks.

    The script tracks the same bunch through TRWAKE with a damped-sine wake after different arrival-time
    shifts and a drift, with 96 fixed bins, 1024 fixed bins and automatic bins, and with the bunch and
    wake table stretched to 4 ps. It compares every slope with a causal pair sum over the particles,
    which is shift-invariant, and for the cases with particles outside the grid also with the pair sum
    restricted to the particles inside it. The zero shift, +28.00 ps (inside the grid), 1024-bin and
    automatic-bin cases are the controls.
    """

    import re
    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _trwake import ATOL, RTOL, N, beam, compare, fmt, pair_sum, track

    C = 299792458.0  # SDDS constants.h
    P = 37.0  # m_e c
    NBINS = 96
    # Six time groups of three particles, centered on the bin centers of a grid of spacing `scale`*2 ps.
    GROUPS = np.array([-26, -16, -6, 6, 16, 26]) * 1e-12

    # (label, bins, arrival-time shift in s, drift in m, scale of times and wake spacing)
    CASES = (
        ("96 bins, no shift", NBINS, 0.0, 0.0, 1),
        ("96 bins, +28 ps", NBINS, 28e-12, 0.0, 1),
        ("96 bins, +80 ps", NBINS, 80e-12, 0.0, 1),
        ("96 bins, +0.7 ns", NBINS, 0.7e-9, 0.0, 1),
        ("96 bins, 0.21 m drift", NBINS, 0.0, 0.21, 1),
        ("1024 bins, +0.7 ns", 1024, 0.7e-9, 0.0, 1),
        ("automatic bins, +0.7 ns", 0, 0.7e-9, 0.0, 1),
        ("96 bins, 4 ps, +1.4 ns", NBINS, 1.4e-9, 0.0, 2),
        ("automatic bins, 4 ps, +1.4 ns", 0, 1.4e-9, 0.0, 2),
    )

    defect = False
    for mode in modes():
        for label, bins, shift, drift, scale in CASES:
            spacing = scale * 2e-12
            times = np.repeat(GROUPS * scale, 3) + shift
            columns = beam(times, 0.6e-3, -0.4e-3, P)
            observed, log = track(mode, columns, n_bins=bins, drift=drift, spacing=spacing)
            # A pair sum depends on arrival-time differences only, so the expected kicks do not depend
            # on the shift or the drift.
            expected = pair_sum(columns, spacing=spacing)
            # Particles Elegant bins: with the grid centered on time zero, tmin = -bins spacing/2 and
            # bin k is centered on tmin + k spacing. The index is (t - tmin)/spacing + 0.5 truncated
            # toward zero and accepted for 0 <= k <= bins-1, i.e. tmin - 1.5 spacing < t <
            # tmin + (bins - 0.5) spacing. A drift adds drift/(beta c) to the arrival time.
            arrival = times + drift * np.sqrt(1 + P**2) / (C * P)
            tmin = -bins * spacing / 2
            inside = np.ones(N, dtype=bool)
            if bins > 0:
                inside = (arrival > tmin - 1.5 * spacing) & (arrival < tmin + (bins - 0.5) * spacing)
            warned = "not binned in TRWAKE" in log
            if inside.all():
                bad = compare(f"{mode} {label}", observed, expected)
                if bad or warned:
                    sys.exit(f"{mode} {label}: control disagrees with the pair sum")
                continue
            # Cause: particles outside the grid are skipped as sources and as probes.
            predicted = pair_sum(columns, spacing=spacing, binned=inside)
            bad = compare(f"{mode} {label}, {inside.sum()} of {N} in the grid", observed, expected)
            if inside.any():
                middle = np.flatnonzero(inside)[-1]
                print(
                    f"{mode} {label}: last particle in the grid (t={arrival[middle] * 1e12:g} ps): "
                    f"observed={fmt(observed[middle])} expected={fmt(expected[middle])}"
                )
            if bad:
                # A mismatch must be explained by the restricted pair sum and the warning to count.
                if not np.allclose(observed, predicted, rtol=RTOL, atol=ATOL):
                    sys.exit(f"{mode} {label}: kicks differ from the pair sum restricted to the grid")
                if not warned:
                    sys.exit(f"{mode} {label}: no unbinned-particle warning")
                # The serial warning states the count of binned particles.
                count = re.search(r"Only (\d+) of \d+ particles were binned", log)
                if mode == "serial" and (count is None or int(count[1]) != inside.sum()):
                    sys.exit(f"{mode} {label}: Elegant bins a different number of particles")
                defect = True
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g} on the slopes of all {N} particles")
    verdict(defect)

    ```

??? example "Reproducer: COL-24-mpi.py"

    [Download COL-24-mpi.py](COL-24-mpi.py)

    ```python
    """COL-24: MPI fixed-bin controls with unequal bunch means across workers

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
| Found (single-bunch zero time origin) | `gpt-6-astra` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Found (multiple-bunch rank-local time origins) | `gpt-6.1-sol` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source reading, not executed for this claim |
| Confirmed (single-bunch zero time origin) | `gpt-6.1-sol` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed (multiple-bunch rank-local time origins) | `gpt-6-astra` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial control and 3-rank Pelegant |

