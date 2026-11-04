---
title: "COL-19: Empty SCMULT slice centroids break translation symmetry"
hide:
  - toc
---

# COL-19 — Empty SCMULT slice centroids break translation symmetry

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [calloc, insertSCeffects.cc:386–390](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L386-L390)<br>[occupied-bin guard, insertSCeffects.cc:411](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L411)<br>[interpolation, insertSCeffects.cc:464–466](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L464-L466)<br>[slice parameters, elegant.tex:4504–4511](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4504-L4511)<br>[SDDS loading, sdds_beam.c:446](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_beam.c#L446)<br>[drift, compute_matrices.c:614](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L614)<br>[`computeTimeCoordinatesOnly`, trwake.c:539–549](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L539-L549) |

Linear SCMULT tracking with time slicing and slice interpolation leaves empty slices' transverse centroids at zero instead of using the whole-bunch centroids specified by the manual. Mixing those values with occupied slices breaks translation symmetry: a common transverse displacement changes the self-kicks, whose mean must remain zero for the symmetric test bunch. For 20 electrons carrying 0.2000 nC in 100.0 ps slices with one empty neighbor, a displacement of (1.700, −2.200) mm produces a mean horizontal slope of 5.597×10⁻⁶ instead of zero. Sliced ring simulations that interpolate with an empty neighbor are affected; extrapolation can also reverse the erroneous mean.

??? example "Reproducer: COL-19.py"

    [Download COL-19.py](COL-19.py)

    ```python
    """COL-19: Empty SCMULT slice centroids break translation symmetry

    Linear SCMULT tracking with time slicing and slice interpolation leaves empty slices' transverse
    centroids at zero instead of using the whole-bunch centroids specified by the manual. Mixing those
    values with occupied slices breaks translation symmetry: a common transverse displacement changes
    the self-kicks, whose mean must remain zero for the symmetric test bunch. For 20 electrons carrying
    0.2000 nC in 100.0 ps slices with one empty neighbor, a displacement of (1.700, −2.200) mm produces a
    mean horizontal slope of 5.597×10⁻⁶ instead of zero. Sliced ring simulations that interpolate with
    an empty neighbor are affected; extrapolation can also reverse the erroneous mean.

    The script tracks one turn of a four-sector ring with SCMULT and a cold, transversely symmetric
    bunch, centered and shifted, and compares the shifted kicks and their mean with the centered ones,
    which translation symmetry requires to be equal. It reconstructs Elegant's slice grid to predict the
    spurious mean from the empty neighbor's interpolation weight. Controls are slice widths that leave
    no empty bin, interpolation off and no slicing; an interior-gap bunch shows the effect with
    interpolation fractions reaching 1.5.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _scmult_ring import QE, RE, STRAIGHT, SX, SY, C, P, fmt, track, train

    CHARGE = 2e-10  # C
    SHIFT = (1.7e-3, -2.2e-3)  # m, common transverse translation
    ATOL, RTOL = 2e-17, 3e-12  # slope; the translated and centered kicks of a control agree to ~1e-20

    # Twenty particles: the four transverse corners at five arrival times, or six for the gap bunch.
    TIMES = np.array([-2, -1, 0, 1, 2]) * 9e-12
    GAP_TIMES = np.array([-250, -240, -230, 230, 240, 250]) * 1e-12


    def slice_grid(times, duration):
        """Reconstruct SCMULT's slices for a cold bunch: counts, interpolation fractions, mean factor.

        The arrival time SCMULT uses is not `times`: the SDDS loader stores s = t c p/gamma, the cold
        drift adds the straight's length, and SCMULT recomputes t = s gamma/(c p). Rounding then
        decides the slice of particles exactly on a slice boundary (the earliest particles always are:
        tmin is half a slice before them). Slice j is centered on tmin + j duration; a particle in it
        is interpolated between slice j and j+1 (j-1 and j if it is before the center), with
        fraction = offset from the lower center / duration.
        The mean factor is mean(q (1 - c)) over particles, where q is the interpolated charge fraction
        and c the interpolated centroid factor, 0 for an empty slice and 1 for an occupied one.
        """
        gamma = np.sqrt(P * P + 1)
        times = np.repeat(times, 4)  # four transverse corners per arrival time
        s = times * C * P / gamma + STRAIGHT
        t = s * gamma / (C * P)
        tmin, tmax = t.min() - duration / 2, t.max() + duration / 2
        n_slices = int((tmax - tmin) / duration + 1)
        bins = ((t - tmin) / duration + 0.5).astype(int)
        counts = np.bincount(bins, minlength=n_slices)
        if len(counts) != n_slices:
            raise RuntimeError("particle beyond the last slice")
        lower = bins.copy()
        offset = t - (tmin + lower * duration)
        before = ((offset < 0) & (lower > 0)) | (lower == n_slices - 1)
        lower[before] -= 1
        offset[before] += duration
        fraction = offset / duration
        occupied = (counts != 0).astype(float)
        centroid = (1 - fraction) * occupied[lower] + fraction * occupied[lower + 1]
        charge = ((1 - fraction) * counts[lower] + fraction * counts[lower + 1]) / len(t)
        return counts, fraction, float(np.mean(charge * (1 - centroid)))


    def mean_kick(mode, times, duration, interpolate, shift):
        """Slope kicks (xp, yp) of every particle for a bunch translated by `shift`."""
        columns = train(((1, 1),), times, shift=shift)
        _, exit_ = track(mode, columns, 1, charge=CHARGE, duration=duration, interpolate=interpolate)
        # The bunch starts with zero slopes and the straight is a drift, so the exit slopes are kicks.
        return np.column_stack([exit_[0]["xp"], exit_[0]["yp"]])


    def predicted_mean(duration, shift, mean_factor):
        """Mean slope from the interpolated empty-slice centroid.

        For a cold beam translated by a, the interpolated centroid is c a instead of a, and the
        interpolated slice charge fraction is q. The linear kick of SCMULT's uniform slices
          dx' = K q Q L (x - c a)/(c_light Delta_t sx (sx + sy)),   K = 2 r_e/(e p^2 gamma),
        averaged over the symmetric corners (mean x - a = 0) leaves K Q L a mean(q (1 - c))/(c_light
        Delta_t sx (sx + sy)), with K = sqrt(2 pi) c1 and c1 = sqrt(2/pi) r_e/(e p^2 gamma). The optical
        weight reduces to L because beta is equal at both ends of the full straight.
        """
        c1 = np.sqrt(2 / np.pi) * RE / QE / (P**2 * np.hypot(P, 1))
        scale = c1 * CHARGE / (duration * C) * np.sqrt(2 * np.pi) * STRAIGHT * mean_factor
        return scale * np.array(shift) / (np.array([SX, SY]) * (SX + SY))


    CASES = (
        # label, arrival times, slice duration, interpolation, shifts
        ("100 ps slices, interpolation", TIMES, 100e-12, 1, (SHIFT, (-3.4e-3, 4.4e-3))),
        ("160 ps slices, interpolation", TIMES, 160e-12, 1, (SHIFT,)),
        ("120 ps slices, interpolation (control)", TIMES, 120e-12, 1, (SHIFT,)),
        ("180 ps slices, interpolation (control)", TIMES, 180e-12, 1, (SHIFT,)),
        ("100 ps slices, no interpolation (control)", TIMES, 100e-12, 0, (SHIFT,)),
        ("no slicing (control)", TIMES, 0.0, 0, (SHIFT,)),
        (
            "interior gap, 100 ps slices, interpolation",
            GAP_TIMES,
            100e-12,
            1,
            (SHIFT, (-3.4e-3, 4.4e-3)),
        ),
        ("interior gap, 100 ps slices, no interpolation (control)", GAP_TIMES, 100e-12, 0, (SHIFT,)),
    )

    defect = False
    for mode in modes():
        for label, times, duration, interpolate, shifts in CASES:
            centered = mean_kick(mode, times, duration, interpolate, (0.0, 0.0))
            # The symmetric centered bunch must have zero mean kick; this is the symmetry control.
            if not np.allclose(centered.mean(axis=0), 0, rtol=0, atol=ATOL):
                sys.exit(f"{mode} {label}: the centered bunch's mean kick is not zero")
            means = []
            for shift in shifts:
                kicks = mean_kick(mode, times, duration, interpolate, shift)
                mean = kicks.mean(axis=0)
                # Expected: translation symmetry preserves every kick, so the mean stays that of the
                # centered bunch (zero by symmetry).
                bad = not np.allclose(kicks, centered, rtol=0, atol=ATOL)
                counts = np.array([])
                if duration:
                    counts, fraction, mean_factor = slice_grid(times, duration)
                print(
                    f"{mode} {label}, shift {fmt(np.array(shift) * 1e3)} mm, "
                    f"particles per slice {counts.astype(int).tolist()}: "
                    f"observed mean={fmt(mean)} expected={fmt(centered.mean(axis=0))}"
                )
                if not (duration and interpolate and (counts == 0).any()):
                    if bad:
                        sys.exit(f"{mode} {label}: control breaks translation symmetry")
                    continue
                if not bad:
                    continue  # translation symmetry holds: a valid NOT REPRODUCED outcome
                # Cause: the empty slice's zero centroid, weighted by the interpolation fractions.
                prediction = predicted_mean(duration, shift, mean_factor)
                print(
                    f"{mode} {label}: empty-slice prediction={fmt(prediction)}, "
                    f"interpolation fraction range={fraction.min():.4g}..{fraction.max():.4g}"
                )
                if not np.allclose(mean, prediction, rtol=RTOL, atol=ATOL):
                    sys.exit(f"{mode} {label}: the empty-centroid prediction disagrees")
                means.append(mean)
                defect = True
            if len(means) == 2 and not np.allclose(means[1], -2 * means[0], rtol=RTOL, atol=ATOL):
                sys.exit(f"{mode} {label}: the spurious mean does not reverse and double")
    print(f"comparison: kicks atol={ATOL:g}; prediction rtol={RTOL:g}, atol={ATOL:g}")
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

??? example "Helper: _scmult_ring.py"

    [Download _scmult_ring.py](_scmult_ring.py)

    ```python
    """Shared setup for the SCMULT bunch-train reproducers: a closed four-sector ring, bunch trains in
    bunched mode and the beam-moment helpers.

    The ring has four 0.8 m sector bends of angle pi/2 with K1 = -h^2/2 (h = pi/(2 * 0.8 m)), which
    focus equally in both planes, separated by 0.6 m straights; it is stable and closes geometrically.
    `insert_sceffects` places one SCMULT after a drift DS of length `length` at the start of the first
    straight, and WATCH elements record every particle before DS (entrance) and after the SCMULT
    (exit) on each pass. The SCMULT changes only the slopes.
    """

    from __future__ import annotations

    import itertools
    import re
    import sys

    import numpy as np
    from _elegant import particles, run

    # SDDS constants.h, as Elegant uses them.
    C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
    P = 17.0  # electron momentum, units of m_e c
    BETA = P / np.hypot(1, P)
    SX, SY = 0.7e-3, 1.8e-3  # m, rms sizes of the unscaled bunch
    STRAIGHT, BEND_LENGTH = 0.6, 0.8
    H = np.pi / (2 * BEND_LENGTH)  # curvature of a 90-degree sector bend
    SPACING = 2.3e-9  # s between bunches


    def optics(length: float) -> tuple[float, float, float]:
        """Matched (beta, alpha) at the start of the straight and beta at distance `length` into it.

        One cell is a straight followed by a sector bend with K1 = -h^2/2, which focuses equally in
        both planes with k = h/sqrt(2). The periodic Courant-Snyder parameters follow from the
        one-cell matrix M: sin(mu) = sqrt(1 - (tr M/2)^2), beta0 = M12/sin(mu),
        alpha0 = (M11 - M22)/(2 sin(mu)), gamma0 = -M21/sin(mu); a drift of length s transports
        beta to beta0 - 2 alpha0 s + gamma0 s^2.
        """
        k = H / np.sqrt(2)
        phase = k * BEND_LENGTH
        bend = np.array([[np.cos(phase), np.sin(phase) / k], [-k * np.sin(phase), np.cos(phase)]])
        cell = bend @ np.array([[1, STRAIGHT], [0, 1]])
        sine = np.sqrt(1 - (np.trace(cell) / 2) ** 2)
        beta0 = cell[0, 1] / sine
        alpha0 = (cell[0, 0] - cell[1, 1]) / (2 * sine)
        gamma0 = -cell[1, 0] / sine
        return beta0, alpha0, beta0 - 2 * alpha0 * length + gamma0 * length**2


    def bunch(matched: bool, times: np.ndarray) -> np.ndarray:
        """Particles (x, xp, y, yp, t) of one unscaled bunch, shape (n, 5).

        Every combination of the four transverse phases and the arrival times. matched=True places
        the phases 0, 90, 180, 270 degrees of a matched ellipse of rms sizes SX, SY at the start of
        the straight (exact matched second moments, including the within-plane position-slope
        covariance, and no coupling between the transverse planes); otherwise the particles are
        cold, at x = +-SX, y = +-SY with zero slopes.
        """
        if matched:
            a, b, t = np.array(list(itertools.product(np.arange(4), np.arange(4), times))).T
            beta0, alpha0, _ = optics(STRAIGHT)
            x = np.sqrt(2) * SX * np.cos(a * np.pi / 2)
            y = np.sqrt(2) * SY * np.cos(b * np.pi / 2)
            xp = (-alpha0 * x + np.sqrt(2) * SX * np.sin(a * np.pi / 2)) / beta0
            yp = (-alpha0 * y + np.sqrt(2) * SY * np.sin(b * np.pi / 2)) / beta0
        else:
            x, y, t = np.array(list(itertools.product([-SX, SX], [-SY, SY], times))).T
            xp, yp = np.zeros_like(x), np.zeros_like(y)
        return np.column_stack([x, xp, y, yp, t])


    def train(
        specs: tuple[tuple[float, float], ...],
        times: np.ndarray,
        *,
        matched: bool = False,
        shift: tuple[float, float] = (0.0, 0.0),
        reverse: bool = False,
    ) -> dict[str, np.ndarray | int]:
        """Bunch train as SDDS columns; specs lists (transverse scale, duration scale) per bunch.

        Bunch b holds particle IDs b*n+1 ... (b+1)*n (`IDSlotsPerBunch` = n) and arrives b * SPACING
        after the first. Rows are interleaved so that every MPI worker holds part of each bunch;
        reverse=True reverses the row order, so that bunch order does not follow file order.
        """
        base = bunch(matched, times)
        n = len(base)
        blocks = []
        for index, (scale, zscale) in enumerate(specs):
            x, xp, y, yp, t = base.T
            blocks.append(
                np.column_stack(
                    [
                        scale * x + shift[0],
                        scale * xp,
                        scale * y + shift[1],
                        scale * yp,
                        zscale * t + index * SPACING,
                    ]
                )
            )
        values = np.concatenate(blocks)
        ids = np.arange(1, len(values) + 1)
        values = values.reshape(len(specs), n, 5).transpose(1, 0, 2).reshape(-1, 5)
        ids = ids.reshape(len(specs), n).T.reshape(-1)
        if reverse:
            values, ids = values[::-1], ids[::-1]
        columns: dict[str, np.ndarray | int] = {
            key: values[:, i] for i, key in enumerate(("x", "xp", "y", "yp", "t"))
        }
        columns["p"] = np.full(len(values), P)
        columns["particleID"] = ids
        columns["IDSlotsPerBunch"] = n
        return columns


    def track(
        mode: str,
        columns: dict,
        bunches: int,
        *,
        length: float = STRAIGHT,
        charge: float = 2e-10,
        passes: int = 1,
        nonlinear: int = 0,
        factor: float = 1.0,
        duration: float = 0.0,
        interpolate: int = 0,
    ) -> tuple[list[dict], list[dict]]:
        """Run the ring for `passes` turns; return per-pass coordinates (entrance, exit) by particle ID.

        `charge` is the charge of each bunch. The SCMULT acts after DS, a drift of `length`.
        """
        lattice = (
            f"Q: CHARGE,TOTAL={bunches * charge!r}\n"
            'W0: WATCH,FILENAME="entrance.sdds",MODE="coordinates"\n'
            f"DS: DRIF,L={length!r}\n"
            'W1: WATCH,FILENAME="exit.sdds",MODE="coordinates"\n'
            f"DT: DRIF,L={STRAIGHT - length!r}\n"
            f"DA: DRIF,L={STRAIGHT!r}\n"
            f"B: SBEND,L={BEND_LENGTH!r},ANGLE={np.pi / 2!r},K1={-(H**2) / 2!r}\n"
            "BL: LINE=(Q,W0,DS,W1,DT,B,DA,B,DA,B,DA,B)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            '&insert_sceffects name="DS",skip=1,horizontal=1,vertical=1,'
            f"nonlinear={nonlinear},averaging_factor={factor!r},slice_duration={duration!r},"
            f"slice_interpolation={interpolate} &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},default_order=2,"
            "output=out.sdds &end\n"
            f"&run_control n_steps=1,n_passes={passes} &end\n"
            "&twiss_output matched=1 &end\n"
            "&sdds_beam input=beam.sdds,use_bunched_mode=1,track_pages_separately=0 &end\n"
            "&track soft_failure=0 &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": columns},
            outputs=("entrance.sdds", "exit.sdds"),
        )
        total = len(columns["particleID"])
        pages = []
        for name in ("entrance.sdds", "exit.sdds"):
            sdds = result[name]
            if sdds.n_pages != passes:
                sys.exit(f"{mode}: {name} has {sdds.n_pages} pages for {passes} passes")
            pages.append([particles(sdds, k) for k in range(passes)])
        for page in pages[0] + pages[1]:
            if not np.array_equal(page["particleID"], np.arange(1, total + 1)):
                sys.exit(f"{mode}: particles lost")
            if not all(np.isfinite(page[key]).all() for key in ("x", "xp", "y", "yp", "t")):
                sys.exit(f"{mode}: non-finite coordinates")
        if re.search(r"\b(error|fatal|nan)\b", result["log"], re.IGNORECASE):
            sys.exit(f"{mode}: diagnostics in the native log")
        return pages[0], pages[1]


    def rms(page: dict, rows: slice = slice(None)) -> np.ndarray:
        """Rms sizes (x, y, z) in m of a bunch; z = beta c times the arrival-time spread."""
        return np.array(
            [np.std(page["x"][rows]), np.std(page["y"][rows]), BETA * C * np.std(page["t"][rows])]
        )


    def fmt(values: np.ndarray) -> str:
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values) + 0.0) + "]"

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

