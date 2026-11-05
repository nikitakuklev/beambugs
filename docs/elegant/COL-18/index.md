---
title: "COL-18: SCMULT size-filter history is shared between bunches"
hide:
  - toc
---

# COL-18 — SCMULT size-filter history is shared between bunches

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [bunch loop, insertSCeffects.cc:316–323](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L316-L323)<br>[lastSigma, track.h:3458](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track.h#L3458)<br>[averaging_factor, elegant.tex:4492–4503](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4492-L4503) |

Nonlinear SCMULT bunch-train tracking with size-averaging factor 0 < f < 1 uses one shared size history per element: after the first turn, each bunch filters against history last updated by another bunch instead of its own previous turn. In this independent-bunch model, changing another bunch must not change the kick; an isolated bunch checked against an independent Gaussian-field integral supplies the control. At momentum 17 mₑc, charge 0.2000 pC per bunch and f = 0.4000, making a second bunch 2.500 times wider changes the first bunch's second-turn horizontal slope kick on its first particle from 2.510×10⁻⁹ to 4.532×10⁻⁹; making it 2.500 times longer changes it to 2.733×10⁻⁹.

??? example "Reproducer: COL-18.py"

    [Download COL-18.py](COL-18.py)

    ```python
    """COL-18: SCMULT size-filter history is shared between bunches

    Nonlinear SCMULT bunch-train tracking with size-averaging factor 0 < f < 1 uses one shared size
    history per element: after the first turn, each bunch filters against history last updated by
    another bunch instead of its own previous turn. In this independent-bunch model, changing another
    bunch must not change the kick; an isolated bunch checked against an independent Gaussian-field
    integral supplies the control. At momentum 17 mₑc, charge 0.2000 pC per bunch and f = 0.4000, making
    a second bunch 2.500 times wider changes the first bunch's second-turn horizontal slope kick on its
    first particle from 2.510×10⁻⁹ to 4.532×10⁻⁹; making it 2.500 times longer changes it to
    2.733×10⁻⁹.

    The script tracks two turns of a four-sector ring with SCMULT, first one matched bunch alone and
    then the same bunch followed by a second bunch that is wider or longer, and reads the first bunch's
    slope kick on the second turn. It compares that kick with the isolated bunch's kick, with a
    Poisson-integral Gaussian-field oracle that filters each bunch against its own history, and with
    the same oracle filtering against the other bunch's size. Controls are f = 1.000, which bypasses
    filtering, and an identical second bunch. A reversed bunch order is also run.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _scmult_ring import BETA, QE, RE, STRAIGHT, C, P, fmt, optics, rms, track, train
    from _scring import poisson_field

    TIMES = np.array([-3, -2, -1, 1, 2, 3]) * 17e-12  # s, arrival times within a bunch
    CHARGE = 2e-13  # C per bunch
    N = 96  # particles per bunch: 16 transverse phase combinations at 6 arrival times
    ATOL = 5e-18  # slope; the controls agree exactly
    RTOL_ORACLE, ATOL_ORACLE = 5e-7, 2e-16  # native Gaussian-force kernel accuracy


    def oracle(entrance, exit_, sigma_hat):
        """Slope kick (xp, yp) of every particle of one bunch, with filtered sizes `sigma_hat`.

        SCMULT's nonlinear kick is dx' = K lambda(z) (D/beta_kick) sigma_x (sigma_x + sigma_y) F_x(x, y),
        with F the Gaussian transverse field of rms sizes sigma_hat (normalized to x/[sx (sx + sy)]
        near the axis; here the Poisson integral, independent of Elegant's error-function kernel),
        K lambda = sqrt(2/pi) r_e N_e/(p^2 gamma sz) exp(-z^2/(2 sz^2)), N_e = Q/e, and D the
        trapezoid of beta/[sigma_x (sigma_x + sigma_y)] over the drift from the true (unfiltered)
        entrance and exit moments. The matched beta is the same at both ends of the full straight.
        """
        before, after = rms(entrance)[:2], rms(exit_)[:2]
        beta0, _, beta1 = optics(STRAIGHT)
        weight = STRAIGHT / 2 * (beta0 / (before * before.sum()) + beta1 / (after * after.sum()))
        sx, sy, sz = sigma_hat
        xy = np.column_stack([exit_["x"], exit_["y"]])
        centered = xy - xy.mean(axis=0)
        field = poisson_field(centered[:, 0], centered[:, 1], sx, sy)
        z = BETA * C * (exit_["t"] - exit_["t"].mean())
        strength = np.sqrt(2 / np.pi) * RE * (CHARGE / QE) / (P**2 * np.hypot(P, 1))
        line = strength / sz * np.exp(-(z**2) / (2 * sz**2))
        return line[:, None] * weight / beta1 * np.array([sx, sy]) * (sx + sy) * field


    def second_turn_kick(entrance, exit_, rows):
        """Slope change (xp, yp) of the particles `rows` during the second turn's SCMULT kick."""
        return np.column_stack([exit_[1][key][rows] - entrance[1][key][rows] for key in ("xp", "yp")])


    def pages(page, rows):
        return {key: value[rows] for key, value in page.items()}


    defect = False
    first, other = slice(0, N), slice(N, 2 * N)
    for mode in modes():
        for factor in (0.4, 0.7, 1.0):
            single = train(((1, 1),), TIMES, matched=True)
            s_in, s_out = track(mode, single, 1, charge=CHARGE, passes=2, nonlinear=1, factor=factor)
            expected = second_turn_kick(s_in, s_out, slice(None))
            # Filtered size for the isolated bunch: its own size on this turn and on the last turn.
            own = factor * rms(s_out[1]) + (1 - factor) * rms(s_out[0])
            own_oracle = oracle(s_in[1], s_out[1], own)
            if not np.allclose(expected, own_oracle, rtol=RTOL_ORACLE, atol=ATOL_ORACLE):
                sys.exit(f"{mode} f={factor}: the isolated kick disagrees with the field oracle")
            for label, specs in (
                ("identical second bunch", ((1, 1), (1, 1))),
                ("second bunch 2.5 times wider", ((1, 1), (2.5, 1))),
                ("second bunch 2.5 times longer", ((1, 1), (1, 2.5))),
            ):
                columns = train(specs, TIMES, matched=True)
                e_in, e_out = track(
                    mode, columns, 2, charge=CHARGE, passes=2, nonlinear=1, factor=factor
                )
                # Neither bunch can affect the other's coordinates before the second-turn kick.
                for key in ("x", "xp", "y", "yp"):
                    if not np.allclose(e_in[1][key][first], s_in[1][key], rtol=0, atol=2e-17):
                        sys.exit(f"{mode} f={factor} {label}: the first bunch's entrance changed")
                observed = second_turn_kick(e_in, e_out, first)
                # The shared history holds the second bunch's size from the last turn.
                shared = factor * rms(e_out[1], first) + (1 - factor) * rms(e_out[0], other)
                shared_oracle = oracle(pages(e_in[1], first), pages(e_out[1], first), shared)
                bad = not np.allclose(observed, expected, rtol=0, atol=ATOL)
                # The shared-history oracle diagnoses a deviation from the isolated kick; a bunch
                # that matches the isolated kick is accepted (a corrected own-history filter).
                if bad and not np.allclose(observed, shared_oracle, rtol=RTOL_ORACLE, atol=ATOL_ORACLE):
                    sys.exit(f"{mode} f={factor} {label}: shared-history prediction fails")
                print(
                    f"{mode} f={factor} {label}: observed={fmt(observed[0])} expected={fmt(expected[0])} "
                    f"shared-history prediction={fmt(shared_oracle[0])} "
                    f"max difference={np.max(np.abs(observed - expected)):.6g}"
                )
                if factor == 1.0 or specs[1] == (1, 1):
                    if bad:
                        sys.exit(f"{mode} f={factor} {label}: control differs from the isolated bunch")
                else:
                    defect |= bad
        # Reversed bunch order: the wide bunch has the first IDs and the unit bunch the last.
        factor = 0.4
        single = train(((1, 1),), TIMES, matched=True)
        s_in, s_out = track(mode, single, 1, charge=CHARGE, passes=2, nonlinear=1, factor=factor)
        expected = second_turn_kick(s_in, s_out, slice(None))
        columns = train(((2.5, 1), (1, 1)), TIMES, matched=True, reverse=True)
        e_in, e_out = track(mode, columns, 2, charge=CHARGE, passes=2, nonlinear=1, factor=factor)
        # The unit bunch's incoming coordinates must equal the isolated run's before the kick.
        for key in ("x", "xp", "y", "yp"):
            if not np.allclose(e_in[1][key][other], s_in[1][key], rtol=0, atol=2e-17):
                sys.exit(f"{mode} reversed order: the unit bunch's entrance changed")
        observed = second_turn_kick(e_in, e_out, other)
        # The first-ID bunch reads the last turn's last bunch; the last-ID bunch then reads that
        # bunch's filtered size.
        wide = factor * rms(e_out[1], first) + (1 - factor) * rms(e_out[0], other)
        shared = factor * rms(e_out[1], other) + (1 - factor) * wide
        shared_oracle = oracle(pages(e_in[1], other), pages(e_out[1], other), shared)
        bad = not np.allclose(observed, expected, rtol=0, atol=ATOL)
        if bad and not np.allclose(observed, shared_oracle, rtol=RTOL_ORACLE, atol=ATOL_ORACLE):
            sys.exit(f"{mode} reversed order: shared-history prediction fails")
        print(
            f"{mode} f={factor} reversed bunch order, 2.5 times wider first bunch: "
            f"observed={fmt(observed[0])} expected={fmt(expected[0])} "
            f"shared-history prediction={fmt(shared_oracle[0])} "
            f"max difference={np.max(np.abs(observed - expected)):.6g}"
        )
        defect |= bad
    print(f"comparison: atol={ATOL:g} against the isolated bunch; field oracle rtol={RTOL_ORACLE:g}")
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

??? example "Helper: _scring.py"

    [Download _scring.py](_scring.py)

    ```python
    """Shared setup for the SCMULT space-charge reproducers: a closed four-bend ring, a gridded beam and
    the Gaussian-beam kick computed from the Poisson integral, independently of Elegant's code.

    One turn around a ring of four 1 m sector bends (angle pi/2, K1 = -0.5 m^-2), each preceded by a
    drift of length L. `insert_sceffects` places one SCMULT after the first drift, and a WATCH right
    after it records every particle's slope kick before the first bend.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import particles, run
    from scipy.integrate import quad
    from scipy.special import ndtri

    # SDDS constants.h, as Elegant uses them.
    C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
    P = 10.0  # momentum, units of m_e c (electrons)
    GAMMA = np.sqrt(1 + P**2)
    BETA = P / GAMMA
    CHARGE = 1e-9  # C
    BEND_LENGTH = 1.0  # m, four bends make the ring


    def levels(n: int, uniform: bool = False) -> np.ndarray:
        """n symmetric levels of unit rms: Gaussian quantiles, or equally spaced for a uniform beam."""
        q = (np.arange(n) + 0.5) / n
        q = q - 0.5 if uniform else ndtri(q)
        return q / np.std(q)


    def beam(
        sx: float, sy: float, times: np.ndarray, n: int = 8, copies: int = 1
    ) -> dict[str, np.ndarray]:
        """Particles on the full grid of n levels in x and y and the given arrival times, zero slopes.

        The rms sizes are exactly sx and sy, which is what SCMULT reads from the particles.
        """
        q = levels(n)
        x, y, t = (a.ravel() for a in np.meshgrid(sx * q, sy * q, times, indexing="ij"))
        x, y, t = (np.tile(a, copies) for a in (x, y, t))
        return {"x": x, "y": y, "t": t, "p": np.full(len(x), P), "particleID": np.arange(1, len(x) + 1)}


    def circumference(length: float) -> float:
        return 4 * (BEND_LENGTH + length)


    def track(
        mode: str,
        length: float,
        particles_in: dict[str, np.ndarray],
        *,
        horizontal: int = 1,
        vertical: int = 1,
        nonlinear: int = 1,
        uniform: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slope kicks (xp, yp) of every particle after SCMULT, in input order, and the effective
        integrating lengths of the two planes.

        The effective length of a drift of constant beam size is L (beta_in + beta_out) / (2 beta_out),
        from the trapezoid weighting that SCMULT applies to the matched beta functions at the entrance
        and exit of the drift; the beta functions are read from the matched Twiss output.
        """
        lattice = (
            f"Q: CHARGE,TOTAL={CHARGE!r}\n"
            f"D: DRIF,L={length!r}\n"
            'W: WATCH,FILENAME="watch.sdds",MODE="coordinates"\n'
            f"A: DRIF,L={length!r}\n"
            f"B: SBEND,L={BEND_LENGTH!r},ANGLE={np.pi / 2!r},K1=-0.5\n"
            "BL: LINE=(Q,D,W,B,A,B,A,B,A,B)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f'&insert_sceffects name="D",skip=1,horizontal={horizontal},vertical={vertical},'
            f"nonlinear={nonlinear},uniform_distribution={uniform} &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            '&twiss_output filename="twiss.sdds",matched=1 &end\n'
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": particles_in},
            outputs=("watch.sdds", "twiss.sdds"),
        )
        twiss = result["twiss.sdds"]
        names = list(twiss[0, "ElementName"])
        if not names.index("D") < names.index("MYSC") < names.index("W"):
            raise RuntimeError(f"unexpected element order {names}")
        leff = np.array(
            [
                length * (b[0] + b[names.index("D")]) / (2 * b[names.index("MYSC")])
                for b in (twiss[0, "betax"], twiss[0, "betay"])
            ]
        )
        if not (np.all(np.isfinite(leff)) and np.all(leff > 0)):
            raise RuntimeError(f"{mode}: invalid effective lengths {leff}")
        out = particles(result["watch.sdds"])
        if not np.array_equal(out["particleID"], particles_in["particleID"]):
            raise RuntimeError(f"{mode}: particles lost before the kick")
        # The initial slopes are zero, so the slopes after SCMULT are the kicks.
        return np.column_stack([out["xp"], out["yp"]]), leff


    def poisson_field(x: np.ndarray, y: np.ndarray, sx: float, sy: float) -> np.ndarray:
        """Transverse field (Fx, Fy), in 1/m, of a Gaussian charge density with rms sizes sx and sy.

        Heat-kernel solution of the 2D Poisson equation, written with positive real integrands:
          Fx = x/2 int_0^inf exp[-x^2/(2(sx^2+t)) - y^2/(2(sy^2+t))] / [(sx^2+t)^(3/2) (sy^2+t)^(1/2)] dt
        and Fy with the planes exchanged. Round beams give Gauss's law,
        F = (1 - exp[-r^2/(2 sigma^2)]) (x, y)/r^2, and the origin gives the linear force,
        Fx = x/[sx (sx + sy)].
        """
        r2 = (sy / sx) ** 2  # t = sx^2 u makes the integrals dimensionless

        def integral(a: float, b: float, plane: int) -> float:
            def integrand(u: float) -> float:
                common = np.exp(-(a**2) / (2 * (1 + u)) - b**2 / (2 * (r2 + u)))
                common /= 2 * np.sqrt((1 + u) * (r2 + u))
                return common * (a / (1 + u) if plane == 0 else b / (r2 + u)) / sx

            return quad(integrand, 0, np.inf, epsabs=0, epsrel=1e-13)[0]

        pairs, inverse = np.unique(np.column_stack([x, y]), axis=0, return_inverse=True)
        unique = np.array(
            [[integral(px / sx, py / sx, 0), integral(px / sx, py / sx, 1)] for px, py in pairs]
        )
        return unique[inverse.reshape(-1)]


    def kick(
        particles_in: dict[str, np.ndarray],
        leff: np.ndarray,
        *,
        nonlinear: bool,
        uniform: bool = False,
    ) -> np.ndarray:
        """Physical slope kick (xp, yp) of every particle, shape (n, 2).

        dx' = K L lambda F, with K = 2 r_e / (beta^2 gamma^3) and L the effective length of the plane;
        K multiplies the particle number per metre lambda, not a charge density.
        beta^2 gamma^3 = P^2 sqrt(1 + P^2). The longitudinal density lambda = N f(z), in particles per
        metre with N = CHARGE/e, is Gaussian, f = exp(-z^2/(2 sz^2))/(sqrt(2 pi) sz), or uniform over the
        interval of the same rms, f = 1/(sqrt(12) sz). The field F is the Gaussian transverse field
        (nonlinear) or its linear part near the axis.
        """
        sx, sy = np.std(particles_in["x"]), np.std(particles_in["y"])
        x = particles_in["x"] - np.mean(particles_in["x"])
        y = particles_in["y"] - np.mean(particles_in["y"])
        z = BETA * C * (particles_in["t"] - np.mean(particles_in["t"]))
        sz = np.std(z)
        shape = np.full_like(z, 1 / (np.sqrt(12) * sz)) if uniform else _gauss(z, sz)
        strength = 2 * (CHARGE / QE) * RE / (P**2 * np.sqrt(1 + P**2))
        if nonlinear:
            field = poisson_field(x, y, sx, sy)
        else:
            field = np.column_stack([x / (sx * (sx + sy)), y / (sy * (sx + sy))])
        return strength * shape[:, None] * leff[None, :] * field


    def _gauss(z: np.ndarray, sz: float) -> np.ndarray:
        return np.exp(-(z**2) / (2 * sz**2)) / (np.sqrt(2 * np.pi) * sz)


    def fmt(values: np.ndarray) -> str:
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"


    def require_finite(label: str, *arrays: np.ndarray) -> None:
        """Fail the run when an array holds a NaN or infinity; NaN would otherwise pass as a mismatch."""
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise RuntimeError(f"{label}: nonfinite values")


    def compare(
        label: str, observed: np.ndarray, expected: np.ndarray, rtol: float, atol: float = 1e-18
    ) -> bool:
        """Print the first particle's (xp, yp) and return True when any particle disagrees.

        Arrays of different shape or with a nonfinite entry make the comparison invalid: the run fails
        instead of returning a verdict.
        """
        observed, expected = np.asarray(observed), np.asarray(expected)
        if observed.shape != expected.shape:
            raise RuntimeError(f"{label}: shapes {observed.shape} and {expected.shape} differ")
        require_finite(label, observed, expected)
        print(f"{label}: observed={fmt(observed[0])} expected={fmt(expected[0])}")
        return not np.allclose(observed, expected, rtol=rtol, atol=atol)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

