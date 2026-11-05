---
title: "COL-17: Later SCMULT bunches use the exit beta function for the entrance weight"
hide:
  - toc
---

# COL-17 — Later SCMULT bunches use the exit beta function for the entrance weight

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`accumulateSCMULT` entrance optics, insertSCeffects.cc:762](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L762)<br>[entrance weights, insertSCeffects.cc:837–839](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L837-L839)<br>[exit-optics assignment, insertSCeffects.cc:849](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L849)<br>[bunch-mode section, elegant.tex:484](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L484)<br>[SCMULT's model, scmult.tex:10–24](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/scmult.tex#L10-L24) |

When Elegant tracks a bunch train with SCMULT space-charge kicks (`insert_sceffects` in bunch mode), every bunch after the first integrates the space-charge strength over each element with the element's exit beta function β in place of its entrance value, so identical bunches receive different kicks. A bunch's kick depends only on its own charge distribution and the optics; the manual's linear force integrated with independently computed matched optics reproduces the first bunch's kick. Where β falls from 0.9141 m to 0.8018 m along a 0.3 m drift, the second and third of three identical bunches receive 6.542% less kick than the first, with linear and nonlinear forces alike.

??? example "Reproducer: COL-17.py"

    [Download COL-17.py](COL-17.py)

    ```python
    """COL-17: Later SCMULT bunches use the exit beta function for the entrance weight

    When Elegant tracks a bunch train with SCMULT space-charge kicks (`insert_sceffects` in bunch mode),
    every bunch after the first integrates the space-charge strength over each element with the
    element's exit beta function β in place of its entrance value, so identical bunches receive
    different kicks. A bunch's kick depends only on its own charge distribution and the optics; the
    manual's linear force integrated with independently computed matched optics reproduces the first
    bunch's kick. Where β falls from 0.9141 m to 0.8018 m along a 0.3 m drift, the second and third of
    three identical bunches receive 6.542% less kick than the first, with linear and nonlinear forces
    alike.

    The script tracks one bunch, then three identical bunches, through a straight of a stable
    four-sector ring where `insert_sceffects` places an SCMULT after a drift of length L, and reads
    each particle's slope kick just after it. It compares the linear kicks with the manual's Gaussian
    force integrated over the drift with the matched beta functions of an independent matrix
    calculation, and the elliptical nonlinear kicks with those of the isolated bunch. The control is
    a drift spanning the whole straight, where the entrance and exit beta functions are equal.
    """

    import itertools
    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h, as Elegant uses them.
    C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
    P = 17.0  # electron momentum, units of m_e c
    BETA = P / np.hypot(1, P)
    SX, SY = 0.7e-3, 1.8e-3  # m
    TIMES = np.array([-3, -2, -1, 1, 2, 3]) * 17e-12  # s, arrival times within a bunch
    # A cold bunch: every combination of x = +-SX, y = +-SY and the arrival times, zero slopes.
    X, Y, T = np.array(list(itertools.product([-SX, SX], [-SY, SY], TIMES))).T
    N = len(X)
    SPACING = 2.3e-9  # s between bunches
    CHARGE = 2e-10  # C per bunch
    STRAIGHT, BEND_LENGTH = 0.6, 0.8
    H = np.pi / (2 * BEND_LENGTH)  # curvature of a 90-degree sector bend
    RTOL, ATOL = 3e-12, 2e-17
    NB = 3


    def optics(length):
        """Matched beta at the start of the straight and at distance `length` into it.

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
        return beta0, beta0 - 2 * alpha0 * length + gamma0 * length**2


    def kicks(mode, length, nonlinear, bunches):
        """Slope kicks (xp, yp) after SCMULT, shape (bunches, particles per bunch, 2)."""
        ids = np.arange(1, N * bunches + 1)
        stack = {
            "x": np.tile(X, bunches),
            "y": np.tile(Y, bunches),
            "particleID": ids,
        }
        stack["t"] = (T + SPACING * np.arange(bunches)[:, None]).ravel()
        # Interleave the bunches row by row, so that every MPI worker holds part of each bunch, and
        # reverse the row order, so that bunch order does not follow file order.
        order = np.arange(N * bunches).reshape(bunches, N).T.ravel()[::-1]
        beam = {key: value[order] for key, value in stack.items()}
        beam["p"] = np.full(N * bunches, P)
        beam["IDSlotsPerBunch"] = N  # SDDS parameter: bunch b holds IDs b*N+1 ... (b+1)*N
        lattice = (
            f"Q: CHARGE,TOTAL={bunches * CHARGE!r}\n"
            f"DS: DRIF,L={length!r}\n"
            'W1: WATCH,FILENAME="exit.sdds",MODE="coordinates"\n'
            f"DT: DRIF,L={STRAIGHT - length!r}\n"
            f"DA: DRIF,L={STRAIGHT!r}\n"
            f"B: SBEND,L={BEND_LENGTH!r},ANGLE={np.pi / 2!r},K1={-(H**2) / 2!r}\n"
            "BL: LINE=(Q,DS,W1,DT,B,DA,B,DA,B,DA,B)\n"
        )
        # insert_sceffects places one SCMULT after DS; it integrates the optics over DS alone.
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f'&insert_sceffects name="DS",skip=1,horizontal=1,vertical=1,nonlinear={nonlinear} &end\n'
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&twiss_output matched=1 &end\n"
            "&sdds_beam input=beam.sdds,use_bunched_mode=1,track_pages_separately=0 &end\n"
            "&track &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": beam},
            outputs=("exit.sdds",),
        )
        out = particles(result["exit.sdds"])
        if not np.array_equal(out["particleID"], np.arange(1, N * bunches + 1)):
            sys.exit(f"{mode}: particles lost before the kick")
        # Initial slopes are zero, so the slopes after SCMULT are the kicks.
        slopes = np.stack([out["xp"], out["yp"]], axis=-1)
        if slopes.shape != (bunches * N, 2) or not np.all(np.isfinite(slopes)):
            sys.exit(f"{mode}: expected {bunches * N} finite kicks, got shape {slopes.shape}")
        return slopes.reshape(bunches, N, 2)


    def linear_kick(length):
        """Manual's linear Gaussian force integrated over the drift, for the 24-particle bunch.

        dx' = K L exp(-z^2/2 sz^2)/(sqrt(2 pi) sz) * x/[sx (sx + sy)], K = 2 N r_e/(gamma^3 beta^2),
        gamma^3 beta^2 = p^2 sqrt(1 + p^2). SCMULT weights the optics by the trapezoid over the drift
        and converts at the kick with the local beta, so a beam of constant size receives
        L_eff = L (beta_in + beta_out)/(2 beta_out) in place of L.
        """
        beta_in, beta_out = optics(length)
        leff = length * (beta_in + beta_out) / (2 * beta_out)
        z = BETA * C * (T - T.mean())
        sz = np.std(z)
        strength = 2 * (CHARGE / QE) * RE / (P**2 * np.sqrt(1 + P**2))
        line = strength * leff * np.exp(-(z**2) / (2 * sz**2)) / (np.sqrt(2 * np.pi) * sz)
        return np.column_stack([line * X / (SX * (SX + SY)), line * Y / (SY * (SX + SY))])


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"


    def compare(label, observed, expected):
        """Print the first particle's (xp, yp) kick; compare all particles."""
        bad = not np.allclose(observed, expected, rtol=RTOL, atol=ATOL)
        print(f"{label}: observed={fmt(observed[0])} expected={fmt(expected[0])}")
        return bad


    defect = False
    print("kicks (xp, yp) of the first particle of each bunch")
    for mode in modes():
        for length, nonlinear in ((0.3, 0), (0.3, 1), (STRAIGHT, 0)):
            beta_in, beta_out = optics(length)
            name = f"{mode} L={length} {'nonlinear' if nonlinear else 'linear'}"
            alone = kicks(mode, length, nonlinear, 1)[0]
            train = kicks(mode, length, nonlinear, NB)
            if nonlinear:
                # Identical bunches in the same optics must receive the isolated bunch's kicks.
                expected = alone
            else:
                expected = linear_kick(length)
                if compare(f"{name}, single bunch", alone, expected):
                    sys.exit("The single-bunch kick disagrees with the analytic force.")
            # The first bunch is unaffected by the faulty pointer: it must match the isolated bunch.
            if compare(f"{name}, bunch 1 against isolated bunch (control)", train[0], alone):
                sys.exit("The first bunch of the train disagrees with the isolated bunch.")
            # The faulty pointer gives later bunches beta_out at both ends of the trapezoid.
            prediction = 2 * beta_out / (beta_in + beta_out)
            first = train[0, :, 0]
            if not (np.all(np.isfinite(first)) and np.all(first != 0)):
                sys.exit(f"{name}: the first-bunch reference kicks are not finite and nonzero.")
            for b in range(NB):
                bad = compare(f"{name}, bunch {b + 1} of {NB}", train[b], expected)
                ratio = train[b, :, 0] / first
                if b:
                    print(
                        f"{name}, bunch {b + 1}: later/first ratio={ratio.min():.16g}..{ratio.max():.16g}, "
                        f"faulty-pointer prediction 2*beta_out/(beta_in+beta_out)={prediction:.16g}"
                    )
                if length == STRAIGHT:
                    # Equal entrance and exit beta: every bunch is a required control.
                    if bad:
                        sys.exit("An equal-endpoint control bunch disagrees with the expected kick.")
                elif bad:
                    # A failing 0.3 m bunch must follow the faulty-pointer ratio, or the diagnosis
                    # of the cause fails. A ratio of one (a correct map) is accepted unflagged.
                    if not np.allclose(ratio, prediction, rtol=RTOL, atol=0):
                        sys.exit("The error of a later bunch does not follow the beta_out weight.")
                    defect = True
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

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

