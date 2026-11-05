---
title: "COL-29: insert_sceffects clear leaves a negative specification count, crashing or silently dropping every SCMULT"
hide:
  - toc
---

# COL-29 — insert_sceffects clear leaves a negative specification count, crashing or silently dropping every SCMULT

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [clearSCSpecs, get_beamline.c:2364–2377](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/get_beamline.c#L2364-L2377)<br>[clear and disable, insertSCeffects.cc:81–87](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L81-L87)<br>[warning, insertSCeffects.cc:112–116](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L112-L116)<br>[addSCSpec, get_beamline.c:2339–2357](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/get_beamline.c#L2339-L2357)<br>[disable and clear, elegant.tex:4480–4481](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4480-L4481) |

Setting clear = 1 on insert_sceffects decrements the count of insertion specifications to −1 instead of 0, so the next specification is written one entry before its array. Clearing and replacing an earlier command with the same selector then completes with no SCMULT and no warning, so the run has no space charge; clearing in a fresh run, clearing alone, and clearing with disable = 1 after an earlier command crash. The manual says clear deletes prior insertions and disable makes a command ignored, so each run must complete, and all but the lone clear must insert one SCMULT. In a one-segment ring with a 0.8000 nC bunch, the replacement's expected linear slope kick of 2.379×10⁻⁴ becomes exactly zero.

??? example "Reproducer: COL-29.py"

    [Download COL-29.py](COL-29.py)

    ```python
    """COL-29: insert_sceffects clear leaves a negative specification count, crashing or silently dropping every SCMULT

    Setting clear = 1 on insert_sceffects decrements the count of insertion specifications to −1 instead
    of 0, so the next specification is written one entry before its array. Clearing and replacing an
    earlier command with the same selector then completes with no SCMULT and no warning, so the run has
    no space charge; clearing in a fresh run, clearing alone, and clearing with disable = 1 after an
    earlier command crash. The manual says clear deletes prior insertions and disable makes a command
    ignored, so each run must complete, and all but the lone clear must insert one SCMULT. In a
    one-segment ring with a 0.8000 nC bunch, the replacement's expected linear slope kick of 2.379×10⁻⁴
    becomes exactly zero.

    The script runs a one-segment ring (a 0.8 nC bunch of 243 particles and a 1 m EMATRIX quarter
    turn) with insert_sceffects commands that use clear: clear and replace a prior command, clear on a
    fresh run, clear alone, and clear together with disable after a prior command. The expected outcome
    follows the manual's clear and disable descriptions, and the expected kick is the manual's linear
    formula for one SCMULT after the segment. A prior command followed by a disabled one is the control.
    """

    import numpy as np
    from _elegant import verdict
    from _scinsert import N, failure_line, linear_kick, spec, track

    EXPECTED = linear_kick()
    PROBE = int(np.argmax(EXPECTED[:, 0]))  # the reported particle: largest horizontal kick
    ATOL = 1e-18
    PRIOR = spec(name="R")
    # (label, commands, SCMULT insertions the manual requires, control?)
    CASES = (
        ("prior command, then disable=1 (control)", PRIOR + spec(name="R", disable=1), 1, True),
        (
            "prior command, then clear=1 with the same selector",
            PRIOR + spec(name="R", clear=1),
            1,
            False,
        ),
        ("clear=1 with a selector, fresh run", spec(name="R", clear=1), 1, False),
        ("clear=1 alone, fresh run", "&insert_sceffects clear=1 &end\n", 0, False),
        ("prior command, then clear=1,disable=1", PRIOR + spec(name="R", clear=1, disable=1), 1, False),
    )

    defect = False
    # Serial only: after the corrupting write, Pelegant can spin without finishing, so a 3-rank run
    # would wait for the native timeout.
    for mode in ("serial",):
        for label, commands, count, control in CASES:
            result = track(commands, mode)
            expected = f"exit 0, {count} SCMULT" + (
                f", kick={EXPECTED[PROBE, 0]:.16g}" if count else ", no kick"
            )
            if result["status"]:
                observed = f"exit {result['status']} ({failure_line(result['log'])})"
                ok = False
            else:
                kick = result["kick"]
                observed = f"exit 0, {len(result['insertions'])} SCMULT, kick={kick[PROBE, 0]:.16g}"
                want = EXPECTED if count else np.zeros((N, 2))
                ok = len(result["insertions"]) == count and bool(
                    np.allclose(kick, want, rtol=0, atol=ATOL)
                )
            print(f"{mode} {label}: observed={observed} expected={expected} (particle {PROBE + 1})")
            if control and not ok:
                raise SystemExit(f"{mode} {label}: the control fails")
            defect |= not ok
    print(f"comparison: kicks atol={ATOL:g} against the manual's linear kick for one SCMULT")
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

??? example "Helper: _scinsert.py"

    [Download _scinsert.py](_scinsert.py)

    ```python
    """Shared setup for the insert_sceffects command reproducers: a one-segment ring, a bunch with exact
    second moments, the manual's linear space-charge kick and a runner that reports the SCMULT
    insertions.

    The ring is a 0.8 nC electron bunch (p = 10 mₑc) followed by R, a 1 m symplectic EMATRIX quarter
    turn matched at beta = 1 m in both planes. With one linear SCMULT after R, every particle's slope
    change over one turn is the manual's linear kick evaluated with the rms sizes after R.
    """

    from __future__ import annotations

    import itertools
    import re

    import numpy as np
    from _elegant import RunFailure, particles, run

    # SDDS constants.h, as Elegant uses them.
    C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
    P = 10.0  # electron momentum, units of m_e c
    GAMMA = np.hypot(1, P)
    BETA_C = C * P / GAMMA
    CHARGE = 0.8e-9  # C
    SX, SY, SZ = 1.5e-3, 0.8e-3, 0.025  # m
    LENGTH = 1.0  # m

    # R: (x, xp) -> (xp, -x) and (y, yp) -> (yp, -y), a quarter turn at beta = 1 m, over 1 m.
    LATTICE = (
        f"Q: CHARGE,TOTAL={CHARGE!r}\n"
        f"R: EMATRIX,L={LENGTH!r},ORDER=1,R12=1,R21=-1,R34=1,R43=-1,R55=1,R66=1\n"
        "RING: LINE=(Q,R)\n"
    )
    TAIL = (
        f"&run_setup lattice=lattice.lte,use_beamline=RING,p_central={P!r},default_order=1,"
        "output=out.sdds &end\n"
        '&twiss_output matched=1,filename="twiss.sdds" &end\n'
        "&run_control n_steps=1,n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )

    # Every combination of -sqrt(1.5), 0, +sqrt(1.5) in (x, xp, y, yp, z): exact unit variances,
    # zero means and no correlations, so the bunch has rms sizes SX, SY, SZ and is matched at 1 m.
    _UNIT = np.array(list(itertools.product([-np.sqrt(1.5), 0.0, np.sqrt(1.5)], repeat=5)))
    N = len(_UNIT)
    _x, _xp, _y, _yp, _z = _UNIT.T
    BEAM = {
        "x": SX * _x,
        "xp": SX * _xp,
        "y": SY * _y,
        "yp": SY * _yp,
        "t": SZ * _z / BETA_C,
        "p": np.full(N, P),
        "particleID": np.arange(1, N + 1),
    }
    # Coordinates (x, xp, y, yp) after R without space charge.
    IDEAL = np.column_stack([BEAM["xp"], -BEAM["x"], BEAM["yp"], -BEAM["y"]])


    def linear_kick() -> np.ndarray:
        """The manual's linear kick K L lambda(z) x_i/[sigma_i (sigma_x + sigma_y)] for one SCMULT after R.

        K = 2 N r_e/(beta^2 gamma^3) and lambda = exp(-z^2/(2 sigma_z^2))/(sqrt(2 pi) sigma_z); the
        sizes are those after R (the quarter turn exchanges positions and slopes, whose spreads are
        equal at beta = 1 m). Returns shape (N, 2).
        """
        k = 2 * CHARGE / QE * RE / (P * P * GAMMA)
        sigma = np.array([IDEAL[:, 0].std(), IDEAL[:, 2].std()])
        s = SZ * _z
        density = np.exp(-0.5 * ((s - s.mean()) / s.std()) ** 2) / (np.sqrt(2 * np.pi) * s.std())
        return k * LENGTH * density[:, None] * IDEAL[:, [0, 2]] / (sigma * sigma.sum())


    def spec(**options) -> str:
        """One insert_sceffects command with both transverse planes on and the given options."""
        options = {"horizontal": 1, "vertical": 1, **options}
        text = ",".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in options.items())
        return f"&insert_sceffects {text} &end\n"


    def track(commands: str, mode: str = "serial") -> dict:
        """Run the ring with the given insert_sceffects commands.

        Returns {"status", "log"} and, when the run completes, "insertions" (names of the elements
        that precede each SCMULT in the Twiss output) and "kick" (slope change per particle, shape
        (N, 2)). A run that exits nonzero (an abort or a crash) returns its status and log.
        """
        files = {
            "lattice.lte": LATTICE,
            "run.ele": "&global_settings echo_namelists=0 &end\n" + commands + TAIL,
        }
        try:
            result = run(files, mode, {"beam.sdds": BEAM}, outputs=("out.sdds", "twiss.sdds"))
        except RunFailure as failure:
            return {"status": failure.status, "log": failure.log}
        twiss = result["twiss.sdds"]
        names = [str(n) for n in twiss[0, "ElementName"]]
        types = [str(t) for t in twiss[0, "ElementType"]]
        insertions = [names[i - 1] for i, t in enumerate(types) if t == "SCMULT"]
        out = particles(result["out.sdds"])
        kick = np.column_stack([out["xp"] - IDEAL[:, 1], out["yp"] - IDEAL[:, 3]])
        return {"status": 0, "log": result["log"], "insertions": insertions, "kick": kick}


    def failure_line(log: str) -> str:
        """The last line of a failed run that names the failure, for printing."""
        lines = [
            ln.strip()
            for ln in log.splitlines()
            if re.search(r"error|signal|SIG|abort|fault", ln, re.IGNORECASE)
        ]
        return lines[-1] if lines else log.strip().splitlines()[-1]

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

