---
title: "TRK-19: TMCF fiducial timing uses transverse positions as momentum components"
hide:
  - toc
---

# TRK-19 — TMCF fiducial timing uses transverse positions as momentum components

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`set_up_derivatives`, T_TMCF fiducial, motion.cc:617–624](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/motion.cc#L617-L624)<br>[`add_to_particle_energy`, simple_rfca.c:1140](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1140)<br>[setup comment, motion.cc:609–614](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/motion.cc#L609-L614)<br>[`FIDUCIAL` dictionary entry, track_data.c:881](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L881)<br>[fields are spatially constant, track_data.c:860–862](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L860-L862)<br>[bare `average` mode, motion.cc:1540–1566](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/motion.cc#L1540-L1566)<br>[`t,median`, track.h:1419–1423](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track.h#L1419-L1423) |

A TMCF cavity using a particle fiducial, including the serial default t,median, estimates transit time using transverse positions instead of slopes. Translating a parallel beam through a transversely uniform field must leave its energy unchanged. For electrons of momentum p = 17 mₑc in a 0.16 m, 0.9 GHz, 250 kV/m cavity, translation by (0.12, −0.08) m produces a 0.01597 rad phase error and increases exit momentum by 7.643×10⁻⁴ mₑc; adding the analytic phase correction restores the on-axis result. For small offsets, the leading error is quadratic: 1.512 μrad at x = y = 1 mm. Bare average and light-speed fiducials remain invariant.

??? example "Reproducer: TRK-19.py"

    [Download TRK-19.py](TRK-19.py)

    ```python
    """TRK-19: TMCF fiducial timing uses transverse positions as momentum components

    A TMCF cavity using a particle fiducial, including the serial default t,median, estimates transit
    time using transverse positions instead of slopes. Translating a parallel beam through a
    transversely uniform field must leave its energy unchanged. For electrons of momentum p = 17 mₑc in
    a 0.16 m, 0.9 GHz, 250 kV/m cavity, translation by (0.12, −0.08) m produces a 0.01597 rad phase
    error and increases exit momentum by 7.643×10⁻⁴ mₑc; adding the analytic phase correction restores
    the on-axis result. For small offsets, the leading error is quadratic: 1.512 μrad at x = y = 1 mm.
    Bare average and light-speed fiducials remain invariant.

    The script sends twelve identical electrons parallel to the axis through a TMCF with a uniform
    longitudinal field, on the axis and translated by (0.12, −0.08) m, and compares the exit momentum,
    time and slopes. The translated first-particle case is rerun with PHASE increased by the analytic
    phase error ωL/(2βc)(1/√(1−x²−y²)−1), which restores the on-axis result. Controls are the light-speed
    and beam-average fiducials, which do not use a particle's position; the default fiducial is
    also run.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    C = 299792458.0  # m/s, c_mks of SDDS
    P, LENGTH, FREQUENCY, EZ, PHASE = 17.0, 0.16, 0.9e9, 250000.0, 0.4
    X, Y = 0.12, -0.08  # m, translation of the parallel beam
    STEPS = 2400
    COPIES = 12
    P_ATOL, T_ATOL, SLOPE_ATOL = 5e-12, 5e-22, 5e-12


    def track(mode, x, y, fiducial, phase=PHASE):
        """Exit (p, t, xp, yp) of the first particle of a beam translated to (x, y), parallel to the axis.

        fiducial is the FIDUCIAL string, or None for the element's default. The field is purely
        longitudinal and uniform in the transverse plane, so a translation cannot change the motion.
        """
        fid = "" if fiducial is None else f'FIDUCIAL="{fiducial}",'
        lattice = (
            f"K: TMCF,L={LENGTH},FREQUENCY={FREQUENCY!r},EZ={EZ!r},ER=0,BPHI=0,PHASE={phase!r},{fid}"
            f'N_STEPS={STEPS},METHOD="non-adaptive runge-kutta",ACCURACY=1e-10\n'
            "BL: LINE=(K)\n"
        )
        commands = (
            "&change_particle name=electron &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P!r}, "
            "output=out.sdds, random_number_seed=752913 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {"x": np.full(COPIES, x), "y": np.full(COPIES, y), "p": np.full(COPIES, P)}
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if len(out["p"]) != COPIES or not all(np.all(np.isfinite(v)) for v in out.values()):
            sys.exit(f"{mode} {fiducial}: particles were lost or not finite")
        return np.array([out["p"][0], out["t"][0], out["xp"][0], out["yp"][0]])


    def close(observed, expected):
        atol = np.array([P_ATOL, T_ATOL, SLOPE_ATOL, SLOPE_ATOL])
        return bool(np.all(np.abs(observed - expected) <= atol))


    def compare(label, observed, expected):
        print(
            f"{label}: observed=[{', '.join(map(repr, observed.tolist()))}] "
            f"expected=[{', '.join(map(repr, expected.tolist()))}]"
        )


    def main():
        # The code places the fiducial's field phase at the cavity centre from the time the fiducial
        # needs to get there, L/(2 Pz) in units of the fiducial's own beta. It takes Pz = P sqrt(1 - x^2
        # - y^2) from the positions in metres; the longitudinal momentum is P/sqrt(1 + xp^2 + yp^2).
        # For this beam xp = yp = 0, so Pz = P and a translation must change nothing. The wrong Pz
        # lengthens the centre-to-entrance transit by 1/sqrt(1 - x^2 - y^2) and moves the phase by
        #   omega*L/(2 beta c) * (1/sqrt(1 - x^2 - y^2) - 1),
        # which adding to PHASE undoes (omega = 2 pi f, beta = P/sqrt(1 + P^2)).
        beta = P / np.sqrt(1 + P * P)
        correction = float(
            2 * np.pi * FREQUENCY / C * LENGTH / (2 * beta) * (1 / np.sqrt(1 - X * X - Y * Y) - 1)
        )
        print(f"phase error from the position-based momentum: {correction!r} rad")
        defect = False
        for mode in modes():
            axis = track(mode, 0.0, 0.0, "t,first")
            offset = track(mode, X, Y, "t,first")
            compare(f"{mode} first-particle fiducial, translated beam", offset, axis)
            bad = not close(offset, axis)
            defect |= bad
            compensated = track(mode, X, Y, "t,first", PHASE + correction)
            compare(
                f"{mode} first-particle fiducial, translated beam, PHASE+correction", compensated, axis
            )
            if not close(compensated, axis):
                sys.exit("The phase error is not explained by the position-based momentum.")
            # Controls: modes that do not use a particle's transverse position are translation invariant.
            for fiducial in ("t,light", "average"):
                base = track(mode, 0.0, 0.0, fiducial)
                moved = track(mode, X, Y, fiducial)
                compare(f"{mode} {fiducial} fiducial, translated beam", moved, base)
                if not close(moved, base):
                    sys.exit(f"The {fiducial} control is not translation invariant.")
            base = track(mode, 0.0, 0.0, None)
            moved = track(mode, X, Y, None)
            compare(f"{mode} default fiducial, translated beam", moved, base)
        print(f"comparison: atol p={P_ATOL:g}, t={T_ATOL:g} s, slopes={SLOPE_ATOL:g}")
        verdict(defect)


    main()

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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |

