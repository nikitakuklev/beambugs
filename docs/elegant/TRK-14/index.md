---
title: "TRK-14: POLAR random-direction modes give NaN spins along a longitudinal axis"
hide:
  - toc
---

# TRK-14 — POLAR random-direction modes give NaN spins along a longitudinal axis

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`polarizeBeam`, generate_bunch.c:1367–1508](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/generate_bunch.c#L1367-L1508)<br>[axis normalization and two-state routing, generate_bunch.c:1381–1391](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/generate_bunch.c#L1381-L1391)<br>[narrow-cone rotation, generate_bunch.c:1451–1453](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/generate_bunch.c#L1451-L1453)<br>[wide-cone rotation, generate_bunch.c:1472–1474](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/generate_bunch.c#L1472-L1474)<br>[von Mises–Fisher rotation, generate_bunch.c:1491–1493](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/generate_bunch.c#L1491-L1493)<br>[`performQuaternionRotation`, spinUpdate.c:127–163](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/spinUpdate.c#L127-L163)<br>[POLAR manual, polar.tex:1–19](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/polar.tex#L1-L19)<br>[POLAR dispatch, do_tracking.c:2001–2004](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2001-L2004) |

With spin tracking enabled, POLAR's narrow-cone, wide-cone and von Mises–Fisher modes return NaN (not-a-number) spin components for every tested particle when the polarization axis is (0, 0, ±1). At requested polarizations P = 0.25 and 0.8, all 12 spins become nonfinite without warning; the tilted axis (0.6, 0, 0.8) gives 12 finite unit spins. Rotations should preserve unit spin length, and a narrow cone should satisfy S·v = P, where S is the spin and v the unit axis. The shared singular rotation establishes the extension to all 0 < P < 1 by source inspection, affecting preparation of longitudinally polarized beams.

??? example "Reproducer: TRK-14.py"

    [Download TRK-14.py](TRK-14.py)

    ```python
    """TRK-14: POLAR random-direction modes give NaN spins along a longitudinal axis

    With spin tracking enabled, POLAR's narrow-cone, wide-cone and von Mises–Fisher modes return NaN
    (not-a-number) spin components for every tested particle when the polarization axis is (0, 0, ±1).
    At requested polarizations P = 0.25 and 0.8, all 12 spins become nonfinite without warning; the
    tilted axis (0.6, 0, 0.8) gives 12 finite unit spins. Rotations should preserve unit spin length,
    and a narrow cone should satisfy S·v = P, where S is the spin and v the unit axis. The shared
    singular rotation establishes the extension to all 0 < P < 1 by source inspection, affecting
    preparation of longitudinally polarized beams.

    The script tracks 12 identical electrons with spin tracking enabled through a POLAR element for each
    mode, P of 0.25 and 0.8, and the axes (0, 0, 1) and (0, 0, -1), and counts the particles whose three
    spin components are finite and of unit length; a narrow cone must also satisfy S·v = P. The controls
    are the tilted axis (0.6, 0, 0.8) in every mode, and the longitudinal axis with the two-state mode,
    with P = 0 and with P = 1, which bypass the cone rotation.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    N = 12  # particles, enough for every MPI worker to hold some
    P0 = 30.0  # reference momentum, units of m_e c; POLAR does not change the orbit
    TOL = 5e-15  # spin norm and S.v - P; a quaternion rotation is exact to rounding


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def polarize(mode, kind, polarization, axis):
        """Spins of N particles after a POLAR element; initial spins are (0, 0, 1)."""
        vx, vy, vz = axis
        lattice = (
            f'P: POLAR,POLARIZATION={polarization!r},VX={vx!r},VY={vy!r},VZ={vz!r},MODE="{kind}"\n'
            "BL: LINE=(P)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds,"
            " spin_tracking=1, random_number_seed=64783 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        beam = {
            "p": np.full(N, P0),
            "spx": np.zeros(N),
            "spy": np.zeros(N),
            "spz": np.ones(N),
        }
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if len(out["p"]) != N:
            sys.exit(f"{mode}: {N - len(out['p'])} particles lost")
        return np.column_stack([out["spx"], out["spy"], out["spz"]])


    def check(label, spins, axis, polarization, cone):
        """Print finite count and spin norm deviation (and S.v - P for a narrow cone).

        Expected: every spin finite (12 of 12), |S| = 1, and for a narrow cone S.v = P exactly,
        because the sampled polar angle satisfies cos(theta) = P and the rotation to the axis keeps it.
        Returns False when the invariants hold and True when every component of every spin is NaN,
        the recorded defect; any other state is a failed run, not a verdict.
        """
        finite = int(np.count_nonzero(np.all(np.isfinite(spins), axis=1)))
        norm_deviation = np.max(np.abs(np.linalg.norm(spins, axis=1) - 1))
        observed, expected = [finite, norm_deviation], [N, 0.0]
        if cone:
            v = np.asarray(axis) / np.linalg.norm(axis)
            observed.append(np.max(np.abs(spins @ v - polarization)))
            expected.append(0.0)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        if finite == N and np.all(np.abs(np.array(observed[1:])) <= TOL):
            return False
        if np.all(np.isnan(spins)):
            return True
        sys.exit(f"{label}: the spins are neither valid nor all NaN")


    defect = []
    print("columns: finite spins, max | |S| - 1 |" + ", max |S.v - P| (narrow cone)")
    for mode in modes():
        for kind in ("narrowcone", "widecone", "vonmises-fisher"):
            for polarization in (0.25, 0.8):
                for axis in ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.6, 0.0, 0.8)):
                    label = f"{mode} {kind} P={polarization} axis={axis}"
                    spins = polarize(mode, kind, polarization, axis)
                    bad = check(label, spins, axis, polarization, kind == "narrowcone")
                    if axis[0]:
                        # The tilted axis has VX != 0, so the rotation is regular.
                        if bad:
                            sys.exit("The tilted-axis control fails: the comparison is not valid.")
                    else:
                        defect.append(bad)
        # Longitudinal axis, but the cone rotation is bypassed: P = 0 or P = 1 select the two-state
        # branch, and mode "twostate" never rotates.
        for kind, polarization in (("twostate", 0.5), ("narrowcone", 0.0), ("widecone", 1.0)):
            spins = polarize(mode, kind, polarization, (0.0, 0.0, 1.0))
            label = f"{mode} control {kind} P={polarization} axis=(0.0, 0.0, 1.0)"
            if check(label, spins, (0.0, 0.0, 1.0), polarization, False):
                sys.exit("A longitudinal control fails: the comparison is not valid.")
    print(f"comparison: unit norm and S.v = P to {TOL:g}; finite count")
    verdict(any(defect))

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
| Found | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

