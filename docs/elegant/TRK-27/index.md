---
title: "TRK-27: CSBEND with EXPAND_HAMILTONIAN=1 does not transport the spin through the bend body"
hide:
  - toc
---

# TRK-27 — CSBEND with EXPAND_HAMILTONIAN=1 does not transport the spin through the bend body

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`track_through_csbend`, csbend.c:1496–1502](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1496-L1502)<br>[expanded integrator, csbend.c:2064–2255](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2064-L2255)<br>[spin update in the exact integrator, csbend.c:1930–1935](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1930-L1935)<br>[EXPAND_HAMILTONIAN, track_data.c:1977](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1977)<br>[spin_tracking, elegant.tex:7291–7292](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7291-L7292)<br>[CSBEND among spin-aware elements, elegant.tex:11191–11193](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L11191-L11193) |

With spin tracking enabled, a CSBEND using EXPAND_HAMILTONIAN=1 omits spin transport through the magnet body, although the exact-Hamiltonian branch applies the Thomas–BMT equation. For a reference electron with initially longitudinal spin and momentum 450 mₑc through a 0.8 m bend with fringes disabled and angle θ = 0.18 rad, the correct spin rotation in the exit frame is −aγθ = −0.09393 rad, where a is the electron's anomalous magnetic moment and γ its Lorentz factor. The exact-Hamiltonian control agrees, while the expanded branch returns zero rotation without a warning. Users combining spin tracking with the leading-order Hamiltonian option are affected; fringe rotations are separate.

??? example "Reproducer: TRK-27.py"

    [Download TRK-27.py](TRK-27.py)

    ```python
    """TRK-27: CSBEND with EXPAND_HAMILTONIAN=1 does not transport the spin through the bend body

    With spin tracking enabled, a CSBEND using EXPAND_HAMILTONIAN=1 omits spin transport through the
    magnet body, although the exact-Hamiltonian branch applies the Thomas–BMT equation. For a reference
    electron with initially longitudinal spin and momentum 450 mₑc through a 0.8 m bend with fringes
    disabled and angle θ = 0.18 rad, the correct spin rotation in the exit frame is −aγθ = −0.09393 rad,
    where a is the electron's anomalous magnetic moment and γ its Lorentz factor. The exact-Hamiltonian
    control agrees, while the expanded branch returns zero rotation without a warning. Users combining
    spin tracking with the leading-order Hamiltonian option are affected; fringe rotations are separate.

    The script tracks 12 identical on-reference electrons with spin (0, 0, 1) through a pure CSBEND with
    edge effects off, for the exact and the expanded Hamiltonian, at two momenta and two bend angles of
    both signs, and compares the final spin with the analytic precession of a spin along the reference
    orbit. The exact-Hamiltonian run of each case is the control.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    A_E = 0.00115965218128  # electron anomalous magnetic moment, particleAnomalousMagneticMoment
    LENGTH, SLICES = 0.8, 256
    N = 12  # identical particles, so that every MPI worker tracks some
    # Absolute tolerance on dimensionless sx and sz; the fourth-order exact integration is within
    # about 2e-14 here.
    TOL = 1e-9
    CASES = ((450.0, 0.18), (900.0, 0.18), (450.0, -0.09))  # (p0 in m_e c, bend angle in rad)


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def spin_after(mode, p0, angle, expand):
        """Final spin (sx, sz) of the reference electrons through a pure dipole body."""
        lattice = (
            f"B: CSBEND,L={LENGTH!r},ANGLE={angle!r},N_SLICES={SLICES},INTEGRATION_ORDER=4,"
            f"EXPAND_HAMILTONIAN={expand},EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,SYNCH_RAD=0,ISR=0,"
            "REFERENCE_CORRECTION=0\nBL: LINE=(B)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, output=out.sdds,"
            " spin_tracking=1 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        zero = np.zeros(N)
        beam = {"p": np.full(N, p0), "spx": zero, "spy": zero, "spz": np.ones(N)}
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})
        out = particles(result["out.sdds"])
        if len(out["p"]) != N:
            sys.exit(f"{mode}: {N - len(out['p'])} particles lost")
        if not all(np.all(np.isfinite(out[k])) for k in ("p", "spx", "spy", "spz")):
            sys.exit(f"{mode}: nonfinite particle momentum or spin in the output: invalid run")
        if np.max(np.abs(out["spx"] - out["spx"][0])) > 0 or np.max(np.abs(out["spy"])) > 1e-15:
            sys.exit(f"{mode}: identical reference particles leave with different spins")
        return np.array([out["spx"][0], out["spz"][0]]), result["log"]


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected differ by more than TOL."""
        if not (np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))):
            sys.exit(f"{label}: nonfinite comparison operand: invalid run")
        bad = bool(np.any(np.abs(observed - expected) > TOL))
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    defect = []
    print("columns: sx, sz after the bend; expected from the Thomas-BMT precession")
    for mode in modes():
        for p0, angle in CASES:
            # On the reference orbit the field is perpendicular to the velocity, so Thomas-BMT rotates
            # the spin by -(1 + a*gamma)*theta in the laboratory frame, and transforming to the exit
            # reference frame adds +theta to this signed laboratory spin angle: the sum is
            # -a*gamma*theta (gamma = sqrt(1 + p0^2) for p0 in m_e c).
            theta = -A_E * np.sqrt(1 + p0 * p0) * angle
            expected = np.array([np.sin(theta), np.cos(theta)])
            label = f"{mode} p0={p0:g} angle={angle}"
            exact, _ = spin_after(mode, p0, angle, 0)
            if compare(f"{label} exact Hamiltonian (control)", exact, expected):
                sys.exit("The exact-Hamiltonian control disagrees with the precession: invalid.")
            expanded, log = spin_after(mode, p0, angle, 1)
            defect.append(compare(f"{label} EXPAND_HAMILTONIAN=1", expanded, expected))
            print(
                f"{label}: expected angle {theta:.16g} rad; warnings in expanded log: {'warn' in log.lower()}"
            )
    print(f"comparison: abs tolerance {TOL:g} on sx and sz")
    verdict(all(defect))

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
| Found | `gpt-6.1-sol` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

