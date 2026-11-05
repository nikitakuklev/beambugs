---
title: "COL-14: BEAMBEAM kicks change arrival time because s is not rescaled"
hide:
  - toc
---

# COL-14 — BEAMBEAM kicks change arrival time because s is not rescaled

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [Gaussian branch update, do_tracking.c:6322](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L6322)<br>[helper update, ionEffects.cc:3617](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ionEffects.cc#L3617)<br>[coordinate definition, elegant.tex:372–379](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L372-L379) |

A BEAMBEAM kick changes momentum but leaves the stored longitudinal coordinate s = βct unchanged, where t is arrival time, c is the speed of light and β is speed divided by c. The output time therefore changes by the ratio of incoming to outgoing speed, although the zero-length element must preserve time and rescale s. For an electron of momentum 1 mₑc at horizontal position x = 0.2300 mm, a 1.300 nC uniform opposing bunch of radius 1.300 mm changes an input time of 1 µs by −1.936×10⁻¹¹ s, a relative error of −1.936×10⁻⁵. Zero input time masks the error.

??? example "Reproducer: COL-14.py"

    [Download COL-14.py](COL-14.py)

    ```python
    """COL-14: BEAMBEAM kicks change arrival time because s is not rescaled

    A BEAMBEAM kick changes momentum but leaves the stored longitudinal coordinate s = βct unchanged,
    where t is arrival time, c is the speed of light and β is speed divided by c. The output time
    therefore changes by the ratio of incoming to outgoing speed, although the zero-length element must
    preserve time and rescale s. For an electron of momentum 1 mₑc at horizontal position x = 0.2300 mm,
    a 1.300 nC uniform opposing bunch of radius 1.300 mm changes an input time of 1 µs by −1.936×10⁻¹¹
    s, a relative error of −1.936×10⁻⁵. Zero input time masks the error.

    The script sends electrons at x = 0.23 mm with six arrival times (zero, positive and negative,
    from 1 ns to 0.25 s) through one BEAMBEAM element of each distribution at p = 1, 100 and 1000 mₑc.
    It compares the output time with the input time and with the prediction for an unchanged s,
    t_in β_old/β_new, where β is taken from the momentum Elegant returns. The control is zero
    opposing charge, which must leave both momentum and time unchanged.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    TIMES = np.array([0, 1e-9, -2e-9, 1e-6, 0.125, -0.25] * 2)  # s; the 1 us entry is index 3
    N = len(TIMES)
    X0, SIZE = 0.23e-3, 1.3e-3  # m
    RTOL, ATOL = 2e-15, 2e-24  # relative, and absolute in s (zero-time entries)


    def beta(p):
        return p / np.sqrt(1 + p * p)


    def track(mode, distribution, p0, charge):
        lattice = (
            f"K: BEAMBEAM,DISTRIBUTION={distribution},CHARGE={charge!r},XSIZE={SIZE!r},YSIZE={SIZE!r}\n"
            "BL: LINE=(K)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0!r},output=out.sdds,"
            "losses=lost.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {"x": np.full(N, X0), "t": TIMES, "p": np.full(N, p0)}
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": beam},
            outputs=("out.sdds", "lost.sdds"),
        )
        out, lost = result["out.sdds"], result["lost.sdds"]
        if len(out[0, "p"]) != N or len(lost[0, "particleID"]) != 0:
            sys.exit(f"{mode} {distribution} p={p0}: particles were lost")
        # The global_settings echo prints "warning_limit"; any other mention is a warning.
        if "warning" in result["log"].lower().replace("warning_limit", ""):
            sys.exit(f"{mode} {distribution} p={p0}: Elegant printed a warning")
        p, t = np.asarray(out[0, "p"]), np.asarray(out[0, "t"])
        # Invalid coordinates are a failed run, never evidence for the defect.
        if not (np.all(np.isfinite(p)) and np.all(p > 0) and np.all(np.isfinite(t))):
            sys.exit(f"{mode} {distribution} p={p0}: non-finite times or non-positive momentum")
        return p, t


    defect = False
    for mode in modes():
        for p0 in (1.0, 100.0, 1000.0):
            for distribution in ("uniform", "parabolic", "gaussian"):
                # The charge scales with p to keep the angular kick small but resolvable.
                for charge in (0.0, 1.3e-9 * p0):
                    p, t = track(mode, distribution, p0, charge)
                    name = f"{mode} {distribution} p_in={p0:g} charge={charge:.3g} C"
                    # The correct map conserves t (s = beta c t is rescaled with the velocity).
                    changed = not np.allclose(t, TIMES, rtol=RTOL, atol=ATOL)
                    # An unchanged s = beta c t predicts t_out = t_in beta_old/beta_new. It is the
                    # diagnosis of a changed time only; a map that preserves t needs no diagnosis.
                    s_unchanged = TIMES * beta(p0) / beta(p)
                    if changed and not np.allclose(t, s_unchanged, rtol=RTOL, atol=ATOL):
                        sys.exit(f"{name}: output time does not follow the unchanged-s prediction.")
                    print(
                        f"{name}: p_out={p[0]:.16g} t_in={TIMES[3]:.16g} "
                        f"observed t_out={t[3]:.16g} expected t_out={TIMES[3]:.16g}, "
                        f"observed t_out/t_in={t[3] / TIMES[3]:.16g}"
                    )
                    if charge == 0:
                        if changed or not np.array_equal(p, np.full(N, p0)):
                            sys.exit("The zero-charge control changed the momentum or the time.")
                    else:
                        defect |= changed
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g} s")
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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |

