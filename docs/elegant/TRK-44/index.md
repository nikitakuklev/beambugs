---
title: "TRK-44: RFCA with Q ramps its voltage as the square root of a stored-energy fill, not as a resonant fill"
hide:
  - toc
---

# TRK-44 — RFCA with Q ramps its voltage as the square root of a stored-energy fill, not as a resonant fill

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Tags | <span class="bb-tag">contract-unsettled</span> [?](../../methodology.md#tags) |
| Source | [`trackRfCavityWithWakes`, simple_rfca.c:521](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L521)<br>[simple_rfca.c:723–733](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L723-L733)<br>[simple_rfca.c:1010](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1010)<br>[parameter table, track_data.c:603](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L603) |

An RFCA with nonzero Q, described as a cavity that charges up to its voltage from zero, ramps the voltage as VOLT·√(1 − e^(−ωt/Q)), with ω the angular RF frequency and t the time since the reference time. The envelope equation of a resonant mode driven at its frequency from rest gives VOLT·(1 − e^(−ωt/2Q)); the square-root law rises with infinite initial slope and stays above it. For a 350 MHz, 200 kV cavity with Q = 3500, the kick after 100 RF periods (285.7 ns) is 81.07 kV instead of 17.17 kV, 4.722 times too large, and 153.9 kV instead of 72.32 kV after 1.429 µs. Both laws reach VOLT in steady state.

??? example "Reproducer: TRK-44.py"

    [Download TRK-44.py](TRK-44.py)

    ```python
    """TRK-44: RFCA with Q ramps its voltage as the square root of a stored-energy fill, not as a resonant fill

    An RFCA with nonzero Q, described as a cavity that charges up to its voltage from zero, ramps the
    voltage as VOLT·√(1 − e^(−ωt/Q)), with ω the angular RF frequency and t the time since the reference
    time. The envelope equation of a resonant mode driven at its frequency from rest gives VOLT·(1 −
    e^(−ωt/2Q)); the square-root law rises with infinite initial slope and stays above it. For a 350
    MHz, 200 kV cavity with Q = 3500, the kick after 100 RF periods (285.7 ns) is 81.07 kV instead of
    17.17 kV, 4.722 times too large, and 153.9 kV instead of 72.32 kV after 1.429 µs. Both laws reach
    VOLT in steady state.

    The script sends electrons of p = 300 mₑc through a 350 MHz, 200 kV RFCA with Q = 3500 and
    T_REFERENCE = 0, timed to reach the kick at whole RF periods (0 to 2000) after the fill starts, both
    as a zero-length cavity (kick method) and as a 0.2 m cavity with N_KICKS = 0 (matrix method). It
    compares the energy gain with the voltage envelope of a resonantly driven cavity filling from rest
    and prints Elegant's square-root law beside it. The same cavity with Q = 0 is the control.
    """

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h, as Elegant uses them.
    C, ME_MEV = 299792458.0, 0.51099906
    FREQ, VOLT, Q = 350e6, 200000.0, 3500.0  # Hz, V
    P0 = 300.0  # m_e c
    PERIODS = np.array([0, 1, 25, 100, 500, 2000])  # elapsed RF periods at the kick
    OMEGA = 2 * np.pi * FREQ
    TOL = 1e-3  # eV; Elegant's own law is matched to 1e-7 eV


    def gains(mode, length, q):
        """Energy gains in eV of particles reaching the kick at whole RF periods after t = 0.

        With T_REFERENCE = 0 the fill starts at t = 0 and the carrier phase is omega t + PHASE. The
        zero-length cavity kicks at the entrance; the matrix method kicks at the cavity center, reached
        L/(2 beta c) after the entrance, so the entrance times are shifted back by that flight time.
        """
        beta = P0 / np.hypot(P0, 1)
        t_kick = PERIODS / FREQ
        lattice = (
            f"R: RFCA,L={length!r},VOLT={VOLT!r},FREQ={FREQ!r},Q={q!r},PHASE=90,N_KICKS=0,"
            "T_REFERENCE=0,CHANGE_P0=0,END1_FOCUS=0,END2_FOCUS=0\n"
            "BL: LINE=(R)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
            "output=out.sdds, default_order=1 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {
            "t": t_kick - length / (2 * beta * C),
            "p": np.full(len(PERIODS), P0),
            "particleID": np.arange(1, len(PERIODS) + 1),
        }
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if len(out["p"]) != len(PERIODS):
            raise SystemExit(f"{mode}: particles were lost")
        return (np.hypot(out["p"], 1) - np.hypot(P0, 1)) * ME_MEV * 1e6, t_kick


    defect = False
    for mode in modes():
        for label, length in (
            ("zero-length (kick method)", 0.0),
            ("L=0.2 m, N_KICKS=0 (matrix method)", 0.2),
        ):
            observed, t = gains(mode, length, Q)
            # A resonant mode driven at its frequency by a constant coherent drive has the envelope
            # equation dA/dt + A omega/(2Q) = omega V/(2Q) with A(0) = 0, so A = V (1 - exp(-omega t/(2Q))):
            # the voltage fills with time constant 2Q/omega (the stored energy decays with Q/omega).
            expected = VOLT * (1 - np.exp(-OMEGA * t / (2 * Q)))
            # Elegant's law: tau = Q/omega and V sqrt(1 - exp(-t/tau)), the square root of an
            # energy that relaxes with tau.
            elegant_law = VOLT * np.sqrt(1 - np.exp(-OMEGA * t / Q))
            if np.max(np.abs(observed - elegant_law)) > TOL:
                raise SystemExit(
                    f"{mode} {label}: the gain does not follow V sqrt(1 - exp(-omega t/Q))"
                )
            for n, obs, exp_, law in zip(PERIODS, observed, expected, elegant_law, strict=True):
                ratio = f"ratio {obs / exp_:.6f}" if exp_ else "ratio -"
                print(
                    f"{mode} {label}, {n} periods ({n / FREQ * 1e9:.4f} ns): observed={obs:.6f} eV "
                    f"expected={exp_:.6f} eV ({ratio}); square-root law {law:.6f} eV"
                )
                defect |= abs(obs - exp_) > TOL
            control, _ = gains(mode, length, 0.0)
            print(
                f"{mode} {label}, Q=0 (control): gains {np.array2string(control, precision=6)} eV, expected {VOLT:.6f}"
            )
            if np.max(np.abs(control - VOLT)) > TOL:
                raise SystemExit(f"{mode} {label}: control fails: Q=0 does not give VOLT at every time")
    print(f"comparison: tolerance {TOL:g} eV")
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
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed (zero-length cavity (kick method), judged against resonant-fill physics without an explicit manual equation) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

