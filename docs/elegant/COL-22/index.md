---
title: "COL-22: Constant RFMODE frequency waveforms repeatedly scale stored voltage"
hide:
  - toc
---

# COL-22 — Constant RFMODE frequency waveforms repeatedly scale stored voltage

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [binned tracking, rfmode.cc:712–717](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/rfmode.cc#L712-L717)<br>[binless tracking, rfmode.cc:1633–1638](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/rfmode.cc#L1633-L1638)<br>[caller, do_tracking.c:2387–2395](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2387-L2395)<br>[setup, rfmode.cc:1232](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/rfmode.cc#L1232)<br>[loads the frequency table, rfmode.cc:1413–1431](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/rfmode.cc#L1413-L1431)<br>[waveforms, rfmode.tex:87–99](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/rfmode.tex#L87-L99)<br>[parameter metadata, track_data.c:2187](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2187) |

In single-bunch RFMODE tracking, a constant frequency multiplier a supplied through FREQ_WAVEFORM, with FREQ divided by a to preserve the resonant frequency, scales the stored beam-induced voltage by √a on every traversal. A stationary passive cavity should follow the same damped response as the direct-frequency setting, which agrees with an independent parallel-RLC circuit calculation. For quality factor Q = 11.00, shunt impedance RA = 200.0 kΩ, frequency 0.8300 GHz and a 0.3000 nC electron bunch represented by 18 macroparticles, the fifth-traversal tail energy loss is 9366 eV for a = 1.440 or 1.247×10⁴ eV for a = 0.2500, instead of 1.015×10⁴ eV.

??? example "Reproducer: COL-22.py"

    [Download COL-22.py](COL-22.py)

    ```python
    """COL-22: Constant RFMODE frequency waveforms repeatedly scale stored voltage

    In single-bunch RFMODE tracking, a constant frequency multiplier a supplied through FREQ_WAVEFORM,
    with FREQ divided by a to preserve the resonant frequency, scales the stored beam-induced voltage by
    √a on every traversal. A stationary passive cavity should follow the same damped response as the
    direct-frequency setting, which agrees with an independent parallel-RLC circuit calculation. For
    quality factor Q = 11.00, shunt impedance RA = 200.0 kΩ, frequency 0.8300 GHz and a 0.3000 nC
    electron bunch represented by 18 macroparticles, the fifth-traversal tail energy loss is 9366 eV for
    a = 1.440 or 1.247×10⁴ eV for a = 0.2500, instead of 1.015×10⁴ eV.

    The script tracks the bunch through RFMODE and a drift five times, exciting the cavity by rigid
    tracking until the last traversal, and compares the fifth traversal's energy losses with a lumped
    parallel-RLC circuit driven by the same charge impulses, cross-checked against the closed-form
    resonator wake. The controls are the direct frequency, a unit waveform, a single traversal, and
    SINGLE_PASS = 1, which discards the stored field. It also shows that scaling the circuit state by √a
    at each traversal reproduces the observed losses.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from scipy.linalg import expm

    # SDDS constants.h, as Elegant uses them.
    C, REST_EV = 299792458.0, 510999.06
    N = 18  # electron macroparticles, spaced 2 ps
    FREQ, Q, RA = 0.83e9, 11.0, 2e5  # actual resonant frequency (Hz), Q, shunt impedance (ohm)
    CHARGE, P0 = 0.3e-9, 64.0  # C, m_e c
    FLIGHT = 2.7e-9  # s between traversals, supplied by a drift
    TIMES = 0.43e-9 + (np.arange(N) - (N - 1) / 2) * 2e-12  # s, arrival times of the first traversal
    LENGTH = float(C * P0 / np.sqrt(1 + P0**2) * FLIGHT)  # drift length for FLIGHT, m

    # (traversals, constant waveform factor a or None for no waveform, SINGLE_PASS, label)
    CASES = (
        (5, None, 0, "direct frequency, 5 passes"),
        (5, 1.0, 0, "unit waveform, 5 passes"),
        (5, 1.44, 0, "factor 1.44, 5 passes"),
        (5, 0.25, 0, "factor 0.25, 5 passes"),
        (3, 1.44, 0, "factor 1.44, 3 passes"),
        (1, 1.44, 0, "factor 1.44, first pass"),
        (5, 1.44, 1, "factor 1.44, single pass"),
    )


    def circuit_losses(times, multiplier=1.0, single=False):
        """Energy loss in eV of each particle at each traversal in a parallel RLC circuit.

        The resonator is R = RA/2 in parallel with C = 2Q/(RA omega) and L = 1/(omega^2 C). Its
        state z = (V, sqrt(L/C) I_L) has voltage units and obeys dz/dt = omega [[-1/Q, -1], [1, 0]] z.
        A particle of charge q deposits q/C into V, and loses energy q (V + half its own deposit)
        (the fundamental theorem of beam loading). `multiplier` scales the state at the start of
        each traversal; the physical circuit has multiplier 1, and `single` clears it each time.
        """
        omega = 2 * np.pi * FREQ
        capacitance = 2 * Q / (RA * omega)
        generator = omega * np.array([[-1 / Q, -1], [1, 0]])
        deposit = CHARGE / N / capacitance
        state, last = np.zeros(2), float(times[0, 0])
        loss = np.zeros_like(times)
        for traversal, arrivals in enumerate(times):
            state = np.zeros(2) if single else state * multiplier
            for i in np.argsort(arrivals):
                state = expm(generator * (float(arrivals[i]) - last)) @ state
                loss[traversal, i] = state[0] + deposit / 2  # volts, times the unit charge e
                state[0] += deposit
                last = float(arrivals[i])
        return loss


    def green_losses(arrivals, previous):
        """Closed-form energy loss in eV of the last traversal from the resonator wake of `previous`.

        W(t) = RA omega/(2Q) exp(-alpha t) [cos(omega_d t) - alpha/omega_d sin(omega_d t)],
        alpha = omega/(2Q), omega_d = sqrt(omega^2 - alpha^2), with half weight at coincidence.
        """
        omega = 2 * np.pi * FREQ
        alpha = omega / (2 * Q)
        damped = np.sqrt(omega**2 - alpha**2)
        delays = arrivals[:, None] - previous[None, :]
        wake = (
            RA * omega / (2 * Q) * np.exp(-alpha * delays)
            * (np.cos(damped * delays) - alpha / damped * np.sin(damped * delays))
        )  # fmt: skip
        weight = (delays > 1e-18) + 0.5 * (np.abs(delays) <= 1e-18)
        return CHARGE / N * np.sum(wake * weight, axis=1)


    def last_traversal_losses(mode, binless, passes, factor, single):
        """Observed energy loss (eV) of each particle on the last traversal, and all arrival times."""
        waveform = ',FREQ_WAVEFORM="frequency.sdds=time+factor"' if factor is not None else ""
        scale = 1.0 if factor is None else factor
        # Actual resonant frequency FREQ = FREQ_parameter * factor.
        lattice = (
            f"CH: CHARGE,TOTAL={CHARGE!r}\n"
            'B: WATCH,FILENAME="before.sdds",MODE="coordinates"\n'
            f"R: RFMODE,FREQ={FREQ / scale!r},Q={Q},RA={RA!r},BIN_SIZE=2e-12,N_BINS=64,"
            f"BINLESS={binless},RIGID_UNTIL_PASS={passes - 1},SINGLE_PASS={single},"
            f"PRELOAD=0,INITIAL_V=0,DRIVE_FREQUENCY=0{waveform}\n"
            'A: WATCH,FILENAME="after.sdds",MODE="coordinates"\n'
            f"D: DRIF,L={LENGTH!r}\n"
            "BL: LINE=(CH,B,R,A,D)\n"
        )
        table = (
            "SDDS1\n"
            "&column name=time, units=s, type=double &end\n"
            "&column name=factor, type=double &end\n"
            f"&data mode=ascii, no_row_counts=1 &end\n-1e-06 {scale!r}\n1e-06 {scale!r}\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0!r},output=out.sdds &end\n"
            f"&run_control n_steps=1,n_passes={passes} &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {"t": TIMES, "p": np.full(N, P0), "particleID": np.arange(1, N + 1)}
        result = run(
            {"lattice.lte": lattice, "run.ele": commands, "frequency.sdds": table},
            mode,
            {"beam.sdds": beam},
            outputs=("before.sdds", "after.sdds"),
        )
        before = [particles(result["before.sdds"], k) for k in range(passes)]
        after = [particles(result["after.sdds"], k) for k in range(passes)]
        if any(len(page["p"]) != N for page in before + after):
            sys.exit(f"{mode}: particles lost")
        for k in range(passes - 1):  # rigid tracking until the last traversal
            if not np.allclose(after[k]["p"], before[k]["p"], rtol=0, atol=2e-13):
                sys.exit(f"{mode}: the cavity acted before the last traversal")
        energy = lambda p: REST_EV * np.sqrt(1 + p**2)
        loss = energy(before[-1]["p"]) - energy(after[-1]["p"])
        return loss, np.array([page["t"] for page in before])


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"


    RTOL, ATOL = 1e-10, 2e-8  # eV
    defect = False
    for mode in modes():
        for binless in (0, 1) if mode == "serial" else (0,):  # MPI Pelegant has no BINLESS
            for passes, factor, single, label in CASES:
                name = f"{mode} {'binless' if binless else 'binned'} {label}"
                observed, arrivals = last_traversal_losses(mode, binless, passes, factor, single)
                # Expected: the physical circuit, which the Green-function sum cross-checks.
                expected = circuit_losses(arrivals, single=bool(single))[-1]
                histories = arrivals[-1:] if single else arrivals
                green = sum(green_losses(arrivals[-1], past) for past in histories)
                if not np.allclose(green, expected, rtol=2e-13, atol=2e-9):
                    sys.exit(f"{name}: the two oracles disagree")
                # Cause: the circuit state scaled by sqrt(a) at the start of every traversal.
                scale = 1.0 if factor is None else factor
                predicted = circuit_losses(arrivals, np.sqrt(scale), bool(single))[-1]
                error = np.max(np.abs(observed - expected))
                fault = scale != 1.0 and not single and passes > 1
                print(
                    f"{name}: observed={fmt(observed[-1])} expected={fmt(expected[-1])} "
                    f"max error={error:.6g} eV, residual of sqrt(a) prediction="
                    f"{np.max(np.abs(observed - predicted)):.3g} eV"
                )
                physical = np.allclose(observed, expected, rtol=RTOL, atol=ATOL)
                if fault and not physical:
                    # A mismatch must be explained by the sqrt(a) scaling to count as the defect.
                    if not np.allclose(observed, predicted, rtol=RTOL, atol=ATOL):
                        sys.exit(f"{name}: losses follow neither the circuit nor the sqrt(a) scaling")
                    defect |= error > 100  # eV; the physical circuit differs by under 1e-7
                elif not physical:
                    sys.exit(f"{name}: control disagrees with the circuit; the comparison is invalid")
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g} eV")
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
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

