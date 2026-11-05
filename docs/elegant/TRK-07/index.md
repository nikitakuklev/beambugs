---
title: "TRK-07: MODRF damped phase modulation keeps an undamped local frequency"
hide:
  - toc
---

# TRK-07 — MODRF damped phase modulation keeps an undamped local frequency

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">contract-unsettled</span> [?](../../methodology.md#tags) |
| Source | [phase and local frequency, mod_rfca.c:183–188](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/mod_rfca.c#L183-L188)<br>[energy kick, mod_rfca.c:254](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/mod_rfca.c#L254)<br>[manual phase model, modrf.tex:23–37](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/modrf.tex#L23-L37) |

With damped phase modulation, MODRF uses a local angular frequency that is not the derivative of its bunch-centroid phase. The frequency omits both exponential attenuation and the derivative of the damping envelope, so the energy variation within the bunch disagrees with the specified phase waveform. For a 200 MHz, 10 kV cavity with 25° phase modulation at 90 MHz, decay rate α = 4 × 10⁸ s⁻¹ and bunch-centre time T = 2 ns, the slope dγ/dt is −2.383 × 10⁷ s⁻¹ instead of −2.255 × 10⁷ s⁻¹, 5.711% too steep; γ is the Lorentz factor. The manual specifies the same local-frequency formula.

??? example "Reproducer: TRK-07.py"

    [Download TRK-07.py](TRK-07.py)

    ```python
    """TRK-07: MODRF damped phase modulation keeps an undamped local frequency

    With damped phase modulation, MODRF uses a local angular frequency that is not the derivative of its
    bunch-centroid phase. The frequency omits both exponential attenuation and the derivative of the
    damping envelope, so the energy variation within the bunch disagrees with the specified phase
    waveform. For a 200 MHz, 10 kV cavity with 25° phase modulation at 90 MHz, decay rate α = 4 × 10⁸
    s⁻¹ and bunch-centre time T = 2 ns, the slope dγ/dt is −2.383 × 10⁷ s⁻¹ instead of −2.255 × 10⁷ s⁻¹,
    5.711% too steep; γ is the Lorentz factor. The manual specifies the same local-frequency formula.

    The script places six particles 10 fs ahead and six 10 fs behind a bunch centre at 2 ns, which is
    a narrow bunch whose centroid phase is Φ(T), and reads the energy slope dγ/dt from the exit
    momenta. It compares the slope with V/(m c²/e) cos Φ(T) Φ'(T) evaluated from the derivative of the
    phase. Controls: with PMDECAY = 0 the local frequency equals Φ' and the slope agrees, with and
    without modulation frequency; PMFREQ = 0 with damping isolates the envelope derivative.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    ME_MEV = 0.51099906  # electron rest energy in MeV, me_mev of SDDS constants.h
    P0 = 2.0  # momentum, units of m_e c
    VOLT, FREQ, PHASE = 1e4, 2e8, 15.0  # V, Hz, degrees
    PMMAG, PMPHASE = 25.0, 30.0  # degrees
    T, H = 2e-9, 1e-14  # s, bunch centre and half separation of the two groups
    RTOL = 1e-7  # the undamped controls agree to 3e-9 (finite difference and rounding)


    def slope(mode, pmfreq, pmdecay):
        """Energy slope dgamma/dt (1/s) across the bunch, from the two groups' exit momenta."""
        lattice = (
            f'E: MODRF,L=0,VOLT={VOLT!r},FREQ={FREQ!r},PHASE={PHASE!r},FIDUCIAL="light",'
            f"PMMAG={PMMAG!r},PMPHASE={PMPHASE!r},PMFREQ={pmfreq!r},PMDECAY={pmdecay!r}\n"
            "BL: LINE=(E)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
            "output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        # Particles 1-6 arrive at T - H, particles 7-12 at T + H.
        beam = {
            "t": np.repeat([T - H, T + H], 6),
            "p": np.full(12, P0),
            "particleID": np.arange(1, 13),
        }
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if len(out["t"]) != 12:
            sys.exit(f"{mode}: {12 - len(out['t'])} particles lost")
        # The element has L = 0 and must leave arrival times unchanged (they do to within two floating-point steps of a 2 ns time, 8.3e-25 s); a changed
        # time separation would invalidate the slope.
        if not np.allclose(out["t"], beam["t"], rtol=0, atol=1e-24):
            sys.exit(f"{mode}: the element changed the arrival times: comparison not valid.")
        gamma = np.sqrt(1 + out["p"] ** 2)
        if not (np.ptp(gamma[:6]) == 0 and np.ptp(gamma[6:]) == 0):
            sys.exit(f"{mode}: identical particles left with different energies")
        separation = out["t"][6] - out["t"][0]
        value = (gamma[6] - gamma[0]) / separation
        if not (np.isfinite(separation) and separation != 0 and np.isfinite(value)):
            sys.exit(f"{mode}: the time separation or energy slope is not finite and nonzero")
        return value


    def expected_slopes(pmfreq, pmdecay):
        """dgamma/dt of the kick V' sin(Phi(t)) at the bunch centre, for the phase derivative and for
        the undamped local frequency that MODRF uses.

        The energy kick is dgamma = V/(m c^2/e) sin(omega (t - T) + Phi(T)); at t = T its slope is
        V/(m c^2/e) cos(Phi(T)) omega. The correct omega is the derivative Phi'(T) of
        Phi(T) = w0 T + phi0 + A exp(-a T) sin(W T + psi). The fiducial "light" of an element at
        z = 0 puts the reference time at 0, so T is the arrival time of the bunch centre.
        """
        w0, a_amp = 2 * np.pi * FREQ, np.radians(PMMAG)
        big_w, psi = 2 * np.pi * pmfreq, np.radians(PMPHASE)
        envelope = np.exp(-pmdecay * T)
        arg = big_w * T + psi
        phase = w0 * T + np.radians(PHASE) + a_amp * envelope * np.sin(arg)
        true_omega = w0 + a_amp * envelope * (big_w * np.cos(arg) - pmdecay * np.sin(arg))
        used_omega = w0 + a_amp * big_w * np.cos(arg)
        scale = VOLT / (ME_MEV * 1e6) * np.cos(phase)
        return scale * true_omega, scale * used_omega


    defect = False
    print(
        "energy slope d(gamma)/dt (1/s) across a bunch at 2 ns; used = slope for the undamped local frequency"
    )
    for mode in modes():
        for pmfreq, pmdecay in [(9e7, 0.0), (0.0, 0.0), (9e7, 4e8), (0.0, 4e8)]:
            observed = slope(mode, pmfreq, pmdecay)
            expected, used = expected_slopes(pmfreq, pmdecay)
            bad = not np.isclose(observed, expected, rtol=RTOL, atol=0)
            print(
                f"{mode} PMFREQ={pmfreq:g} PMDECAY={pmdecay:g}: observed={observed:.10g} "
                f"expected={expected:.10g} used={used:.10g} observed/expected-1={observed / expected - 1:.4e}"
            )
            if pmdecay == 0:
                if bad:
                    sys.exit("The undamped control disagrees: the comparison is not valid.")
            elif bad:
                # The damped slope must differ from the derivative-consistent one by the omitted damping
                # terms, i.e. equal the slope for the undamped local frequency; anything else is an
                # unexplained discrepancy, not this defect.
                if not np.isclose(observed, used, rtol=RTOL, atol=0):
                    sys.exit(
                        "The damped slope matches neither the derivative nor the undamped frequency."
                    )
                defect = True
    print(f"comparison: rtol={RTOL:g}")
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

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

