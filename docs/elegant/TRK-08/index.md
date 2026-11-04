---
title: "TRK-08: Finite matrix RFCA assigns the whole transit to the entrance speed"
hide:
  - toc
---

# TRK-08 — Finite matrix RFCA assigns the whole transit to the entrance speed

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`trackRfCavityWithWakes` matrix branch, simple_rfca.c:994–1078](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L994-L1078)<br>[matrix-method selection, simple_rfca.c:525–530](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L525-L530)<br>[parameter table, track_data.c:597–617](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L597-L617)<br>[element description, track_data.c:233](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L233)<br>[RFCA manual, rfca.tex](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/rfca.tex)<br>[longitudinal coordinate, elegant.tex:364–382](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L364-L382) |

An RFCA of nonzero length using the default matrix method (N_KICKS = 0) times the entire cavity at the entrance speed despite modeling acceleration in its transverse response. Without transverse focusing, that response bounds the transit time for monotonic acceleration; kick tracking satisfies its corresponding bounds. For a 0.75 m, 160 kV cavity at crest and electron momentum p = 2.3 mₑc, the 2.728 ns transit exceeds the allowed upper limit by 27.45 ps. The minimum timing errors fall to 0.3778 ps at 10 mₑc and 0.3902 fs at 100 mₑc, so the main impact is on low-energy beams.

??? example "Reproducer: TRK-08.py"

    [Download TRK-08.py](TRK-08.py)

    ```python
    """TRK-08: Finite matrix RFCA assigns the whole transit to the entrance speed

    An RFCA of nonzero length using the default matrix method (N_KICKS = 0) times the entire cavity at
    the entrance speed despite modeling acceleration in its transverse response. Without transverse
    focusing, that response bounds the transit time for monotonic acceleration; kick tracking satisfies
    its corresponding bounds. For a 0.75 m, 160 kV cavity at crest and electron momentum p = 2.3 mₑc,
    the 2.728 ns transit exceeds the allowed upper limit by 27.45 ps. The minimum timing errors fall to
    0.3778 ps at 10 mₑc and 0.3902 fs at 100 mₑc, so the main impact is on low-energy beams.

    The script sends electrons of 2.3, 10 and 100 mₑc, on axis and with slopes of plus and minus 10⁻⁷,
    through a 0.75 m RFCA of 160 kV (zero frequency or 1 MHz, crest phase, no end or body focusing,
    default N_KICKS = 0). The slope pair measures the mean of 1/p along the cavity, and the convexity
    of the flight time per length turns it into an interval for the transit time; the on-axis time is
    compared with that interval. Controls are zero voltage, an exact drift, and kick tracking with one
    and 128 kicks, which stay inside the interval; the uniform-field flight time is printed for
    reference.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h, as Elegant uses them.
    C, ME_MEV = 299792458.0, 0.51099906
    LENGTH, VOLT = 0.75, 160000.0  # m, V
    EPS = 1e-7  # slope pair measuring the transverse response
    MOMENTA = (2.3, 10.0, 100.0)  # entrance momentum, units of m_e c; the reference is the first
    TOL = 3e-22  # s, agreement with the interval for the controls; a larger excess is the defect
    # (label, length, voltage, FREQ, N_KICKS)
    CASES = [
        ("matrix, FREQ=0", LENGTH, VOLT, 0.0, 0),
        ("matrix, FREQ=1 MHz", LENGTH, VOLT, 1e6, 0),
        ("matrix, zero voltage", LENGTH, 0.0, 0.0, 0),
        ("kicks, N_KICKS=1", LENGTH, VOLT, 0.0, 1),
        ("kicks, N_KICKS=128", LENGTH, VOLT, 0.0, 128),
    ]


    def track(mode, length, volt, freq, kicks):
        """Exit x, t, p of the nine particles: per momentum, slopes 0, +EPS and -EPS, all at x = 0."""
        lattice = (
            f"R: RFCA,L={length!r},VOLT={volt!r},PHASE=90,FREQ={freq!r},N_KICKS={kicks},"
            "END1_FOCUS=0,END2_FOCUS=0,CHANGE_T=0,CHANGE_P0=0,T_REFERENCE=0\n"
            "BL: LINE=(R)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={MOMENTA[0]!r}, "
            "output=out.sdds, default_order=2 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {"p": np.repeat(MOMENTA, 3), "xp": np.tile([0.0, EPS, -EPS], len(MOMENTA))}
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if len(out["p"]) != len(beam["p"]):
            sys.exit(f"{mode}: particles were lost")
        return out


    def allowed_interval(p_in, p_out, rbar, length):
        """Transit time bounds from the mean of r = 1/p along the cavity, for any profile p(z).

        c dt/dz = gamma/p = f(r) with f(r) = sqrt(1 + r^2), convex. For a monotonic p(z), r(z) lies
        between r_out = 1/p_out and r_in = 1/p_in. Jensen's inequality gives the mean of f(r) >=
        f(rbar); the chord of f through the end values lies above f and is linear in r, so the mean of
        f(r) <= chord(rbar). The time is L times the mean of f divided by c.
        """
        r_in, r_out = 1 / p_in, 1 / p_out
        weight = (rbar - r_out) / (r_in - r_out)
        if not -1e-9 < weight < 1 + 1e-9:
            sys.exit("the measured mean of 1/p lies outside the end values")
        f_in, f_out = np.sqrt(1 + r_in**2), np.sqrt(1 + r_out**2)
        lower = length / C * np.sqrt(1 + rbar**2)
        upper = length / C * (weight * f_in + (1 - weight) * f_out)
        return float(lower), float(upper)


    defect = False
    for mode in modes():
        for label, length, volt, freq, kicks in CASES:
            out = track(mode, length, volt, freq, kicks)
            for index, p_in in enumerate(MOMENTA):
                on_axis, plus, minus = 3 * index, 3 * index + 1, 3 * index + 2
                p_out, observed = float(out["p"][on_axis]), float(out["t"][on_axis])
                if not np.isfinite([p_out, observed]).all():
                    sys.exit(f"{mode} {label}, p_in={p_in:g}: nonfinite time or momentum")
                gamma_in, gamma_out = np.hypot(p_in, 1), np.hypot(p_out, 1)
                gain = volt / (1e6 * ME_MEV)  # energy gain in units of m_e c^2
                if not freq and abs(gamma_out - gamma_in - gain) > 7e-13:
                    sys.exit(f"{mode} {label}: the energy gain is not V/(m c^2/e)")
                name = f"{mode} {label}, p_in={p_in:g}"
                if not volt:
                    # No field: an exact drift, t = L gamma/(p c).
                    expected = float(length * gamma_in / (p_in * C))
                    print(f"{name}: observed=[{observed!r}] expected=[{expected!r}]")
                    if abs(observed - expected) > TOL:
                        sys.exit(f"{name}: the zero-voltage control is not an exact drift")
                    continue
                # Without focusing p_x is conserved, so to first order x = p_x * integral dz/p(z).
                # With p_x = p_in*EPS/sqrt(1 + EPS^2) (units of m_e c), the slope pair gives
                # rbar = (x_plus - x_minus)/(2 p_x L), the mean of 1/p over the cavity.
                px = p_in * EPS / np.sqrt(1 + EPS**2)
                rbar = float(out["x"][plus] - out["x"][minus]) / (2 * px * length)
                lower, upper = allowed_interval(p_in, p_out, rbar, length)
                if not all(np.isfinite([observed, rbar, lower, upper])):
                    sys.exit(f"{mode} {label}, p_in={p_in:g}: nonfinite time or interval endpoint")
                late = max(observed - upper, lower - observed, 0.0)
                note = f"outside by {late:.4g} s" if late > TOL else "inside"
                if freq == 0.0 and gain:
                    # Uniform field: dgamma/dz = gain/L, so dt/dz = gamma/(c p) integrates to
                    # L (p_out - p_in)/(c gain).
                    uniform = float(length * (p_out - p_in) / (gain * C))
                    note += f"; uniform-field time {uniform!r}"
                print(f"{name}: observed=[{observed!r}] expected=[{lower!r}, {upper!r}] {note}")
                if kicks:
                    if late > TOL:
                        sys.exit(f"{name}: the kick-tracking control violates the interval")
                else:
                    defect |= late > TOL
    print(f"comparison: interval tolerance {TOL:g} s")
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
| Found (source observation of the entrance-speed timing) | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source reading, not executed for this claim |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

