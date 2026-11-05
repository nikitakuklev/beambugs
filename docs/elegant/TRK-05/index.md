---
title: "TRK-05: Zero-voltage MODRF and RAMPRF advance the arrival time by L instead of the slanted path length"
hide:
  - toc
---

# TRK-05 — Zero-voltage MODRF and RAMPRF advance the arrival time by L instead of the slanted path length

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [zero-voltage branch, mod_rfca.c:102–111](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/mod_rfca.c#L102-L111)<br>[zero-voltage branch, ramped_rfca.c:67–77](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ramped_rfca.c#L67-L77)<br>[MODRF half drifts, mod_rfca.c:246–262](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/mod_rfca.c#L246-L262)<br>[RAMPRF half drifts, ramped_rfca.c:170–186](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ramped_rfca.c#L170-L186) |

With VOLT = 0, MODRF and RAMPRF move particles along straight lines but advance the longitudinal coordinate by cavity length L, omitting the extra path L(√(1 + x′² + y′²) − 1) for transverse slopes x′ and y′. A nonzero nominal voltage with exactly zero effective field gives the full slanted-path time, as an exact-drift control confirms. For L = 1.7 m and electron momentum 2 mₑc with x′ = 0.03 and y′ = −0.02, arrival is at 6.340 ns instead of 6.344 ns, 4.120 ps early.

??? example "Reproducer: TRK-05.py"

    [Download TRK-05.py](TRK-05.py)

    ```python
    """TRK-05: Zero-voltage MODRF and RAMPRF advance the arrival time by L instead of the slanted path length

    With VOLT = 0, MODRF and RAMPRF move particles along straight lines but advance the longitudinal
    coordinate by cavity length L, omitting the extra path L(√(1 + x′² + y′²) − 1) for transverse slopes
    x′ and y′. A nonzero nominal voltage with exactly zero effective field gives the full slanted-path
    time, as an exact-drift control confirms. For L = 1.7 m and electron momentum 2 mₑc with x′ = 0.03
    and y′ = −0.02, arrival is at 6.340 ns instead of 6.344 ns, 4.120 ps early.

    The script tracks particles with three slope pairs through a 1.7 m MODRF and a 1.7 m RAMPRF with
    VOLT = 0, and compares their arrival times with the time of flight over the slanted path at constant
    speed. Controls: the on-axis pair, which has no extra path, a MODRF and a RAMPRF with a nonzero VOLT whose effective field is
    exactly zero (AMOFFSET = -1, or a zero waveform), and an EDRIFT.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    C = 299792458.0  # m/s, c_mks of SDDS
    P0 = 2.0  # particle momentum, units of m_e c
    LENGTH = 1.7  # m
    # Slopes (xp, yp): on axis, the pair of the summary, and a larger pair; four copies of each pair
    # so that every MPI worker has some.
    SLOPES = np.repeat([[0.0, 0.0], [0.03, -0.02], [0.1, 0.04]], 4, axis=0)
    BEAM = {"xp": SLOPES[:, 0], "yp": SLOPES[:, 1], "p": np.full(12, P0)}
    RTOL = 1e-13  # the exact cases agree to 3e-16
    # Constant waveform factors for RAMPRF, as an ASCII SDDS table.
    WAVE = "SDDS1\n&column name=t, type=double, units=s &end\n&column name=v, type=double &end\n&data mode=ascii, no_row_counts=1 &end\n-1 {v}\n1 {v}\n"
    CAVITY = 'FREQ=1e8,PHASE=0,FIDUCIAL="light"'
    LATTICES = {
        "EDRIFT": f"E: EDRIFT,L={LENGTH!r}\n",
        "MODRF, VOLT=0": f"E: MODRF,L={LENGTH!r},VOLT=0,{CAVITY}\n",
        "RAMPRF, VOLT=0": f'E: RAMPRF,L={LENGTH!r},VOLT=0,{CAVITY},VOLT_WAVEFORM="wave.sdds=t+v"\n',
        "MODRF, VOLT=10000, AMOFFSET=-1": f"E: MODRF,L={LENGTH!r},VOLT=10000,{CAVITY},AMOFFSET=-1\n",
        "RAMPRF, VOLT=10000, zero waveform": f'E: RAMPRF,L={LENGTH!r},VOLT=10000,{CAVITY},VOLT_WAVEFORM="wave.sdds=t+v"\n',
    }


    def arrival_times(mode, name):
        """Arrival time (s) of the three slope pairs after the element; particles start at t = 0."""
        lattice = LATTICES[name] + "BL: LINE=(E)\n"
        wave = WAVE.format(v=0 if "zero waveform" in name else 1)
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
            "output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        files = {"lattice.lte": lattice, "run.ele": commands, "wave.sdds": wave}
        out = particles(run(files, mode, {"beam.sdds": BEAM})["out.sdds"])
        if len(out["t"]) != 12:
            sys.exit(f"{mode} {name}: {12 - len(out['t'])} particles lost")
        if not np.allclose(out["p"], P0, rtol=1e-14, atol=0):
            sys.exit(f"{mode} {name}: the momentum changed")
        # The expected flight time assumes the field-free straight ray: slopes unchanged and
        # x = L xp, y = L yp from the origin. Anything else invalidates the comparison.
        straight = (out["x"], out["xp"], out["y"], out["yp"])
        ray = (LENGTH * SLOPES[:, 0], SLOPES[:, 0], LENGTH * SLOPES[:, 1], SLOPES[:, 1])
        if not all(np.allclose(o, e, rtol=0, atol=1e-14) for o, e in zip(straight, ray, strict=True)):
            sys.exit(f"{mode} {name}: the transverse coordinates are not the straight ray")
        times = out["t"].reshape(3, 4)
        if not (times == times[:, :1]).all():
            sys.exit(f"{mode} {name}: identical particles arrived at different times")
        return times[:, 0]


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected disagree."""
        bad = not np.allclose(observed, expected, rtol=RTOL, atol=0)
        fmt = lambda a: "[" + ", ".join(f"{v:.16g}" for v in a) + "]"
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    # Constant speed over the slanted path: t = L sqrt(1 + xp^2 + yp^2) / (beta c), with
    # beta = p/sqrt(1 + p^2) for p in units of m_e c.
    path = LENGTH * np.sqrt(1 + SLOPES[::4, 0] ** 2 + SLOPES[::4, 1] ** 2)
    expected = path / (P0 / np.sqrt(1 + P0**2) * C)
    defect = False
    print("columns: arrival time (s) of the slope pairs (0, 0), (0.03, -0.02), (0.1, 0.04)")
    for mode in modes():
        for name in LATTICES:
            observed = arrival_times(mode, name)
            # On axis the slanted path is L: the first pair must agree for every element.
            if not np.isclose(observed[0], expected[0], rtol=RTOL, atol=0):
                sys.exit(f"{mode} {name}: the on-axis control disagrees, the comparison is not valid.")
            bad = compare(f"{mode} {name}", observed, expected)
            if "VOLT=0" in name:
                defect |= bad
            elif bad:
                sys.exit(f"{mode} {name}: the control disagrees with the slanted path.")
    print(f"comparison: rtol={RTOL:g}")
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
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

