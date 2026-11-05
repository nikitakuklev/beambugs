---
title: "COL-46: WAKE gives each particle the full W(0) of its own bin, so full-origin wakes such as CORGPIPE's overstate the loss"
hide:
  - toc
---

# COL-46 — WAKE gives each particle the full W(0) of its own bin, so full-origin wakes such as CORGPIPE's overstate the loss

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Tags | <span class="bb-tag">contract-unsettled</span> <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [binTimeDistribution, wake.c:539–545](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L539-L545)<br>[convolution and kick, wake.c:273–290](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L273-L290)<br>[wake.c:514–529](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L514-L529)<br>[WAKE manual, wake.tex:1–12](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/wake.tex#L1-L12)<br>[CORGPIPE table, wake.c:678–686](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L678-L686)<br>[CORGPIPE manual, corgpipe.tex:1–5](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/corgpipe.tex#L1-L5)<br>[CORGPLATES table, wake.c:816–820](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L816-L820)<br>[cwake.c:399](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/cwake.c#L399) |

WAKE gives every particle the table's W(0) times all charge in its own time bin, its own included. When W(0) is the wake just behind a charge, W(0⁺), as in CORGPIPE's table, built from Bane's model, a charge should receive W(0⁺)/2 from itself and nothing from charge behind it (beam-loading theorem), so the mean loss is too large by W(0⁺)/2 times the average charge of a particle's bin. For a uniform 0.8 nC, 4 ps electron bunch in a 0.1 m CORGPIPE with the default time step, the mean loss is 4.288% above the exact particle sum; a quarter step leaves 1.058%. The manual does not say whether a table's W(0) is W(0⁺) or W(0⁺)/2; halving it removes the excess.

??? example "Reproducer: COL-46.py"

    [Download COL-46.py](COL-46.py)

    ```python
    """COL-46: WAKE gives each particle the full W(0) of its own bin, so full-origin wakes such as CORGPIPE's overstate the loss

    WAKE gives every particle the table's W(0) times all charge in its own time bin, its own included.
    When W(0) is the wake just behind a charge, W(0⁺), as in CORGPIPE's table, built from Bane's model,
    a charge should receive W(0⁺)/2 from itself and nothing from charge behind it (beam-loading
    theorem), so the mean loss is too large by W(0⁺)/2 times the average charge of a particle's bin. For
    a uniform 0.8 nC, 4 ps electron bunch in a 0.1 m CORGPIPE with the default time step, the mean loss
    is 4.288% above the exact particle sum; a quarter step leaves 1.058%. The manual does not say
    whether a table's W(0) is W(0⁺) or W(0⁺)/2; halving it removes the excess.

    The script tracks a uniform 0.8 nC electron bunch through a WAKE whose table is a constant
    3e13 V/C from t = 0, with the full value and with half of it at t = 0, and compares the mean loss
    with the beam-loading result Q W/2, which holds for any bunch shape. It then tracks a uniform 4 ps
    bunch through a 0.1 m CORGPIPE with its default time step and with a quarter of it, compares the
    mean loss with the loss of Bane's cosine wake summed exactly over the particles, and reproduces
    the CORGPIPE result with a WAKE table of the same cosine, full and halved at t = 0.
    """

    import numpy as np
    from _elegant import verdict
    from _wake import beam, loss, table, track

    P0, CHARGE = 2000.0, 0.8e-9  # m_e c, C
    C, Z0 = 2.99792458e8, 4e-7 * np.pi * 2.99792458e8  # as in wake.c
    CONTROL_RTOL = 1e-3  # half-origin controls against the exact expectations (binning error ~3e-4)
    MIN_EXCESS = 0.01  # defect if the default CORGPIPE loss exceeds the expectation by more than 1%


    def wake_loss(times: np.ndarray, t: np.ndarray, w: np.ndarray, extra: str = "") -> np.ndarray:
        """Loss (eV) of each particle through CHARGE + WAKE reading the table (t, w)."""
        files = {"wake.sdds": table({"t": t, "W": w}, {"t": "s", "W": "V/C"})}
        out, _ = track(
            f'Q: CHARGE,TOTAL={CHARGE!r}\nD: DRIFT,L=0.05\nW1: WAKE,INPUTFILE="wake.sdds",'
            f'TCOLUMN="t",WCOLUMN="W"{extra}',
            "Q,D,W1,D",
            beam(times, P0),
            P0,
            files,
        )
        return loss(out, P0)


    def report(label: str, observed: float, expected: float) -> float:
        print(
            f"serial {label}: mean loss observed={observed:.10g} eV expected={expected:.10g} eV "
            f"(ratio {observed / expected:.7f})"
        )
        return observed / expected - 1


    # 1. Constant wake W+ = 3e13 V/C on 0 <= t <= 40 ps, step 0.2 ps; 8192 particles spread
    # uniformly over 32 ps. Each particle receives W+ from every particle ahead of it and, by the
    # beam-loading theorem, W+/2 from itself, so the mean loss is q W+ [N(N-1)/2 + N/2]/N = Q W+/2.
    # A convolution that gives each bin's whole charge, own particle included, the full W(0) adds
    # Q W+/2 * sum_b (n_b/N)^2 over the bins; halving W(0) removes it exactly.
    W_FLAT, DT_FLAT, N_FLAT = 3e13, 0.2e-12, 8192
    times = (np.arange(N_FLAT) + 0.5) / N_FLAT * 32e-12
    grid = np.arange(201) * DT_FLAT
    flat = np.full(len(grid), W_FLAT)
    half = flat.copy()
    half[0] /= 2
    expected_flat = CHARGE * W_FLAT / 2
    # Elegant's bins: centers at min(t) - dt + i dt, each particle in its nearest center.
    counts = np.bincount(np.floor((times - (times.min() - DT_FLAT)) / DT_FLAT + 0.5).astype(int))
    print(
        f"flat wake: full-origin prediction Q W+/2 (1 + sum (n_b/N)^2) = "
        f"{expected_flat * (1 + np.sum((counts / N_FLAT) ** 2)):.10g} eV"
    )
    report("WAKE flat, full W(0)", wake_loss(times, grid, flat).mean(), expected_flat)
    control = report(
        "WAKE flat, W(0) halved (control)", wake_loss(times, grid, half).mean(), expected_flat
    )
    if abs(control) > CONTROL_RTOL:
        raise SystemExit("the half-origin flat-wake control misses Q W+/2")

    # 2. CORGPIPE: Bane's single-mode wake W(t) = 2 kappa L cos(omega t) for t > 0, with
    # kappa = Z0 c/(2 pi a^2) and omega = c sqrt(2 p/(a h g)) (radius a, period p, depth h, gap g).
    # Summing the pair wakes with the self term W(0+)/2 gives the mean loss of N equal particles
    #   (q/2N) sum_ij W(|t_i - t_j|) = Q kappa L |<exp(i omega t)>|^2.
    RADIUS, PERIOD, DEPTH, GAP, LENGTH = 0.01, 1e-4, 2e-4, 5e-5, 0.1
    kappa = Z0 * C / (2 * np.pi * RADIUS**2)
    omega = C * np.sqrt(2 * PERIOD / (RADIUS * DEPTH * GAP))
    times = (np.arange(32768) + 0.5) / 32768 * 4e-12
    expected_pipe = CHARGE * kappa * LENGTH * abs(np.mean(np.exp(1j * omega * times))) ** 2
    pipe = f"L={LENGTH},RADIUS={RADIUS},PERIOD={PERIOD},DEPTH={DEPTH},GAP={GAP}"
    # The default step is min(0.1/omega, bunch duration/30), here 4 ps/30, with 1.5*30 = 45 fixed
    # bins centered on the mean time (wake.c, track_through_corgpipe).
    step = float(np.ptp(times)) / 30
    out, log = track(f"Q: CHARGE,TOTAL={CHARGE!r}\nC1: CORGPIPE,{pipe}", "Q,C1", beam(times, P0), P0)
    pipe_loss = loss(out, P0)
    excess = report("CORGPIPE, default step", pipe_loss.mean(), expected_pipe)
    out, _ = track(
        f"Q: CHARGE,TOTAL={CHARGE!r}\nC1: CORGPIPE,{pipe},DT={step / 4!r}", "Q,C1", beam(times, P0), P0
    )
    report("CORGPIPE, a quarter of the default step", loss(out, P0).mean(), expected_pipe)
    cosine = 2 * kappa * LENGTH * np.cos(omega * np.arange(45) * step)
    same = wake_loss(times, np.arange(45) * step, cosine, ",N_BINS=45")
    print(
        f"WAKE with the same cosine table, full W(0): max |loss - CORGPIPE loss| = {np.max(np.abs(same - pipe_loss)):.3g} eV"
    )
    cosine[0] /= 2
    control = report(
        "WAKE with the same cosine table, W(0) halved (control)",
        wake_loss(times, np.arange(45) * step, cosine, ",N_BINS=45").mean(),
        expected_pipe,
    )
    if abs(control) > CONTROL_RTOL:
        raise SystemExit("the half-origin cosine control misses the exact expectation")
    print(
        f"control tolerance {CONTROL_RTOL:g} relative; defect if the default CORGPIPE loss is more than "
        f"{MIN_EXCESS:.0%} above the expectation"
    )
    verdict(excess > MIN_EXCESS)

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

??? example "Helper: _wake.py"

    [Download _wake.py](_wake.py)

    ```python
    """Shared setup for the WAKE, TRWAKE and CORGPIPE reproducers: an ASCII SDDS wake table, one serial
    run of an electron beam through a lattice that reads it, and the energy change of each particle.

    Times are in s, momenta p = beta*gamma in units of m_e c and energies in eV. A positive wake loss
    is energy removed from an electron.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import RunFailure, particles, run

    ME_EV = 0.51099906e6  # SDDS constants.h, as Elegant uses it


    def table(columns: dict[str, np.ndarray], units: dict[str, str]) -> str:
        """ASCII SDDS page with the given double columns (all of one length)."""
        names = list(columns)
        head = "SDDS1\n" + "".join(
            f"&column name={name}, units={units[name]}, type=double &end\n" for name in names
        )
        rows = "".join(
            " ".join(repr(float(columns[name][i])) for name in names) + "\n"
            for i in range(len(columns[names[0]]))
        )
        return head + "&data mode=ascii, no_row_counts=1 &end\n" + rows


    def beam(times: np.ndarray, p: float, x: float = 0.0) -> dict[str, np.ndarray]:
        """Particles at the given arrival times, all at momentum p and offset x, zero slopes."""
        n = len(times)
        return {
            "x": np.full(n, x),
            "t": np.asarray(times, dtype=float),
            "p": np.full(n, p),
            "particleID": np.arange(1, n + 1),
        }


    def track(
        elements: str, line: str, particles_in: dict, p0: float, files: dict[str, str] | None = None
    ) -> tuple[dict[str, np.ndarray], str]:
        """Serial run through BL: LINE=(line); (final coordinates by particle ID, run log).

        `files` holds extra inputs such as wake tables. A nonzero exit raises RunFailure.
        """
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        inputs = {"lattice.lte": elements.rstrip("\n") + f"\nBL: LINE=({line})\n", "run.ele": commands}
        result = run({**inputs, **(files or {})}, beams={"beam.sdds": particles_in})
        out = particles(result["out.sdds"])
        if len(out["p"]) != len(particles_in["p"]):
            raise SystemExit("particles were lost")
        return out, result["log"]


    def loss(out: dict[str, np.ndarray], p0: float) -> np.ndarray:
        """Energy lost by each particle (eV), from its final momentum and the initial p0."""
        return ME_EV * (np.sqrt(1 + p0**2) - np.sqrt(1 + out["p"] ** 2))


    __all__ = ["ME_EV", "RunFailure", "beam", "loss", "table", "track"]

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed (half-bin endpoint excess for full-origin wake data, including CORGPIPE's built-in table; not that every WAKE input is wrong) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

