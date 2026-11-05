---
title: "DOC-07: WAKE manual says ALLOW_LONG_BEAM ignores the beam's far part, but Elegant keeps the whole beam"
hide:
  - toc
---

# DOC-07 — WAKE manual says ALLOW_LONG_BEAM ignores the beam's far part, but Elegant keeps the whole beam

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [WAKE manual, wake.tex:34–38](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/wake.tex#L34-L38)<br>[long-beam check, wake.c:154–168](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L154-L168)<br>[wake.c:176–181](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L176-L181)<br>[bin range, wake.c:200–207](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L200-L207)<br>[wake.c:514–529](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L514-L529)<br>[track_data.c:2540](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2540) |

The WAKE manual says that when the beam is longer than the wake data and ALLOW_LONG_BEAM is nonzero, Elegant warns and ignores the part of the beam furthest from the mean arrival time in computing the wake. Elegant warns but bins and kicks the whole beam; the trimming is commented out in the source. For a 0.8 nC beam of 90 electrons at t = 0 and satellite groups of 10 at 50 ps and 51 ps, with 20 ps of wake data, the manual's algorithm keeps only the main group and gives no loss, whereas the 51 ps group loses 2182 eV, the whole-beam pair sum.

??? example "Reproducer: DOC-07.py"

    [Download DOC-07.py](DOC-07.py)

    ```python
    """DOC-07: WAKE manual says ALLOW_LONG_BEAM ignores the beam's far part, but Elegant keeps the whole beam

    The WAKE manual says that when the beam is longer than the wake data and ALLOW_LONG_BEAM is nonzero, Elegant warns and ignores the part of the beam furthest from the mean arrival time in computing the wake. Elegant warns but bins and kicks the whole beam; the trimming is commented out in the source. For a 0.8 nC beam of 90 electrons at t = 0 and satellite groups of 10 at 50 ps and 51 ps, with 20 ps of wake data, the manual's algorithm keeps only the main group and gives no loss, whereas the 51 ps group loses 2182 eV, the whole-beam pair sum.

    The script tracks 90 electrons at t = 0 and two satellite groups of 10 at 50 ps and 51 ps through
    a WAKE whose table covers only 20 ps (W(0) = 0, W = 3e13 V/C for 0 < t <= 20 ps), with
    ALLOW_LONG_BEAM=1 and, as the control, with ALLOW_LONG_BEAM=0. It compares the losses with the
    manual's algorithm, which trims the beam to the 20 ps nearest its mean arrival time and so
    ignores both satellite groups, and with the direct pair sum over the whole beam.
    """

    import numpy as np
    from _elegant import verdict
    from _wake import RunFailure, beam, loss, table, track

    P0, CHARGE, W_PLUS = 2000.0, 0.8e-9, 3e13  # m_e c, C, V/C
    grid = np.arange(201) * 0.1e-12  # s; the wake data last 20 ps
    wake = np.where(grid > 0, W_PLUS, 0.0)  # W(0) = 0 keeps the own-bin term out of the comparison
    TABLE = table({"t": grid, "W": wake}, {"t": "s", "W": "V/C"})
    TIMES = np.r_[np.zeros(90), np.full(10, 50e-12), np.full(10, 51e-12)]
    Q = CHARGE / len(TIMES)  # macroparticle charge

    # Manual (doc/wake.tex, step 2): the beam spans 51 ps > 20 ps, and "that part of the beam that is
    # furthest from t_mean is ignored". t_mean = (10*50 + 10*51)/110 = 9.18 ps, so the far end is the
    # tail: keeping the 20 ps from t_min retains the 90 particles at t = 0 and ignores both satellite
    # groups; with W(0) = 0 nobody then loses energy.
    expected_manual = np.zeros(len(TIMES))
    # Whole-beam pair sum: a particle loses Q W(t_i - t_j) for each particle j ahead of it within the
    # 20 ps of data; only the 51 ps group has such sources (the 50 ps group, 1 ps ahead).
    delay = TIMES[:, None] - TIMES[None, :]
    pair_sum = Q * np.where((delay > 0) & (delay <= grid[-1]), W_PLUS, 0.0).sum(axis=1)

    element = 'W1: WAKE,INPUTFILE="wake.sdds",TCOLUMN="t",WCOLUMN="W"'
    try:
        track(
            f"Q: CHARGE,TOTAL={CHARGE!r}\n{element},ALLOW_LONG_BEAM=0",
            "Q,W1",
            beam(TIMES, P0),
            P0,
            {"wake.sdds": TABLE},
        )
        raise SystemExit("ALLOW_LONG_BEAM=0 accepted a beam longer than the wake data")
    except RunFailure as failure:
        print(f"serial ALLOW_LONG_BEAM=0 (control): fatal error, exit status {failure.status}")

    out, log = track(
        f"Q: CHARGE,TOTAL={CHARGE!r}\n{element},ALLOW_LONG_BEAM=1",
        "Q,W1",
        beam(TIMES, P0),
        P0,
        {"wake.sdds": TABLE},
    )
    observed = loss(out, P0)
    groups = [slice(0, 90), slice(90, 100), slice(100, 110)]
    summary = ", ".join(f"{observed[g].mean():.10g}" for g in groups)
    print(
        f"serial ALLOW_LONG_BEAM=1: loss of the groups at 0, 50, 51 ps observed=[{summary}] eV "
        f"expected (manual, trimmed)=[0, 0, 0] eV; whole-beam pair sum=[0, 0, {pair_sum[-1]:.10g}] eV; "
        f"long-beam warning printed: {'longer than the longitudinal wake' in log}"
    )
    if not np.allclose(observed, pair_sum, rtol=1e-9, atol=1e-6):
        raise SystemExit("the losses differ from the whole-beam pair sum")
    verdict(bool(np.max(np.abs(observed - expected_manual)) > 1.0))

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
| Confirmed (a WAKE documentation defect; the non-trimming implementation is not shown to be physically wrong) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

