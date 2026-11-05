---
title: "OPT-24: closed_orbit ignores start_from_centroid and start_from_dp_centroid"
hide:
  - toc
---

# OPT-24 — closed_orbit ignores start_from_centroid and start_from_dp_centroid

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`run_closed_orbit`, src/closed_orbit.c:302–313](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/closed_orbit.c#L302-L313)<br>[src/elegant.c:1355](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L1355)<br>[src/elegant.c:1506](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L1506)<br>[the namelist defaults, doc/elegant.tex:1728–1729](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L1728-L1729)<br>[`start_from_centroid` forces the computation to start from the beam centroids, `start_from_dp_centroid` uses the momentum centroid and can allow computing the closed orbit for an off-momentum beam, doc/elegant.tex:1754–1764](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L1754-L1764) |

With `closed_orbit` in its default `fixed_length=0` mode, `start_from_centroid` and `start_from_dp_centroid` are ignored, so an off-momentum beam gets the on-momentum orbit. The manual promises to use the beam's momentum centroid for either option; `start_from_centroid` also supplies the transverse starting point. For a 2.4 GeV, 70.00 m ring and beam momentum offset δ = 0.001, independent tracking gives x = 2.458 mm and x′ = 0.4474 mrad at the orbit start. Either option instead returns δ = 0 and x = 2.799 nm. Fixed-length modes can vary δ, although they also ignore these centroid flags.

??? example "Reproducer: OPT-24.py"

    [Download OPT-24.py](OPT-24.py)

    ```python
    """OPT-24: closed_orbit ignores start_from_centroid and start_from_dp_centroid

    With `closed_orbit` in its default `fixed_length=0` mode, `start_from_centroid` and
    `start_from_dp_centroid` are ignored, so an off-momentum beam gets the on-momentum orbit. The
    manual promises to use the beam's momentum centroid for either option; `start_from_centroid`
    also supplies the transverse starting point. For a 2.4 GeV, 70.00 m ring and beam momentum
    offset δ = 0.001, independent tracking gives x = 2.458 mm and x′ = 0.4474 mrad at the orbit
    start. Either option instead returns δ = 0 and x = 2.799 nm. Fixed-length modes can vary δ,
    although they also ignore these centroid flags.

    The script runs serial Elegant `closed_orbit` before tracking a one-particle beam with x = 0.3 mm,
    x′ = 0.1 mrad and δ = 0.001 through a 2.4 GeV, 70 m ring of 20 FODO cells with CSBENDs, with the
    flags (start_from_centroid, start_from_dp_centroid) set to (0, 0), (1, 0) and (0, 1), and reads
    the orbit at the ring start. The expected orbits for δ = 0.001 and, as a control, δ = 0 come from
    Newton iteration on single-turn tracking without `closed_orbit`; the (0, 0) run must give the
    on-momentum orbit.
    """

    import sys

    import numpy as np
    from _elegant import particles, run, verdict

    P0 = 2400 / 0.51099906  # reference momentum, units of m_e c (Elegant's electron mass)
    DELTA = 1e-3
    CENTROID = {"x": 3e-4, "xp": 1e-4}
    LATTICE = f"""F: QUAD,L=0.3,K1=0.9
    G: QUAD,L=0.3,K1=-0.85
    D: DRIF,L=0.65
    E: DRIF,L=0.4
    B: CSBEND,L=1.2,ANGLE={2 * np.pi / 20!r},N_SLICES=40,EXPAND_HAMILTONIAN=0
    C: LINE=(F,D,B,E,G,D)
    R: LINE=(20*C)
    """
    STEP = 2e-6


    def shoot(delta):
        """Periodic (x, xp) at the ring start for momentum offset delta, by Newton iteration."""
        orbit = np.zeros(2)
        probes = np.array([[0, 0], [STEP, 0], [-STEP, 0], [0, STEP], [0, -STEP]])
        commands = (
            "&run_setup lattice=ring.lte, use_beamline=R, p_central_mev=2400, output=out.sdds &end\n"
            "&run_control &end\n&sdds_beam input=beam.sdds &end\n&track &end\n"
        )
        for _ in range(10):
            start = orbit + probes
            beam = {
                "x": start[:, 0],
                "xp": start[:, 1],
                "p": np.full(len(probes), P0 * (1 + delta)),
                "particleID": np.arange(1, len(probes) + 1),
            }
            o = particles(
                run({"ring.lte": LATTICE, "run.ele": commands}, beams={"beam.sdds": beam})["out.sdds"]
            )
            end = np.c_[o["x"], o["xp"]]
            residual = end[0] - orbit
            if np.abs(residual).max() < 1e-14:
                return orbit
            jacobian = np.c_[(end[1] - end[2]) / (2 * STEP), (end[3] - end[4]) / (2 * STEP)] - np.eye(2)
            orbit = orbit - np.linalg.solve(jacobian, residual)
        sys.exit("Newton iteration for the closed orbit did not converge.")


    def closed_orbit(centroid, dp_centroid):
        commands = (
            "&run_setup lattice=ring.lte, use_beamline=R, p_central_mev=2400 &end\n"
            "&twiss_output filename=twiss.sdds, matched=1 &end\n"
            f"&closed_orbit output=orbit.sdds, start_from_centroid={centroid},"
            f" start_from_dp_centroid={dp_centroid} &end\n"
            "&run_control &end\n"
            "&bunched_beam n_particles_per_bunch=1, bunch=bunch.sdds,"
            f" centroid[0]={CENTROID['x']!r}, centroid[1]={CENTROID['xp']!r}, centroid[5]={DELTA!r} &end\n"
            "&track &end\n"
        )
        result = run(
            {"ring.lte": LATTICE, "run.ele": commands},
            outputs=("orbit.sdds", "bunch.sdds", "twiss.sdds"),
        )
        orbit = result["orbit.sdds"]
        bunch = particles(result["bunch.sdds"])
        eta = float(result["twiss.sdds"][0, "etax"][0])
        start = [float(orbit[0, "x"][0]), float(orbit[0, "xp"][0])]
        beam = [float(bunch["x"][0]), float(bunch["xp"][0]), float(bunch["p"][0]) / P0 - 1]
        return start, float(orbit["delta"][0]), beam, eta


    on = shoot(0.0)
    off = shoot(DELTA)
    print(
        f"tracked closed orbit at the ring start: delta=0 x, xp={on.tolist()}; delta={DELTA:g} x, xp={off.tolist()}"
    )
    defect = False
    for centroid, dp_centroid in ((0, 0), (1, 0), (0, 1)):
        start, delta, beam, eta = closed_orbit(centroid, dp_centroid)
        label = f"start_from_centroid={centroid} start_from_dp_centroid={dp_centroid}"
        if not np.allclose(beam, [CENTROID["x"], CENTROID["xp"], DELTA], rtol=1e-9, atol=1e-15):
            sys.exit(f"{label}: the beam centroid is not the requested one: {beam}")
        expected = on if (centroid, dp_centroid) == (0, 0) else off
        expected_delta = 0.0 if (centroid, dp_centroid) == (0, 0) else DELTA
        print(
            f"{label}: x, xp, delta observed=[{start[0]:.10g}, {start[1]:.10g}, {delta:.10g}]"
            f" expected=[{expected[0]:.10g}, {expected[1]:.10g}, {expected_delta:.10g}]"
        )
        bad = (
            not np.allclose(start, expected, rtol=1e-6, atol=1e-11)
            or abs(delta - expected_delta) > 1e-12
        )
        if (centroid, dp_centroid) == (0, 0):
            if bad:
                sys.exit(
                    "The on-momentum control disagrees with the tracked closed orbit: invalid comparison."
                )
            print(f"control: dispersion at the start {eta:.10g} m; eta*delta = {eta * DELTA:.10g} m")
        else:
            defect |= bad
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
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

