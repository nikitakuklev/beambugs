---
title: "TRK-15: GKICKMAP applies classical radiation loss with SYNCH_RAD disabled"
hide:
  - toc
---

# TRK-15 — GKICKMAP applies classical radiation loss with SYNCH_RAD disabled

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`trackKickMap`, kickmap.c:92–99](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/kickmap.c#L92-L99)<br>[`radCoef` and `isrCoef`, kickmap.c:60–61](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/kickmap.c#L60-L61)<br>[parameter dictionary, track_data.c:3469–3470](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3469-L3470)<br>[UKICKMAP coefficients, ukickmap.c:56–59](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ukickmap.c#L56-L59)<br>[GKICKMAP manual, gkickmap.tex:1–22](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/gkickmap.tex#L1-L22) |

Selecting ISR = 1 with SYNCH_RAD = 0 in GKICKMAP also applies classical radiation loss, despite the switches independently selecting excitation and mean loss. In a one-slice, 0.6 m map at reference momentum p₀ = 4200 mₑc, with entrance slope x′ = 0.004 and horizontal slope kick 0.025, the unwanted relative-momentum loss is 1.450×10⁻⁷. A separate small-angle radiation calculation agrees with the classical-only control within 10⁻¹⁶, while the radiation-free control preserves momentum. With paired random seeds, the ISR-only and both-enabled runs give identical particle momenta instead of differing by that loss.

??? example "Reproducer: TRK-15.py"

    [Download TRK-15.py](TRK-15.py)

    ```python
    """TRK-15: GKICKMAP applies classical radiation loss with SYNCH_RAD disabled

    Selecting ISR = 1 with SYNCH_RAD = 0 in GKICKMAP also applies classical radiation loss, despite the
    switches independently selecting excitation and mean loss. In a one-slice, 0.6 m map at reference
    momentum p₀ = 4200 mₑc, with entrance slope x′ = 0.004 and horizontal slope kick 0.025, the
    unwanted relative-momentum loss is 1.450×10⁻⁷. A separate small-angle radiation calculation agrees
    with the classical-only control within 10⁻¹⁶, while the radiation-free control preserves momentum.
    With paired random seeds, the ISR-only and both-enabled runs give identical particle momenta
    instead of differing by that loss.

    The script tracks 24 identical electrons (p0 = 4200 mₑc, entrance slope 0.004) through one slice of
    a 0.6 m GKICKMAP whose map is a constant kick of 0.025, for the four combinations of the
    SYNCH_RAD and ISR switches, all with the same random seed. It checks the classical-only case
    against a paraxial one-slice radiation model (curvature κ = KICK/L, path weighted by the post-kick
    slope; not an exact constant-curvature arc), and the no-radiation case against zero loss; these
    controls validate the oracle. It then compares the case with only ISR against the case with both
    switches, which must differ by exactly that classical loss.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h, as Elegant uses them; eps_0 = 1/(mu_0 c^2) with mu_0 = 4 pi 1e-7.
    C, QE, ME = 299792458.0, 1.60217733e-19, 9.1093897e-31
    EPS0 = 1e7 / (4 * np.pi * C**2)
    RE = QE**2 / (4 * np.pi * EPS0 * ME * C**2)  # classical electron radius, m
    P0, LENGTH, SLOPE, KICK = 4200.0, 0.6, 0.004, 0.025  # m_e c, m, rad, rad
    COPIES = 24
    SEED = 24681357
    ATOL = 2e-16  # delta is recovered from p/p0 - 1, which rounds at about 1e-16

    # Constant kick (xpFactor) on a 5 x 5 grid, y outer and x inner as the manual documents.
    GRID = [-0.05, -0.025, 0.0, 0.025, 0.05]
    MAP = (
        "SDDS1\n"
        '&column name=x, type=double, units="m" &end\n'
        '&column name=y, type=double, units="m" &end\n'
        "&column name=xpFactor, type=double &end\n"
        "&column name=ypFactor, type=double &end\n"
        "&data mode=ascii &end\n"
        f"{len(GRID) ** 2}\n" + "".join(f"{x!r} {y!r} {KICK!r} 0\n" for y in GRID for x in GRID)
    )


    def classical_loss():
        """Fractional momentum loss of one pass in the paraxial one-slice radiation model.

        The slope change KICK over L is taken as a curvature kappa = KICK/L, a small-angle
        approximation (a slope change is not exactly an angular bend). An electron radiates classically
        d(delta) = -(2/3) r_e p0^3 (1 + delta)^2 kappa^2 ds per path length ds (p0 in units of
        m_e c; the power (2/3) r_e m c^3 gamma^4 kappa^2 divided by the energy gamma m c^2). The path
        weight is L sqrt(1 + xp^2) with the slope xp = SLOPE + KICK after the kick, as for one slice
        in the code, not the exact circular path integral.
        """
        kappa = KICK / LENGTH
        return 2 / 3 * RE * P0**3 * kappa**2 * LENGTH * np.sqrt(1 + (SLOPE + KICK) ** 2)


    def momentum_change(mode, synch_rad, isr):
        """delta = p/p0 - 1 of each particle, ordered by particle ID."""
        lattice = (
            f'K: GKICKMAP, L={LENGTH!r}, N_KICKS=1, INPUT_FILE="map.sdds", '
            f"SYNCH_RAD={synch_rad}, ISR={isr}\n"
            "BL: LINE=(K)\n"
        )
        commands = (
            "&change_particle name=electron &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
            f"output=out.sdds, random_number_seed={SEED} &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {"p": np.full(COPIES, P0), "xp": np.full(COPIES, SLOPE)}
        result = run(
            {"lattice.lte": lattice, "run.ele": commands, "map.sdds": MAP}, mode, {"beam.sdds": beam}
        )
        out = particles(result["out.sdds"])
        if len(out["p"]) != COPIES:
            sys.exit(f"{mode}: {COPIES - len(out['p'])} particles lost")
        if not np.all(np.isfinite(out["p"])):
            sys.exit(f"{mode} SYNCH_RAD={synch_rad} ISR={isr}: nonfinite output momenta")
        return out["p"] / P0 - 1


    def compare(label, observed, expected):
        """Print the mean over the particles; return True when every particle agrees."""
        print(f"{label}: observed=[{float(np.mean(observed))!r}] expected=[{float(expected)!r}]")
        return bool(np.allclose(observed, expected, rtol=0, atol=ATOL))


    loss = classical_loss()
    defect = False
    print("mean fractional momentum change delta of 24 electrons")
    for mode in modes():
        delta = {(sr, isr): momentum_change(mode, sr, isr) for sr in (0, 1) for isr in (0, 1)}
        if not compare(f"{mode} SYNCH_RAD=0 ISR=0", delta[0, 0], 0.0):
            sys.exit("The radiation-free control loses momentum: the comparison is not valid.")
        if not compare(f"{mode} SYNCH_RAD=1 ISR=0", delta[1, 0], -loss):
            sys.exit("The classical-only control disagrees with the curvature oracle: not valid.")
        # With one slice the random draw does not depend on the classical switch, so the particles
        # of the ISR-only run (where no classical loss should occur) sit exactly `loss` above those
        # of the run with both switches.
        print(f"{mode} SYNCH_RAD=0 ISR=1: mean delta={float(np.mean(delta[0, 1]))!r}")
        print(f"{mode} SYNCH_RAD=1 ISR=1: mean delta={float(np.mean(delta[1, 1]))!r}")
        if compare(f"{mode} ISR only minus both, paired", delta[0, 1] - delta[1, 1], loss):
            continue  # the ISR-only run lacks the classical loss: correct
        # Otherwise the two runs must be identical, the recorded unwanted loss; anything else is a
        # failed run, not a verdict.
        if not np.allclose(delta[0, 1], delta[1, 1], rtol=0, atol=ATOL):
            sys.exit(f"{mode}: the ISR-only run matches neither the paired prediction nor the both run")
        defect = True
    print(f"comparison: atol={ATOL:g}")
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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |

