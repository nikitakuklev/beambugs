---
title: "SCT-17: ibsEmittance stops the equilibrium emittance at ε₀/(1+κ) when τx > τy and reports convergence"
hide:
  - toc
---

# SCT-17 — ibsEmittance stops the equilibrium emittance at ε₀/(1+κ) when τx > τy and reports convergence

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [simplex limits, ibsEmittance.c:806–812](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibsEmittance.c#L806-L812)<br>[simplex call, ibsEmittance.c:871–873](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibsEmittance.c#L871-L873)<br>[balance, ibsEmittance.c:1085–1095](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibsEmittance.c#L1085-L1095)<br>[`emitxInput`, ibsEmittance.c:740–741](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibsEmittance.c#L740-L741)<br>[ibsEmittance.c:910–914](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibsEmittance.c#L910-L914)<br>[description, elegant.tex:9456–9458](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L9456-L9458)<br>[`-coupling`, elegant.tex:9493](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L9493)<br>[sddsbrightness `-coupling`, elegant.tex:9782–9783](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L9782-L9783) |

ibsEmittance, run for equilibrium emittances at coupling ratio κ = εy/εx, never returns a horizontal emittance below ε₀/(1+κ), with ε₀ the uncoupled natural emittance. When the horizontal damping time τx exceeds the vertical τy (horizontal damping partition Jx < 1) and IBS is weak, the radiation–IBS balance the program solves has its root below that bound, at ε₀/(1+κτx/τy) without IBS; the program returns the bound and reports "Emittance converged". For a 3 GeV, 288.0 m ring with Jx = 0.9634, κ = 0.1 and 1 nC, it returns 48.96 nm versus 48.79 nm, 0.3408% high; the transverse balance misses zero by 0.3408% of its excitation term. This lower-bound obstruction is absent for weak IBS when τx ≤ τy.

??? example "Reproducer: SCT-17.py"

    [Download SCT-17.py](SCT-17.py)

    ```python
    """SCT-17: ibsEmittance stops the equilibrium emittance at ε₀/(1+κ) when τx > τy and reports convergence

    ibsEmittance, run for equilibrium emittances at coupling ratio κ = εy/εx, never returns a
    horizontal emittance below ε₀/(1+κ), with ε₀ the uncoupled natural emittance. When the
    horizontal damping time τx exceeds the vertical τy (horizontal damping partition Jx < 1) and IBS
    is weak, the radiation–IBS balance the program solves has its root below that bound, at
    ε₀/(1+κτx/τy) without IBS; the program returns the bound and reports "Emittance converged". For
    a 3 GeV, 288.0 m ring with Jx = 0.9634, κ = 0.1 and 1 nC, it returns 48.96 nm versus 48.79 nm,
    0.3408% high; the transverse balance misses zero by 0.3408% of its excitation term. This
    lower-bound obstruction is absent for weak IBS when τx ≤ τy.

    The script builds a 3 GeV periodic model of 12 FODO cells, with 55.2 m total length and sector
    bends totaling π, writes its Twiss and radiation integrals with elegant, and runs ibsEmittance for the equilibrium at κ = 0.1, at 10⁻¹² nC (no IBS)
    and at 1 nC. It compares the returned εx with the root of the program's own radiation balance,
    ε₀/(1+κτx/τy), and evaluates that balance, IBS terms included, at the returned emittances. The
    control is the same model with a defocusing gradient in the bends, which makes τx < τy; there the
    program returns the root.
    """

    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import pysdds
    from _elegant import TIMEOUT, RunFailure, verdict

    KAPPA = 0.1  # coupling ratio epsilon_y/epsilon_x given to -coupling
    CELLS = 12


    def lattice(bend_k1):
        # FODO ring of 12 cells, 24 sector bends of 1 m (circumference 12 x 4.6 m = 55.2 m). With
        # bend_k1 = 0 the bends have no gradient and the damping partition is Jx = 1 - D < 1, so
        # tau_x > tau_y; a defocusing gradient in the bends raises Jx above 1 (tau_x < tau_y).
        return (
            "D: DRIF,L=0.5\n"
            "QF: QUAD,L=0.3,K1=1.2\n"
            "QD: QUAD,L=0.3,K1=-1.2\n"
            f"B: SBEN,L=1,ANGLE={3.141592653589793 / (2 * CELLS)!r},K1={bend_k1!r}\n"
            "CELL: LINE=(QF,D,B,D,QD,D,B,D)\n"
            f"RING: LINE=({CELLS}*CELL)\n"
        )


    TWISS = (
        '&run_setup lattice="ring.lte",use_beamline=RING,p_central_mev=3000,default_order=2 &end\n'
        '&twiss_output filename="ring.twi",matched=1,radiation_integrals=1 &end\n'
        "&run_control &end\n"
        "&bunched_beam &end\n"
        "&track &end\n"
    )


    def equilibrium(bend_k1, charge_nc):
        """Parameters of ibsEmittance's equilibrium output for the ring (no -integrate, no -growthRatesOnly)."""
        if "RPN_DEFNS" not in os.environ:
            sys.exit("Set RPN_DEFNS to the defns.rpn file of SDDS (see the elegant installation).")
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="elegant-repro-") as tmp:
            work = Path(tmp)
            (work / "ring.lte").write_text(lattice(bend_k1))
            (work / "twiss.ele").write_text(TWISS)
            for argv in (
                ["elegant", "twiss.ele"],
                # -length is the rms bunch length in mm; no RF is needed for a fixed length.
                [
                    "ibsEmittance",
                    "ring.twi",
                    "ring.ibs",
                    f"-charge={charge_nc!r}",
                    f"-coupling={KAPPA!r}",
                    "-length=5",
                ],
            ):
                done = subprocess.run(
                    argv,
                    cwd=work,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT,
                    check=False,
                )
                if done.returncode:
                    raise RunFailure(argv[0], done.returncode, done.stdout + done.stderr)
            out = pysdds.read(work / "ring.ibs")
        return {p.name: p.data[0] for p in out.parameters}


    def case(label, bend_k1, charge_nc):
        p = equilibrium(bend_k1, charge_nc)
        ex0, taux, tauy = float(p["emitx0"]), float(p["taux"]), float(p["tauy"])
        ex, ey = float(p["emitx"]), float(p["emity"])
        # Equilibrium balance solved by IBSequations (ibsEmittance.c:1085-1090): summing the x and y
        # action equations with epsilon_y = kappa epsilon_x,
        #   0 = -2 ex/taux - 2 ey/tauy + 2 ex0/taux + gx ex + gy ey.
        # Without IBS (gx = gy = 0) its root is ex = ex0/(1 + kappa taux/tauy), the value the program
        # itself reports as emitxInput (ibsEmittance.c:747) and the manual gives for sddsbrightness.
        floor = ex0 / (1 + KAPPA)  # the simplex lower limit, ibsEmittance.c:806,810
        # With IBS, the root of the same balance with the rates gx, gy the program returns; at these
        # charges gx/(2/taux) < 1e-7, so holding the rates at their returned values is exact to that order.
        gx, gy = float(p["xGrowthRate"]), float(p["yGrowthRate"])
        root = (2 * ex0 / taux) / (2 / taux + 2 * KAPPA / tauy - gx - KAPPA * gy)
        # Balance at the returned point, normalized by the excitation term 2 ex0/taux as the program's
        # objective is.
        residual = (-2 * ex / taux - 2 * ey / tauy + 2 * ex0 / taux + gx * ex + gy * ey) / (
            2 * ex0 / taux
        )
        if gx * taux / 2 > 1e-7:
            raise SystemExit(f"{label}: IBS is not weak; the fixed-rate root is not exact")
        print(f"{label}: taux/tauy={taux / tauy:.6f} gx={gx:.4g}/s 2/taux={2 / taux:.6g}/s")
        print(f"  emitx observed={ex:.10g} expected={root:.10g} (relative {ex / root - 1:+.4%}), "
              f"floor ex0/(1+kappa)={floor:.10g}")  # fmt: skip
        print(f"  balance residual={residual:+.4%}; emitxInput={float(p['emitxInput']):.10g}; "
              f"Convergence=\"{p['Convergence']}\"")  # fmt: skip
        return ex, root, floor, residual, str(p["Convergence"])


    # Control: bends with K1 = -0.02 m^-2 give tau_x < tau_y; the root lies above the floor.
    ex, root, floor, residual, _ = case("control, bend K1=-0.02, 1e-12 nC", -0.02, 1e-12)
    if not root > floor or abs(ex / root - 1) > 1e-9 or abs(residual) > 1e-9:
        raise SystemExit("control: the equilibrium misses the radiation balance")

    defect = True
    for charge in (1e-12, 1.0):
        ex, root, floor, residual, status = case(f"sector bends, {charge:g} nC", 0.0, charge)
        clamped = abs(ex / floor - 1) < 1e-12 and ex / root - 1 > 1e-3
        defect &= clamped and abs(residual) > 1e-3 and status == "Emittance converged"
    print(
        "comparison: control rtol=1e-9; floor equality rtol=1e-12; defect needs >0.1% excess and residual"
    )
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
| Confirmed (equilibria below the ε₀/(1+κ) bound, that is weak IBS with τx > τy) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

