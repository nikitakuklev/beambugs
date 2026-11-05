---
title: "OPT-11: MULT pole-tip optics use electron rigidity for other species"
hide:
  - toc
---

# OPT-11 — MULT pole-tip optics use electron rigidity for other species

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`mult_matrix`, rigidity `H = P me c/e` and the pole-tip conversion, compute_matrices.c:3251–3254](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L3251-L3254)<br>[`particleCharge/(particleMass c Po)`, multipole.c:531–535](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L531-L535)<br>[`change_particle` sets the species mass and signed charge, elegant.c:3244–3290](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3244-L3290)<br>[K_nL = n! B_tip L/(rⁿ Bρ), doc/mult.tex:5–15](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/mult.tex#L5-L15)<br>[`BORE` and `BTIPL` parameters, track_data.c:508–509](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L508-L509)<br>[`change_particle`, which must precede `run_setup` and has had limited testing, doc/elegant.tex:1493–1518](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L1493-L1518) |

For `MULT` quadrupoles specified by pole-tip fields, analytic optics use electron rigidity even when `change_particle` selects positrons or protons. For a 0.4 m magnet with bore radius 0.02 m, integrated pole-tip field 2.000×10⁻⁵ T·m and dimensionless momentum p/(mc) = 10, the physical horizontal slope response R₂₁ is −0.05844 m⁻¹ for electrons, +0.05890 m⁻¹ for positrons and +3.195×10⁻⁵ m⁻¹ for protons; the analytic matrix gives −0.05844 m⁻¹ for all three. The positron and proton focusing signs are reversed, and the proton |R₂₁| is 1829 times too large. Direct tracking and explicit integrated strengths agree with the physical quadrupole matrix.

??? example "Reproducer: OPT-11.py"

    [Download OPT-11.py](OPT-11.py)

    ```python
    """OPT-11: MULT pole-tip optics use electron rigidity for other species

    For `MULT` quadrupoles specified by pole-tip fields, analytic optics use electron rigidity even when
    `change_particle` selects positrons or protons. For a 0.4 m magnet with bore radius 0.02 m,
    integrated pole-tip field 2.000×10⁻⁵ T·m and dimensionless momentum p/(mc) = 10, the physical
    horizontal slope response R₂₁ is −0.05844 m⁻¹ for electrons, +0.05890 m⁻¹ for positrons and
    +3.195×10⁻⁵ m⁻¹ for protons; the analytic matrix gives −0.05844 m⁻¹ for all three. The positron and
    proton focusing signs are reversed, and the proton |R₂₁| is 1829 times too large. Direct tracking and
    explicit integrated strengths agree with the physical quadrupole matrix.

    The script builds a 0.4 m `MULT` quadrupole specified by its pole-tip field for an electron, a
    positron and a proton beam in serial Elegant (and with `--mpi` in 3-rank Pelegant). It compares the
    analytic R21 written by `matrix_output` and the R21 measured by tracking small-amplitude particles
    with the transport of the physical field and rigidity. Controls: the same magnet with the physically
    correct integrated strength `KNL` given explicitly, and the convergence of the tracking error with
    the number of slices, show that the matrix code and the integrator are sound.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # Constants of SDDS constants.h and the particle table of elegant.c.
    C, QE, ME, MP = 299792458.0, 1.60217733e-19, 9.1093897e-31, 1.6726485e-27
    LENGTH, BORE, BTIPL, P0 = 0.4, 0.02, 2e-5, 10.0  # m, m, T m, p/(m c) of the species
    CHARGE = {"electron": -QE, "positron": QE, "proton": QE}  # physical charge (C)
    MASS = {"electron": ME, "positron": ME, "proton": MP}  # kg
    X0 = np.array([-6.0, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]) * 1e-7  # amplitudes (m)
    ANALYTIC_ATOL = 1e-12  # explicit-KNL matrix against the oracle: rounding
    TRACK_ATOL = 2e-9  # tracking at 1024 slices: second-order integrator error is about 2e-10


    def expected_knl(species):
        """Physical integrated quadrupole strength in Elegant's sign convention (focusing > 0).

        The pole-tip field gradient is BTIPL/(BORE L) per unit length, so K1 L = (BTIPL/BORE)/(B rho)
        with rigidity B rho = m c p0/|q|, where p0 = p/(m c) is the dimensionless momentum of the
        selected species. Elegant's focusing sign is that of an electron: positive for q < 0.
        """
        rigidity = MASS[species] * C * P0 / abs(CHARGE[species])
        return -np.sign(CHARGE[species]) * (BTIPL / BORE) / rigidity


    def expected_r21(knl):
        """R21 of a hard-edge quadrupole for small amplitudes: x'' = -k x with k = KNL/L."""
        k = knl / LENGTH
        root = np.sqrt(abs(k))
        return -root * np.sin(root * LENGTH) if k > 0 else root * np.sinh(root * LENGTH)


    def case(mode, species, strength, slices):
        """(analytic R21, tracked R21) of the MULT with the given strength specification."""
        lattice = (
            f"E: MULT,L={LENGTH!r},N_SLICES={slices},ORDER=1,MATRIX_ORDER=1,{strength}\nBL: LINE=(E)\n"
        )
        commands = (
            f"&change_particle name={species} &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds,\n"
            " default_order=2 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&matrix_output SDDS_output=matrix.sdds &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        beam = {
            "x": X0,
            "p": np.full(len(X0), P0),
            "particleID": np.arange(1, len(X0) + 1),
        }
        out = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            beams={"beam.sdds": beam},
            outputs=("matrix.sdds", "out.sdds"),
        )
        analytic = out["matrix.sdds"][0, "R21"][-1]
        final = particles(out["out.sdds"])
        # Least-squares response of the output slope to the (symmetric) input amplitudes.
        tracked = np.sum(X0 * final["xp"]) / np.sum(X0**2)
        return analytic, tracked


    def compare(label, observed, expected, atol):
        """Print one case; return True when observed and expected disagree."""
        if not (np.isfinite(observed) and np.isfinite(expected)):
            sys.exit(f"{label}: nonfinite value, the comparison is not valid.")
        bad = abs(observed - expected) > atol
        print(f"{label}: observed=[{observed:.12g}] expected=[{expected:.12g}]")
        return bad


    defect = False
    print("R21 (1/m): analytic matrix from matrix_output, and tracked response at 1024 slices")
    for mode in modes():
        errors = {}
        for species in ("electron", "positron", "proton"):
            knl = expected_knl(species)
            oracle = expected_r21(knl)
            # Control: the same strength given as KNL explicitly. The analytic matrix is right, and
            # tracking agrees with the oracle to the integrator's error.
            analytic, tracked = case(mode, species, f"KNL={float(knl)!r}", 1024)
            if compare(f"{mode} {species} KNL analytic", analytic, oracle, ANALYTIC_ATOL) or compare(
                f"{mode} {species} KNL tracked", tracked, oracle, TRACK_ATOL
            ):
                sys.exit(
                    "The explicit-KNL control disagrees with the oracle: the comparison is not valid."
                )
            # Pole-tip specification: tracking uses the species rigidity, the analytic matrix does not.
            analytic, tracked = case(mode, species, f"BORE={BORE!r},BTIPL={BTIPL!r}", 1024)
            if compare(f"{mode} {species} pole-tip tracked", tracked, oracle, TRACK_ATOL):
                sys.exit("Pole-tip tracking disagrees with the oracle: the comparison is not valid.")
            defect |= compare(f"{mode} {species} pole-tip analytic", analytic, oracle, ANALYTIC_ATOL)
            if species == "positron":
                errors[1024] = abs(tracked - oracle)
        # Control: the tracking error is the second-order integration error, which shrinks fourfold
        # when the slice count doubles; it cannot explain the analytic discrepancy of order 0.1.
        _, coarse = case(mode, "positron", f"BORE={BORE!r},BTIPL={BTIPL!r}", 512)
        ratio = abs(coarse - expected_r21(expected_knl("positron"))) / errors[1024]
        print(
            f"{mode} positron tracking error, 512 slices / 1024 slices: observed={ratio:.6g} expected=4"
        )
        if not 3.9 < ratio < 4.1:
            sys.exit("The tracking error does not converge as the square of the slice count.")
    print(f"comparison: analytic tolerance {ANALYTIC_ATOL:g}, tracking tolerance {TRACK_ATOL:g}")
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
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

