---
title: "OPT-03: CCBEND radiation totals omit repeated periods"
hide:
  - toc
---

# OPT-03 — CCBEND radiation totals omit repeated periods

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P1 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`incrementRadIntegrals`, CCBEND branch, src/twiss.cc:3091–3102](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3091-L3102)<br>[the generic bend branch, src/twiss.cc:3331–3335](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3331-L3335)<br>[the common per-element output, src/twiss.cc:3346–3351](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3346-L3351)<br>[`radiation_integrals` option, manual, doc/elegant.tex:8092–8095](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8092-L8095) |

With `twiss_output`, `radiation_integrals=1` and `n_periods` greater than 1, Elegant adds, for CCBEND (canonical combined-function bend) elements, the integrals I₁ to I₅ of one period to the totals without multiplying by `n_periods`, although the per-element contributions are multiplied. For a cell of two 1 m bends with curvature 0.1 m⁻¹ at 6000 mₑc, I₂ must be 0.02000 m⁻¹ for one period and 0.04000 m⁻¹ for two, but Elegant writes 0.02000 m⁻¹ for both. The energy loss per turn then stays at 24.88 keV instead of doubling to 49.77 keV, and the damping times double instead of staying unchanged. The same cell with SBEN bends is correct.

??? example "Reproducer: OPT-03.py"

    [Download OPT-03.py](OPT-03.py)

    ```python
    """OPT-03: CCBEND radiation totals omit repeated periods

    With `twiss_output`, `radiation_integrals=1` and `n_periods` greater than 1, Elegant adds, for
    CCBEND (canonical combined-function bend) elements, the integrals I₁ to I₅ of one period to the
    totals without multiplying by `n_periods`, although the per-element contributions are multiplied.
    For a cell of two 1 m bends with curvature 0.1 m⁻¹ at 6000 mₑc, I₂ must be 0.02000 m⁻¹ for one
    period and 0.04000 m⁻¹ for two, but Elegant writes 0.02000 m⁻¹ for both. The energy loss per turn
    then stays at 24.88 keV instead of doubling to 49.77 keV, and the damping times double instead of
    staying unchanged. The same cell with SBEN bends is correct.

    The script runs the same three-meter cell (two quadrupoles and two 1 m bends, 6000 mₑc) with
    `n_periods` = 1 and 2, in serial Elegant (and with `--mpi` in 3-rank Pelegant). For one period it
    checks I₂ = Σ L/ρ², the energy loss and the damping times against their closed-form definitions;
    for two periods the totals and the energy loss must double and the damping times be unchanged.
    The same cell with SBEN bends is the control, and the sum of the written per-element I₂
    contributions is printed.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    # SDDS constants.h, as Elegant uses them.
    C, RE, ELECTRON_MEV = 299792458.0, 2.81794092e-15, 0.51099906
    P_CENTRAL = 6000.0  # m_e c
    BEND_L, BEND_ANGLE = 1.0, 0.1  # two bends per cell; curvature h = angle/L = 0.1 1/m
    CELL_L = 0.5 + 1.0 + 0.5 + 1.0  # m: QF, bend, QD, bend
    RTOL = 1e-6  # Elegant's rounded constants and relativistic energy

    NAMES = ("I1", "I2", "I3", "I4", "I5", "U0", "taux", "tauy", "taudelta")


    def lattice(bend):
        element = {
            "CCBEND": f"D: CCBEND,L={BEND_L!r},ANGLE={BEND_ANGLE!r},N_SLICES=16",
            "SBEN": f"D: SBEN,L={BEND_L!r},ANGLE={BEND_ANGLE!r}",
        }[bend]
        return (
            "QF: QUAD,L=0.5,K1=0.6,TILT=0\nQD: QUAD,L=0.5,K1=-0.54,TILT=0\n"
            f"{element}\nBL: LINE=(QF,D,QD,D)\n"
        )


    def totals(mode, bend, periods):
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P_CENTRAL!r}, "
            "default_order=2 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            f"&twiss_output filename=twiss.sdds, radiation_integrals=1, n_periods={periods} &end\n"
            "&bunched_beam n_particles_per_bunch=12 &end\n"
            "&track &end\n"
        )
        twiss = run({"lattice.lte": lattice(bend), "run.ele": commands}, mode, outputs=("twiss.sdds",))[
            "twiss.sdds"
        ]
        values = np.array([twiss[name][0] for name in NAMES])
        return values, np.sum(twiss[0, "dI2"])


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.atleast_1d(values)) + "]"


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected disagree."""
        bad = not np.allclose(observed, expected, rtol=RTOL, atol=0)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    def single_period_oracle(i4_over_i2):
        """Closed-form I2, U0 and damping times of one cell (E in MeV, times in s).

        I2 = sum of L/rho^2 = 2 L h^2 for two uniform bends. U0 = C_gamma E^4 I2/(2 pi) with
        C_gamma = 4 pi r_e/(3 (m c^2)^3). The damping times are tau_u = 2 E T0/(J_u U0) with
        T0 = cell length/c; Jx = 1 - I4/I2, Jy = 1, Jdelta = 2 + I4/I2.
        """
        i2 = 2 * BEND_L * (BEND_ANGLE / BEND_L) ** 2
        energy = np.sqrt(1 + P_CENTRAL**2) * ELECTRON_MEV
        c_gamma = 4 * np.pi / 3 * RE / ELECTRON_MEV**3  # m/MeV^3
        u0 = c_gamma * energy**4 * i2 / (2 * np.pi)
        t0 = CELL_L / C
        return i2, u0, [2 * energy * t0 / (j * u0) for j in (1 - i4_over_i2, 1, 2 + i4_over_i2)]


    defect = False
    print(f"columns: {', '.join(NAMES)}; energies in MeV, times in s")
    for mode in modes():
        for bend in ("SBEN", "CCBEND"):
            one, sum_one = totals(mode, bend, 1)
            two, sum_two = totals(mode, bend, 2)
            i2, u0, taus = single_period_oracle(one[3] / one[1])
            # One period (the control for the oracle): I2, U0 and the damping times follow from the
            # curvature, the energy and the cell length.
            if compare(
                f"{mode} {bend} 1 period (I2, U0, taux, tauy, taudelta)",
                one[[1, 5, 6, 7, 8]],
                [i2, u0, *taus],
            ):
                sys.exit("The one-period totals disagree with the closed form: comparison not valid.")
            # Two identical periods: every integral and U0 add, T0 doubles, so each damping time
            # 2 E T0/(J U0) is unchanged.
            expected = np.concatenate([2 * one[:6], one[6:]])
            bad = compare(f"{mode} {bend} 2 periods", two, expected)
            print(
                f"{mode} {bend} sum of dI2 over elements: 1 period={sum_one:.10g} 2 periods={sum_two:.10g}"
            )
            if bend == "SBEN" and bad:
                sys.exit("The SBEN control does not scale with n_periods: comparison not valid.")
            defect |= bad
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
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

