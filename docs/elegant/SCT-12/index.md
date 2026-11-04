---
title: "SCT-12: Thin forced-MCS MATTER omits the flight distance from the transverse angles"
hide:
  - toc
---

# SCT-12 — Thin forced-MCS MATTER omits the flight distance from the transverse angles

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [forced thin multiple scattering, matter.c:204–213](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L204-L213)<br>[coordinate update, matter.c:283](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L283)<br>[standard multiple scattering, matter.c:203](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L203)<br>[forced multiple scattering, matter.tex:40–46](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/matter.tex#L40-L46)<br>[coordinate definition, elegant.tex:373–376](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L373-L376)<br>[MATTER parameters, track_data.c:2137–2138](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2137-L2138) |

With FORCE_MCS=1 in a MATTER element thinner than 0.001 radiation lengths and ENERGY_DECAY=0, Elegant advances the longitudinal coordinate by the axial length L alone. At speed βc, the flight time must be at least √(L² + Δx² + Δy²)/(βc), where Δx and Δy are transverse displacements between the endpoints. In dilute helium with L = 0.4 m, electrons at momentum 2.000 mₑc with slopes (0.001, 0.003) violate this bound by 7.459×10⁻¹⁵ s; at 2.400 mₑc with slopes (−0.02, 0.015), the deficit is 4.516×10⁻¹³ s. Arrival-time calculations through thin targets or residual gas with forced scattering are affected.

??? example "Reproducer: SCT-12.py"

    [Download SCT-12.py](SCT-12.py)

    ```python
    """SCT-12: Thin forced-MCS MATTER omits the flight distance from the transverse angles

    With FORCE_MCS=1 in a MATTER element thinner than 0.001 radiation lengths and ENERGY_DECAY=0, Elegant
    advances the longitudinal coordinate by the axial length L alone. At speed βc, the flight time must
    be at least √(L² + Δx² + Δy²)/(βc), where Δx and Δy are transverse displacements between the
    endpoints. In dilute helium with L = 0.4 m, electrons at momentum 2.000 mₑc with slopes (0.001,
    0.003) violate this bound by 7.459×10⁻¹⁵ s; at 2.400 mₑc with slopes (−0.02, 0.015), the deficit is
    4.516×10⁻¹³ s. Arrival-time calculations through thin targets or residual gas with forced scattering
    are affected.

    The script sends 24 electrons through helium of length 0.4 m, with slopes (0, 0), (0.001, 0.003) and
    (−0.02, 0.015) and momenta 0.8, 1.0 and 1.2 times 2, 20 or 300 mₑc, and compares each particle's
    arrival time with the straight-line bound for its observed entrance and exit points. Forced Gaussian
    scattering in a dilute density, as residual gas and at a finite density is compared with three
    controls that must meet the bound exactly: single scattering, an element that starts on a later
    pass, and a zero-length scattering impulse.
    """

    import math
    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h and the MATTER source, as Elegant uses them.
    C, KB, MP = 299792458.0, 1.380658e-23, 1.6726485e-27
    LENGTH, COUNT = 0.4, 24
    ATOL = 2e-24  # s; arrival times are about 1e-9 s, so this is about 1e-15 relative
    XP = np.tile([0.0, 0.001, -0.02], COUNT // 3)
    YP = np.tile([0.0, 0.003, 0.015], COUNT // 3)
    GAS = "PRESSURE=1e-9,TEMPERATURE=300,MULTIPLICITY=1"
    MODES = (  # name, element parameters, length; the first three are the faulty path
        ("forced", f"L={LENGTH},RHO=1e-28,FORCE_MCS=1", LENGTH),
        ("gas", f"L={LENGTH},{GAS},FORCE_MCS=1", LENGTH),
        ("finite", f"L={LENGTH},RHO=1e-6,FORCE_MCS=1", LENGTH),
        ("individual", f"L={LENGTH},RHO=1e-28,FORCE_MCS=0", LENGTH),
        ("inactive", f"L={LENGTH},RHO=1e-28,FORCE_MCS=1,START_PASS=1", LENGTH),
        ("impulse", f"L=0,LEFFECTIVE={LENGTH},RHO=1e-28,FORCE_MCS=1", 0.0),
    )


    def track(mode, p0, element):
        """Columns of the 24 electrons after one MATTER element of helium (Z=2, A=4), by particleID."""
        lattice = f"M: MATTER,Z=2,A=4,ENERGY_DECAY=0,{element}\nBL: LINE=(M)\n"
        commands = (
            "&change_particle name=electron &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, output=out.sdds, "
            "default_order=2, random_number_seed=94821 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        columns = {
            "x": np.zeros(COUNT),
            "xp": XP,
            "y": np.zeros(COUNT),
            "yp": YP,
            "t": np.zeros(COUNT),
            "p": p0 * np.tile([0.8, 1.0, 1.2], COUNT // 3),
            "particleID": np.arange(1, COUNT + 1),
        }
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})["out.sdds"]
        )
        # particles() rejects nonfinite x, y, xp, yp, t and p; NaNs would make every comparison false.
        if not np.array_equal(out["particleID"], columns["particleID"]):
            sys.exit(f"{mode}: the output particle IDs differ from the input")
        if not np.allclose(out["p"], columns["p"], rtol=0, atol=2e-13):
            sys.exit(f"{mode}: the momentum changed")
        return columns, out


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.ravel(values)) + "]"


    defect = False
    print("endpoint bound minus arrival time in s, first three electrons (expected: not above 0)")
    for mode in modes():
        for p0 in (2.0, 20.0, 300.0):
            kicks = {}
            for name, element, length in MODES:
                columns, out = track(mode, p0, element)
                beta = columns["p"] / np.hypot(columns["p"], 1)
                # Whatever path a particle takes between its entrance (x = y = 0) and its exit point at
                # unchanged momentum, it needs at least the straight-chord time: t >= sqrt(L^2 + x^2 + y^2)/(beta c).
                bound = np.sqrt(length**2 + out["x"] ** 2 + out["y"] ** 2) / (beta * C)
                deficit = bound - out["t"]
                control = name in ("individual", "inactive", "impulse")
                bad = bool(np.max(deficit) > ATOL)
                # A scattered path can be longer than its chord, so the faulty cases only need deficit <= 0;
                # the controls follow the chord exactly.
                expected = f"={fmt(np.zeros(3))}" if control else " <= 0"
                print(
                    f"{mode} p0={p0:g} {name}: observed={fmt(deficit[:3])} expected{expected} "
                    f"({np.count_nonzero(deficit > ATOL)} of {COUNT} electrons violating the arrival-time lower bound)"
                )
                if control and (bad or np.max(-deficit) > ATOL):
                    sys.exit(
                        f"{mode}: the {name} control does not meet the bound: the comparison is not valid"
                    )
                if name in ("gas", "finite"):
                    kicks[name] = np.stack([out["xp"][0::3] - XP[0::3], out["yp"][0::3] - YP[0::3]])
                defect |= bad
            # Control: a gas of density rho and a material of density rho draw the same Gaussian
            # deflections, which scale as sqrt(rho) at fixed length.
            rho_gas = 1e-9 / (KB * 300) * MP * 4
            if not np.allclose(
                kicks["finite"], kicks["gas"] * math.sqrt(1e-6 / rho_gas), rtol=2e-14, atol=1e-18
            ):
                sys.exit(f"{mode}: the gas and density deflections do not scale as sqrt(density)")
    print(f"comparison: atol={ATOL:g} s")
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

