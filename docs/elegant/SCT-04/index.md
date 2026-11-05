---
title: "SCT-04: MATTER collision slices omit incoming transverse drift"
hide:
  - toc
---

# SCT-04 — MATTER collision slices omit incoming transverse drift

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [collision slice, matter.c:232–240](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L232-L240)<br>[no-collision slice, matter.c:241–245](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L241-L245)<br>[multiple-scattering branch, matter.c:197–201](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L197-L201)<br>[slice model, matter.tex:21–37](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/matter.tex#L21-L37)<br>[MATTER parameters, track_data.c:2137–2138](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2137-L2138) |

When MATTER uses single scattering in a finite-length target, each slice containing a collision omits the transverse drift from the slope entering that slice. Correct transport must include that slope times the slice length, in addition to the displacement from the scattering kick. In a one-slice, 0.01 m helium target, shifting the incoming horizontal and vertical slopes by 0.002 and −0.003 leaves all scattered electrons' exit positions unchanged, instead of shifting them by 20.00 µm and −30.00 µm; unscattered electrons drift correctly. Earlier kicks can also expose the error in later slices.

??? example "Reproducer: SCT-04.py"

    [Download SCT-04.py](SCT-04.py)

    ```python
    """SCT-04: MATTER collision slices omit incoming transverse drift

    When MATTER uses single scattering in a finite-length target, each slice containing a collision
    omits the transverse drift from the slope entering that slice. Correct transport must include that
    slope times the slice length, in addition to the displacement from the scattering kick. In a
    one-slice, 0.01 m helium target, shifting the incoming horizontal and vertical slopes by 0.002 and
    −0.003 leaves all scattered electrons' exit positions unchanged, instead of shifting them by 20.00
    µm and −30.00 µm; unscattered electrons drift correctly. Earlier kicks can also expose the error in later slices.

    The script tracks 2400 electrons through the target twice with the same random seed, once on axis
    and once with the incoming slopes shifted, so that both runs draw identical collisions and kicks. It
    compares the shift of the exit positions with the geometric requirement L·(a, b), separately for the
    electrons that scattered and for those that did not; the unscattered electrons are the control and
    must drift correctly.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _matter import LENGTH, beam, track

    COUNT, SEED = 2400, 56731
    SHIFT = np.array([0.002, -0.003])  # incoming slope shift (xp, yp)
    # One slice (PLIMIT above the mean 0.3 collisions per electron); no energy loss.
    ELEMENT = f"L={LENGTH!r},PLIMIT=0.5,ENERGY_DECAY=0,NUCLEAR_BREMSSTRAHLUNG=0,ELECTRON_RECOIL=0"
    ATOL = 1e-18  # m; positions are about 1e-5 m, so this is about 1e-13 relative


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.ravel(values)) + "]"


    def compare(label, observed, expected):
        bad = not np.allclose(observed, expected, rtol=0, atol=ATOL)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    defect = False
    print(
        "exit-position shift in m, as (x min, x max, y min, y max) over electrons, for shifted slopes"
    )
    for mode in modes():
        zero = track(mode, ELEMENT, SEED, beam(COUNT))
        sloped = track(mode, ELEMENT, SEED, beam(COUNT, *SHIFT))
        # The same seed gives the same collisions and kicks in both runs, so the outgoing slopes differ
        # by exactly the incoming shift. Otherwise the electrons cannot be paired.
        kick = np.column_stack([sloped["xp"] - SHIFT[0], sloped["yp"] - SHIFT[1]])
        if not np.allclose(kick, np.column_stack([zero["xp"], zero["yp"]]), rtol=0, atol=1e-17):
            sys.exit(f"{mode}: the paired runs drew different kicks")
        scattered = (zero["xp"] != 0) | (zero["yp"] != 0)
        if not 0 < scattered.sum() < COUNT:
            sys.exit(f"{mode}: need both scattered and unscattered electrons")
        shift = np.column_stack([sloped["x"] - zero["x"], sloped["y"] - zero["y"]])
        # A kick at depth z of a slice of length L gives x_out = x_in + z a + (L - z)(a + k)
        # = x_in + L a + (L - z) k, so the incoming slope a adds L a whatever z and k are.
        expected = LENGTH * SHIFT
        for name, rows in (("unscattered control", ~scattered), ("scattered", scattered)):
            label = f"{mode} {name} ({rows.sum()} electrons)"
            observed = np.ravel([shift[rows].min(axis=0), shift[rows].max(axis=0)], order="F")
            bad = compare(label, observed, np.repeat(expected, 2))
            if name == "unscattered control" and bad:
                sys.exit(f"{mode}: the unscattered control disagrees: the comparison is not valid")
            defect |= bad
    print(f"comparison: atol={ATOL:g} m")
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

??? example "Helper: _matter.py"

    [Download _matter.py](_matter.py)

    ```python
    """Shared setup for the MATTER reproducers: a thin helium target tracked by Elegant's single-scattering model.

    One MATTER element of length 0.01 m filled with helium (Z = 2, A = 4, density 0.02813 kg/m^3),
    traversed by electrons of momentum 1000 m_e c. PLIMIT bounds the scattering probability of one
    section, and the element is divided into the number of sections that keeps it below the bound.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import particles, run

    P0 = 1000.0  # reference and beam momentum, units of m_e c
    LENGTH = 0.01  # m
    RHO, Z, A = 0.0281327461321535, 2, 4  # kg/m^3 (dilute helium), atomic number, mass number

    # SDDS constants.h and the MATTER source, as Elegant uses them.
    RE, AMU = 2.81794092e-15, 1.6605e-27  # classical electron radius (m), atomic mass unit (kg)
    BS_Y0 = 1e-8  # lower cutoff of the recoil and bremsstrahlung energy-transfer samplers


    # Mean nuclear-bremsstrahlung events per electron: 4 L/(3 X0) [-ln y0 - (1 - y0) + 3/8 (1 - y0^2)],
    # with the helium radiation length X0 = 94.32 g/cm^2 (PDG atomic and nuclear properties of helium
    # gas) divided by the density, converted from cm to m. Bremsstrahlung is rare in this thin target.
    X0 = 94.32 / (RHO * 1e-3) * 1e-2  # m
    BREMS_PER_ELECTRON = 4 * LENGTH / (3 * X0) * (-np.log(BS_Y0) - (1 - BS_Y0) + 3 / 8 * (1 - BS_Y0**2))


    def track(mode, element, seed, columns):
        """Track `columns` (beam arrays with p in m_e c) through `MATTER` + element text; return columns.

        `element` holds the MATTER parameters after the fixed Z, A and RHO. Rows are returned ordered by
        particleID, which the beam must carry (MPI output order varies).
        """
        lattice = f"M: MATTER,Z={Z},A={A},RHO={RHO!r},{element}\nBL: LINE=(M)\n"
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
            f"output=out.sdds, random_number_seed={seed} &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})
        # particles() rejects nonfinite coordinates; the output must also hold exactly the input particles.
        out = particles(result["out.sdds"])
        if len(out["p"]) != len(columns["p"]):
            raise SystemExit(f"{mode}: {len(columns['p']) - len(out['p'])} particles lost")
        if not np.array_equal(out["particleID"], np.sort(columns["particleID"])):
            raise SystemExit(f"{mode}: the output particleIDs differ from the input ones")
        return out


    def beam(count, xp=0.0, yp=0.0):
        """`count` electrons on axis at momentum P0 with the given slopes, numbered from 1."""
        return {
            "x": np.zeros(count),
            "xp": np.full(count, xp),
            "y": np.zeros(count),
            "yp": np.zeros(count) + yp,
            "t": np.zeros(count),
            "p": np.full(count, P0),
            "particleID": np.arange(1, count + 1),
        }


    def gamma(p):
        """Lorentz factor of momentum p in m_e c."""
        return np.sqrt(1 + np.asarray(p) ** 2)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

