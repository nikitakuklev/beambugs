---
title: "TRK-31: A custom particle keeps the anomalous magnetic moment of the previously selected species"
hide:
  - toc
---

# TRK-31 — A custom particle keeps the anomalous magnetic moment of the previously selected species

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`process_particle_command`, elegant.c:3264–3309](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3264-L3309)<br>[custom branch, elegant.c:3292–3298](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3292-L3298)<br>[anomalous moments, elegant.c:3249–3250](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3249-L3250)<br>[initial value, track_data.c:27](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L27)<br>[spinUpdate.c:205](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/spinUpdate.c#L205)<br>[namelist, elegant.nl:66–70](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.nl#L66-L70)<br>[change_particle, elegant.tex:1497–1517](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L1497-L1517) |

Selecting a custom particle with change_particle sets its mass and charge but not its anomalous magnetic moment a, which keeps the value of the species selected before, or the electron's in a fresh run. Spin tracking of the same custom particle therefore depends on earlier change_particle commands. For a custom particle with the electron's mass and charge at momentum 200 mₑc, an on-axis spin (1, 0, 0) through a 0.65 m SOLE with KS = 0.9000 m⁻¹ ends at sᵧ = −0.5528 when custom is the only selection, but at −0.9980 after a proton selection, the Thomas–BMT result for the proton's a = 1.793. No input sets a for a custom particle.

??? example "Reproducer: TRK-31.py"

    [Download TRK-31.py](TRK-31.py)

    ```python
    """TRK-31: A custom particle keeps the anomalous magnetic moment of the previously selected species

    Selecting a custom particle with change_particle sets its mass and charge but not its anomalous
    magnetic moment a, which keeps the value of the species selected before, or the electron's in a
    fresh run. Spin tracking of the same custom particle therefore depends on earlier change_particle
    commands. For a custom particle with the electron's mass and charge at momentum 200 mₑc, an on-axis
    spin (1, 0, 0) through a 0.65 m SOLE with KS = 0.9000 m⁻¹ ends at sᵧ = −0.5528 when custom is the
    only selection, but at −0.9980 after a proton selection, the Thomas–BMT result for the proton's a =
    1.793. No input sets a for a custom particle.

    The script runs the same custom particle, lattice and beam after four selection histories (none,
    proton, muon, and proton then electron) and requires identical exit spins, with the custom-only run
    as the reference. The reference must equal the analytic precession with the electron moment and the
    proton-electron-custom history must restore it (controls); each departure is matched to the
    analytic precession with the earlier species' moment.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _sole import A_ELECTRON, A_MUON, A_PROTON, axial_rotation, compare, on_axis_angle, track

    LENGTH, P0, KS = 0.65, 200.0, 0.9
    ELEMENT = f"SOLE,L={LENGTH!r},KS={KS!r}"
    CUSTOM = "&change_particle name=custom, mass_ratio=1, charge_ratio=1 &end\n"
    ATOL = 1e-13
    HISTORIES = {  # label: change_particle commands before the final custom selection
        "custom only": [],
        "proton, then custom": ["proton"],
        "muon, then custom": ["muon"],
        "proton, electron, then custom": ["proton", "electron"],
    }


    def final_spin(mode, earlier):
        before = "".join(f"&change_particle name={name} &end\n" for name in earlier) + CUSTOM
        out, _ = track(mode, ELEMENT, P0, {"spx": 1.0}, before=before)
        return np.array([out["spx"], out["spy"], out["spz"]])


    def bmt(anomaly):
        """On-axis Thomas-BMT spin from (1, 0, 0) for anomalous moment a (see _sole.on_axis_angle).

        The angle -KS L (1 + a) depends on the species only through a, because KS is given.
        """
        return axial_rotation(on_axis_angle(KS, LENGTH, anomaly))


    defect = False
    for mode in modes():
        spins = {label: final_spin(mode, earlier) for label, earlier in HISTORIES.items()}
        baseline = spins["custom only"]
        # The final inputs (species definition, lattice, beam) are identical in every history, so the
        # spin must be the same; the custom-only run is the reference.
        if not compare(
            f"{mode} custom only (reference) vs BMT with the electron a",
            baseline,
            bmt(A_ELECTRON),
            atol=ATOL,
        ):
            sys.exit("The custom-only reference is not the electron-moment rotation: unexpected.")
        if not compare(
            f"{mode} proton, electron, then custom (control)",
            spins["proton, electron, then custom"],
            baseline,
            atol=ATOL,
        ):
            sys.exit("Reselecting the electron did not restore the reference: the control fails.")
        for label, anomaly in (("proton, then custom", A_PROTON), ("muon, then custom", A_MUON)):
            defect |= not compare(f"{mode} {label}", spins[label], baseline, atol=ATOL)
            # The departure is the rotation with the earlier species' anomalous moment.
            species = label.split(",")[0]
            if not compare(
                f"{mode} {label} vs BMT with the {species} a={anomaly:g}",
                spins[label],
                bmt(anomaly),
                atol=ATOL,
            ):
                sys.exit(f"{label}: the spin is not the rotation with the {species} moment")
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

??? example "Helper: _sole.py"

    [Download _sole.py](_sole.py)

    ```python
    """Shared runner for the SOLE spin-tracking reproducers: one solenoid, one pass, spin tracking on.

    `track` sends identical copies of one particle (so that each MPI worker tracks some) through a
    lattice holding a single element and returns that particle's final coordinates and spin. Spin
    components (spx, spy, spz) are in the fixed straight element frame; p is βγ. Constants are those of
    SDDS `constants.h` and of the species table of Elegant's `change_particle` command, so that a
    different choice of constants cannot pass for an algorithm defect.
    """

    from __future__ import annotations

    import sys

    import numpy as np
    from _elegant import particles, run

    C, QE, ME = 299792458.0, 1.60217733e-19, 9.1093897e-31  # SDDS constants.h
    A_ELECTRON = 0.00115965218128  # electron and positron anomalous magnetic moment in Elegant
    A_PROTON = 1.792847
    A_MUON = 1.165921e-3
    COPIES = 6
    COORDINATES = ("x", "xp", "y", "yp", "t", "p")
    SPIN = ("spx", "spy", "spz")


    def track(
        mode: str,
        element: str,
        p0: float,
        start: dict[str, float],
        before: str = "",
        setup: str = "",
        after_setup: str = "",
        outputs: tuple[str, ...] = ("out.sdds",),
    ) -> tuple[dict[str, float], dict]:
        """Track COPIES copies of one particle through `S: <element>` and return its exit state.

        start: initial x, xp, y, yp, spx, spy, spz (missing values are zero; p is p0). before: commands
        ahead of run_setup (change_particle); setup: extra run_setup fields; after_setup: commands
        between run_control and the beam (matrix_output). Returns ({name: value}, run result).
        """
        lattice = f"S: {element}\nBL: LINE=(S)\n"
        commands = (
            before
            + f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, output=out.sdds, "
            f"spin_tracking=1, default_order=2{setup} &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            + after_setup
            + "&sdds_beam input=beam.sdds &end\n&track &end\n"
        )
        beam = {n: np.full(COPIES, float(start.get(n, 0.0))) for n in ("x", "xp", "y", "yp", *SPIN)}
        beam["p"] = np.full(COPIES, p0)
        result = run(
            {"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam}, outputs=outputs
        )
        out = particles(result["out.sdds"])
        if len(out["p"]) != COPIES:
            sys.exit(f"{mode}: {COPIES - len(out['p'])} particles lost")
        state = np.column_stack([out[n] for n in (*COORDINATES, *SPIN)])
        if not np.all(np.isfinite(state)):
            sys.exit(f"{mode}: nonfinite orbit or spin")
        if np.ptp(state, axis=0).max() != 0:
            sys.exit(f"{mode}: identical particles ended in different states")
        if abs(np.linalg.norm(state[0, 6:]) - 1) > 1e-14:
            sys.exit(f"{mode}: the spin norm is not 1")
        return {n: float(v) for n, v in zip((*COORDINATES, *SPIN), state[0], strict=True)}, result


    def axial_rotation(theta: float, spin=(1.0, 0.0, 0.0)) -> np.ndarray:
        """Spin rotated by the angle theta about +z (counterclockwise seen from +z)."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([c * spin[0] - s * spin[1], s * spin[0] + c * spin[1], spin[2]])


    def on_axis_angle(ks: float, length: float, anomaly: float) -> float:
        """Spin rotation angle about +z on the axis of a solenoid of strength KS, on momentum.

        KS = q B/(m c p0) for the signed physical charge q (Elegant's "-Bs/(B*Rho)" with its legacy
        charge sign). With the velocity along B, magnetic Thomas-BMT reduces to
        dS/dt = -(q/m)(1 + a) B/gamma z-hat x S; dt = gamma dz/(beta gamma c) turns it into
        dtheta/dz = -q (1 + a) B/(m c p) = -KS (1 + a).
        """
        return -ks * length * (1 + anomaly)


    def fmt(values) -> str:
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def compare(label: str, observed, expected, atol: float, rtol: float = 0.0) -> bool:
        """Print one case; return True when observed and expected agree within atol + rtol*|expected|."""
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bool(np.allclose(observed, expected, rtol=rtol, atol=atol))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

