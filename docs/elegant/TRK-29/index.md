---
title: "TRK-29: SOLE tracks with B when KS is also set, while its matrix and Twiss output use KS"
hide:
  - toc
---

# TRK-29 — SOLE tracks with B when KS is also set, while its matrix and Twiss output use KS

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [B branch in tracking, do_tracking.c:2359–2375](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2359-L2375)<br>[`updateSpinForSolenoid` strength choice, spinUpdate.c:194–197](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/spinUpdate.c#L194-L197)<br>[SOLE matrix, compute_matrices.c:1552–1558](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1552-L1558)<br>[SOLE parameters, track_data.c:555–556](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L555-L556) |

A SOLE solenoid given both a nonzero strength KS and a nonzero field B is tracked with the strength that B implies, while its matrix and Twiss output use KS; the parameter description says B is used only if KS is zero. For an electron at momentum 200 mₑc in a 0.65 m solenoid with KS = 0.9000 m⁻¹ and B = −0.02500 T (equivalent to KS = 0.07333 m⁻¹), the matrix output gives the documented transfer element R₁₁ = cos²(KS·L/2) = 0.9169, with L the length, but tracked particles follow R₁₁ = 0.9994. With spin tracking, an on-axis spin turns by 0.04772 rad instead of 0.5857 rad. Either parameter alone gives consistent tracking and optics.

??? example "Reproducer: TRK-29.py"

    [Download TRK-29.py](TRK-29.py)

    ```python
    """TRK-29: SOLE tracks with B when KS is also set, while its matrix and Twiss output use KS

    A SOLE solenoid given both a nonzero strength KS and a nonzero field B is tracked with the strength
    that B implies, while its matrix and Twiss output use KS; the parameter description says B is used
    only if KS is zero. For an electron at momentum 200 mₑc in a 0.65 m solenoid with KS = 0.9000 m⁻¹
    and B = −0.02500 T (equivalent to KS = 0.07333 m⁻¹), the matrix output gives the documented
    transfer element R₁₁ = cos²(KS·L/2) = 0.9169, with L the length, but tracked particles follow R₁₁ =
    0.9994. With spin tracking, an on-axis spin turns by 0.04772 rad instead of 0.5857 rad. Either
    parameter alone gives consistent tracking and optics.

    The script runs a 0.65 m SOLE with KS alone, B alone and both, and for each compares the tracked
    R₁₁ (x_out/x_in of one particle), the R₁₁ of matrix_output and the spin of an on-axis particle with
    the analytic solenoid matrix and Thomas–BMT precession for the strength the parameter description
    selects. The KS-only and B-only runs are the controls; with both set, the matrix output follows KS
    and tracking follows B.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _sole import A_ELECTRON, ME, QE, C, axial_rotation, compare, on_axis_angle, track

    LENGTH, P0 = 0.65, 200.0
    KS, FIELD = 0.9, -0.025  # m^-1 and T
    X0 = 1e-6  # m, initial x of the particle that measures R11
    ATOL = 1e-13
    MATRIX = '&matrix_output SDDS_output="matrix.sdds", SDDS_output_order=1 &end\n'


    def r11(ks):
        """R11 = cos^2(KS L/2) of the solenoid transfer matrix (rotation by KS L/2 times focusing)."""
        return np.cos(ks * LENGTH / 2) ** 2


    def configuration(mode, label, element):
        """Tracked R11, matrix-output R11 and the spin of an on-axis particle with spin (1, 0, 0)."""
        orbit, result = track(
            mode,
            element,
            P0,
            {"x": X0, "spz": 1.0},
            after_setup=MATRIX,
            outputs=("out.sdds", "matrix.sdds"),
        )
        matrix = result["matrix.sdds"]
        names = list(np.asarray(matrix[0, "ElementName"]))
        if names[-1] != "S":
            sys.exit(f"{label}: the matrix output does not end at the solenoid")
        printed = float(np.asarray(matrix[0, "R11"])[-1])
        # The symmetry (x, y) -> (-x, -y) of a solenoid excludes purely transverse second-order terms,
        # and the chromatic ones vanish on momentum, so x_out/x_in is R11 to rounding.
        tracked = orbit["x"] / X0
        spin, _ = track(mode, element, P0, {"spx": 1.0})
        return tracked, printed, np.array([spin["spx"], spin["spy"], spin["spz"]])


    def check(mode, label, element, ks):
        """Compare tracking and matrix output with the strength ks; return per-quantity agreement."""
        tracked, printed, spin = configuration(mode, label, element)
        ok_track = compare(f"{mode} {label}: tracked R11", tracked, r11(ks), atol=ATOL)
        ok_matrix = compare(f"{mode} {label}: matrix-output R11", printed, r11(ks), atol=ATOL)
        ok_spin = compare(
            f"{mode} {label}: spin",
            spin,
            axial_rotation(on_axis_angle(ks, LENGTH, A_ELECTRON)),
            atol=ATOL,
        )
        return ok_track, ok_matrix, ok_spin


    # The strength that B stands for: KS = q B/(m c p0) with the electron's physical charge -e.
    KS_OF_B = -QE * FIELD / (ME * C * P0)
    print(f"KS = {KS} /m; B = {FIELD} T stands for KS = {KS_OF_B:.16g} /m")
    defect = False
    for mode in modes():
        if not all(check(mode, "KS only (control)", f"SOLE,L={LENGTH!r},KS={KS!r}", KS)):
            sys.exit("The KS-only control is inconsistent: it is not valid.")
        if not all(check(mode, "B only (control)", f"SOLE,L={LENGTH!r},B={FIELD!r}", KS_OF_B)):
            sys.exit("The B-only control is inconsistent: it is not valid.")
        # Both given: the parameter description of B reads "field strength (used if KS is zero)", so
        # KS governs; the matrix output follows it, tracking does not.
        ok_track, ok_matrix, ok_spin = check(
            mode, "KS and B, expected per KS", f"SOLE,L={LENGTH!r},KS={KS!r},B={FIELD!r}", KS
        )
        if not ok_matrix:
            sys.exit("The matrix output with both strengths does not follow KS: unexpected behaviour.")
        defect |= not ok_track or not ok_spin
    print(f"comparison: atol={ATOL:g}")
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

