---
title: "TRK-30: Matrix concatenation drops the spin rotation of SOLE and ROTATE"
hide:
  - toc
---

# TRK-30 — Matrix concatenation drops the spin rotation of SOLE and ROTATE

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [absorption test, concat_beamline.c:81–82](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/concat_beamline.c#L81-L82)<br>[sequence replaced, concat_beamline.c:113–121](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/concat_beamline.c#L113-L121)<br>[final sequence, concat_beamline.c:162–173](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/concat_beamline.c#L162-L173)<br>[MATR dispatch, do_tracking.c:2055–2060](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2055-L2060)<br>[orbit-only transport, do_tracking.c:4719–4720](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L4719-L4720)<br>[SOLE flags, track_data.c:4172](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L4172)<br>[ROTATE flags, track_data.c:4200](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L4200)<br>[SOLE, do_tracking.c:2380–2382](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2380-L2382)<br>[ROTATE, do_tracking.c:1988–1992](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L1988-L1992)<br>[warning, elegant.c:990–992](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L990-L992)<br>[concat_order, elegant.tex:7248–7249](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7248-L7249)<br>[spin_tracking, elegant.tex:7291–7292](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7291-L7292)<br>[spin-aware elements, elegant.tex:11191–11192](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L11191-L11192) |

With spin tracking enabled and a nonzero concat_order in run_setup, Elegant replaces each sequence of matrix elements of at most that order, even a single element, by one generic matrix element that transports only the orbit, so the spin rotations of absorbed SOLE solenoids and ROTATE elements are lost. For an on-axis electron at momentum 200 mₑc in a 0.65 m SOLE with KS = 0.9000 m⁻¹, the Thomas–BMT precession of 0.5857 rad, which concat_order = 0 reproduces, vanishes at concat_order = 2 and 3 while the orbit is unchanged. A ROTATE with TILT = 0.3000 rad leaves the spin unrotated at concat_order = 1. The only warning says that concatenation reduces accuracy.

??? example "Reproducer: TRK-30.py"

    [Download TRK-30.py](TRK-30.py)

    ```python
    """TRK-30: Matrix concatenation drops the spin rotation of SOLE and ROTATE

    With spin tracking enabled and a nonzero concat_order in run_setup, Elegant replaces each sequence
    of matrix elements of at most that order, even a single element, by one generic matrix element that
    transports only the orbit, so the spin rotations of absorbed SOLE solenoids and ROTATE elements are
    lost. For an on-axis electron at momentum 200 mₑc in a 0.65 m SOLE with KS = 0.9000 m⁻¹, the
    Thomas–BMT precession of 0.5857 rad, which concat_order = 0 reproduces, vanishes at concat_order =
    2 and 3 while the orbit is unchanged. A ROTATE with TILT = 0.3000 rad leaves the spin unrotated at
    concat_order = 1. The only warning says that concatenation reduces accuracy.

    The script tracks an on-axis electron with spin (1, 0, 0) through one order-2 SOLE at concat_order
    0 to 3 and compares the exit spin with the exact on-axis Thomas–BMT precession; concat_order 0 and
    1, which leave the element in place, are the controls. It checks that an off-axis orbit is
    identical with and without concatenation, and repeats the comparison for a ROTATE element against
    its own unconcatenated frame rotation.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _sole import A_ELECTRON, axial_rotation, compare, on_axis_angle, track

    LENGTH, P0, KS = 0.65, 200.0, 0.9
    TILT = 0.3  # rad
    ATOL = 1e-13


    def spin_through(mode, element, concat_order, start=None):
        """Final spin and orbit of one particle through `element` with the given concat_order."""
        out, result = track(
            mode, element, P0, start or {"spx": 1.0}, setup=f", concat_order={concat_order}"
        )
        if concat_order and "concat_order is non-zero" not in result["log"]:
            sys.exit("The concatenation warning is missing: the run did not concatenate as assumed.")
        return np.array([out["spx"], out["spy"], out["spz"]]), out


    defect = False
    sole = f"SOLE,L={LENGTH!r},KS={KS!r},ORDER=2"
    # Thomas-BMT on the axis of the solenoid (see _sole.on_axis_angle).
    expected_sole = axial_rotation(on_axis_angle(KS, LENGTH, A_ELECTRON))
    # ROTATE turns the coordinate frame by TILT about +z, so the components of a fixed spin turn
    # by -TILT.
    expected_rotate = axial_rotation(-TILT)
    for mode in modes():
        # concat_order = 0 (no concatenation) and 1 (the order-2 SOLE matrix is not absorbed) keep SOLE.
        for order in (0, 1):
            spin, _ = spin_through(mode, sole, order)
            if not compare(
                f"{mode} SOLE, concat_order={order} (control): spin", spin, expected_sole, atol=ATOL
            ):
                sys.exit("The unconcatenated SOLE control disagrees with Thomas-BMT: it is not valid.")
        for order in (2, 3):
            spin, _ = spin_through(mode, sole, order)
            defect |= not compare(
                f"{mode} SOLE, concat_order={order}: spin", spin, expected_sole, atol=ATOL
            )
        # The orbit is not affected: an off-axis particle leaves concat_order 0 and 2 identically.
        start = {"x": 1e-3, "xp": -2e-4, "y": 5e-4, "yp": 3e-4, "spz": 1.0}
        _, plain = spin_through(mode, sole, 0, start)
        _, merged = spin_through(mode, sole, 2, start)
        names = ("x", "xp", "y", "yp", "t", "p")
        diff = max(abs(plain[n] - merged[n]) for n in names)
        print(f"{mode} SOLE orbit, concat_order 2 minus 0: max|difference|={diff:.3g}")
        if diff != 0:
            sys.exit("Concatenation changed the orbit: the comparison is not isolated to the spin.")
        spin, _ = spin_through(mode, f"ROTATE,TILT={TILT!r}", 0)
        if not compare(
            f"{mode} ROTATE, concat_order=0 (control): spin", spin, expected_rotate, atol=ATOL
        ):
            sys.exit("The ROTATE control disagrees with the frame rotation: it is not valid.")
        spin, _ = spin_through(mode, f"ROTATE,TILT={TILT!r}", 1)
        defect |= not compare(f"{mode} ROTATE, concat_order=1: spin", spin, expected_rotate, atol=ATOL)
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
| Found (ROTATE case) | `claude-opus-5-5` | high | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed (ROTATE case) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

