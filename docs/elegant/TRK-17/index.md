---
title: "TRK-17: Yawed kick maps sample half a slice behind the particle"
hide:
  - toc
---

# TRK-17 — Yawed kick maps sample half a slice behind the particle

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`trackKickMap`, kickmap.c:76–77](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/kickmap.c#L76-L77)<br>[`trackUndulatorKickMap`, ukickmap.c:155–156](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ukickmap.c#L155-L156)<br>[GKICKMAP yaw, gkickmap.tex:25–32](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/gkickmap.tex#L25-L32)<br>[UKICKMAP yaw, ukickmap.tex:50–57](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ukickmap.tex#L50-L57)<br>[YAW, track_data.c:3466](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3466) |

In GKICKMAP and UKICKMAP, yaw alignment evaluates the map-axis displacement at the slice entrance after the particle has drifted to the slice midpoint. The resulting horizontal sampling error is h tan ψ/2, where h is the slice length and ψ the yaw angle, so a particle following the map's zero-force axis receives a false kick. For a 0.7 m device with ψ = 0.013 rad and slope kick Δx′ = −(0.08 m⁻¹)x at local horizontal offset x, three slices give slope error −1.203×10⁻⁴ instead of zero; 12 slices give −3.005×10⁻⁵. Both map types, all three tested pivots and both yaw signs are affected; zero yaw passes.

??? example "Reproducer: TRK-17.py"

    [Download TRK-17.py](TRK-17.py)

    ```python
    """TRK-17: Yawed kick maps sample half a slice behind the particle

    In GKICKMAP and UKICKMAP, yaw alignment evaluates the map-axis displacement at the slice entrance
    after the particle has drifted to the slice midpoint. The resulting horizontal sampling error is
    h tan ψ/2, where h is the slice length and ψ the yaw angle, so a particle following the map's
    zero-force axis receives a false kick. For a 0.7 m device with ψ = 0.013 rad and slope kick
    Δx′ = −(0.08 m⁻¹)x at local horizontal offset x, three slices give slope error −1.203×10⁻⁴ instead
    of zero; 12 slices give −3.005×10⁻⁵. Both map types, all three tested pivots and both yaw signs are
    affected; zero yaw passes.

    The script launches electrons exactly along the axis of a yawed GKICKMAP or UKICKMAP whose map is a
    linear kick, which is zero on that axis, so the exit slope must equal the entrance slope. It runs
    both elements for three pivots, both yaw signs and 3 or 12 slices, and compares the slope change
    with zero and with a recurrence that places the sampling point half a slice behind the particle.
    The control is zero yaw, where the two sampling points coincide.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h, as Elegant uses them.
    C, QE, ME = 299792458.0, 1.60217733e-19, 9.1093897e-31
    P0, LENGTH, GRADIENT = 1900.0, 0.7, 0.08  # m_e c, m, kick = -GRADIENT*x for the whole element
    COPIES = 6
    ATOL = 1e-17  # rad; xp near 0.013 has a spacing of 1.7e-18
    GRID = [-0.04, -0.02, 0.0, 0.02, 0.04]
    # yaw (rad), pivot (YAW_END: -1 entrance, 0 center, 1 exit), slice count
    CASES = (
        (0.0, 0, 3),
        (0.013, -1, 3),
        (0.013, 0, 3),
        (0.013, 1, 3),
        (-0.013, 0, 3),
        (0.013, 0, 12),
    )


    def kick_map(undulator):
        """Linear kick map -GRADIENT*x, zero on the axis x = 0 for every y.

        GKICKMAP takes dimensionless factors; UKICKMAP takes (T m)^2 and divides by the squared
        rigidity H^2, H = p0 m c/e, so the factors are scaled by H^2 to give the same kick.
        """
        scale = (P0 * ME * C / QE) ** 2 if undulator else 1.0
        units = '"(T*m)$a2$n"' if undulator else '""'
        rows = "".join(f"{x!r} {y!r} {-GRADIENT * x * scale!r} 0\n" for y in GRID for x in GRID)
        return (
            "SDDS1\n"
            '&column name=x, type=double, units="m" &end\n'
            '&column name=y, type=double, units="m" &end\n'
            f"&column name=xpFactor, type=double, units={units} &end\n"
            f"&column name=ypFactor, type=double, units={units} &end\n"
            "&data mode=ascii &end\n"
            f"{len(GRID) ** 2}\n" + rows
        )


    def slope_change(mode, element, yaw, pivot, count):
        """Exit xp minus entrance xp of electrons launched along the yawed device axis."""
        slope = np.tan(yaw)
        # The axis passes through x = -yawOffset at the entrance, where yawOffset is the distance
        # from the entrance to the pivot times tan(yaw).
        entrance_x = -LENGTH * (pivot + 1) / 2 * slope
        lattice = (
            f"K: {element}, L={LENGTH!r}, N_KICKS={count}, YAW={yaw!r}, YAW_END={pivot}, "
            'INPUT_FILE="map.sdds"\n'
            "BL: LINE=(K)\n"
        )
        commands = (
            "&change_particle name=electron &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
            "output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {
            "x": np.full(COPIES, entrance_x),
            "xp": np.full(COPIES, slope),
            "p": np.full(COPIES, P0),
        }
        files = {
            "lattice.lte": lattice,
            "run.ele": commands,
            "map.sdds": kick_map(element == "UKICKMAP"),
        }
        out = particles(run(files, mode, {"beam.sdds": beam})["out.sdds"])
        if len(out["p"]) != COPIES or not np.array_equal(out["p"], np.full(COPIES, P0)):
            sys.exit(f"{mode} {element}: particles were lost or changed momentum")
        return out["xp"] - slope


    def half_slice_prediction(yaw, count):
        """Slope change when the map is sampled h tan(yaw)/2 away from the particle at every kick.

        With h = L/count, a particle on the axis has local offset e (starting at 0) from the axis and
        slope error u. At a slice midpoint the code samples the map at e + h tan(yaw)/2, because it
        subtracts i h tan(yaw) where the axis is at (i + 1/2) h tan(yaw). The kick is
        du = -GRADIENT/count * sampled offset; the two half drifts give e += h u + h du/2.
        """
        h, e, u = LENGTH / count, 0.0, 0.0
        for _ in range(count):
            du = -GRADIENT / count * (e + h * u / 2 + h * np.tan(yaw) / 2)
            e += h * u + h * du / 2
            u += du
        return u


    def compare(label, observed, expected):
        """Print the first particle; return True when any particle disagrees."""
        bad = not np.allclose(observed, expected, rtol=0, atol=ATOL)
        print(f"{label}: observed=[{float(observed[0])!r}] expected=[{float(expected)!r}]")
        return bad


    defect = False
    print("exit xp minus entrance xp on the zero-force axis; expected is 0")
    for mode in modes():
        for element in ("GKICKMAP", "UKICKMAP"):
            for yaw, pivot, count in CASES:
                label = f"{mode} {element} yaw={yaw}, pivot={pivot}, N_KICKS={count}"
                observed = slope_change(mode, element, yaw, pivot, count)
                bad = compare(label, observed, 0.0)
                if yaw == 0.0:
                    if bad:
                        sys.exit("The zero-yaw control receives a kick: the comparison is not valid.")
                    continue
                if not bad:
                    continue  # no kick on the zero-force axis: correct
                # A nonzero kick must be the half-slice sampling offset; anything else is a failed run.
                if compare(
                    label + " vs half-slice offset", observed, half_slice_prediction(yaw, count)
                ):
                    sys.exit("The false kick is not explained by a half-slice sampling offset.")
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

