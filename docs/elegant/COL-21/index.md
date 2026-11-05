---
title: "COL-21: Sliced linear SCMULT normalizes every slice by the whole-bunch beam size"
hide:
  - toc
---

# COL-21 — Sliced linear SCMULT normalizes every slice by the whole-bunch beam size

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`linearSCKick`, insertSCeffects.cc:510–529](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L510-L529)<br>[`accumulateSCMULT`, insertSCeffects.cc:837–856](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L837-L856)<br>[slice branch of `trackThroughSCMULT`, insertSCeffects.cc:377–472](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L377-L472)<br>[slicing options, elegant.tex:4504–4511](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4504-L4511)<br>[linear field, scmult.tex:10–24](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/scmult.tex#L10-L24) |

With constant matched optics, linear SCMULT space-charge forces, slicing on (`slice_duration` > 0), interpolation off and slice populations meeting `slice_threshold`, Elegant normalizes each slice's field gradient by the whole bunch's rms widths. The manual instead treats each slice independently; for transverse rms widths σₓ and σᵧ, its horizontal gradient is proportional to 1/[σₓ(σₓ + σᵧ)], as equal-width and isolated-slice controls confirm. For two equal-charge slices whose widths differ by a factor of 2, the narrow slice's gradient is 9.742×10⁻⁷ m⁻¹ instead of 2.436×10⁻⁶ m⁻¹, 2.500 times too weak, while the wide slice's is 1.600 times too strong.

??? example "Reproducer: COL-21.py"

    [Download COL-21.py](COL-21.py)

    ```python
    """COL-21: Sliced linear SCMULT normalizes every slice by the whole-bunch beam size

    With constant matched optics, linear SCMULT space-charge forces, slicing on (`slice_duration` > 0),
    interpolation off and slice populations meeting `slice_threshold`, Elegant normalizes each slice's
    field gradient by the whole bunch's rms widths. The manual instead treats each slice independently;
    for transverse rms widths σₓ and σᵧ, its horizontal gradient is proportional to 1/[σₓ(σₓ + σᵧ)], as
    equal-width and isolated-slice controls confirm. For two equal-charge slices whose widths differ by a
    factor of 2, the narrow slice's gradient is 9.742×10⁻⁷ m⁻¹ instead of 2.436×10⁻⁶ m⁻¹, 2.500 times
    too weak, while the wide slice's is 1.600 times too strong.

    The script tracks one turn around a ring of sixteen identical sector bends, with constant matched
    beta functions, and a single SCMULT at its end; the beam is two clouds of 128 matched particles,
    20.25 ns apart, so each cloud fills one slice. It fits each slice's kick gradient (the kick
    divided by the offset from the slice centroid, from runs with and without charge) and compares
    it with the manual's linear Gaussian field evaluated with that slice's own sizes. The controls
    are two equal slices, each slice alone, and a slice threshold above the slice population, for
    which the manual itself prescribes whole-bunch sizes.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h, as Elegant uses them.
    C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
    P0 = 100.0  # electron momentum, units of m_e c
    RADIUS, K1 = 8.0, -0.0075  # m, 1/m^2
    LENGTH = 2 * np.pi * RADIUS  # ring circumference, m
    # A sector bend of radius R with gradient K1 focuses with k_x = 1/R^2 + K1 and k_y = -K1; a
    # ring of identical bends has constant matched beta = 1/sqrt(k).
    BETA = np.array([1 / np.sqrt(1 / RADIUS**2 + K1), 1 / np.sqrt(-K1)])
    SIGMA = np.array([1.0e-3, 1.6e-3])  # rms x, y of the narrowest slice, m
    N = 128  # particles per slice
    Q = 1e-12  # C per slice
    DT = 2.0**-30  # slice duration, s; a binary fraction keeps slice edges exact
    SEPARATION = 20.25e-9  # s between the slice centers
    RTOL, ATOL = 4e-10, 1e-16  # 1/m; floor from subtracting the transported slopes


    def unit_cloud():
        """N points with zero mean and exactly unit covariance in four dimensions."""
        z = np.random.default_rng(88210).normal(size=(N, 4))
        z -= z.mean(axis=0)
        return np.linalg.qr(z)[0] * np.sqrt(N)


    def beam(widths):
        """One matched cloud per slice, scaled by its width factor, at arrival times j*SEPARATION."""
        z = unit_cloud()
        clouds = [
            {
                "x": w * SIGMA[0] * z[:, 0],
                "xp": w * SIGMA[0] / BETA[0] * z[:, 1],
                "y": w * SIGMA[1] * z[:, 2],
                "yp": w * SIGMA[1] / BETA[1] * z[:, 3],
                "t": np.full(N, j * SEPARATION),
                "p": np.full(N, P0),
            }
            for j, w in enumerate(widths)
        ]
        return {key: np.concatenate([c[key] for c in clouds]) for key in clouds[0]}


    def gradients(mode, widths, threshold):
        """Fitted kick gradients (x, y) of each slice, in 1/m."""
        columns = beam(widths)
        outputs = []
        for charge in (0.0, Q * len(widths)):
            lattice = (
                f"Q: CHARGE,TOTAL={charge!r}\n"
                f"B: SBEN,L={LENGTH / 16!r},ANGLE={2 * np.pi / 16!r},K1={K1!r}\n"
                "BL: LINE=(Q,16*B)\n"
            )
            # skip=0 inserts one SCMULT at the end of the beamline; it integrates the whole ring.
            commands = (
                "&insert_sceffects name=B,skip=0,horizontal=1,vertical=1,nonlinear=0,"
                f"slice_duration={DT!r},slice_threshold={threshold},slice_interpolation=0 &end\n"
                f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0!r},default_order=1,"
                "output=out.sdds &end\n"
                "&run_control n_steps=1,n_passes=1 &end\n"
                "&twiss_output matched=1 &end\n"
                "&sdds_beam input=beam.sdds &end\n"
                "&track &end\n"
            )
            result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})
            out = particles(result["out.sdds"])
            if len(out["p"]) != len(columns["p"]):
                sys.exit(f"{mode}: particles lost")
            outputs.append(out)
        before, after = outputs
        # NaN slopes would turn every comparison below into a spurious mismatch.
        for name in ("xp", "yp"):
            if not (np.all(np.isfinite(before[name])) and np.all(np.isfinite(after[name]))):
                sys.exit(f"{mode}: non-finite {name} in the uncharged or charged run")
        # Space charge changes only the slopes.
        for name in ("x", "y", "t", "p"):
            if not np.array_equal(after[name], before[name]):
                sys.exit(f"{mode}: space charge changed {name}")
        fitted = []
        for j in range(len(widths)):
            rows = slice(j * N, (j + 1) * N)
            offsets, kicks = [], []
            for plane, slope in (("x", "xp"), ("y", "yp")):
                offset = before[plane][rows] - before[plane][rows].mean()
                offsets.append(offset)
                kicks.append(after[slope][rows] - before[slope][rows])
            offsets, kicks = np.array(offsets), np.array(kicks)
            # Least-squares slope of kick against offset from the slice centroid.
            fitted.append(np.sum(offsets * kicks, axis=1) / np.sum(offsets**2, axis=1))
        fitted = np.array(fitted)
        if not np.all(np.isfinite(fitted)):
            sys.exit(f"{mode}: non-finite fitted gradient")
        return fitted


    def field_gradient(sigma):
        """Manual's linear field for a slice of charge Q, uniform over DT, sizes sigma = (sx, sy).

        dx' = K L lambda x/[sx (sx + sy)], K = 2 N_e r_e/(gamma^3 beta^2), with the line density
        1/(c DT) of a uniform slice in place of the Gaussian factor, N_e = Q/e and
        gamma^3 beta^2 = p^2 sqrt(1 + p^2). With constant beta the optical weight SCMULT integrates
        over the ring, L beta/[sx (sx + sy)], divided by the beta at the kick, leaves exactly L.
        """
        k = 2 * (Q / QE) * RE / (P0**2 * np.sqrt(1 + P0**2))
        return k * LENGTH / (C * DT) / (sigma * sigma.sum())


    def fmt(values):
        return "[" + ", ".join(f"{v:.15g}" for v in np.ravel(values)) + "]"


    defect = False
    print("kick gradients (x, y) in 1/m")
    for mode in modes():
        cases = (
            ("slice widths 1 and 2", (1, 2), 100, False),
            ("control, slice widths 1 and 1", (1, 1), 100, True),
            ("control, slice width 1 alone", (1,), 100, True),
            ("control, slice width 2 alone", (2,), 100, True),
            ("control, threshold 1000, widths 1 and 2", (1, 2), 1000, True),
        )
        for label, widths, threshold, control in cases:
            observed = gradients(mode, widths, threshold)
            # Whole-bunch rms sizes of equal-population, co-centered clouds: sqrt(mean(w^2))*SIGMA.
            whole = field_gradient(np.sqrt(np.mean(np.square(widths))) * SIGMA)
            if threshold > N:
                # Below the threshold the manual replaces slice sizes with whole-bunch sizes.
                expected = np.array([whole for _ in widths])
            else:
                expected = np.array([field_gradient(w * SIGMA) for w in widths])
            bad = not np.allclose(observed, expected, rtol=RTOL, atol=ATOL)
            for j, w in enumerate(widths):
                print(
                    f"{mode} {label}, slice {j + 1} (width {w}): "
                    f"observed={fmt(observed[j])} expected={fmt(expected[j])}"
                )
            if control and bad:
                sys.exit(f"{mode} {label}: control disagrees with the field; the comparison is invalid")
            if not control:
                # Cause: every slice receives the gradient of the whole-bunch sizes.
                print(f"{mode} {label}: whole-bunch-size gradient={fmt(whole)}")
                if bad and not np.allclose(
                    observed, np.broadcast_to(whole, observed.shape), rtol=RTOL, atol=ATOL
                ):
                    sys.exit(
                        f"{mode} {label}: gradients follow neither the slice field nor the bunch's"
                    )
                defect |= bad
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g} 1/m")
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
| Found (source-only lead) | `gpt-6-astra` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source reading, not executed for this claim |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

