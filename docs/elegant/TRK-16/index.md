---
title: "TRK-16: UKICKMAP loss compaction gives survivors extra radiation updates"
hide:
  - toc
---

# TRK-16 — UKICKMAP loss compaction gives survivors extra radiation updates

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`trackUndulatorKickMap`, ukickmap.c:155–165](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ukickmap.c#L155-L165)<br>[radiation block, ukickmap.c:181–190](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ukickmap.c#L181-L190)<br>[`swapParticles`, elegant.c:2979–2986](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L2979-L2986)<br>[single-particle radiation in tracking, ukickmap.tex:34–45](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ukickmap.tex#L34-L45) |

With classical radiation enabled in UKICKMAP, removing an out-of-map particle can apply one extra radiation update to the survivor moved into its place. The same identified survivors therefore lose different momentum when unrelated lost particles are listed before them rather than after them, although nominal loss must be independent of row order. For electrons at reference momentum p₀ = 5100 mₑc in a 0.16 m map with undulator parameter K = 2.4, one period gives relative momentum offset δ = −1.362×10⁻⁸ instead of −6.808×10⁻⁹; two periods give −4.085×10⁻⁸ instead of −2.723×10⁻⁸. Independent radiation-integral recurrences and no-loss and loss-last controls establish the extra update.

??? example "Reproducer: TRK-16.py"

    [Download TRK-16.py](TRK-16.py)

    ```python
    """TRK-16: UKICKMAP loss compaction gives survivors extra radiation updates

    With classical radiation enabled in UKICKMAP, removing an out-of-map particle can apply one extra
    radiation update to the survivor moved into its place. The same identified survivors therefore lose
    different momentum when unrelated lost particles are listed before them rather than after them,
    although nominal loss must be independent of row order. For electrons at reference momentum
    p₀ = 5100 mₑc in a 0.16 m map with undulator parameter K = 2.4, one period gives relative momentum
    offset δ = −1.362×10⁻⁸ instead of −6.808×10⁻⁹; two periods give −4.085×10⁻⁸ instead of
    −2.723×10⁻⁸. Independent radiation-integral recurrences and no-loss and loss-last controls
    establish the extra update.

    The script tracks twelve electrons (p0 = 5100 mₑc) through a UKICKMAP of one or two periods with
    classical radiation on: six survivors near the axis and six particles outside the map's aperture,
    which are lost. The same survivors are tracked with the lost particles before them, after them and
    replaced by six in-map particles in the no-loss control. Each survivor's momentum loss is compared
    with the recurrence of the manual's radiation integral, applied once per period. The no-loss and
    lost-last cases are the controls.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h, as Elegant uses them.
    C, QE, ME, RE = 299792458.0, 1.60217733e-19, 9.1093897e-31, 2.81794092e-15
    P0, LENGTH, K = 5100.0, 0.16, 2.4  # m_e c, m, undulator parameter KREF
    ATOL = 2e-16  # delta is recovered from p/p0 - 1, which rounds at about 1e-16
    GOOD_IDS, BAD_IDS = np.arange(101, 107), np.arange(201, 207)
    GOOD_X = np.linspace(-0.0002, 0.0002, 6)  # inside the map
    BAD_X = np.full(6, 0.2)  # outside the map (it extends to +-0.04 m): lost in the first kick
    GRID = [-0.04, -0.02, 0.0, 0.02, 0.04]


    def evolve(updates, decrement):
        """delta after the given number of radiation updates, each delta -> delta - a (1 + delta)^2."""
        delta = 0.0
        for _ in range(updates):
            delta -= decrement * (1 + delta) ** 2
        return delta


    def decrement_per_period(periods):
        """a = (2/3) r_e p0^3 I2 for one period of a sinusoidal planar wiggler.

        The orbit curvature is kappa(z) = kappa_peak sin(k_w z) with the ultrarelativistic
        approximation kappa_peak = 2 pi K/(gamma lambda), gamma = sqrt(p0^2 + 1); the exact magnetic
        curvature has the normalized momentum beta gamma = p0 in the denominator. Then
        I2 = integral of kappa^2 over one period = lambda kappa_peak^2/2. The row-order independence
        and the extra-update diagnosis do not depend on this model.
        """
        wavelength = LENGTH / periods
        kappa_peak = 2 * np.pi * K / (np.hypot(P0, 1) * wavelength)
        return 2 / 3 * RE * P0**3 * wavelength * kappa_peak**2 / 2


    def kick_map(periods):
        """Ideal planar-wiggler map: no force on the midplane y = 0, vertical focusing off it.

        The ponderomotive potential of a planar wiggler with peak field B0 gives the integrated
        kick -B0^2 L sinh(2 k_w y)/(4 k_w) in (T m)^2, with B0 = K m c k_w/e.
        """
        k_w = 2 * np.pi * periods / LENGTH
        b0 = K * ME * C * k_w / QE
        rows = "".join(
            f"{x!r} {y!r} 0 {float(-(b0**2) * LENGTH * np.sinh(2 * k_w * y) / (4 * k_w))!r}\n"
            for y in GRID
            for x in GRID
        )
        return (
            "SDDS1\n"
            '&column name=x, type=double, units="m" &end\n'
            '&column name=y, type=double, units="m" &end\n'
            '&column name=xpFactor, type=double, units="(T*m)$a2$n" &end\n'
            '&column name=ypFactor, type=double, units="(T*m)$a2$n" &end\n'
            "&data mode=ascii &end\n"
            f"{len(GRID) ** 2}\n" + rows
        )


    def survivor_deltas(mode, periods, order):
        """delta = p/p0 - 1 of the six survivors, identified by particle ID."""
        lattice = (
            f"U: UKICKMAP, L={LENGTH!r}, PERIODS={periods}, N_KICKS={periods}, KREF={K!r}, "
            'SYNCH_RAD=1, ISR=0, INPUT_FILE="map.sdds"\n'
            "BL: LINE=(U)\n"
        )
        commands = (
            "&change_particle name=electron &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
            "output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        good, bad = (GOOD_X, GOOD_IDS), (BAD_X, BAD_IDS)
        groups = {"no loss": [good], "lost first": [bad, good], "lost last": [good, bad]}[order]
        x = np.concatenate([g[0] for g in groups])
        ids = np.concatenate([g[1] for g in groups])
        if order == "no loss":  # the same survivors, with the lost particles removed
            x, ids = np.r_[x, x], np.r_[ids, BAD_IDS]
        beam = {"x": x, "p": np.full(len(x), P0), "particleID": ids}
        files = {"lattice.lte": lattice, "run.ele": commands, "map.sdds": kick_map(periods)}
        out = particles(run(files, mode, {"beam.sdds": beam})["out.sdds"])
        expected_count = len(x) if order == "no loss" else len(GOOD_IDS)
        if len(out["p"]) != expected_count:
            sys.exit(f"{mode} {order}: {len(out['p'])} particles survived, expected {expected_count}")
        survivors = np.isin(out["particleID"], GOOD_IDS)
        if not np.array_equal(out["particleID"][survivors], GOOD_IDS):
            sys.exit(f"{mode} {order}: the survivors are not particles {GOOD_IDS[0]}-{GOOD_IDS[-1]}")
        return out["p"][survivors] / P0 - 1


    def compare(label, observed, expected):
        """Print the first survivor; return True when any survivor disagrees."""
        bad = not np.allclose(observed, expected, rtol=0, atol=ATOL)
        print(f"{label}: observed=[{float(observed[0])!r}] expected=[{float(expected)!r}]")
        return bad


    defect = False
    print("delta of each survivor; expected is one radiation update per period")
    for mode in modes():
        for periods in (1, 2):
            a = decrement_per_period(periods)
            for order in ("no loss", "lost last", "lost first"):
                label = f"{mode} periods={periods}, {order}"
                observed = survivor_deltas(mode, periods, order)
                bad = compare(label, observed, evolve(periods, a))
                if order != "lost first":
                    if bad:
                        sys.exit("A control disagrees with the radiation recurrence: not valid.")
                    continue
                if not bad:
                    continue  # the nominal loss, independent of the row order: correct
                # Otherwise the loss must be one extra update, applied to the swapped-in survivor
                # before its own turn; any other result is a failed run, not a verdict.
                if compare(label + " vs one extra update", observed, evolve(periods + 1, a)):
                    sys.exit("The excess loss is not one extra radiation update.")
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

