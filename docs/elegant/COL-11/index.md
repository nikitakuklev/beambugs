---
title: "COL-11: BIN_ONCE clears the grid required by attenuation-mode CSRDRIFT"
hide:
  - toc
---

# COL-11 — BIN_ONCE clears the grid required by attenuation-mode CSRDRIFT

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [final grid preparation, csbend.c:3757](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3757)<br>[zeroed grid, csbend.c:3809–3813](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3809-L3813)<br>[wake marked valid, csbend.c:3960](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3960)<br>[hand-off to CSRDRIFT, csbend.c:4007–4010](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L4007-L4010)<br>[spacing read, csbend.c:4338](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L4338)<br>[gather, csbend.c:4530–4535](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L4530-L4535)<br>[Stupakov dispatch, csbend.c:4332–4333](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L4332-L4333)<br>[parameter table, track_data.c:2060](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2060)<br>[CSRDRIFT manual, csrdrift.tex:54–60](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csrdrift.tex#L54-L60) |

With `BIN_ONCE=1`, a CSRCSBEND passes zero wake-grid spacing to a following exponential-attenuation CSRDRIFT. In the reproduction the drift bins no particles and applies no kick, warning at every step; the zero-spacing failure can behave differently on other builds. For a one-slice 0.5 m, 50 mrad bend followed by a 0.1 m drift with 1000 m attenuation length, a 1 nC, 1 mm rms Gaussian electron bunch at 2000 mₑc should lose a further 681.0 eV per electron on average, as integration of the exported wake confirms, but loses zero. A drift with `USE_STUPAKOV=1` gives identical nonzero kicks for both binning settings.

??? example "Reproducer: COL-11.py"

    [Download COL-11.py](COL-11.py)

    ```python
    """COL-11: BIN_ONCE clears the grid required by attenuation-mode CSRDRIFT

    With `BIN_ONCE=1`, a CSRCSBEND passes zero wake-grid spacing to a following exponential-attenuation CSRDRIFT. In the reproduction the drift bins no particles and applies no kick, warning at every step; the zero-spacing failure can behave differently on other builds. For a one-slice 0.5 m, 50 mrad bend followed by a 0.1 m drift with 1000 m attenuation length, a 1 nC, 1 mm rms Gaussian electron bunch at 2000 mₑc should lose a further 681.0 eV per electron on average, as integration of the exported wake confirms, but loses zero. A drift with `USE_STUPAKOV=1` gives identical nonzero kicks for both binning settings.

    The script tracks 16384 electrons (p = 2000 m_e c, 1 nC, 1 mm rms) through a one-slice CSRCSBEND (0.5 m, 50 mrad) and a 0.1 m CSRDRIFT with exponential attenuation, once with BIN_ONCE=0 and once with BIN_ONCE=1. The drift kick of the BIN_ONCE=0 run is compared with an independent integration of the wake the bend exports, and the BIN_ONCE=1 run with it. The BIN_ONCE=1 drift must leave the momenta at the bend exit, which is the same for both settings, while
    the oracle's kick is nonzero. The control is a Stupakov-model drift, which builds its own grid and must
    not depend on BIN_ONCE.
    """

    import sys
    from statistics import NormalDist

    import numpy as np
    from _elegant import modes, particles, run, verdict

    C = 299792458.0  # SDDS constants.h
    COUNT, P0, CHARGE, SIGMA_CT = 16384, 2000.0, 1e-9, 1e-3
    BEND_L, BEND_ANGLE, BINS = 0.5, 0.05, 512
    DRIFT_L, KICKS, ATTENUATION = 0.1, 10, 1000.0
    ATOL_P = 1e-11  # m_e c; the oracle reproduces the tracked momenta to ~1e-12
    MIN_P = 1e-4  # m_e c; the BIN_ONCE=1 drift omits kicks of order 1e-3 m_e c

    # Deterministic Gaussian quantiles in ct = c t; particle IDs fix the row order in every mode.
    CT = SIGMA_CT * np.array([NormalDist().inv_cdf((i + 0.5) / COUNT) for i in range(COUNT)])
    BEAM = {"t": CT / C, "p": np.full(COUNT, P0), "particleID": np.arange(1, COUNT + 1)}


    def track(mode, bin_once, drift, output=False):
        """Final (p, t, xp, yp) of every particle, plus the exported wake when `output` is set."""
        wake = ',OUTPUT_FILE="wake.sdds"' if output else ""
        lattice = (
            f"Q: CHARGE,TOTAL={CHARGE!r}\n"
            f"B: CSRCSBEND,L={BEND_L!r},ANGLE={BEND_ANGLE!r},N_SLICES=1,BINS={BINS},"
            f"BIN_ONCE={bin_once},STEADY_STATE=1,SYNCH_RAD=0,ISR=0,SG_HALFWIDTH=0,"
            f'SGDERIV_HALFWIDTH=1,DERBENEV_CRITERION_MODE="evaluate"{wake}\n'
            f"D: {drift}\n"
            "BL: LINE=(Q,B,D)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": BEAM},
            outputs=("out.sdds", "wake.sdds") if output else ("out.sdds",),
        )
        out = particles(result["out.sdds"])
        if not np.array_equal(out["particleID"], BEAM["particleID"]):
            sys.exit(f"{mode}: particles lost")
        if not all(np.all(np.isfinite(out[name])) for name in ("p", "t", "xp", "yp")):
            sys.exit(f"{mode}: nonfinite particle coordinates")
        return out, result.get("wake.sdds"), result["log"]


    def gamma(p):
        return np.sqrt(1 + p**2)


    def beta(p):
        return p / np.sqrt(1 + p**2)


    def cached_wake_drift(exit_beam, wake):
        """Momenta after the attenuating drift if the bend's exported wake is reused.

        CSRDRIFT with ATTENUATION_LENGTH a applies the frozen wake field dgamma(ct), the bend's total
        energy change per bin, at the middle of each of the KICKS steps: a drift of half a step, then
        kicks separated by whole steps. A kick at distance z from the drift entrance is
            dp = dgamma(ct - ct_origin - z) exp(-z/a) (step / bend length),
        with the field read by linear interpolation in bins of width BinSize. The field travels with
        the light front, so the particle's arrival ct (converted from path length s with its velocity
        beta = p/sqrt(1+p^2)) is measured from the wake origin plus z. The origin is the exit ct
        window: its center minus 1.2 times its half-width.
        """
        ct_exit = C * exit_beam["t"]
        lower = (ct_exit.min() + ct_exit.max()) / 2 - 1.2 * np.ptp(ct_exit) / 2
        dct = next(par.data[0] for par in wake.parameters if par.name == "BinSize")
        field = np.asarray(wake[0, "DeltaGamma"])
        if not (np.isfinite(dct) and dct > 0 and np.all(np.isfinite(field))):
            sys.exit("The exported wake has a nonfinite field or a nonpositive BinSize.")
        p = exit_beam["p"].copy()
        s = ct_exit * beta(p)
        path = np.sqrt(1 + exit_beam["xp"] ** 2 + exit_beam["yp"] ** 2)
        step = DRIFT_L / KICKS
        distance = 0.0
        for k in range(KICKS):
            advance = step / 2 if k == 0 else step
            distance += advance
            ct = (s + advance * path) / beta(p)
            coordinate = (ct - lower - distance) / dct
            if coordinate.min() <= 0 or coordinate.max() >= len(field) - 1:
                sys.exit("A particle lies outside the exported wake.")
            p += (
                np.interp(coordinate, np.arange(len(field)), field)
                * np.exp(-distance / ATTENUATION)
                * (step / BEND_L)
            )
            s = ct * beta(p)
        return p


    def show(mode, label, observed, expected):
        print(f"{mode} {label}: observed={observed:.16g} expected={expected:.16g}")


    ATTENUATION_DRIFT = f"CSRDRIFT,L={DRIFT_L!r},N_KICKS={KICKS},ATTENUATION_LENGTH={ATTENUATION!r}"
    STUPAKOV_DRIFT = f"CSRDRIFT,L={DRIFT_L!r},N_KICKS={KICKS},USE_STUPAKOV=1"
    defect = False
    for mode in modes():
        # Bend exit (zero-length drift) and the wake the bend exports. The bend exit must not depend on
        # BIN_ONCE: with it the momenta of the BIN_ONCE=1 attenuation drift are compared below.
        exit_beam, wake, _ = track(mode, 0, "DRIFT,L=0", output=True)
        if not np.array_equal(track(mode, 1, "DRIFT,L=0")[0]["p"], exit_beam["p"]):
            sys.exit("The bend exit depends on BIN_ONCE.")
        oracle = cached_wake_drift(exit_beam, wake)
        if not np.all(np.isfinite(oracle)):
            sys.exit("The wake oracle is nonfinite.")
        if np.max(np.abs(oracle - exit_beam["p"])) <= MIN_P:
            sys.exit("The oracle's added kick is too small to separate the cases.")
        mean_oracle = np.mean(gamma(oracle) - gamma(exit_beam["p"]))
        for bin_once in (0, 1):
            out, _, log = track(mode, bin_once, ATTENUATION_DRIFT)
            mean_gain = np.mean(gamma(out["p"]) - gamma(exit_beam["p"]))
            error = np.max(np.abs(out["p"] - oracle))
            label = f"attenuation drift, BIN_ONCE={bin_once}, mean added gamma"
            show(mode, label, mean_gain, mean_oracle)
            print(
                f"{mode} attenuation drift, BIN_ONCE={bin_once}: max |p - oracle|={error:.10g} m_e c; "
                f"'particles binned' warnings: {log.count('only 0 of')}"
            )
            if bin_once == 0 and not np.allclose(out["p"], oracle, rtol=0, atol=ATOL_P):
                sys.exit("The BIN_ONCE=0 drift disagrees with the wake oracle.")
            if bin_once == 1:
                # The omitted kick: the momenta equal the bend exit and differ from the oracle.
                same = np.array_equal(out["p"], exit_beam["p"])
                print(f"{mode} attenuation drift, BIN_ONCE=1: momenta equal the bend exit: {same}")
                defect |= same and bool(error > MIN_P)
        # Control: the Stupakov drift builds its own grid, so BIN_ONCE does not matter.
        stupakov = [track(mode, b, STUPAKOV_DRIFT)[0]["p"] for b in (0, 1)]
        gain = np.mean(gamma(stupakov[1]) - gamma(exit_beam["p"]))
        if not np.isfinite(gain) or not np.isfinite(
            np.mean(gamma(stupakov[0]) - gamma(exit_beam["p"]))
        ):
            sys.exit("The Stupakov control gains are nonfinite.")
        show(
            mode,
            "Stupakov drift, mean added gamma BIN_ONCE=1 vs 0",
            gain,
            np.mean(gamma(stupakov[0]) - gamma(exit_beam["p"])),
        )
        if not np.array_equal(stupakov[0], stupakov[1]) or gain == 0:
            sys.exit("The Stupakov control depends on BIN_ONCE or has no kick.")
    print(
        f"oracle tolerance {ATOL_P:g} m_e c; defect if the BIN_ONCE=1 drift differs by more than {MIN_P:g} m_e c"
    )
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
| Found | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Found | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

