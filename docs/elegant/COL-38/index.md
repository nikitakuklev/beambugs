---
title: "COL-38: CSRDRIFT spread mode with WAVELENGTH_MODE=\"peak-to-peak\" uses an uninitialized wavelength"
hide:
  - toc
---

# COL-38 — CSRDRIFT spread mode with WAVELENGTH_MODE="peak-to-peak" uses an uninitialized wavelength

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [reset, csbend.c:4616–4619](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L4616-L4619)<br>[declaration, csbend.c:3921](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3921)<br>[wavelength, csbend.c:3953–3955](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3953-L3955)<br>[hand-off, csbend.c:4007–4009](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L4007-L4009)<br>[index_min_max, findMinMax.c:116–123](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/mdbmth/findMinMax.c#L116-L123)<br>[wavelength mode, csbend.c:4381–4383](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L4381-L4383)<br>[radiation angle, csbend.c:4389–4392](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L4389-L4392)<br>[radiation-only factor, csbend.c:4493–4495](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L4493-L4495)<br>[WAVELENGTH_MODE, track_data.c:2700](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2700)<br>[CSRDRIFT manual, csrdrift.tex:62–65](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csrdrift.tex#L62-L65) |

With SPREAD=1 and WAVELENGTH_MODE="peak-to-peak", CSRDRIFT's beam-spreading model takes its wavelength from two wake indices that are never set: the extremum search runs on the wake array before the bend hands it over and returns without writing them. For a 0.1 pC, 35 μm rms bunch after a 0.6 m, 0.3 rad bend, the radiation-only spread formula with the 0.1791 mm peak-to-peak wavelength gives an attenuation factor of 0.003415 for a 0.4 m drift, but the kick is applied unattenuated (factor 1), and Valgrind reports the use of uninitialized values in the drift. The default sigmaz mode agrees with the formula within 1.0×10⁻⁸ relative.

??? example "Reproducer: COL-38.py"

    [Download COL-38.py](COL-38.py)

    ```python
    """COL-38: CSRDRIFT spread mode with WAVELENGTH_MODE="peak-to-peak" uses an uninitialized wavelength

    With SPREAD=1 and WAVELENGTH_MODE="peak-to-peak", CSRDRIFT's beam-spreading model takes its wavelength from two wake indices that are never set: the extremum search runs on the wake array before the bend hands it over and returns without writing them. For a 0.1 pC, 35 μm rms bunch after a 0.6 m, 0.3 rad bend, the radiation-only spread formula with the 0.1791 mm peak-to-peak wavelength gives an attenuation factor of 0.003415 for a 0.4 m drift, but the kick is applied unattenuated (factor 1), and Valgrind reports the use of uninitialized values in the drift. The default sigmaz mode agrees with the formula within 1.0×10⁻⁸ relative.

    The script tracks 40001 electrons (Gaussian quantiles, σ_ct = 35 μm, p = 4000 mₑc, 0.1 pC, x
    spread over ±30 μm) through a one-slice 0.6 m, 0.3 rad CSRCSBEND (400 bins, STEADY_STATE=1) and a
    0.4 m CSRDRIFT with one kick, without SPREAD and with SPREAD=1, SPREAD_MODE="radiation-only" and
    BUNCHLENGTH_MODE="rms" in both wavelength modes. Expected attenuation factors follow the
    model's formula with the bend-exit beam size and the wavelength the mode selects; the sigmaz
    mode is the control. Both spread modes are then run under Valgrind (which must be on PATH),
    which must report no use of uninitialized values in the drift.
    """

    import re

    import numpy as np
    from _csr import C, final, gaussian_beam, page, track, valgrind
    from _elegant import particles, verdict

    P0, SIGMA, RHO, DRIFT = 4000.0, 35e-6, 2.0, 0.4
    BEND = (
        "L=0.6,ANGLE=0.3,N_SLICES=1,INTEGRATION_ORDER=6,SG_HALFWIDTH=1,SGDERIV_HALFWIDTH=3,"
        "SGDERIV_ORDER=3,BINS=400,STEADY_STATE=1"
    )
    BEAM = gaussian_beam(40001, SIGMA, P0)
    BEAM["x"] = 3e-5 * np.cos(np.arange(40001) * 2.399963229728653)  # golden-angle spread in x
    LINE = "Q,B,W,D"


    def lattice(spread: str) -> str:
        drift = f"D: CSRDRIFT,L={DRIFT},N_KICKS=1" + (f",{spread}" if spread else "")
        return (
            "Q: CHARGE,TOTAL=1e-13\n"
            f'B: CSRCSBEND,{BEND},OUTPUT_FILE="wake.sdds"\n'
            'W: WATCH,FILENAME="bend.sdds",MODE="coordinates"\n' + drift
        )


    def spread(mode: str) -> str:
        return f'SPREAD=1,SPREAD_MODE="radiation-only",WAVELENGTH_MODE="{mode}",BUNCHLENGTH_MODE="rms"'


    def drift_kick(spread_text: str):
        result = track(
            lattice(spread_text), LINE, BEAM, P0, outputs=("out.sdds", "bend.sdds", "wake.sdds")
        )
        exit_beam = particles(result["bend.sdds"])
        return np.mean(final(result)["p"] - exit_beam["p"]), exit_beam, page(result["wake.sdds"])


    def factor(wavelength: float, sigma_x: float) -> float:
        """Radiation-only spread factor of the drift kick at its middle, zTravel = L/2.

        The CSRDRIFT spread model widens the radiation by the angle
            theta = 0.5463e-3/(p 0.511e-3) (lambda_c/lambda)^(-1/3), lambda_c = 4.19 rho/p^3,
        of synchrotron radiation of wavelength lambda in a bend of radius rho, and scales the
        unattenuated wake by sigma_x/sqrt(sigma_x^2 + (zTravel theta)^2).
        """
        critical = 4.19 * RHO / P0**3
        theta = 0.5463e-3 / (P0 * 0.511e-3) / (critical / wavelength) ** (1 / 3)
        return sigma_x / np.sqrt(sigma_x**2 + (DRIFT / 2 * theta) ** 2)


    base, exit_beam, (wake, info) = drift_kick("")
    sigma_x = np.std(exit_beam["x"])
    # The selected wavelengths: the rms bunch length (sigmaz), and twice the distance between the
    # extrema of the bend's wake (peak-to-peak), from the exported wake.
    lengths = {
        "sigmaz": np.std(C * exit_beam["t"]),
        "peak-to-peak": 2
        * abs(int(np.argmax(wake["DeltaGamma"])) - int(np.argmin(wake["DeltaGamma"])))
        * info["BinSize"],
    }
    print(f"serial no SPREAD: mean drift kick {base:.10g} m_e c; bend-exit sigma_x={sigma_x:.6g} m")
    defect = False
    for mode, wavelength in lengths.items():
        kick, _, _ = drift_kick(spread(mode))
        observed, expected = kick / base, factor(wavelength, sigma_x)
        good = np.isclose(observed, expected, rtol=1e-6)
        print(
            f"serial SPREAD=1 {mode} (wavelength {wavelength:.6g} m): attenuation factor "
            f"observed={observed:.10g} expected={expected:.10g}"
        )
        if mode == "sigmaz" and not good:
            raise SystemExit("The sigmaz control disagrees with the spread formula.")
        defect |= mode == "peak-to-peak" and not good
    # Uninitialized values used by the drift, reported with the source line of the first use.
    USE = re.compile(
        r"(?:Conditional jump or move depends on|Use of) uninitialised value.*\n(?:==\d+==\s+at .*\n)?==\d+==\s+(?:at|by) [^(]*track_through_driftCSR \(([^)]*)\)"
    )
    for mode in lengths:
        _, log = valgrind(lattice(spread(mode)), LINE, BEAM, P0)
        uses = list(dict.fromkeys(USE.findall(log)))
        shown = (
            f"uninitialized values used at {', '.join(uses)}"
            if uses
            else "no uninitialized value used in track_through_driftCSR"
        )
        print(f"serial SPREAD=1 {mode} under Valgrind: {shown}; expected none")
        if mode == "sigmaz" and uses:
            raise SystemExit("The sigmaz control uses uninitialized values.")
        defect |= mode == "peak-to-peak" and bool(uses)
    verdict(defect)

    ```

??? example "Helper: _csr.py"

    [Download _csr.py](_csr.py)

    ```python
    """Shared setup for the CSRCSBEND and CSRDRIFT reproducers: Elegant's physical constants,
    deterministic electron beams, one-run tracking with the wake files Elegant exports, the analytic
    steady-state CSR wake of a Gaussian bunch, and a serial Valgrind runner for the memory defects.

    Lengths are in m, momenta p in units of m_e c, and the longitudinal coordinate z = c t of a
    particle grows toward the tail of the bunch, as in Elegant's wake output column `s`.
    """

    from __future__ import annotations

    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np
    from _elegant import RunFailure, particles, run, write_beam
    from scipy.integrate import quad
    from scipy.special import ndtri

    # SDDS constants.h, the values Elegant computes with: the comparisons below are then not limited
    # by revisions of the CODATA constants.
    C = 2.99792458e8
    E_CHARGE = 1.60217733e-19
    M_E = 9.1093897e-31
    EPSILON_0 = 1e7 / (4 * np.pi * C**2)
    R_E = E_CHARGE**2 / (4 * np.pi * EPSILON_0 * M_E * C**2)
    TIMEOUT = 600  # seconds per Valgrind run


    def gaussian_beam(count: int, sigma_ct: float, p0: float, chirp: float = 0.0, x0: float = 0.0):
        """Gaussian quantiles in z = c t with rms sigma_ct, all at p0 (1 + chirp z) and x = x0.

        Quantiles (i + 1/2)/count place the particles deterministically and symmetrically about z = 0.
        """
        z = sigma_ct * ndtri((np.arange(count) + 0.5) / count) if count > 1 else np.zeros(1)
        return {
            "x": np.full(count, x0),
            "t": z / C,
            "p": p0 * (1 + chirp * z),
            "particleID": np.arange(1, count + 1),
        }


    def commands(p0: float, steps: int = 1, passes: int = 1, seed: int = 0) -> str:
        """Command file: track beam.sdds through BL, final coordinates in out.sdds."""
        random = f",random_number_seed={seed}" if seed else ""
        return (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0!r},output=out.sdds{random} &end\n"
            f"&run_control n_steps={steps},n_passes={passes} &end\n"
            "&sdds_beam input=beam.sdds,reuse_bunch=1 &end\n"
            "&track &end\n"
        )


    def track(elements: str, line: str, beam, p0: float, outputs=("out.sdds",), **kw):
        """Track `beam` through BL: LINE=(line) defined after `elements`; returns run()'s dict."""
        lattice = elements.rstrip("\n") + f"\nBL: LINE=({line})\n"
        return run(
            {"lattice.lte": lattice, "run.ele": commands(p0, **kw)},
            beams={"beam.sdds": beam},
            outputs=outputs,
        )


    def status(elements: str, line: str, beam, p0: float, **kw) -> int:
        """Exit status of a plain serial run (0 on success): the memory defects end in a signal."""
        try:
            track(elements, line, beam, p0, outputs=(), **kw)
        except RunFailure as failure:
            return failure.status
        return 0


    def page(sdds, index: int = 0) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Columns and parameters of one page of an SDDS file."""
        columns = {c.name: np.asarray(sdds[index, c.name]) for c in sdds.columns}
        parameters = {p.name: p.data[index] for p in sdds.parameters}
        return columns, parameters


    def final(result: dict) -> dict[str, np.ndarray]:
        """Final particle coordinates of a run, ordered by particle ID."""
        return particles(result["out.sdds"])


    def steady_state_wake(z: np.ndarray, sigma: float, k: float) -> np.ndarray:
        """Steady-state CSR energy change of a Gaussian bunch, per particle, in units of m_e c^2.

        Saldin, Schneidmiller and Yurkov (NIM A 398, 1997) and Stupakov and Emma (LCLS-TN-01-12, 2001):
            Delta gamma(z) = k int_0^inf lambda'(z + u) u^(-1/3) du,
        lambda the line density normalized to 1 and k = 2 N r_e L / (3 rho^2)^(1/3) for N electrons
        in a bend of length L and radius rho. With z = a sigma and u = sigma t^3 the integrable
        endpoint singularity disappears:
            Delta gamma = k sigma^(-4/3) int_0^inf 3 t lambda_1'(a + t^3) dt,
        lambda_1(a) = exp(-a^2/2)/sqrt(2 pi), lambda_1'(a) = -a lambda_1(a); the integrand vanishes
        beyond a + t^3 = 12.
        """

        def one(a: float) -> float:
            upper = max(0.0, 12.0 - a) ** (1 / 3)
            integrand = lambda t: -3 * t * (a + t**3) * np.exp(-0.5 * (a + t**3) ** 2)
            return quad(integrand, 0, upper, epsabs=1e-13, epsrel=1e-12, limit=200)[0]

        values = np.array([one(a) for a in np.asarray(z) / sigma])
        return k * sigma ** (-4 / 3) * values / np.sqrt(2 * np.pi)


    def valgrind(elements: str, line: str, beam, p0: float, **kw) -> tuple[int, str]:
        """Serial Elegant under `valgrind --track-origins=yes`: (exit status, Valgrind and run log)."""
        if shutil.which("valgrind") is None or shutil.which("elegant") is None:
            sys.exit("valgrind and elegant must be on PATH.")
        if "RPN_DEFNS" not in os.environ:
            sys.exit("Set RPN_DEFNS to the defns.rpn file of SDDS (see the elegant installation).")
        env = dict(os.environ, OMP_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="elegant-repro-") as tmp:
            work = Path(tmp)
            (work / "lattice.lte").write_text(elements.rstrip("\n") + f"\nBL: LINE=({line})\n")
            (work / "run.ele").write_text(commands(p0, **kw))
            write_beam(work / "beam.sdds", beam)
            try:
                done = subprocess.run(
                    ["valgrind", "--error-limit=no", "--track-origins=yes", "elegant", "run.ele"],
                    cwd=work,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                sys.exit(f"valgrind elegant exceeded {TIMEOUT} s")
        return done.returncode, done.stdout + done.stderr

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
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source reading, not executed for this claim |
| Found (executed verification) | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed (uninitialized wavelength indices; not a reliably zero wavelength or a NULL dereference) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

