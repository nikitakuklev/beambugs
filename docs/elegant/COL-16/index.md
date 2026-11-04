---
title: "COL-16: Dipole resonators create momentum in the disabled plane"
hide:
  - toc
---

# COL-16 — Dipole resonators create momentum in the disabled plane

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [binned kick, trfmode.c:561–563](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trfmode.c#L561-L563)<br>[binless kick, trfmode.c:901–903](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trfmode.c#L901-L903)<br>[FTRFMODE kick, ftrfmode.c:556–558](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ftrfmode.c#L556-L558)<br>[transverse-momentum convention, simple_rfca.c:1139–1144](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1139-L1144)<br>[TRFMODE manual, trfmode.tex:1](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/trfmode.tex#L1)<br>[FTRFMODE manual, ftrfmode.tex:1](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ftrfmode.tex#L1) |

When TRFMODE or FTRFMODE tracks a dipole mode in one plane only, its longitudinal kick creates momentum in the other plane in proportion to the incoming slope there. The code updates longitudinal momentum p∥ before recovering incoming transverse momenta from slopes. For a horizontal mode the voltage is independent of vertical position, so vertical momentum pᵧ must be conserved. For 16 electrons with momentum 750 mₑc, horizontal and vertical slopes (0.01200, 0.02400), horizontal offset 8.000 mm and charge 0.7000 nC, the last particle's vertical momentum instead changes by Δpᵧ = −7.165×10⁻⁴ mₑc, matching vertical slope times Δp∥ within 10⁻¹⁴ mₑc. Resonator simulations with nonzero slopes in the disabled plane are affected.

??? example "Reproducer: COL-16.py"

    [Download COL-16.py](COL-16.py)

    ```python
    """COL-16: Dipole resonators create momentum in the disabled plane

    When TRFMODE or FTRFMODE tracks a dipole mode in one plane only, its longitudinal kick creates
    momentum in the other plane in proportion to the incoming slope there. The code updates longitudinal
    momentum p∥ before recovering incoming transverse momenta from slopes. For a horizontal mode the
    voltage is independent of vertical position, so vertical momentum pᵧ must be conserved. For 16
    electrons with momentum 750 mₑc, horizontal and vertical slopes (0.01200, 0.02400), horizontal
    offset 8.000 mm and charge 0.7000 nC, the last particle's vertical momentum instead changes by
    Δpᵧ = −7.165×10⁻⁴ mₑc, matching vertical slope times Δp∥ within 10⁻¹⁴ mₑc. Resonator simulations
    with nonzero slopes in the disabled plane are affected.

    The script tracks 16 electrons, one per time bin, through each of the three resonator paths with
    only the horizontal or only the vertical mode enabled and with both signs of the incoming slopes.
    A pairwise causal Green-function sum, independent of Elegant's amplitude and phase recurrence,
    gives the longitudinal and transverse impulses; the disabled-plane momentum change must be zero.
    Controls are the agreement of the predicted and tracked p_z and a run with zero incoming slopes,
    for which the error vanishes.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h, as Elegant uses them.
    C, ME_MEV = 299792458.0, 0.51099906
    FREQ, QUALITY, RS = 8e8, 12.0, 6e7  # Hz, dimensionless, Ohm/m
    CHARGE, P0, N = 7e-10, 750.0, 16  # C, units of m_e c
    SPACING, DT, BINS = 1.25e-11, 5e-13, 1024  # s between particles, bin size, bin count
    OFFSET = 8e-3  # m, offset in the active plane
    SLOPES = (0.012, 0.024)  # incoming (xp, yp), times the sign of the case
    MODE_TABLE = (
        "SDDS1\n"
        '&column name=Frequency, type=double, units="Hz" &end\n'
        '&column name=Q, type=double, units="" &end\n'
        '&column name=ShuntImpedance, type=double, units="Ohm/m" &end\n'
        "&data mode=ascii &end\n"
        f"1\n{FREQ!r} {QUALITY!r} {RS!r}\n"
    )
    PZ_TOLERANCE = 5e-13  # m_e c; the oracle's p_z agrees to 2.3e-13
    MOMENTUM_TOLERANCE = 3e-14  # m_e c; disabled-plane error against slope * Delta p_z


    def track(mode, kind, binless, plane, slopes):
        """Output particle columns for the resonator with only the `plane` dipole mode enabled."""
        params = f"BIN_SIZE={DT!r},N_BINS={BINS},RIGID_UNTIL_PASS=0"
        if kind == "TRFMODE":
            params += f',FREQ={FREQ!r},Q={QUALITY!r},RS={RS!r},PLANE="{plane}",BINLESS={binless}'
        else:
            params += ',FILENAME="modes.sdds",' + ("YFACTOR=0" if plane == "x" else "XFACTOR=0")
        lattice = f"C: CHARGE,TOTAL={CHARGE!r}\nK: {kind},{params}\nBL: LINE=(C,K)\n"
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {
            "x": np.full(N, OFFSET if plane == "x" else 0.0),
            "y": np.full(N, OFFSET if plane == "y" else 0.0),
            "xp": np.full(N, slopes[0]),
            "yp": np.full(N, slopes[1]),
            "t": np.arange(N) * SPACING,
            "p": np.full(N, P0),
        }
        result = run(
            {"lattice.lte": lattice, "run.ele": commands, "modes.sdds": MODE_TABLE},
            mode,
            {"beam.sdds": beam},
        )
        out = particles(result["out.sdds"])
        if len(out["p"]) != N or not all(np.all(np.isfinite(v)) for v in out.values()):
            sys.exit(f"{mode} {kind}: particles lost or non-finite output")
        return out


    def momenta(out):
        """Momentum components (px, py, pz) in units of m_e c: pz = p/sqrt(1 + xp^2 + yp^2)."""
        pz = out["p"] / np.sqrt(1 + out["xp"] ** 2 + out["yp"] ** 2)
        return np.array([pz * out["xp"], pz * out["yp"], pz])


    def green_impulses(times):
        """Transverse and longitudinal impulses (m_e c) of a dipole mode on each particle.

        Mode: omega_r = 2 pi f, omega_d = omega_r sqrt(1 - 1/(4Q^2)), alpha = omega_r/(2Q). A source
        of charge q = CHARGE/N at offset x_s leaves the transverse voltage
        V_perp(t) = omega_r Rs/Q (omega_r/omega_d) q x_s exp(-alpha t) sin(omega_d t) (circuit
        impedance convention of the manual) and the longitudinal voltage
        V_z(t) = -(omega_d/c) x_probe amplitude exp(-alpha t) cos(omega_d t), with half the
        self-contribution at t = 0. Sums are over earlier particles, which sit in distinct bins.
        """
        wr = 2 * np.pi * FREQ
        wd = wr * np.sqrt(1 - 1 / (4 * QUALITY**2))
        alpha = wr / (2 * QUALITY)
        amplitude = wr * RS / QUALITY * (wr / wd) * CHARGE / N * OFFSET
        transverse, longitudinal = np.zeros(N), np.zeros(N)
        for j in range(N):
            delays = times[j] - times[:j]
            transverse[j] = amplitude * np.sum(np.exp(-alpha * delays) * np.sin(wd * delays))
            longitudinal[j] = (
                -wd
                / C
                * OFFSET
                * amplitude
                * (np.sum(np.exp(-alpha * delays) * np.cos(wd * delays)) + 0.5)
            )
        # Volts to m_e c: divide by the rest energy in eV.
        return transverse / (1e6 * ME_MEV), longitudinal / (1e6 * ME_MEV)


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in values) + "]"


    defect = False
    cases = (("TRFMODE", 0), ("TRFMODE", 1), ("FTRFMODE", 0))
    for mode in modes():
        for kind, binless in cases:
            for plane in ("x", "y"):
                for sign in (1, -1):
                    slopes = (sign * SLOPES[0], sign * SLOPES[1])
                    out = track(mode, kind, binless, plane, slopes)
                    # The mode excites each particle at its bin centre when binned, at its own time
                    # when binless. The tracked time is the arrival time before the kick.
                    times = out["t"].copy()
                    if not binless:
                        low = times.mean() - DT * BINS / 2
                        width = DT
                        index = np.floor((times - low) / width).astype(int)
                        if len(np.unique(index)) != N:
                            sys.exit("Two particles share a time bin; the oracle needs distinct bins.")
                        times = low + (index + 0.5) * width
                    transverse, longitudinal = green_impulses(times)
                    pz0 = P0 / np.sqrt(1 + slopes[0] ** 2 + slopes[1] ** 2)
                    incoming = np.array([slopes[0] * pz0, slopes[1] * pz0, pz0])[:, None]
                    physical = np.broadcast_to(incoming, (3, N)).copy()
                    physical[2] += longitudinal
                    physical[0 if plane == "x" else 1] += transverse
                    actual = momenta(out)
                    # The oracle must reproduce the tracked p_z before the disabled plane is judged.
                    pz_residual = np.max(np.abs(actual[2] - physical[2]))
                    if pz_residual > PZ_TOLERANCE:
                        sys.exit(f"{kind} binless={binless} plane={plane}: oracle disagrees on p_z.")
                    inactive = 1 if plane == "x" else 0
                    slope = slopes[inactive]
                    # The voltage does not depend on the disabled coordinate, so its momentum is
                    # conserved: expected change 0. The code adds slope * Delta p_z.
                    observed = actual[inactive] - incoming[inactive, 0]
                    spurious = slope * longitudinal
                    name = f"{mode} {kind} binless={binless} plane={plane} slope_sign={sign:+d}"
                    print(
                        f"{name}: observed last-particle disabled-plane dp={fmt(observed[-1:])} "
                        f"expected={fmt([0.0])} slope*dpz={fmt(spurious[-1:])} "
                        f"max|observed-slope*dpz|={np.max(np.abs(observed - spurious)):.3g} "
                        f"max|pz-oracle|={pz_residual:.3g}"
                    )
                    # Only a changed disabled-plane momentum needs the diagnosis; a corrected map
                    # conserves it and is a valid NOT REPRODUCED outcome.
                    bad = bool(np.max(np.abs(observed)) > MOMENTUM_TOLERANCE)
                    if bad and np.max(np.abs(observed - spurious)) > MOMENTUM_TOLERANCE:
                        sys.exit("The disabled-plane error does not follow slope * Delta p_z.")
                    defect |= bad
        # Control: with zero incoming slopes the disabled-plane momentum change is exactly zero.
        out = track(mode, "FTRFMODE", 0, "x", (0.0, 0.0))
        print(
            f"{mode} FTRFMODE zero incoming slopes: observed max|dpy|={np.max(np.abs(out['yp'] * out['p'])):.3g} "
            "expected=0"
        )
        if np.max(np.abs(out["yp"])) != 0:
            sys.exit("The zero-slope control changes the disabled plane.")
    print(f"comparison: tolerance {MOMENTUM_TOLERANCE:g} mₑc")
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

