---
title: "TRK-18: LSRMDLTR radiation loss scales with the cube of the reference momentum"
hide:
  - toc
---

# TRK-18 — LSRMDLTR radiation loss scales with the cube of the reference momentum

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`stochastic_laserModulator`, motion.cc:2413–2423](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/motion.cc#L2413-L2423)<br>[`radCoef`, motion.cc:213](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/motion.cc#L213)<br>[LSRMDLTR manual, lsrmdltr.tex:1–6](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/lsrmdltr.tex#L1-L6)<br>[`SYNCH_RAD` dictionary entry, track_data.c:3328](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3328) |

With classical radiation enabled in the LSRMDLTR laser modulator, an electron's calculated momentum loss changes when only the reference momentum changes. For particle momentum p and reference momentum p₀ in units of mₑc, the implemented local loss rate is (p₀/γ)³ times the classical rate, where γ = √(1 + p²) is the Lorentz factor. At p = 1200 in a 0.8 m, eight-period, 0.8 T undulator, classical loss on the same integration grid is 2.383×10⁻⁴ mₑc; tracking gives 2.979×10⁻⁵ at p₀ = 600, 2.312×10⁻⁴ at 1188 and 1.906×10⁻³ at 2400.

??? example "Reproducer: TRK-18.py"

    [Download TRK-18.py](TRK-18.py)

    ```python
    """TRK-18: LSRMDLTR radiation loss scales with the cube of the reference momentum

    With classical radiation enabled in the LSRMDLTR laser modulator, an electron's calculated momentum
    loss changes when only the reference momentum changes. For particle momentum p and reference
    momentum p₀ in units of mₑc, the implemented local loss rate is (p₀/γ)³ times the classical rate,
    where γ = √(1 + p²) is the Lorentz factor. At p = 1200 in a 0.8 m, eight-period, 0.8 T undulator,
    classical loss on the same integration grid is 2.383×10⁻⁴ mₑc; tracking gives 2.979×10⁻⁵ at p₀ =
    600, 2.312×10⁻⁴ at 1188 and 1.906×10⁻³ at 2400.

    The script tracks twelve identical electrons of momentum 1200 mₑc through an 8-period, 0.8 m
    LSRMDLTR with radiation on and off, changing only the run's reference momentum (600, 1188, 1200,
    1212, 2400 mₑc); the momentum loss is the difference of the two runs. It compares the loss with the
    classical radiation power summed on the integrator's own step grid, and with the same sum scaled by
    (p₀/γ)³, which explains every observed value. The run whose reference equals the particle momentum
    is the control.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from scipy.integrate import quad, solve_ivp

    # SDDS constants.h, as Elegant uses them.
    C, QE, ME, RE = 299792458.0, 1.60217733e-19, 9.1093897e-31, 2.81794092e-15
    P = 1200.0  # particle momentum, m_e c
    LENGTH, PERIODS, FIELD, STEPS = 0.8, 8, 0.8, 6400
    REFERENCES = (600.0, 1188.0, 1200.0, 1212.0, 2400.0)
    COPIES = 12
    RAD_OFF_ATOL = 5e-10  # momentum change with radiation off
    ORACLE_ATOL = 5e-11  # observed loss against the discrete oracle that follows the code's formula
    CONTROL_RTOL = 1e-5  # on-reference loss against the classical loss (beta^3 = 1 - 1.04e-6)

    KU = 2 * np.pi * PERIODS / LENGTH
    B_SCALE = QE * FIELD / (ME * C)  # eB0/(m c), 1/m
    GAMMA = np.sqrt(1 + P * P)
    AMPLITUDE = B_SCALE / KU  # peak of Px = p_x in the planar field, in m_e c


    def element(rad):
        # Explicit wavelength, zero laser power and no transverse gradient keep the physical field
        # independent of the reference momentum.
        return (
            f"LSRMDLTR,L={LENGTH},PERIODS={PERIODS},BU={FIELD},N_STEPS={STEPS},"
            'METHOD="non-adaptive runge-kutta",ACCURACY=1e-9,FIELD_EXPANSION="exact",'
            "LASER_WAVELENGTH=2e-6,LASER_PEAK_POWER=0,POLE_FACTOR1=1,POLE_FACTOR2=1,POLE_FACTOR3=1,"
            f"SYNCH_RAD={rad},ISR=0"
        )


    def tracked_momentum(mode, reference, rad):
        lattice = f"K: {element(rad)}\nBL: LINE=(K)\n"
        commands = (
            "&change_particle name=electron &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={reference!r}, "
            "output=out.sdds, random_number_seed=752913 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {"p": np.full(COPIES, P)}
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if len(out["p"]) != COPIES or not np.all(np.isfinite(out["p"])):
            sys.exit(f"{mode} reference={reference} rad={rad}: particles were lost or not finite")
        return float(out["p"][0])


    def orbit_grid():
        """Radiation-free orbit z(ct) and the integrator's time grid h = c*dt, as the code steps it.

        In the planar field B_y = B0 cos(ku z) the transverse momentum is P_x = A sin(ku z) with
        A = eB0/(m c ku), so dz/d(ct) = sqrt(p^2 - P_x^2)/gamma. The non-adaptive integrator advances
        by h = 1.5*L*gamma/(p*STEPS) in c*dt and drops the last fractional step that brackets z = L,
        so the sum over its grid, not the continuum integral, is what the code can be compared with.
        """
        h = 1.5 * LENGTH * GAMMA / P / STEPS
        orbit = solve_ivp(
            lambda t, z: [np.sqrt(P * P - (AMPLITUDE * np.sin(KU * z[0])) ** 2) / GAMMA],
            (0, 1.5 * LENGTH * GAMMA / P),
            [0.0],
            events=lambda t, z: z[0] - LENGTH,
            dense_output=True,
            method="DOP853",
            max_step=LENGTH / 500,
            rtol=2.3e-14,
            atol=2e-15,
        )
        ticks = np.arange(1, int(np.floor(orbit.t_events[0][0] / h)) + 1) * h
        return h, orbit.sol(ticks)[0]


    def classical_loss_continuum():
        """First-order classical loss of a particle on the radiation-free orbit.

        Lienard power for a purely magnetic force gives dp/dz = -(2 r_e/3) gamma^2 [eB(z)/(m c)]^2 p/P_z
        with P_z = sqrt(p^2 - P_x^2) and B(z) = B0 cos(ku z).
        """
        integral = quad(
            lambda z: np.cos(KU * z) ** 2 / np.sqrt(1 - (AMPLITUDE / P * np.sin(KU * z)) ** 2),
            0,
            LENGTH,
            epsabs=2e-14,
            epsrel=2e-14,
            limit=200,
        )[0]
        return (2 * RE / 3) * (1 + P * P) * B_SCALE**2 * integral


    def main():
        h, zs = orbit_grid()
        grid_sum = np.sum(np.cos(KU * zs) ** 2) * h  # sum of cos^2(ku z) c*dt over the integrator grid
        # Classical rate per c*dt: dp/d(ct) = -(2 r_e/3) gamma p [eB/(m c)]^2, summed over the grid.
        # The code replaces gamma by p0^3/gamma^2, so its loss is the classical one times (p0/gamma)^3.
        classical_grid = (2 * RE / 3) * GAMMA * P * B_SCALE**2 * grid_sum

        def implemented(reference):
            # Loss of the code's formula; the loss^2/(2p) term is the effect of the code applying
            # the loss as a factor (1 + delta) at every step.
            first = (2 * RE / 3) * reference**3 * P / GAMMA**2 * B_SCALE**2 * grid_sum
            return first + first**2 / (2 * P)

        print(f"classical loss along the orbit: continuum={classical_loss_continuum()!r}")
        print(f"classical loss on the integrator grid: {float(classical_grid)!r}")
        defect = False
        for mode in modes():
            for reference in REFERENCES:
                on = tracked_momentum(mode, reference, 1)
                off = tracked_momentum(mode, reference, 0)
                if abs(off - P) > RAD_OFF_ATOL:
                    sys.exit(f"{mode} reference={reference}: momentum changes with radiation off")
                observed = off - on
                print(
                    f"{mode} reference={reference:g}: observed=[{observed!r}] "
                    f"expected=[{float(classical_grid)!r}] code formula=[{float(implemented(reference))!r}]"
                )
                bad = abs(observed - classical_grid) > CONTROL_RTOL * classical_grid
                if not bad:
                    continue  # the loss does not depend on the reference momentum: correct
                if reference == P:
                    sys.exit("The on-reference control differs from the classical loss.")
                # A discrepant loss must follow the code's (p0/gamma)^3 scaling; anything else is a
                # failed run, not a verdict.
                if abs(observed - implemented(reference)) > ORACLE_ATOL:
                    sys.exit("The loss is not explained by the (p0/gamma)^3 scaling.")
                defect = True
        print(f"comparison: control rtol={CONTROL_RTOL:g}, atol(rad off)={RAD_OFF_ATOL:g}")
        verdict(defect)


    main()

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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |

