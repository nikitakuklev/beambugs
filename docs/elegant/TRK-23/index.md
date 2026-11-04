---
title: "TRK-23: CSBEND dipole-only strength error rescales its octupole and higher multipoles"
hide:
  - toc
---

# TRK-23 — CSBEND dipole-only strength error rescales its octupole and higher multipoles

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`track_through_csbend`, csbend.c:1029–1081](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1029-L1081)<br>[curvature radius, csbend.c:1111–1116](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1111-L1116)<br>[body kicks, csbend.c:1921–1927](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1921-L1927)<br>[`computeCSBENDFieldCoefficients`, csbend.c:355–359](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L355-L359)<br>[parameter metadata, track_data.c:1961–1962](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1961-L1962)<br>[manual, csbend.tex:26–32](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L26-L32) |

A CSBEND using FSE_DIPOLE to change only the dipole field should leave higher multipole strengths scaled solely by the common error FSE. Orders 3–8 instead receive an extra factor 1 + FSE_DIPOLE/(1 + FSE), while quadrupole and sextupole terms are compensated. For a 0.6 m, 0.09 rad bend with octupole strength K₃ = −3000 m⁻⁴, FSE = 0.07 and FSE_DIPOLE = 0.04, an electron of momentum 49 mₑc exits with horizontal slope x′ = −3.610×10⁻³ instead of −3.700×10⁻³, an error of 8.984×10⁻⁵. Independent Lorentz integration and an equivalent common-error configuration establish the expected trajectory.

??? example "Reproducer: TRK-23.py"

    [Download TRK-23.py](TRK-23.py)

    ```python
    """TRK-23: CSBEND dipole-only strength error rescales its octupole and higher multipoles

    A CSBEND using FSE_DIPOLE to change only the dipole field should leave higher multipole strengths
    scaled solely by the common error FSE. Orders 3–8 instead receive an extra factor 1 + FSE_DIPOLE/(1
    + FSE), while quadrupole and sextupole terms are compensated. For a 0.6 m, 0.09 rad bend with
    octupole strength K₃ = −3000 m⁻⁴, FSE = 0.07 and FSE_DIPOLE = 0.04, an electron of momentum 49 mₑc
    exits with horizontal slope x′ = −3.610×10⁻³ instead of −3.700×10⁻³, an error of 8.984×10⁻⁵.
    Independent Lorentz integration and an equivalent common-error configuration establish the expected
    trajectory.

    The script tracks 12 electrons through the bend, once with the dipole error as FSE_DIPOLE and once
    with the same physical field written as FSE alone and a rescaled multipole strength, and compares
    both with an integration of the Lorentz equations in the laboratory frame. Controls are the
    equivalent FSE-only run, a zero dipole error, and K1 and K2, which the code compensates. A further
    check shows that the result equals the oracle with the multipole strength raised by the excess
    factor.
    """

    import sys
    from math import factorial

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from _multipole import P0, beam
    from scipy.integrate import solve_ivp

    LENGTH, ANGLE = 0.6, 0.09
    ATOL = 2e-12  # fourth-order integration with 384 slices; the correct field agrees to ~3e-13
    SPLIT = 1e-8  # the smallest tested defect produces a slope error of 2.555e-6, well above SPLIT=1e-8
    P_ATOL = 2e-13  # no radiation, so the momentum is conserved
    # (order n, K_n, FSE, FSE_DIPOLE); n = 1, 2 and zero dipole error are the controls.
    CASES = [
        (3, -3000.0, 0.07, 0.04),
        (3, -3000.0, 0.07, -0.04),
        (5, 1e8, 0.07, 0.04),
        (3, -3000.0, 0.0, 0.001),
        (3, -3000.0, 0.07, 0.0),
        (1, -0.8, 0.07, 0.04),
        (2, 30.0, 0.07, -0.04),
    ]


    class ExitPlane:
        """Event: the particle reaches the plane through the end of the reference arc."""

        terminal = True
        direction = 1

        def __init__(self, reference, tangent):
            self.reference, self.tangent = reference, tangent

        def __call__(self, _, state):
            return (state[:2] - self.reference) @ self.tangent


    def oracle(initial, n, k, fse, dipole):
        """Exit (x, xp) of each particle in the specified field, integrated in the laboratory frame.

        The orbit (X, Z, alpha) is followed against path length s in the plane of a circle of radius
        rho = L/angle centred at (-rho, 0): dX/ds = sin(alpha), dZ/ds = cos(alpha),
        dalpha/ds = -B_y p0/p, where B_y in units of the reference rigidity is
        B_y = (1 + FSE + FSE_DIPOLE)/rho + (1 + FSE) K_n r^n/n!, with r the radial offset from the
        reference arc. FSE scales every component; FSE_DIPOLE only the dipole. The orbit stops at the
        radial plane through the end of the reference arc, and x is the offset along that plane's
        normal, xp = tan(alpha + angle).
        """
        rho = LENGTH / ANGLE
        normal = np.array([np.cos(ANGLE), np.sin(ANGLE)])
        tangent = np.array([-np.sin(ANGLE), np.cos(ANGLE)])
        reference = rho * np.array([np.cos(ANGLE) - 1, np.sin(ANGLE)])
        result = []
        for x, xp, p in zip(initial["x"], initial["xp"], initial["p"], strict=True):

            def rhs(_, state, ratio=p / P0):
                xx, zz, alpha = state
                radial = np.hypot(xx + rho, zz) - rho
                by = (1 + fse + dipole) / rho + (1 + fse) * k * radial**n / factorial(n)
                return [np.sin(alpha), np.cos(alpha), -by / ratio]

            sol = solve_ivp(
                rhs,
                (0, 1.5 * LENGTH),
                [x, 0.0, np.arctan(xp)],
                events=ExitPlane(reference, tangent),
                method="DOP853",
                rtol=3e-13,
                atol=2e-15,
                max_step=0.02,  # resolve the exit-plane crossing
            )
            if not (sol.success and len(sol.t_events[0]) == 1):
                raise RuntimeError("laboratory-frame integration failed")
            xx, zz, alpha = sol.y_events[0][0]
            result.append([(np.array([xx, zz]) - reference) @ normal, np.tan(alpha + ANGLE)])
        return np.array(result)


    def track(mode, initial, n, k, fse, dipole):
        lattice = (
            f"B: CSBEND,L={LENGTH},ANGLE={ANGLE},K{n}={k!r},FSE={fse!r},FSE_DIPOLE={dipole!r},"
            "N_SLICES=384,INTEGRATION_ORDER=4,EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,SYNCH_RAD=0,ISR=0,"
            "REFERENCE_CORRECTION=0,EXPANSION_ORDER=10\n"
            "BL: LINE=(B)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0}, output=out.sdds, "
            "random_number_seed=753219 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        files = {"lattice.lte": lattice, "run.ele": commands}
        out = particles(run(files, mode, {"beam.sdds": initial})["out.sdds"])
        if len(out["p"]) != len(initial["p"]):
            sys.exit(f"{mode}: particles were lost")
        if np.max(np.abs(out["y"])) > 0 or np.max(np.abs(out["yp"])) > 0:
            sys.exit(f"{mode}: the midplane beam left the midplane")
        if not np.allclose(out["p"], initial["p"], rtol=0, atol=P_ATOL, equal_nan=False):
            sys.exit(f"{mode}: the momentum is not conserved")
        return np.column_stack((out["x"], out["xp"]))


    def compare(label, observed, expected):
        """Print the last particle's x, xp and the largest difference over the beam.

        Nonfinite values make the comparison invalid: the run terminates instead of returning a NaN
        that every later `> ATOL` test would accept.
        """
        if not (np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))):
            sys.exit(f"{label}: a nonfinite value makes the comparison invalid")
        error = float(np.max(np.abs(observed - expected)))
        print(
            f"{label}: observed={[float(v) for v in observed[-1]]} "
            f"expected={[float(v) for v in expected[-1]]} max|diff|={error:.4g}"
        )
        return error


    defect = False
    initial = beam() | {"y": np.zeros(12), "yp": np.zeros(12)}  # midplane beam
    for mode in modes():
        for n, k, fse, dipole in CASES:
            label = f"{mode} K{n}={k:g} FSE={fse:g} FSE_DIPOLE={dipole:g}"
            expected = oracle(initial, n, k, fse, dipole)
            component = track(mode, initial, n, k, fse, dipole)
            # The same field with the dipole error folded into FSE: B_y/rigidity has the same dipole
            # term (1 + g + d)/rho, and the multipole (1 + g) K_n becomes (1 + g + d) K_n'.
            k_equivalent = k * (1 + fse) / (1 + fse + dipole)
            equivalent = track(mode, initial, n, k_equivalent, fse + dipole, 0.0)
            if compare(f"{label} equivalent FSE-only", equivalent, expected) > ATOL:
                sys.exit(f"{label}: the equivalent-field control disagrees with the oracle")
            error = compare(label, component, expected)
            if n <= 2 or dipole == 0:
                if error > ATOL:
                    sys.exit(f"{label}: the control disagrees with the oracle")
                continue
            # What the code does: the multipole is multiplied by (1 + g + d)/(1 + g).
            excess = oracle(initial, n, k * (1 + fse + dipole) / (1 + fse), fse, dipole)
            if not np.all(np.isfinite(excess)):
                sys.exit(f"{label}: the excess-field oracle is not finite")
            if not np.allclose(component, excess, rtol=0, atol=ATOL, equal_nan=False):
                sys.exit(f"{label}: the result is not the excess-field prediction")
            defect |= error > SPLIT
    print(f"comparison: atol={ATOL:g} for agreement, {SPLIT:g} for disagreement")
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

??? example "Helper: _multipole.py"

    [Download _multipole.py](_multipole.py)

    ```python
    """Shared setup for the KQUAD and KSEXT multipole-table reproducers: a 12-electron beam, inline
    multipole tables, and an independent integrator for a straight magnet with steering fields.

    The integrator follows the particles through a straight hard-edge magnet in canonical transverse
    momenta q = p_perp/p0, solving the Lorentz equations with scipy, without any Elegant code.
    """

    from __future__ import annotations

    from math import factorial

    import numpy as np
    from scipy.integrate import solve_ivp

    P0 = 45.0  # reference momentum, units of m_e c
    N = 12  # particles, enough for both workers of a 3-rank run
    RADIUS = 0.027  # m, reference radius of the multipole tables


    def beam() -> dict[str, np.ndarray]:
        """Twelve electrons with transverse offsets and slopes and momenta 43 to 49 m_e c."""
        return {
            "x": np.linspace(-0.018, 0.021, N),
            "xp": np.linspace(0.003, -0.004, N),
            "y": np.linspace(0.008, -0.011, N),
            "yp": np.linspace(-0.002, 0.003, N),
            "t": np.zeros(N),
            "p": np.linspace(43.0, 49.0, N),
        }


    def table_text(orders, normal, skew=None) -> str:
        """ASCII SDDS multipole table: parameter referenceRadius, columns order, normal, (skew)."""
        columns = "&column name=order, type=long &end\n&column name=normal, type=double &end\n"
        if skew is not None:
            columns += "&column name=skew, type=double &end\n"
        rows = "".join(
            f"{n} {a!r}" + (f" {b!r}" if skew is not None else "") + "\n"
            for n, a, b in zip(orders, normal, skew if skew is not None else normal, strict=True)
        )
        return (
            "SDDS1\n"
            '&parameter name=referenceRadius, type=double, units="m" &end\n'
            + columns
            + "&data mode=ascii &end\n"
            + f"{RADIUS!r}\n{len(orders)}\n"
            + rows
        )


    def canonical(b: dict[str, np.ndarray]) -> np.ndarray:
        """Rows x, y, qx, qy with q = (p/p0) * slope / sqrt(1 + xp^2 + yp^2)."""
        ratio = b["p"] / P0
        qs = ratio / np.sqrt(1 + b["xp"] ** 2 + b["yp"] ** 2)
        return np.stack((b["x"], b["y"], qs * b["xp"], qs * b["yp"]))


    def observable(state: np.ndarray, b: dict[str, np.ndarray]) -> np.ndarray:
        """Columns x, xp, y, yp from canonical rows x, y, qx, qy."""
        x, y, qx, qy = state
        qs = np.sqrt((b["p"] / P0) ** 2 - qx**2 - qy**2)
        return np.column_stack((x, qx / qs, y, qy / qs))


    def straight_body(
        state: np.ndarray,
        b: dict[str, np.ndarray],
        main_order: int,
        strength: float,
        length: float,
        factor: float = 0.0,
        kick: tuple[float, float] = (0.0, 0.0),
        harmonics=((2, 0.07), (4, -0.025)),
    ) -> np.ndarray:
        """Canonical rows after a straight magnet of the given length (m).

        Complex field C = By + iBx divided by the reference rigidity, with z = x + iy:
        C = K_m z^m/m! - (H - iV)/L - factor/L sum_n a_n (H - i(-1)^(n/2) V) (z/R)^n,
        where K_m is `strength`, (H, V) = `kick` the steering angles spread over the length and a_n
        the normal steering fractions at order n (zero steering and zero factor leave the main field).
        The equations of motion are dx/ds = qx/qs, dy/ds = qy/qs, dqx/ds = -Re C, dqy/ds = Im C with
        qs = sqrt((p/p0)^2 - qx^2 - qy^2).
        """
        ratio = b["p"] / P0

        def rhs(_, flat):
            x, y, qx, qy = flat.reshape(4, N)
            z = x + 1j * y
            field = strength * z**main_order / factorial(main_order)
            field -= (kick[0] - 1j * kick[1]) / length
            for n, a in harmonics:
                coefficient = -factor * a * (kick[0] - 1j * (-1) ** (n // 2) * kick[1])
                field += coefficient / length * (z / RADIUS) ** n
            qs = np.sqrt(ratio**2 - qx**2 - qy**2)
            return np.stack((qx / qs, qy / qs, -field.real, field.imag)).ravel()

        sol = solve_ivp(rhs, (0, length), state.ravel(), method="DOP853", rtol=3e-13, atol=2e-15)
        if not sol.success:
            raise RuntimeError("straight-magnet integration failed")
        return sol.y[:, -1].reshape(4, N)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

