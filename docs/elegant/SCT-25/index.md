---
title: "SCT-25: touschek_scatter converts normalized emittance with γ instead of βγ"
hide:
  - toc
---

# SCT-25 — touschek_scatter converts normalized emittance with γ instead of βγ

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant master (`0355dd6`) |
| Source | [`TouschekEffect`, touschekScatter.c:97–102](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L97-L102)<br>[`TouschekRate`, touschekScatter.c:203–225](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L203-L225)<br>[`TouschekRate`, touschekScatter.c:238–242](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L238-L242)<br>[`touschek_scatter`, elegant.tex:7763–7765](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7763-L7765)<br>[`exact_normalized_emittance`, elegant.tex:4143–4145](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4143-L4145)<br>[`touschek_scatter`, elegant.tex:7693–7697](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7693-L7697) |

When normalized transverse emittances are supplied to touschek_scatter at constant energy, their conversion uses the Lorentz factor γ instead of βγ, where β is speed divided by the speed of light. The inferred geometric emittances are therefore too small by a factor β, so equivalent input descriptions give different Piwinski rates. For a 1 pC electron bunch at momentum p = 3mc, where m is the electron mass and c is the speed of light, the rate is 6.561×10¹⁰ s⁻¹ instead of 6.295×10¹⁰ s⁻¹, a 4.223% excess. Independent integration validates the geometric-input rate. The emittance error falls below 0.1% for p > 22.34mc.

??? example "Reproducer: SCT-25.py"

    [Download SCT-25.py](SCT-25.py)

    ```python
    """SCT-25: touschek_scatter converts normalized emittance with γ instead of βγ

    When normalized transverse emittances are supplied to touschek_scatter at constant energy, their conversion uses the Lorentz factor γ instead of βγ, where β is speed divided by the speed of light. The inferred geometric emittances are therefore too small by a factor β, so equivalent input descriptions give different Piwinski rates. For a 1 pC electron bunch at momentum p = 3mc, where m is the electron mass and c is the speed of light, the rate is 6.561×10¹⁰ s⁻¹ instead of 6.295×10¹⁰ s⁻¹, a 4.223% excess. Independent integration validates the geometric-input rate. The emittance error falls below 0.1% for p > 22.34mc.

    The script runs serial Elegant on two TSCATTER sites separated by a 1 m drift,
    with fixed Gaussian beam parameters and a momentum aperture ten times the rms spread.
    It compares geometric and equivalent normalized inputs at three momenta, checks the
    geometric rates by independent quadrature, and identifies the erroneous conversion
    using two rescaled inputs. Only the deterministic Piwinski rates enter the verdict.
    """

    import os

    # Cap numerical threads before importing NumPy/SciPy, as well as in the native runner.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"

    import numpy as np
    from _elegant import run, verdict
    from scipy.integrate import quad
    from scipy.special import i0e

    EX, EY = 1e-7, 1e-8  # m, geometric transverse emittances
    SIGMA_S, SIGMA_DP, APERTURE = 0.01, 1e-5, 1e-4  # m, fractional rms spread, fractional half-aperture
    CHARGE = 1e-12  # C
    BX0, BY0, LENGTH = 10.0, 5.0, 1.0  # m; initial alpha functions and dispersion are zero
    # Match the native physical constants so their rounding cannot mimic an input-convention error.
    C, E_CHARGE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15  # m/s, C, m


    def case(p0: float, kind: str) -> float:
        """Piwinski rate at the second TSCATTER; p0 is momentum in units of mc."""
        gamma = np.hypot(1.0, p0)
        beta = p0 / gamma
        emit = {
            "geometric": f"emit_x={EX:.17g},emit_y={EY:.17g}",
            "normalized": f"emit_nx={p0 * EX:.17g},emit_ny={p0 * EY:.17g}",
            "gamma_input": f"emit_nx={gamma * EX:.17g},emit_ny={gamma * EY:.17g}",
            "smaller_geometric": f"emit_x={beta * EX:.17g},emit_y={beta * EY:.17g}",
        }[kind]
        files = {
            "line.lte": f"T0: TSCATTER\nD: DRIF,L={LENGTH}\nT1: TSCATTER\nBL: LINE=(T0,D,T1)\n",
            "aperture.sdds": f"""SDDS1
    &column name=s,units=m,type=double &end
    &column name=deltaPositive,type=double &end
    &column name=deltaNegative,type=double &end
    &column name=ElementName,type=string &end
    &column name=ElementOccurence,type=long &end
    &column name=ElementType,type=string &end
    &data mode=ascii,no_row_counts=1 &end
    0 {APERTURE} {-APERTURE} T0 1 TSCATTER
    {LENGTH} {APERTURE} {-APERTURE} T1 1 TSCATTER
    """,
            "run.ele": f"""&run_setup lattice=line.lte,use_beamline=BL,p_central={p0:.17g},
    random_number_seed=987 &end
    &run_control &end
    &twiss_output matched=0,beta_x={BX0},beta_y={BY0} &end
    &touschek_scatter charge={CHARGE},{emit},sigma_dp={SIGMA_DP},sigma_s={SIGMA_S},
    Momentum_Aperture=aperture.sdds,Momentum_Aperture_scale=1,n_simulated=2000,
    ignored_portion=0,do_track=0,bunch="run.bun" &end
    """,
        }
        bunch = run(files, outputs=("run.bun",))["run.bun"]
        # The first TSCATTER begins the integration interval; the second stores the local rate.
        rate = float(bunch["PLocalRate"].data[1])
        if not np.isfinite(rate) or rate <= 0:
            raise RuntimeError(f"Invalid native rate for p0={p0}, {kind}: {rate}")
        return rate


    def piwinski(p0: float) -> float:
        """Zero-dispersion Piwinski total rate N/tau with finite beta retained (DESY 98-179)."""
        gamma = np.hypot(1.0, p0)
        beta = p0 / gamma
        # In a drift from alpha=0, beta(s)=beta(0)+s^2/beta(0).
        bx, by = BX0 + LENGTH**2 / BX0, BY0 + LENGTH**2 / BY0
        b1 = (bx / EX + by / EY) / (2 * p0**2)
        b2 = abs(bx / EX - by / EY) / (2 * p0**2)
        tm = (beta * APERTURE) ** 2

        def integrand(u):
            # tau=tm*exp(u), d(tau)=tau*du. The scaled Bessel function avoids overflow.
            t = tm * np.exp(u)
            bracket = (
                (2 + 1 / t) ** 2 * (t / tm / (1 + t) - 1)
                + 1
                - np.sqrt(tm * (1 + t) / t)
                - (4 + 1 / t) / (2 * t) * np.log(t / tm / (1 + t))
            )
            return t * bracket * np.sqrt(t / (1 + t)) * i0e(b2 * t) * np.exp(-(b1 - b2) * t)

        integral, error = quad(integrand, 0, 60, limit=500, epsabs=0, epsrel=1e-11)
        if not np.isfinite(integral) or integral <= 0 or error > 1e-8 * abs(integral):
            raise RuntimeError(f"Piwinski quadrature failed: integral={integral}, error={error}")
        population = CHARGE / E_CHARGE
        # One particle exits each side of the symmetric aperture, so the total is N/tau.
        return (
            RE**2 * C * population**2 * integral
            / (8 * np.sqrt(np.pi) * beta**2 * gamma**4 * EX * EY * SIGMA_S)
        )


    def main() -> None:
        defect = True
        for p0 in (3.0, 10.0, 100.0):
            rates = {
                kind: case(p0, kind)
                for kind in ("geometric", "normalized", "gamma_input", "smaller_geometric")
            }
            geometric, observed = rates["geometric"], rates["normalized"]
            independent = piwinski(p0)
            control_error = geometric / independent - 1
            print(
                f"p={p0:g} mc geometric control: observed={geometric:.12g} /s; "
                f"independent Piwinski={independent:.12g} /s; relative difference={control_error:.12g}"
            )
            if abs(control_error) > 5e-8:
                raise RuntimeError("Geometric control disagrees with independent Piwinski quadrature")
            # Multiplying x' by p0 multiplies the 2D covariance determinant by p0^2:
            # epsilon_n=sqrt(det Cov(x,p0*x'))=p0*epsilon. These inputs describe the same beam.
            relative_error = observed / geometric - 1
            print(
                f"p={p0:g} mc normalized input: observed={observed:.12g} /s; "
                f"expected={geometric:.12g} /s (same beam, epsilon_n=p0*epsilon); "
                f"relative error={100 * relative_error:.12g}%"
            )
            gamma_residual = rates["gamma_input"] / geometric - 1
            beta_residual = rates["smaller_geometric"] / observed - 1
            print(
                f"p={p0:g} mc conversion diagnosis: R(gamma*epsilon)/R(geometric)-1="
                f"{gamma_residual:.12g}; R(beta*epsilon)/R(normalized)-1={beta_residual:.12g}; "
                "expected=0 for both if gamma is used instead of p0"
            )
            # The deliberately wrong inputs identify the defect; they are not physical controls.
            # Their failure on a corrected executable should give NOT REPRODUCED, not a run failure.
            defect &= abs(gamma_residual) < 2e-14 and abs(beta_residual) < 2e-14
            defect &= abs(relative_error) > (1e-3 if p0 < 100 else 1e-6)

        # Inferred/physical geometric emittance = beta; solve 1-beta < 0.001 exactly.
        threshold = 0.999 / np.sqrt(1 - 0.999**2)
        print(
            f"emittance error 1-beta < 0.1% for p/(mc) > {threshold:.12g}; "
            "this is not a bound on the rate error"
        )
        print("comparison: quadrature rtol=5e-8; conversion identities rtol=2e-14")
        verdict(bool(defect))


    if __name__ == "__main__":
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
| Found (C4, constant-energy Piwinski rates, serial only) | `gpt-6-astra` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 5-rank Pelegant |
| Confirmed (C4, constant-energy Piwinski rates, serial only) | `gpt-6.1-sol` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

