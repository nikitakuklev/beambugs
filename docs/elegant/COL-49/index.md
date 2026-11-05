---
title: "COL-49: LSC voltage generators omit the source-charge sign and reverse positron kicks"
hide:
  - toc
---

# COL-49 — LSC voltage generators omit the source-charge sign and reverse positron kicks

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant master (`0355dd6`) |
| Source | [`track_through_lscdrift`, lsc.c:319–322](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L319-L322)<br>[`addLSCKick`, lsc.c:616–619](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L616-L619)<br>[`track_through_lscdrift`, lsc.c:361–365](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L361-L365)<br>[`addLSCKick`, lsc.c:645–648](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lsc.c#L645-L648)<br>[`applyLongitudinalWakeKicks`, wake.c:348–360](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L348-L360)<br>[`process_particle_command`, elegant.c:3264–3281](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3264-L3281)<br>[`do_tracking`, do_tracking.c:894–897](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L894-L897)<br>[`track_through_wake`, wake.c:285–290](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L285-L290)<br>[`trackRfCavityAddLSCKick`, simple_rfca.c:404–414](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L404-L414)<br>[`track_through_driftCSR_Stupakov`, csbend.c:5211–5212](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L5211-L5212)<br>[`track_through_driftCSR_Stupakov`, csbend.c:5228–5229](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L5228-L5229) |

After `change_particle name=positron`, longitudinal space-charge kicks reverse in LSCDRIFT and in the space-charge contributions of RFCW and CSRDRIFT. Reversing all source and witness charges must preserve the self-force, but both voltage generators omit the source-charge sign required by the energy-update routine. For a 100.0 pC Gaussian bunch with normalized momentum βγ = 30.00, where β is speed divided by the speed of light and γ is the Lorentz factor, a 1.000 m effective LSCDRIFT kick gives a particle one rms bunch length behind the centre +2799 eV instead of −2799 eV. A zero-charge control gives no kick.

??? example "Reproducer: COL-49.py"

    [Download COL-49.py](COL-49.py)

    ```python
    """COL-49: LSC voltage generators omit the source-charge sign and reverse positron kicks

    After `change_particle name=positron`, longitudinal space-charge kicks reverse in LSCDRIFT and in the space-charge contributions of RFCW and CSRDRIFT. Reversing all source and witness charges must preserve the self-force, but both voltage generators omit the source-charge sign required by the energy-update routine. For a 100.0 pC Gaussian bunch with normalized momentum βγ = 30.00, where β is speed divided by the speed of light and γ is the Lorentz factor, a 1.000 m effective LSCDRIFT kick gives a particle one rms bunch length behind the centre +2799 eV instead of −2799 eV. A zero-charge control gives no kick.

    The script tracks deterministic Gaussian longitudinal quantiles with equal transverse rms sizes through a zero-length LSCDRIFT with a one-metre effective length. It reconstructs exact energy changes particle by particle, compares the two species by charge conjugation, and checks a zero-charge control. Use --mpi to add three-rank Pelegant, with the native programs on PATH and RPN_DEFNS set.
    """

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from scipy.special import ndtri

    C, REST_EV = 299792458.0, 510999.06
    N, P0, SIGMA_CT, SIGMA_R = 8192, 30.0, 1e-3, 1e-4
    CHARGE = 1e-10
    ATOL_EV = 1e-8


    def beam():
        z = SIGMA_CT * ndtri((np.arange(N) + 0.5) / N)
        phi = np.arange(N) * np.pi * (np.sqrt(5) - 1)
        x, y = np.cos(phi), np.sin(phi)
        return {
            "x": (x - x.mean()) / x.std() * SIGMA_R,
            "y": (y - y.mean()) / y.std() * SIGMA_R,
            "t": z / C,
            "p": np.full(N, P0),
            "particleID": np.arange(1, N + 1),
        }


    def case(species, charge, mode):
        lattice = f"""Q: CHARGE,TOTAL={charge:.17g}
    SC: LSCDRIFT,L=0,LEFFECTIVE=1,BINS=256,INTERPOLATE=1,SMOOTHING=0
    BL: LINE=(Q,SC)
    """
        commands = f"""&global_settings echo_namelists=0 &end
    &change_particle name={species} &end
    &run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0},output=out.sdds &end
    &run_control &end
    &sdds_beam input=beam.sdds &end
    &track &end
    """
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam()})["out.sdds"]
        )
        if not np.array_equal(out["particleID"], np.arange(1, N + 1)):
            raise RuntimeError("Unexpected particle loss")
        return (np.hypot(out["p"], 1) - np.hypot(P0, 1)) * REST_EV


    def main():
        defect = True
        z = C * beam()["t"]
        for mode in modes():
            electron = case("electron", CHARGE, mode)
            positron = case("positron", CHARGE, mode)
            zero = case("positron", 0, mode)
            # q_w E_source ~ q_w q_source: reversing both leaves Delta E identical.
            # This equality holds for the implemented periodic FFT model as well as for an
            # isolated bunch. It neither assumes a different boundary condition nor relies on
            # p0 approximately equalling gamma in the impedance.
            error = np.max(np.abs(positron - electron))
            reverse_error = np.max(np.abs(positron + electron))
            print(
                f"{mode} +1 sigma energy_change_eV: positron observed={np.interp(SIGMA_CT, z, positron):.15g} "
                f"expected={np.interp(SIGMA_CT, z, electron):.15g} (electron, charge conjugation)"
            )
            print(
                f"{mode} max_particle_energy_difference_eV={error:.15g} expected=0; "
                f"max_sign_reversal_residual_eV={reverse_error:.15g}"
            )
            print(
                f"{mode} zero-charge control: max_energy_change_eV observed={np.max(abs(zero)):.15g} expected=0"
            )
            if max(abs(electron)) <= ATOL_EV:
                raise RuntimeError("Electron control has no resolved LSC kick")
            if (error > ATOL_EV and reverse_error > ATOL_EV) or max(abs(zero)) > ATOL_EV:
                raise RuntimeError("Controls or sign-reversal prediction failed")
            defect &= error > ATOL_EV
        print(f"comparison: rtol=0, atol={ATOL_EV:g} eV; energy=mc^2 sqrt(1+p^2)")
        verdict(defect)


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
| Found | `gpt-6-astra` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

