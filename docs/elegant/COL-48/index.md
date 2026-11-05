---
title: "COL-48: CSRCSBEND and CSRDRIFT reverse self-CSR kicks for positrons"
hide:
  - toc
---

# COL-48 — CSRCSBEND and CSRDRIFT reverse self-CSR kicks for positrons

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant master (`0355dd6`) |
| Source | [`track_through_csbendCSR`, csbend.c:2762–2764](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2762-L2764)<br>[`track_through_csbendCSR`, csbend.c:3132–3135](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3132-L3135)<br>[`track_through_csbendCSR`, csbend.c:3545–3551](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3545-L3551)<br>[`track_through_csbendCSR`, csbend.c:3467–3471](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3467-L3471)<br>[`track_through_csbendCSR`, csbend.c:3565–3568](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3565-L3568)<br>[`track_through_csbendCSR`, csbend.c:3613–3617](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3613-L3617)<br>[`process_particle_command`, elegant.c:3264–3281](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3264-L3281)<br>[`do_tracking`, do_tracking.c:894–897](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L894-L897)<br>[`track_through_csbendCSR`, csbend.c:4007–4010](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L4007-L4010)<br>[`track_through_csbendCSR`, csbend.c:3968–3969](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3968-L3969)<br>[`track_through_driftCSR`, csbend.c:4530–4535](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L4530-L4535)<br>[`track_through_driftCSR_Stupakov`, csbend.c:5114–5118](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L5114-L5118)<br>[`track_through_driftCSR_Stupakov`, csbend.c:5152–5154](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L5152-L5154) |

After `change_particle name=positron`, CSRCSBEND reverses the coherent synchrotron radiation kick, making a bunch gain energy where an identical electron bunch loses it. Reversing every source and witness charge must preserve the self-force because both the source field and the witness charge reverse. A 100.0 pC Gaussian bunch at 511.0 MeV/c with 35.00 μm rms length gains 104.7 keV per particle on average in a 0.6000 m, 0.3000 rad bend instead of losing 104.7 keV. Disabling coherent radiation restores identical electron and positron transport.

??? example "Reproducer: COL-48.py"

    [Download COL-48.py](COL-48.py)

    ```python
    """COL-48: CSRCSBEND and CSRDRIFT reverse self-CSR kicks for positrons

    After `change_particle name=positron`, CSRCSBEND reverses the coherent synchrotron radiation kick, making a bunch gain energy where an identical electron bunch loses it. Reversing every source and witness charge must preserve the self-force because both the source field and the witness charge reverse. A 100.0 pC Gaussian bunch at 511.0 MeV/c with 35.00 μm rms length gains 104.7 keV per particle on average in a 0.6000 m, 0.3000 rad bend instead of losing 104.7 keV. Disabling coherent radiation restores identical electron and positron transport.

    The script tracks deterministic Gaussian quantiles through a single-slice steady-state CSR bend for electrons and positrons, with coherent radiation enabled and disabled. It compares every momentum kick using charge conjugation and checks a zero-charge control; incoherent radiation and random excitation are disabled. Use --mpi to add three-rank Pelegant, with the native programs on PATH and RPN_DEFNS set.
    """

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from scipy.special import ndtri

    P0 = 1000.0  # Absolute normalized momentum p/(m_e c), equal for electron and positron.
    N = 32768
    SIGMA_CT = 35e-6
    REST_EV = 510999.06
    C = 299792458.0
    CHARGE = 1e-10  # CHARGE requires unsigned magnitude, including for positrons.
    ATOL_P = 5e-12  # Absolute p/(m_e c); approx 2.6 micro-eV per particle.


    def case(species, csr, charge, mode):
        beam = {
            "t": SIGMA_CT / C * ndtri((np.arange(N) + 0.5) / N),
            "p": np.full(N, P0),
            "particleID": np.arange(1, N + 1),
        }
        lattice = f"""Q: CHARGE,TOTAL={charge:.17g}
    B: CSRCSBEND,L=0.6,ANGLE=0.3,N_SLICES=1,INTEGRATION_ORDER=6,BINS=256,STEADY_STATE=1,CSR={csr},SYNCH_RAD=0,ISR=0,SG_HALFWIDTH=1,SGDERIV_HALFWIDTH=3,SGDERIV_ORDER=3
    BL: LINE=(Q,B)
    """
        commands = f"""&global_settings echo_namelists=0 &end
    &change_particle name={species} &end
    &run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0},output=out.sdds &end
    &run_control &end
    &sdds_beam input=beam.sdds &end
    &track &end
    """
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if not np.array_equal(out["particleID"], np.arange(1, N + 1)):
            raise RuntimeError("Unexpected particle loss")
        return out


    def mean_energy_change(on, off):
        # Reconstruct exact energy E = mc^2 sqrt(1+p^2), rather than treating momentum as energy.
        return np.mean(np.hypot(on["p"], 1) - np.hypot(off["p"], 1)) * REST_EV


    def main():
        defect = True
        for mode in modes():
            electron_off = case("electron", 0, CHARGE, mode)
            positron_off = case("positron", 0, CHARGE, mode)
            for name in ("x", "xp", "y", "yp", "t", "p"):
                if not np.array_equal(electron_off[name], positron_off[name]):
                    raise RuntimeError("CSR-off transport differs between species")
            print(f"{mode} CSR-off species control: max_coordinate_difference observed=0 expected=0")
            electron = case("electron", 1, CHARGE, mode)
            positron = case("positron", 1, CHARGE, mode)
            zero = case("positron", 1, 0.0, mode)
            if not np.allclose(zero["p"], positron_off["p"], rtol=0, atol=ATOL_P):
                raise RuntimeError("Zero-charge control failed")
            print(
                f"{mode} zero-charge control: max_delta_p observed={max(abs(zero['p'] - positron_off['p'])):.12g} expected=0"
            )
            electron_kick = electron["p"] - electron_off["p"]
            positron_kick = positron["p"] - positron_off["p"]
            # Maxwell fields are linear in source charge. Self-force is q*E_source ~ q^2:
            # simultaneous charge reversal preserves the entire kick. The single CSR slice
            # ensures both species deposit the same pre-kick density; the cold reference orbit
            # and fixed normalized bend angle are identical, as the off controls establish.
            error = max(abs(positron_kick - electron_kick))
            reverse_error = max(abs(positron_kick + electron_kick))
            print(
                f"{mode} mean_CSR_delta_p: positron observed={np.mean(positron_kick):.15g} "
                f"expected={np.mean(electron_kick):.15g} (equal electron self-field kick)"
            )
            print(
                f"{mode} mean_CSR_energy_change_eV: positron observed={mean_energy_change(positron, positron_off):.15g} "
                f"expected={mean_energy_change(electron, electron_off):.15g}"
            )
            print(
                f"{mode} max_particle_kick_difference={error:.15g} expected=0; "
                f"max_particle_sign_reversal_residual={reverse_error:.15g}"
            )
            if np.mean(electron_kick) >= 0 or max(abs(electron_kick)) <= ATOL_P:
                raise RuntimeError("Electron control does not have a resolved CSR loss")
            if error > ATOL_P and reverse_error > ATOL_P:
                raise RuntimeError("The identified sign-reversal cause does not explain the result")
            defect &= error > ATOL_P
        print(f"comparison: rtol=0, atol={ATOL_P:g} in p/(m_e c); species controls are bit-identical")
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

