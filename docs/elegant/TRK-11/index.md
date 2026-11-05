---
title: "TRK-11: SOLE given by its field B reverses the spin precession of positively charged particles"
hide:
  - toc
---

# TRK-11 — SOLE given by its field B reverses the spin precession of positively charged particles

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`updateSpinForSolenoid`, spinUpdate.c:192–206](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/spinUpdate.c#L192-L206)<br>[charge argument, spinUpdate.c:204](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/spinUpdate.c#L204)<br>[kernel contract, spinUpdate.c:4–19](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/spinUpdate.c#L4-L19)<br>[angular velocity, spinUpdate.c:98–112](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/spinUpdate.c#L98-L112)<br>[`process_particle_command`, elegant.c:3252–3308](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3252-L3308)<br>[SOLE B passed unchanged, spinUpdate.c:194–195](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/spinUpdate.c#L194-L195)<br>[parameter B, track_data.c:556](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L556)<br>[KS branch, spinUpdate.c:197](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/spinUpdate.c#L197)<br>[B to KS for tracking, do_tracking.c:2358–2382](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2358-L2382)<br>[B to KS for the matrix, compute_matrices.c:1552–1557](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1552-L1557)<br>[spin_tracking, elegant.tex:7291–7292](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7291-L7292)<br>[spin-aware elements, elegant.tex:11191–11192](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L11191-L11192) |

With spin tracking enabled, a SOLE solenoid specified by magnetic field B rotates the transverse spin of on-axis positrons and protons in the opposite sense to the Thomas–BMT prediction. For a positron with normalized momentum βγ = 100 in a 0.5000 m solenoid at B = 0.1000 T, initial spin (1, 0, 0) gives vertical spin component sᵧ = +0.2895 instead of −0.2895. Giving the same field as solenoid strength KS agrees with the analytic result, as does the electron control.

??? example "Reproducer: TRK-11.py"

    [Download TRK-11.py](TRK-11.py)

    ```python
    """TRK-11: SOLE given by its field B reverses the spin precession of positively charged particles

    With spin tracking enabled, a SOLE solenoid specified by magnetic field B rotates the transverse
    spin of on-axis positrons and protons in the opposite sense to the Thomas–BMT prediction. For a
    positron with normalized momentum βγ = 100 in a 0.5000 m solenoid at B = 0.1000 T, initial spin
    (1, 0, 0) gives vertical spin component sᵧ = +0.2895 instead of −0.2895. Giving the same field as
    solenoid strength KS agrees with the analytic result, as does the electron control.

    The script sends on-axis particles with spin (1, 0, 0) through a 0.5 m SOLE with B = ±0.1 T, with
    spin tracking on, and compares the final spin with the exact precession angle in a uniform
    longitudinal field. The controls are the electron, whose sign convention is consistent, and the
    positron in the same lattice with the field given as the equivalent solenoid strength KS instead
    of B, which takes a different path in the code and agrees with the exact angle.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h, and the species table of Elegant's change_particle command.
    C, QE, ME = 299792458.0, 1.60217733e-19, 9.1093897e-31
    MP, AE, AP = 1.6726485e-27, 0.00115965218128, 1.792847
    SPECIES = {  # name: (physical charge in C, mass in kg, anomalous magnetic moment)
        "electron": (-QE, ME, AE),
        "positron": (QE, ME, AE),
        "proton": (QE, MP, AP),
    }
    LENGTH, MOMENTUM = 0.5, 100.0  # m, beta*gamma of the beam and of the reference
    COPIES = 6  # identical particles, so that each MPI worker tracks some
    ATOL = 1e-13


    def precession_angle(species, field):
        """Spin rotation angle about +z for a particle on the axis of a uniform longitudinal B.

        The Thomas-BMT equation with the velocity parallel to B reduces to
        dS/dt = -(q/m)(1 + a) B / gamma  S_z-hat x S, because the longitudinal field is the same in
        the particle frame and the laboratory time is gamma times the proper time. With
        dt = gamma dz/(beta gamma c) the angle after length L is
        theta = -q (1 + a) B L / (m c beta gamma), with the signed physical charge q.
        """
        charge, mass, anomaly = SPECIES[species]
        return -charge * (1 + anomaly) * field * LENGTH / (mass * C * MOMENTUM)


    def final_spin(mode, species, element):
        """Exit spin (spx, spy, spz) of the particles that start with spin (1, 0, 0) on the axis."""
        lattice = f"S: {element}\nBL: LINE=(S)\n"
        commands = (
            f"&change_particle name={species} &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={MOMENTUM!r}, "
            "output=out.sdds, spin_tracking=1, default_order=2 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        beam = {
            "p": np.full(COPIES, MOMENTUM),
            "spx": np.ones(COPIES),
            "spy": np.zeros(COPIES),
            "spz": np.zeros(COPIES),
        }
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})
        out = particles(result["out.sdds"])
        if len(out["p"]) != COPIES:
            sys.exit(f"{mode}: {COPIES - len(out['p'])} particles lost")
        if max(np.abs(out[name]).max() for name in ("x", "xp", "y", "yp")) != 0:
            sys.exit(f"{mode}: the orbit left the axis")
        spin = np.column_stack((out["spx"], out["spy"], out["spz"]))
        orbit = np.column_stack([out[name] for name in ("x", "xp", "y", "yp")])
        if not (np.all(np.isfinite(spin)) and np.all(np.isfinite(orbit))):
            sys.exit(f"{mode}: nonfinite spin or orbit")
        if np.max(np.abs(np.linalg.norm(spin, axis=1) - 1)) > ATOL:
            sys.exit(f"{mode}: the spin norm is not 1")
        if np.ptp(spin, axis=0).max() > 1e-15:
            sys.exit(f"{mode}: identical particles ended with different spins")
        return spin[0]


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected agree within ATOL."""
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bool(np.allclose(observed, expected, rtol=0, atol=ATOL))


    def case(mode, species, field, by_ks=False):
        """Compare the final spin with the exact precession.

        Return False when it agrees and True when it is the opposite-sense rotation (cos, -sin, 0),
        the recorded sign reversal; any other result is a failed run, not a verdict.
        """
        theta = precession_angle(species, field)
        expected = np.array([np.cos(theta), np.sin(theta), 0.0])
        reversed_sense = np.array([np.cos(theta), -np.sin(theta), 0.0])
        if by_ks:
            # The solenoid strength that stands for the field B: KS = q B / (m c p0) for the signed
            # physical charge q, with p0 in units of m c.
            charge, mass, _ = SPECIES[species]
            element = f"SOLE,L={LENGTH!r},KS={charge * field / (mass * C * MOMENTUM)!r}"
            label = f"{mode} {species}, B={field} T given as KS"
        else:
            element = f"SOLE,L={LENGTH!r},B={field!r}"
            label = f"{mode} {species}, B={field} T"
        observed = final_spin(mode, species, element)
        if compare(f"{label}: spin", observed, expected):
            return False
        if np.allclose(observed, reversed_sense, rtol=0, atol=ATOL):
            return True
        sys.exit(f"{label}: the spin is neither the exact precession nor its opposite sense")


    defect = False
    print("final spin (spx, spy, spz) from initial (1, 0, 0); expected is the exact precession")
    for mode in modes():
        if case(mode, "electron", 0.1):
            sys.exit("The electron control disagrees with the exact precession: it is not valid.")
        if case(mode, "positron", 0.1, by_ks=True):
            sys.exit("The positron KS control disagrees with the exact precession: it is not valid.")
        defect |= case(mode, "positron", 0.1)
        defect |= case(mode, "positron", -0.1)
        defect |= case(mode, "proton", 0.1)
    print(f"comparison: atol={ATOL:g}")
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
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

