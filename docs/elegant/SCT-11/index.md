---
title: "SCT-11: Approximate CBSCAT recoil depends on the reference momentum"
hide:
  - toc
---

# SCT-11 — Approximate CBSCAT recoil depends on the reference momentum

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [approximate recoil, CBScat.c:313–326](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/CBScat.c#L313-L326)<br>[update of delta, CBScat.c:331–337](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/CBScat.c#L331-L337)<br>[exact recoil, CBScat.c:421](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/CBScat.c#L421)<br>[forced exact recoil, CBScat.c:104–110](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/CBScat.c#L104-L110)<br>[recoil equations, cbscat.tex:14–25](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/cbscat.tex#L14-L25) |

With the approximate recoil of Elegant's CBSCAT element (CROSS_SECTION="thomson", EXACT_RECOIL=0), the momentum an electron loses in a laser Compton scatter depends on the reference momentum P₀ the user chose, although the physics does not. The code computes the fractional change ΔP/P of the electron's momentum P and adds it to δ = P/P₀ − 1, which requires ΔP/P₀. For 12 mₑc electrons and a 30 µm laser, the same 454 scattered electrons lose a mean 2.303×10⁻⁵ mₑc at P₀ = 12 mₑc, 1.152×10⁻⁵ mₑc at P₀ = 6 mₑc and 4.607×10⁻⁵ mₑc at P₀ = 24 mₑc.

??? example "Reproducer: SCT-11.py"

    [Download SCT-11.py](SCT-11.py)

    ```python
    """SCT-11: Approximate CBSCAT recoil depends on the reference momentum

    With the approximate recoil of Elegant's CBSCAT element (CROSS_SECTION="thomson", EXACT_RECOIL=0),
    the momentum an electron loses in a laser Compton scatter depends on the reference momentum P₀ the
    user chose, although the physics does not. The code computes the fractional change ΔP/P of the
    electron's momentum P and adds it to δ = P/P₀ − 1, which requires ΔP/P₀. For 12 mₑc electrons and a
    30 µm laser, the same 454 scattered electrons lose a mean 2.303×10⁻⁵ mₑc at P₀ = 12 mₑc, 1.152×10⁻⁵
    mₑc at P₀ = 6 mₑc and 4.607×10⁻⁵ mₑc at P₀ = 24 mₑc.

    The script scatters 16000 electrons of momentum 12 mₑc off a head-on laser with the Thomson cross
    section, with the same random seed so that the same 454 electrons scatter, changing only the
    reference momentum p_central from 6 to 24 mₑc. The mean and maximum losses are compared with the
    Thomson angular-distribution expectation, and losses are compared event by event with the run whose
    reference equals the electrons' momentum. The same runs with the exact-recoil option, which does not
    use the faulty expression, are the control.
    """

    import math
    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    # SDDS constants.h, as Elegant uses them.
    C, E_CHARGE, H, RE, ELECTRON_MEV = (
        299792458.0,
        1.60217733e-19,
        6.6260755e-34,
        2.81794092e-15,
        0.51099906,
    )
    P = 12.0  # electron momentum, units of m_e c
    COUNT, SPOT = 16000, 1.5e-3
    SIGMA_T = 8 * math.pi / 3 * RE**2
    REFERENCES = (6.0, 11.88, 12.0, 12.12, 24.0)
    SIGMAS = 4  # standard deviations of the sample mean allowed
    INVARIANCE_ATOL = 1e-12  # m_e c; the exact-recoil runs agree within 3e-14

    GAMMA = math.sqrt(1 + P * P)
    BETA = P / GAMMA


    def losses(mode, wavelength, exact, reference):
        """Momentum loss (m_e c) of the scattered electrons, ordered as the electrons are."""
        photons = 0.03 * 2 * math.pi * SPOT**2 / SIGMA_T  # mean 0.03 Thomson scatters per electron
        lattice = (
            f"K: CBSCAT, LASER_WAVELENGTH={wavelength!r}, N_PHOTONS={photons!r}, SIGMA_X={SPOT!r}, "
            f'SIGMA_Y={SPOT!r}, LASER_PULSE_LENGTH=8e-6, N_STEPS=81, CROSS_SECTION="thomson", '
            f"EXACT_RECOIL={exact}\nBL: LINE=(K)\n"
        )
        commands = (
            "&global_settings inhibit_seed_permutation=1 &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={reference!r}, "
            "output=out.sdds, random_number_seed=1357911 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        columns = {name: np.zeros(COUNT) for name in ("x", "xp", "y", "yp", "t")} | {
            "p": np.full(COUNT, P),
            "particleID": np.arange(1, COUNT + 1),
        }
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})
        out = result["out.sdds"]
        if out.n_pages != 1 or len(out[0, "p"]) != COUNT:
            sys.exit(f"{mode}: the output is not one page of {COUNT} particles")
        order = np.argsort(out[0, "particleID"])
        if not np.array_equal(out[0, "particleID"][order], np.arange(1, COUNT + 1)):
            sys.exit(f"{mode}: the output particle IDs differ from the input")
        # NaNs would propagate into the mean and maximum, and comparisons with them are false.
        if not all(np.all(np.isfinite(out[0, name])) for name in ("p", "xp", "yp")):
            sys.exit(f"{mode}: the output holds nonfinite p, xp or yp")
        scattered = np.hypot(out[0, "xp"][order], out[0, "yp"][order]) > 0
        if not scattered.any():
            sys.exit(f"{mode}: no electron scattered")
        return P - out[0, "p"][order][scattered], scattered


    defect = False
    print("momentum loss of the scattered electrons in m_e c: mean, maximum / (2A), residual")
    for mode in modes():
        for wavelength in (3e-5, 3e-3):
            # Head-on (cos theta_0 = -1) photon-momentum conservation, in units of m_e c:
            # Delta P = [gamma^2 E_h (cos theta_0 - beta) - gamma E' cos theta]/(m_e c^2) with
            # E' = gamma E_h (1 - beta cos theta_0), so the loss is -Delta P = A (1 + cos theta) with
            # A = gamma^2 E_h (1 + beta)/(m_e c^2). cos theta follows the 1 + cos^2 theta law, whose
            # mean is 0 and variance 2/5: the mean loss is A, and no loss exceeds 2A.
            energy = H * C / wavelength / E_CHARGE / 1e6  # E_h, MeV
            amplitude = GAMMA**2 * energy * (1 + BETA) / ELECTRON_MEV
            for exact in (0, 1):
                runs = {ref: losses(mode, wavelength, exact, ref) for ref in REFERENCES}
                base, base_mask = runs[P]
                for reference, (loss, mask) in runs.items():
                    if not np.array_equal(mask, base_mask):
                        sys.exit(f"{mode}: the reference changed which electrons scatter")
                    tolerance = SIGMAS * amplitude * math.sqrt(0.4 / len(loss))
                    residual = float(np.max(np.abs(loss - base)))
                    observed = (float(loss.mean()), float(loss.max() / (2 * amplitude)), residual)
                    bad = (
                        abs(observed[0] - amplitude) > tolerance
                        or observed[1] > 1 + 1e-9
                        or residual > INVARIANCE_ATOL
                    )
                    label = f"{mode} wavelength={wavelength:g} exact={exact} reference={reference:g}"
                    print(
                        f"{label} ({len(loss)} events): "
                        f"observed=[{observed[0]:.5g}, {observed[1]:.5g}, {observed[2]:.3g}] "
                        f"expected=[{amplitude:.5g}+-{tolerance:.2g}, <=1, 0]"
                    )
                    if (exact == 1 or reference == P) and bad:
                        sys.exit(f"{mode}: a control disagrees: the comparison is not valid")
                    defect |= bad
    print(f"comparison: {SIGMAS} standard deviations of the mean, residual atol={INVARIANCE_ATOL:g}")
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

