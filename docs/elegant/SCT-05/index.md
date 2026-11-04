---
title: "SCT-05: Standalone electron recoil omits energy loss"
hide:
  - toc
---

# SCT-05 — Standalone electron recoil omits energy loss

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [event probability, matter.c:127–128](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L127-L128)<br>[matter.c:187](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L187)<br>[matter.c:246–251](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L246-L251)<br>[matter.c:258–283](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L258-L283)<br>[`ELECTRON_RECOIL`, track_data.c:2143](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2143)<br>[matter.c:84–85](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L84-L85) |

In MATTER, enabling ELECTRON_RECOIL while ENERGY_DECAY and NUCLEAR_BREMSSTRAHLUNG are disabled omits the documented recoil energy loss. The energy-processing gates ignore standalone recoil, although collisions with target electrons must transfer positive energy. For 20000 electrons of momentum 1000 mₑc crossing 0.01 m of dilute helium, a Poisson approximation to the option's event rate predicts 6894 ± 67.21 electrons losing energy, but recoil-only output is bit-identical to all-off output. Enabling bremsstrahlung activates thousands of recoil losses, isolating the activation failure. Users enabling recoil alone are affected.

??? example "Reproducer: SCT-05.py"

    [Download SCT-05.py](SCT-05.py)

    ```python
    """SCT-05: Standalone electron recoil omits energy loss

    In MATTER, enabling ELECTRON_RECOIL while ENERGY_DECAY and NUCLEAR_BREMSSTRAHLUNG are disabled omits
    the documented recoil energy loss. The energy-processing gates ignore standalone recoil, although
    collisions with target electrons must transfer positive energy. For 20000 electrons of momentum 1000
    mₑc crossing 0.01 m of dilute helium, a Poisson approximation to the option's event rate predicts
    6894 ± 67.21 electrons losing energy, but recoil-only output is bit-identical to all-off output.
    Enabling bremsstrahlung activates thousands of recoil losses, isolating the activation failure.
    Users enabling recoil alone are affected.

    The script tracks 20000 electrons through the target with every combination of ELECTRON_RECOIL and
    NUCLEAR_BREMSSTRAHLUNG and counts the electrons whose momentum drops. It compares each count with
    the number of recoil events the option's own rate predicts, and checks that the recoil-only output
    equals the all-off output coordinate by coordinate. The run with both options enabled shows that the
    recoil sampler works and confirms the expected count; the bremsstrahlung-only run is the control for
    the losses.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _matter import AMU, BREMS_PER_ELECTRON, BS_Y0, LENGTH, P0, RE, RHO, A, Z, beam, track

    COUNT, SEED = 20000, 90131
    THRESHOLD = 1e-9  # m_e c; a momentum decrease above this is a real energy loss, not rounding
    # PLIMIT 0.005 keeps every section's event probability far below one.
    ELEMENT = (
        f"L={LENGTH!r},PLIMIT=0.005,ENERGY_DECAY=0,NUCLEAR_BREMSSTRAHLUNG={{nb}},ELECTRON_RECOIL={{er}}"
    )
    COLUMNS = ("x", "xp", "y", "yp", "t", "p")

    # Mean recoil events per electron in the model's own process (matter.c:128): the collision rate
    # n_atom 2 pi Z r_e^2 L / P (1/y0 - 1), with y0 the sampler's lower cutoff. It only sizes the
    # expected number of electrons that suffer a recoil, in a Poisson approximation (at least one
    # event); that a recoil must lower the momentum follows from the option's documented contract and
    # the positive Moller cross section. The element actually has 61 independent Bernoulli sections of
    # probability MEAN_RECOIL/61 each, whose exact at-least-one expectation is 6913.18 with binomial
    # sigma 67.26; both counts recorded in the ledger pass either 5-sigma comparison.
    MEAN_RECOIL = LENGTH * RHO / (AMU * A) * 2 * np.pi * Z * RE**2 / P0 * (1 / BS_Y0 - 1)
    RECOIL_EXPECTED = COUNT * -np.expm1(-MEAN_RECOIL)
    RECOIL_SIGMA = np.sqrt(RECOIL_EXPECTED * np.exp(-MEAN_RECOIL))  # Poisson-approximation sigma
    BREMS_EXPECTED = COUNT * BREMS_PER_ELECTRON
    # Bremsstrahlung alone may produce a few losses (mean 0.14 events); more than 10 has probability
    # below 1e-16, so it would mean something other than recoil is acting.
    BREMS_MAX = 10

    CASES = (  # label, (nb, er), expected electrons that lose momentum
        ("all off", (0, 0), 0.0),
        ("bremsstrahlung only", (1, 0), BREMS_EXPECTED),
        ("recoil + bremsstrahlung", (1, 1), RECOIL_EXPECTED),
        ("recoil only", (0, 1), RECOIL_EXPECTED),
    )

    defect = False
    print(f"electrons with a momentum decrease above {THRESHOLD:g} m_e c, of {COUNT}")
    for mode in modes():
        outs, counts = {}, {}
        for label, (nb, er), expected in CASES:
            outs[nb, er] = track(mode, ELEMENT.format(nb=nb, er=er), SEED, beam(COUNT))
            counts[nb, er] = int(np.sum(outs[nb, er]["p"] < P0 - THRESHOLD))
            spread = f"±{RECOIL_SIGMA:.2f}" if er else ""
            print(f"{mode} {label}: observed={counts[nb, er]} expected={expected:.2f}{spread}")
        # The controls must be clean before the combined-option losses can be attributed to recoil.
        if counts[0, 0] != 0:
            sys.exit(f"{mode}: the all-off run loses momentum in {counts[0, 0]} electrons")
        if counts[1, 0] > BREMS_MAX:
            sys.exit(f"{mode}: bremsstrahlung alone loses momentum in {counts[1, 0]} electrons")
        # With bremsstrahlung enabled, recoil reaches the sampler: its count must match the model rate.
        # Otherwise the expectation for the standalone case is not valid.
        if abs(counts[1, 1] - RECOIL_EXPECTED) > 5 * RECOIL_SIGMA:
            sys.exit(f"{mode}: recoil with bremsstrahlung gives {counts[1, 1]}, not the model rate")
        same = all(np.array_equal(outs[0, 1][k], outs[0, 0][k]) for k in COLUMNS)
        print(
            f"{mode} recoil only vs all off, six coordinates bit-identical: observed={same} expected=False"
        )
        defect |= same or counts[0, 1] < RECOIL_EXPECTED - 5 * RECOIL_SIGMA
    print(f"comparison: 5 sigma on the recoil count, loss threshold {THRESHOLD:g} m_e c")
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

??? example "Helper: _matter.py"

    [Download _matter.py](_matter.py)

    ```python
    """Shared setup for the MATTER reproducers: a thin helium target tracked by Elegant's single-scattering model.

    One MATTER element of length 0.01 m filled with helium (Z = 2, A = 4, density 0.02813 kg/m^3),
    traversed by electrons of momentum 1000 m_e c. PLIMIT bounds the scattering probability of one
    section, and the element is divided into the number of sections that keeps it below the bound.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import particles, run

    P0 = 1000.0  # reference and beam momentum, units of m_e c
    LENGTH = 0.01  # m
    RHO, Z, A = 0.0281327461321535, 2, 4  # kg/m^3 (dilute helium), atomic number, mass number

    # SDDS constants.h and the MATTER source, as Elegant uses them.
    RE, AMU = 2.81794092e-15, 1.6605e-27  # classical electron radius (m), atomic mass unit (kg)
    BS_Y0 = 1e-8  # lower cutoff of the recoil and bremsstrahlung energy-transfer samplers


    # Mean nuclear-bremsstrahlung events per electron: 4 L/(3 X0) [-ln y0 - (1 - y0) + 3/8 (1 - y0^2)],
    # with the helium radiation length X0 = 94.32 g/cm^2 (PDG atomic and nuclear properties of helium
    # gas) divided by the density, converted from cm to m. Bremsstrahlung is rare in this thin target.
    X0 = 94.32 / (RHO * 1e-3) * 1e-2  # m
    BREMS_PER_ELECTRON = 4 * LENGTH / (3 * X0) * (-np.log(BS_Y0) - (1 - BS_Y0) + 3 / 8 * (1 - BS_Y0**2))


    def track(mode, element, seed, columns):
        """Track `columns` (beam arrays with p in m_e c) through `MATTER` + element text; return columns.

        `element` holds the MATTER parameters after the fixed Z, A and RHO. Rows are returned ordered by
        particleID, which the beam must carry (MPI output order varies).
        """
        lattice = f"M: MATTER,Z={Z},A={A},RHO={RHO!r},{element}\nBL: LINE=(M)\n"
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
            f"output=out.sdds, random_number_seed={seed} &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})
        # particles() rejects nonfinite coordinates; the output must also hold exactly the input particles.
        out = particles(result["out.sdds"])
        if len(out["p"]) != len(columns["p"]):
            raise SystemExit(f"{mode}: {len(columns['p']) - len(out['p'])} particles lost")
        if not np.array_equal(out["particleID"], np.sort(columns["particleID"])):
            raise SystemExit(f"{mode}: the output particleIDs differ from the input ones")
        return out


    def beam(count, xp=0.0, yp=0.0):
        """`count` electrons on axis at momentum P0 with the given slopes, numbered from 1."""
        return {
            "x": np.zeros(count),
            "xp": np.full(count, xp),
            "y": np.zeros(count),
            "yp": np.zeros(count) + yp,
            "t": np.zeros(count),
            "p": np.full(count, P0),
            "particleID": np.arange(1, count + 1),
        }


    def gamma(p):
        """Lorentz factor of momentum p in m_e c."""
        return np.sqrt(1 + np.asarray(p) ** 2)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

