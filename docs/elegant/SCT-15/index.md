---
title: "SCT-15: MATTER converts between mass and atom density with the proton mass"
hide:
  - toc
---

# SCT-15 — MATTER converts between mass and atom density with the proton mass

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [density, matter.c:89–92](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L89-L92)<br>[width, matter.c:141–144](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L141-L144)<br>[constants, matter.c:19–22](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L19-L22)<br>[radiationLength, matter.c:315–349](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L315-L349)<br>[rates, matter.c:121–128](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L121-L128)<br>[MATTER parameters, track_data.c:2145–2149](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2145-L2149)<br>[forced multiple scattering, matter.tex:49–67](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/matter.tex#L49-L67) |

With PRESSURE, TEMPERATURE and MULTIPLICITY, MATTER computes gas density using the proton mass m_p instead of the atomic mass unit u. ENERGY_DECAY and bremsstrahlung rates are consequently 0.7293% too large, and single-scattering and recoil rates, which also use a rounded u, are 0.7316% too large; with RHO supplied directly, the latter rates are 0.002344% too large. For nitrogen at 100 Pa and 300 K over 10 m, the fractional energy loss is 2.978×10⁻⁵ instead of 2.956×10⁻⁵, as an independent ideal-gas atom-density calculation shows. Forced multiple scattering converts RHO to atom density with m_p, making its angles 0.3626% too small; pressure input gives correct angles.

??? example "Reproducer: SCT-15.py"

    [Download SCT-15.py](SCT-15.py)

    ```python
    """SCT-15: MATTER converts between mass and atom density with the proton mass

    With PRESSURE, TEMPERATURE and MULTIPLICITY, MATTER computes gas density using the proton mass
    m_p instead of the atomic mass unit u. ENERGY_DECAY and bremsstrahlung rates are consequently
    0.7293% too large, and single-scattering and recoil rates, which also use a rounded u, are
    0.7316% too large; with RHO supplied directly, the latter rates are 0.002344% too large. For
    nitrogen at 100 Pa and 300 K over 10 m, the fractional energy loss is 2.978×10⁻⁵ instead of
    2.956×10⁻⁵, as an independent ideal-gas atom-density calculation shows. Forced multiple
    scattering converts RHO to atom density with m_p, making its angles 0.3626% too small; pressure
    input gives correct angles.

    The script tracks 200 on-axis electrons at 1000 mₑc through 10 m of nitrogen with ENERGY_DECAY=1
    and FORCE_MCS=1, describing the same gas three ways with one seed: by PRESSURE, TEMPERATURE and
    MULTIPLICITY, by its true mass density RHO = n·A·u, and by RHO = n·A·m_p. It compares each
    attenuation exponent with the ideal-gas prediction, in which A and every mass unit cancel, and
    compares the forced multiple-scattering kicks of the runs, which draw the same random numbers. The
    true-density energy loss and the pressure-input kicks are the controls.
    """

    import numpy as np
    from _elegant import modes, verdict
    from _matter_model import ALPHA_FS, K_BOLTZMANN, M_PROTON, RE, U_KG, track

    SEED, P0, COUNT = 97531, 1000.0, 200  # P0 in m_e c
    Z, A, MULT = 7, 14.0067, 2  # nitrogen: atomic number, atomic mass in AMU, atoms per molecule
    PRESSURE, TEMPERATURE, LENGTH = 100.0, 300.0, 10.0  # Pa, K, m
    N_ATOM = MULT * PRESSURE / (K_BOLTZMANN * TEMPERATURE)  # ideal-gas atom density, 1/m^3

    # Radiation depth from the atom density directly (radiationLength, matter.c:315-349, with
    # rho N_A/(A 1e-3) = n): L/X0 = 4 alpha r_e^2 n L [Z^2 (Lrad - f) + Z Lrad'], Tsai's large-Z
    # logarithms for Z = 7. No mass unit enters.
    a2 = (ALPHA_FS * Z) ** 2
    f_z = a2 * (1 / (1 + a2) + 0.20206 + a2 * (-0.0369 + a2 * (0.0083 - a2 * 0.002)))
    lrad, lradp = np.log(184.15 * Z ** (-1 / 3)), np.log(1194 * Z ** (-2 / 3))
    TAU = 4 * ALPHA_FS * RE**2 * N_ATOM * LENGTH * (Z * Z * (lrad - f_z) + Z * lradp)

    GAS = f"L={LENGTH!r},Z={Z},A={A!r},ENERGY_DECAY=1,FORCE_MCS=1"
    CASES = (  # label, material parameters
        ("pressure input", f"PRESSURE={PRESSURE!r},TEMPERATURE={TEMPERATURE!r},MULTIPLICITY={MULT}"),
        ("RHO=n*A*u (true mass density)", f"RHO={N_ATOM * A * U_KG!r}"),
        ("RHO=n*A*m_p", f"RHO={N_ATOM * A * M_PROTON!r}"),
    )

    defect = False
    gamma_in = np.sqrt(1 + P0**2)
    for mode in modes():
        outs = {}
        for label, material in CASES:
            out = outs[label] = track(mode, f"{GAS},{material}", P0, SEED, COUNT)
            # gamma_out = gamma_in exp(-L/X0) (matter.c:113, 265); recover the exponent stably.
            g_out = np.sqrt(1 + out["p"] ** 2)
            tau = np.log1p((P0 - out["p"]) * (P0 + out["p"]) / (gamma_in + g_out) / g_out)
            print(f"{mode} {label}: L/X0 over ideal-gas prediction observed={np.mean(tau) / TAU:.10f} "
                  f"(spread {np.ptp(tau) / TAU:.1e}) expected=1")  # fmt: skip
            if label.startswith("pressure"):
                defect |= abs(np.mean(tau) / TAU - 1) > 1e-6
            elif "true" in label and abs(np.mean(tau) / TAU - 1) > 1e-9:
                raise SystemExit(f"{mode}: the true-density control misses the radiation length")
        print(f"  proton mass over atomic mass unit m_p/u = {M_PROTON / U_KG:.10f}")
        # The same gas with the same seed must receive the same angular kicks. The forced-MCS width
        # uses n = rho/(m_p A) (matter.c:92, 144), so a RHO input gives kicks scaled by sqrt(u/m_p).
        ref = outs["pressure input"]
        for label in ("RHO=n*A*u (true mass density)", "RHO=n*A*m_p"):
            ratio = outs[label]["xp"] / ref["xp"]
            print(f"{mode} {label}: kick over pressure-input kick observed=[{ratio.min():.10f}, "
                  f"{ratio.max():.10f}] expected=1")  # fmt: skip
            if "true" in label:
                defect |= bool(np.any(abs(ratio - 1) > 1e-9))
        print(f"  sqrt(u/m_p) = {np.sqrt(U_KG / M_PROTON):.10f}")
    print("comparison: L/X0 atol 1e-6 relative (control 1e-9), kick ratios atol 1e-9")
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

??? example "Helper: _matter_model.py"

    [Download _matter_model.py](_matter_model.py)

    ```python
    """Shared setup for the MATTER energy-loss reproducers: any material, momentum and options.

    `track` runs one MATTER element on a beam of electrons that all start on axis at one momentum.
    `model` evaluates the event rates and slice count with the formulas and constants of the MATTER
    source, so a reproducer can predict the counts its own model implies. `gamma_loss` recovers the
    energy loss of each electron from the stored momenta without cancellation.
    """

    from __future__ import annotations

    from dataclasses import dataclass

    import numpy as np
    from _elegant import particles, run

    # SDDS constants.h and the MATTER source (matter.c), as Elegant uses them.
    PI = 3.141592653589793
    RE = 2.81794092e-15  # classical electron radius (m)
    E_MKS, H_MKS, C_MKS = 1.60217733e-19, 6.6260755e-34, 2.99792458e8
    EPSILON0 = 1e7 / (4 * PI * C_MKS**2)
    K_BOLTZMANN = 1.380658e-23  # J/K
    N_AVOGADRO = 6.02214129e23
    AMU_SOURCE = 1.6605e-27  # matter.c's AMU macro (kg), used in the single-scattering rates
    M_PROTON = 1.6726485e-27  # matter.c's mp_mks (kg)
    U_KG = 1e-3 / N_AVOGADRO  # atomic mass unit implied by radiationLength's N_A/A per gram
    ALPHA_FS = E_MKS**2 / (4 * PI * EPSILON0 * H_MKS / (2 * PI) * C_MKS)  # radiationLength's alpha
    Y0 = 1e-8  # lower cutoff of the bremsstrahlung and recoil fractional-transfer samplers
    # Mean photons per radiation length above Y0: (4/3)[-ln y0 - (1 - y0) + 3/8 (1 - y0^2)] (matter.c:125)
    BREMS_PER_X0 = 4 / 3 * (-np.log(Y0) - (1 - Y0) + 3 / 8 * (1 - Y0**2))


    def radiation_length(Z: int, A: float, rho: float) -> float:
        """Radiation length in m by matter.c's radiationLength (Tsai, PDG): Z, A in AMU, rho in kg/m^3."""
        a2 = (ALPHA_FS * Z) ** 2
        f = a2 * (1 / (1 + a2) + 0.20206 + a2 * (-0.0369 + a2 * (0.0083 - a2 * 0.002)))
        light = {1: (5.31, 6.144), 2: (4.79, 5.621), 3: (4.74, 5.805), 4: (4.71, 5.924)}
        lrad, lradp = light.get(Z, (np.log(184.15 * Z ** (-1 / 3)), np.log(1194 * Z ** (-2 / 3))))
        per_atom = 4 * ALPHA_FS * RE**2 * (Z * Z * (lrad - f) + Z * lradp)  # m^2
        return 1 / (per_atom * rho * N_AVOGADRO / (A * 1e-3))


    @dataclass(frozen=True)
    class Rates:
        """Mean events per electron over the whole element, and the slice count, as matter.c sets them."""

        elastic: float  # probScatter
        brems: float  # probBSScatter
        recoil: float  # probERScatter
        sections: int  # sections0


    def model(Z: int, A: float, rho: float, length: float, p0: float, plimit: float = 0.05) -> Rates:
        """Rates of matter.c:118-130 for reference momentum p0 (m_e c) and density rho (kg/m^3)."""
        beta = p0 / np.sqrt(p0**2 + 1)
        k1 = 4 * Z * (Z + 1) * (RE / (beta * p0)) ** 2
        k2 = (Z ** (1 / 3) / 137.036 / p0) ** 2
        n_atom = rho / (AMU_SOURCE * A)
        elastic = n_atom * length * k1 * PI**3 / (k2**2 + k2 * PI**2)
        brems = length / radiation_length(Z, A, rho) * BREMS_PER_X0
        recoil = length * n_atom * 2 * PI * Z * RE**2 / p0 * (1 / Y0 - 1)
        return Rates(elastic, brems, recoil, int(elastic / plimit + 1))


    def beam(count: int, p: float) -> dict[str, np.ndarray]:
        """`count` electrons on axis with zero slopes at momentum p (m_e c), numbered from 1."""
        zeros = np.zeros(count)
        return {"x": zeros, "xp": zeros, "y": zeros, "yp": zeros, "t": zeros, "p": np.full(count, p),
                "particleID": np.arange(1, count + 1)}  # fmt: skip


    def track(mode: str, element: str, p0: float, seed: int, count: int) -> dict[str, np.ndarray]:
        """Track `count` electrons at p0 through `M: MATTER,<element>`; return the columns by particleID.

        The output must hold every input particle (none may stop in the material).
        """
        lattice = f"M: MATTER,{element}\nBL: LINE=(M)\n"
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, "
            f"output=out.sdds, random_number_seed={seed} &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        columns = beam(count, p0)
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": columns})
        out = particles(result["out.sdds"])
        if not np.array_equal(out["particleID"], columns["particleID"]):
            raise SystemExit(f"{mode}: the output particles differ from the input ones (losses?)")
        return out


    def gamma_loss(p_in: float, p_out: np.ndarray) -> np.ndarray:
        """gamma_in - gamma_out from momenta in m_e c: (p_in - p_out)(p_in + p_out)/(gamma_in + gamma_out)."""
        g_in, g_out = np.sqrt(1 + p_in**2), np.sqrt(1 + p_out**2)
        return (p_in - p_out) * (p_in + p_out) / (g_in + g_out)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Found (forced-MCS width with RHO input; elastic and recoil rate factor by source reading) | `claude-opus-5-5` | high | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed (forced-MCS width with RHO input; elastic and recoil rates for pressure and RHO input) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

