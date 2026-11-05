---
title: "SCT-16: ENERGY_STRAGGLE shifts the mean energy loss and narrows its stated spread"
hide:
  - toc
---

# SCT-16 — ENERGY_STRAGGLE shifts the mean energy loss and narrows its stated spread

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [straggling, matter.c:264–273](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matter.c#L264-L273)<br>[simplified energy loss, matter.tex:78–94](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/matter.tex#L78-L94)<br>[`ENERGY_STRAGGLE`, track_data.c:2141](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2141) |

With ENERGY_DECAY=1 and ENERGY_STRAGGLE=1, MATTER multiplies each electron's exponential energy loss by 1 + 0.5·N(0, 1), with N(0, 1) a standard normal deviate, and redraws whenever the loss would be negative. The manual states that the loss then has a standard deviation of half its mean, the exponential loss; the redraw instead gives a truncated normal distribution whose mean is 2.762% larger and whose standard deviation is 0.4581 of that mean. For 200000 electrons at 1000 mₑc crossing hydrogen of 10⁻⁴ radiation lengths, the mean loss is 2.750 ± 0.1051% above the exponential law, and the ratio is 0.4576.

??? example "Reproducer: SCT-16.py"

    [Download SCT-16.py](SCT-16.py)

    ```python
    """SCT-16: ENERGY_STRAGGLE shifts the mean energy loss and narrows its stated spread

    With ENERGY_DECAY=1 and ENERGY_STRAGGLE=1, MATTER multiplies each electron's exponential energy loss
    by 1 + 0.5·N(0, 1), with N(0, 1) a standard normal deviate, and redraws whenever the loss would be
    negative. The manual states that the loss then has a standard deviation of half its mean, the
    exponential loss; the redraw instead gives a truncated normal distribution whose mean is 2.762%
    larger and whose standard deviation is 0.4581 of that mean. For 200000 electrons at 1000 mₑc
    crossing hydrogen of 10⁻⁴ radiation lengths, the mean loss is 2.750 ± 0.1051% above the exponential
    law, and the ratio is 0.4576.

    The script tracks 200000 on-axis electrons at 1000 mₑc through thin hydrogen of 10⁻⁴ radiation
    lengths with ENERGY_DECAY=1 and FORCE_MCS=1, once with ENERGY_STRAGGLE=1 and once without as the
    control. It compares the sample mean and standard deviation of the energy loss with the manual's
    moments (mean equal to the exponential loss, standard deviation half of it) and with the moments
    of the truncated normal distribution that rejecting negative losses produces.
    """

    import numpy as np
    from _elegant import modes, verdict
    from _matter_model import gamma_loss, radiation_length, track
    from scipy.stats import norm

    SEED, P0, COUNT = 86420, 1000.0, 200000  # P0 in m_e c
    RHO = float(1e-4 * radiation_length(1, 1.0, 1.0))  # hydrogen, L = 1 m, L/X0 = 1e-4
    ELEMENT = f"L=1.0,Z=1,A=1.0,RHO={RHO!r},ENERGY_DECAY=1,FORCE_MCS=1,ENERGY_STRAGGLE={{s}}"
    # Mean loss of ENERGY_DECAY: gamma (1 - exp(-L/X0)) (matter.c:113, 265).
    MU = np.sqrt(1 + P0**2) * -np.expm1(-1.0 / radiation_length(1, 1.0, RHO))
    # The source draws X = MU (1 + Z/2), Z ~ N(0, 1), and redraws while X < 0 (matter.c:269-270),
    # i.e. Z conditioned on Z > -2. With h = phi(2)/Phi(2): E[Z | Z > -2] = h and
    # Var[Z | Z > -2] = 1 - 2h - h^2, so E[X] = MU (1 + h/2) and SD[X] = (MU/2) sqrt(1 - 2h - h^2).
    H = norm.pdf(2) / norm.cdf(2)
    TRUNC_MEAN, TRUNC_SD = MU * (1 + H / 2), MU / 2 * np.sqrt(1 - 2 * H - H * H)

    defect = False
    for mode in modes():
        plain = gamma_loss(P0, track(mode, ELEMENT.format(s=0), P0, SEED, COUNT)["p"])
        print(f"{mode} ENERGY_STRAGGLE=0 (control): mean gamma loss observed={plain.mean():.12g} "
              f"expected={MU:.12g}")  # fmt: skip
        if not np.allclose(plain, MU, rtol=1e-9, atol=0):
            raise SystemExit(f"{mode}: the control misses the exponential law")
        loss = gamma_loss(P0, track(mode, ELEMENT.format(s=1), P0, SEED, COUNT)["p"])
        mean, sd = loss.mean(), loss.std(ddof=1)
        se_mean = sd / np.sqrt(COUNT)
        # Standard error of the sample SD from the fourth central moment: Var(s^2) ~ (m4 - s^4)/N.
        m4 = np.mean((loss - mean) ** 4)
        se_sd = np.sqrt((m4 - sd**4) / COUNT) / (2 * sd)
        se_ratio = np.hypot(se_sd / mean, sd * se_mean / mean**2)  # conservative, ignores correlation
        print(f"{mode} ENERGY_STRAGGLE=1: mean gamma loss observed={mean:.8g}±{se_mean:.2g} "
              f"expected (manual)={MU:.8g}, truncated normal {TRUNC_MEAN:.8g}")  # fmt: skip
        print(f"{mode} ENERGY_STRAGGLE=1: SD/mean observed={sd / mean:.6f}±{se_ratio:.2g} "
              f"expected (manual)=0.5, truncated normal {TRUNC_SD / TRUNC_MEAN:.6f}")  # fmt: skip
        print(f"  mean shift over the exponential loss: observed={mean / MU - 1:+.4%}±{se_mean / MU:.4%}, "
              f"truncated normal {H / 2:+.4%}")  # fmt: skip
        if abs(mean - TRUNC_MEAN) > 5 * se_mean or abs(sd - TRUNC_SD) > 5 * se_sd:
            raise SystemExit(f"{mode}: the moments do not follow the redraw rule either")
        defect |= abs(mean - MU) > 5 * se_mean and abs(sd / mean - 0.5) > 5 * se_ratio
    print("comparison: 5 standard errors; control rtol=1e-9")
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
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

