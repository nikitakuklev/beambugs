---
title: "SCT-20: Touschek scattering rates use the electron's classical radius for every species"
hide:
  - toc
---

# SCT-20 — Touschek scattering rates use the electron's classical radius for every species

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`TouschekRate`, touschekScatter.c:147–148](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L147-L148)<br>[touschekScatter.c:972](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L972)<br>[factor, touschekScatter.c:1026–1030](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L1026-L1030)<br>[rate, touschekScatter.c:626–627](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L626-L627)<br>[weights, touschekScatter.c:690](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/touschekScatter.c#L690)<br>[`particleRadius`, elegant.c:3308](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3308)<br>[touschek_scatter, elegant.tex:7693–7697](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7693-L7697)<br>[change_particle, elegant.tex:1497–1517](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L1497-L1517)<br>[usage, touschekLifetime.c:38](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/elegantTools/touschekLifetime.c#L38) |

After change_particle selects a muon, a proton or a custom species, touschek_scatter computes the Touschek scattering rates at TSCATTER elements with the electron's classical radius rₑ instead of the species' r_c = q²/(4πε₀Mc²). At fixed βγ, beam geometry, particle count and momentum aperture, Piwinski's rate scales as r_c², so the correct rate for a singly charged species is the electron's times (mₑ/M)²; Elegant instead writes the electron's rates, from Piwinski's formula and from the Monte Carlo, 4.275×10⁴ times too large for muons and 3.372×10⁶ for protons. For a 1 nC beam at βγ = 100 with a ±10⁻³ momentum aperture, it reports 5.375×10¹¹ s⁻¹ for protons against 1.594×10⁵ s⁻¹ from an independent evaluation of Piwinski's formula.

??? example "Reproducer: SCT-20.py"

    [Download SCT-20.py](SCT-20.py)

    ```python
    """SCT-20: Touschek scattering rates use the electron's classical radius for every species

    After change_particle selects a muon, a proton or a custom species, touschek_scatter computes the
    Touschek scattering rates at TSCATTER elements with the electron's classical radius rₑ instead of
    the species' r_c = q²/(4πε₀Mc²). At fixed βγ, beam geometry, particle count and momentum aperture,
    Piwinski's rate scales as r_c², so the correct rate for a singly charged species is the electron's
    times (mₑ/M)²; Elegant instead writes the electron's rates, from Piwinski's formula and from the
    Monte Carlo, 4.275×10⁴ times too large for muons and 3.372×10⁶ for protons. For a 1 nC beam at βγ =
    100 with a ±10⁻³ momentum aperture, it reports 5.375×10¹¹ s⁻¹ for protons against 1.594×10⁵ s⁻¹ from
    an independent evaluation of Piwinski's formula.

    The script runs touschek_scatter on a 1 m drift between two TSCATTER elements (βx = 10 m, βy = 5 m,
    αx = αy = 0 at the first, no dispersion) with a ±10⁻³ momentum aperture file, once each for the
    electron, the muon and the proton at βγ = 100, with the same seed. It evaluates Piwinski's formula
    independently by quadrature at the second element for each species' classical radius and compares
    it with the written Piwinski rate; the electron is the control and must agree to 10⁻⁶. It also
    checks that the Monte Carlo rate and the per-particle rates of the heavy species equal the
    electron's.
    """

    import numpy as np
    from _elegant import run, verdict
    from scipy.integrate import quad
    from scipy.special import i0e

    # SDDS constants.h and the static species table of elegant.c
    C, E_CHARGE, ME = 2.99792458e8, 1.60217733e-19, 9.1093897e-31  # m/s, C, kg
    RE = 2.81794092e-15  # m, electron classical radius re_mks
    EPS0 = 1e7 / (4 * np.pi * C**2)  # F/m, epsilon_o
    MASS = {"electron": ME, "muon": 1.88353109e-28, "proton": 1.6726485e-27}  # kg
    P0, CHARGE = 100.0, 1e-9  # beta*gamma (p_central, in m c of the species), bunch charge in C
    BETA_X0, BETA_Y0 = 10.0, 5.0  # m, at the first TSCATTER, alpha = 0
    EX, EY, SIGMA_DP, SIGMA_S = 1e-8, 1e-9, 1e-3, 0.01  # m, m, 1, m
    APERTURE = 1e-3  # +/- fractional momentum aperture at both TSCATTER elements
    LENGTH = 1.0  # m, drift between the TSCATTER elements

    LATTICE = f"T0: TSCATTER\nD: DRIF,L={LENGTH!r}\nT1: TSCATTER\nBL: LINE=(T0,D,T1)\n"
    APERTURE_FILE = (
        "SDDS1\n"
        "&column name=s, units=m, type=double &end\n"
        "&column name=deltaPositive, type=double &end\n"
        "&column name=deltaNegative, type=double &end\n"
        "&column name=ElementName, type=string &end\n"
        "&column name=ElementOccurence, type=long &end\n"
        "&column name=ElementType, type=string &end\n"
        "&data mode=ascii, no_row_counts=1 &end\n"
        f"0 {APERTURE!r} {-APERTURE!r} T0 1 TSCATTER\n"
        f"{LENGTH!r} {APERTURE!r} {-APERTURE!r} T1 1 TSCATTER\n"
    )


    def touschek(species):
        """Scattered-bunch file: one page per TSCATTER element with the rates as parameters."""
        commands = (
            f"&change_particle name={species} &end\n"
            f"&run_setup lattice=line.lte, use_beamline=BL, p_central={P0!r}, random_number_seed=987 &end\n"
            "&run_control &end\n"
            f"&twiss_output matched=0, beta_x={BETA_X0!r}, beta_y={BETA_Y0!r} &end\n"
            f"&touschek_scatter charge={CHARGE!r}, emit_x={EX!r}, emit_y={EY!r}, sigma_dp={SIGMA_DP!r},\n"
            f"  sigma_s={SIGMA_S!r}, Momentum_Aperture=aperture.sdds, Momentum_Aperture_scale=1,\n"
            '  n_simulated=2000, ignored_portion=0, bunch="%s.bun" &end\n'
        )
        files = {"line.lte": LATTICE, "aperture.sdds": APERTURE_FILE, "run.ele": commands}
        bunch = run(files, outputs=("run.bun",))["run.bun"]
        page = 1  # T1; the rate at the first TSCATTER element is not computed
        par = {p.name: float(p.data[page]) for p in bunch.parameters if p.name.endswith("Rate")}
        return par, np.asarray(bunch[page, "LRate"])


    def piwinski_rate(radius):
        """Particles scattered beyond the +/- aperture per second at T1, N/tau, by Piwinski's formula.

        A. Piwinski, DESY 98-179 (1998), without dispersion (sigma_h = sigma_p):
          1/tau = r^2 c N sigma_p / (8 sqrt(pi) beta^2 gamma^4 ex ey sigma_s sigma_p) * F,
          F = int_{tau_m}^inf [(2+1/t)^2 (t/tau_m/(1+t) - 1) + 1 - sqrt(tau_m (1+t)/t)
                               - (4+1/t)/(2t) ln(t/tau_m/(1+t))] sqrt(t/(1+t)) exp(-B1 t) I0(B2 t) dt,
          tau_m = beta^2 delta_m^2, B1 = (bx/ex + by/ey)/(2 beta^2 gamma^2),
          B2 = |bx/ex - by/ey|/(2 beta^2 gamma^2), with bx, by the beta functions at the location.
        Each of the two particles of an event leaves through one side of a symmetric aperture, so N/tau
        is the sum of the scattering rates beyond the positive and the negative aperture. In a drift
        from alpha = 0, beta(s) = beta0 + s^2/beta0.
        """
        gamma = np.hypot(P0, 1.0)
        beta = P0 / gamma
        bx, by = BETA_X0 + LENGTH**2 / BETA_X0, BETA_Y0 + LENGTH**2 / BETA_Y0
        n = CHARGE / E_CHARGE
        b1 = (bx / EX + by / EY) / (2 * beta**2 * gamma**2)
        b2 = abs(bx / EX - by / EY) / (2 * beta**2 * gamma**2)
        tm = (beta * APERTURE) ** 2

        def integrand(u):  # t = tau_m e^u, dt = t du; exp(-B1 t) I0(B2 t) = i0e(B2 t) exp(-(B1-B2) t)
            t = tm * np.exp(u)
            log_term = (4 + 1 / t) / (2 * t) * np.log(t / tm / (1 + t))
            bracket = (
                (2 + 1 / t) ** 2 * (t / tm / (1 + t) - 1) + 1 - np.sqrt(tm * (1 + t) / t) - log_term
            )
            return t * bracket * np.sqrt(t / (1 + t)) * i0e(b2 * t) * np.exp(-(b1 - b2) * t)

        f, _ = quad(integrand, 0, 60, limit=500, epsabs=0, epsrel=1e-11)
        inv_tau = radius**2 * C * n * f / (8 * np.sqrt(np.pi) * beta**2 * gamma**4 * EX * EY * SIGMA_S)
        return n * inv_tau


    defect = True
    e_par, e_lrate = touschek("electron")
    for species, mass in MASS.items():
        par, lrate = (e_par, e_lrate) if species == "electron" else touschek(species)
        # Elegant's electron constant; the species radius q^2/(4 pi eps0 M c^2) otherwise (elegant.c).
        radius = RE if species == "electron" else E_CHARGE**2 / (4 * np.pi * EPS0 * mass * C**2)
        expected = piwinski_rate(radius)
        print(f"{species}: PLocalRate at T1 observed={par['PLocalRate']:.10e} /s "
              f"expected={expected:.10e} /s (observed/expected {par['PLocalRate'] / expected:.10g})")  # fmt: skip
        print(f"  SLocalRate={par['SLocalRate']:.10e} /s; LRate column equal to the electron's: "
              f"{np.array_equal(lrate, e_lrate)}")  # fmt: skip
        if species == "electron":
            if not np.isclose(par["PLocalRate"], expected, rtol=1e-6, atol=0):
                raise SystemExit("control: electron Piwinski rate differs from the independent value")
        else:
            defect &= bool(
                par["PLocalRate"] / expected > 1e4
                and par["PLocalRate"] == e_par["PLocalRate"]
                and par["SLocalRate"] == e_par["SLocalRate"]
                and np.array_equal(lrate, e_lrate)
            )
    print("comparison: electron rtol=1e-6; defect needs heavy-species rates identical to the "
          "electron's and above 1e4 times the expected Piwinski rate")  # fmt: skip
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
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`0355dd6aca12`), source reading, not executed for this claim |
| Confirmed (the rate normalization; a species-dependent error in the generated coordinates was not established) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

