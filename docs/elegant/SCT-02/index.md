---
title: "SCT-02: Pelegant writes zero IBS rate columns and an unset vertical growth rate"
hide:
  - toc
---

# SCT-02 — Pelegant writes zero IBS rate columns and an unset vertical growth rate

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`track_IBS` output transfer, ibs_tracking.c:559–610](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibs_tracking.c#L559-L610)<br>[`dump_IBScatter` output, ibs_tracking.c:1262–1281](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibs_tracking.c#L1262-L1281)<br>[IBSCATTER element flags, track_data.c:4237](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L4237) |

A single-slice IBSCATTER in linac mode (SMOOTH=1, ISRING=0) writes incorrect rate diagnostics under three-rank Pelegant. The local columns dIBSRatex, dIBSRatey and dIBSRatez are all zero, and the vertical growth-rate parameter yGrowthRate is uninitialized. Serial diagnostics agree with independent Bjorken–Mtingwa integration, and amplified-update controls give matching serial and parallel particles in this configuration. For a 1 nC electron bunch at momentum 800 mₑc, the correct first-site local rates are −0.9384, −0.8231 and 568.0 s⁻¹, and the vertical growth rate is −0.8024 s⁻¹. Users reading IBS rates from these smooth parallel runs are affected.

??? example "Reproducer: SCT-02.py"

    [Download SCT-02.py](SCT-02.py)

    ```python
    """SCT-02: Pelegant writes zero IBS rate columns and an unset vertical growth rate

    A single-slice IBSCATTER in linac mode (SMOOTH=1, ISRING=0) writes incorrect rate diagnostics under
    three-rank Pelegant. The local columns dIBSRatex, dIBSRatey and dIBSRatez are all zero, and the
    vertical growth-rate parameter yGrowthRate is uninitialized. Serial diagnostics agree with
    independent Bjorken–Mtingwa integration, and amplified-update controls give matching serial and
    parallel particles in this configuration. For a 1 nC electron bunch at momentum 800 mₑc, the correct
    first-site local rates are −0.9384, −0.8231 and 568.0 s⁻¹, and the vertical growth rate is −0.8024
    s⁻¹. Users reading IBS rates from these smooth parallel runs are affected.

    The script tracks a 1024-particle, 1 nC beam at p = 800 m_e c through two 0.2 m drifts and a
    linac-mode IBSCATTER that writes its rates to a file, once with serial elegant and once with
    3-rank Pelegant. It compares the written local rates and growth-rate parameters with an
    independent adaptive quadrature of the Bjorken-Mtingwa integrals; the serial run is the control.
    A second pair of runs multiplies the IBS update by 10^5 and compares the tracked beams, showing
    that Pelegant applies the correct rates to the particles.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from scipy.integrate import quad_vec

    # SDDS constants.h, as Elegant uses them.
    C, QE, RE, ELECTRON_MEV = 299792458.0, 1.60217733e-19, 2.81794092e-15, 0.51099906
    B0, CHARGE, COUNT = 800.0, 1e-9, 1024  # beta*gamma, C, particles
    EX, EY, SD, SZ = 2e-9, 3e-10, 8e-4, 0.004  # m, m, rms dp/p, m
    BX, AX, BY, AY = 8.0, 0.3, 5.0, -0.2  # Twiss parameters at the start, m and 1
    RTOL, ATOL = 3e-5, 1e-12  # Elegant's own quadrature accuracy; see the oracle
    SITES = np.array([0.0, 0.2, 0.4])  # m; the exits of the two drifts and the scattering point
    MIN_GROWTH = 1.05  # the amplified update must raise the rms dp/p by at least this factor


    def beam():
        """Exact Twiss-matched moments: orthonormal columns scaled by sqrt(COUNT) have zero mean and
        unit covariance, so the sample emittances, spread and length equal the requested ones."""
        rng = np.random.default_rng(1912)
        s = rng.normal(size=(COUNT, 6))
        s -= s.mean(axis=0)
        s = np.linalg.qr(s)[0] * np.sqrt(COUNT)
        return {
            "x": np.sqrt(EX * BX) * s[:, 0],
            "xp": np.sqrt(EX / BX) * (s[:, 1] - AX * s[:, 0]),
            "y": np.sqrt(EY * BY) * s[:, 2],
            "yp": np.sqrt(EY / BY) * (s[:, 3] - AY * s[:, 2]),
            "t": SZ / C * s[:, 4],
            "p": B0 * (1 + SD * s[:, 5]),
            "particleID": np.arange(1, COUNT + 1),
        }


    def track(mode, columns, factor=1.0):
        lattice = (
            f"Q: CHARGE,TOTAL={CHARGE!r}\n"
            "D: EDRIFT,L=0.2\n"
            f"I: IBSCATTER,ISRING=0,BUNCHED_BEAM_MODE=0,SMOOTH=1,FORCE_MATCHED_TWISS=1,"
            f'FACTOR={factor!r},FILENAME="ibs.sdds"\n'
            "BL: LINE=(Q,D,D,I)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={B0!r}, "
            "output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            f"&twiss_output matched=0, beta_x={BX!r}, alpha_x={AX!r}, beta_y={BY!r}, "
            f"alpha_y={AY!r} &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": columns},
            outputs=("ibs.sdds", "out.sdds"),
        )
        return result["ibs.sdds"], particles(result["out.sdds"])


    def bm_rates(sz, bx, by):
        """Bjorken-Mtingwa emittance growth rates (x, y, z) in 1/s, zero dispersion.

        Adaptive quadrature of the BM integral, not Elegant's per-decade Simpson sum. Without
        dispersion the BM matrices are diagonal in (horizontal, longitudinal, vertical) velocity
        coordinates: M_x = bx/ex e1 e1^T, M_z = gamma^2/sd^2 e2 e2^T, M_y = by/ey e3 e3^T. With
        a = (bx/ex, gamma^2/sd^2, by/ey) and L = diag(a),
        rate_k = A * integral_0^inf sqrt(u/det(L+u)) [a_k tr (L+u)^-1 - 3 a_k/(a_k+u)] du,
        A = N r_e^2 c lnC/(8 pi beta^3 gamma^4 ex ey sd sz). The Coulomb logarithm lnC follows
        the cutoffs Elegant documents, including its extra sqrt(gamma^2+1) in the temperature.
        """
        gamma = np.sqrt(1 + B0 * B0)
        beta = B0 / gamma
        a = np.array([bx / EX, by / EY, gamma**2 / SD**2])  # ordered (x, y, z) like the rates
        scale = a.max()  # u in units of the largest eigenvalue keeps the quadrature well scaled
        a = a / scale

        def integrand(u):
            inverse = 1 / (a + u)
            return np.sqrt(u * np.prod(inverse)) * (a * inverse.sum() - 3 * a * inverse)

        integrals, error = quad_vec(integrand, 0, np.inf, epsabs=1e-10, epsrel=1e-10)
        if error > 1e-7 * np.linalg.norm(integrals):
            sys.exit("BM quadrature did not converge")
        temperature = 1e6 * (gamma * np.sqrt(gamma**2 + 1) * ELECTRON_MEV - ELECTRON_MEV) * EX / bx
        sx_cm, sy_cm, sz_cm = 100 * np.sqrt(EX * bx), 100 * np.sqrt(EY * by), 100 * sz
        population = CHARGE / QE
        density = population / (8 * np.pi**1.5 * sx_cm * sy_cm * sz_cm)
        maximum = min(sx_cm, 743.4 * np.sqrt(temperature / density))
        minimum = max(
            1.44e-7 / temperature, 1.9732858e-11 / (2 * np.sqrt(1e-6 * temperature * ELECTRON_MEV))
        )
        prefactor = (
            population
            * RE**2
            * C
            * np.log(maximum / minimum)
            / (8 * np.pi * beta**3 * gamma**4 * EX * EY * SD * sz)
        )
        return prefactor * integrals


    def expected_rates(ibs, columns):
        """BM rates at each element exit s listed in the IBS file, and their s-average."""
        # IBSCATTER evaluates the bunch length at the scattering point, 0.4 m downstream.
        p, xp, yp = columns["p"], columns["xp"], columns["yp"]
        t_kick = columns["t"] + 0.4 * np.sqrt(1 + xp * xp + yp * yp) / (p / np.hypot(1, p) * C)
        sz = C * np.std(t_kick)
        s_values = np.asarray(ibs[0, "s"])
        rates = []
        for s in s_values:
            # Drift transport of beta from the start to s: beta - 2 alpha s + gamma s^2.
            bx = BX - 2 * AX * s + (1 + AX * AX) / BX * s * s
            by = BY - 2 * AY * s + (1 + AY * AY) / BY * s * s
            rate = bm_rates(sz, bx, by)
            rate[2] *= 2  # Elegant's linac branch doubles the longitudinal rate
            rates.append(rate)
        rates = np.array(rates)
        # The growth-rate parameters average the local rates over s (trapezoid rule).
        average = np.trapezoid(rates, s_values, axis=0) / np.ptp(s_values)
        return rates, average


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.ravel(values)) + "]"


    def compare(label, observed, expected):
        bad = not np.allclose(observed, expected, rtol=RTOL, atol=ATOL)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    columns = beam()
    defect = False
    expected = None
    print("IBS rates in 1/s: dIBSRate(x,y,z) at each s, then (x,y,z)GrowthRate")
    for mode in ("serial", *modes(mpi_only=True)):
        ibs, _ = track(mode, columns)
        # One page with the three expected sites in every mode, so that the rows compared are like for like.
        sites = np.asarray(ibs[0, "s"])
        if ibs.n_pages != 1 or sites.shape != SITES.shape or not np.allclose(sites, SITES, atol=1e-12):
            sys.exit(f"{mode}: expected one IBS page with sites {SITES}, got {ibs.n_pages} and {sites}")
        rate_columns = [np.asarray(ibs[0, f"dIBSRate{plane}"]) for plane in "xyz"]
        if any(c.shape != SITES.shape for c in rate_columns):
            sys.exit(f"{mode}: the local-rate columns do not match the sites")
        local = np.column_stack(rate_columns)
        growth_values = [np.ravel(ibs[f"{plane}GrowthRate"].data).astype(float) for plane in "xyz"]
        if any(v.size != 1 for v in growth_values):
            sys.exit(f"{mode}: expected one value per growth-rate parameter")
        growth = np.array([v[0] for v in growth_values])
        # Zeros are the defect; nonfinite local rates, or a nonfinite x or z growth rate, are not.
        # A nonfinite yGrowthRate is a valid manifestation of the unset parameter in the MPI run.
        if not np.all(np.isfinite(local)) or not np.all(np.isfinite(growth[[0, 2]])):
            sys.exit(f"{mode}: nonfinite IBS rates: the comparison is not valid")
        if expected is None:
            expected = expected_rates(ibs, columns)
        bad = False
        for s, row, oracle in zip(sites, local, expected[0], strict=True):
            bad |= compare(f"{mode} dIBSRate(x,y,z) at s={s:g} m", row, oracle)
        bad |= compare(f"{mode} (x,y,z)GrowthRate", growth, expected[1])
        if mode == "serial" and bad:
            sys.exit("Serial elegant disagrees with the BM oracle: the comparison is not valid.")
        defect |= bad

    # Control: with FACTOR=1e5 the IBS update visibly widens the momentum spread. The tracked beam
    # does not depend on the execution mode, so Pelegant must reproduce serial elegant.
    final = {mode: track(mode, columns, factor=1e5)[1] for mode in ("serial", *modes(mpi_only=True))}
    reference = final["serial"]
    initial = np.std(columns["p"]) / B0
    print(f"FACTOR=1e5 rms dp/p: initial={initial:.16g}")
    if np.std(reference["p"]) / B0 < MIN_GROWTH * initial:
        sys.exit("FACTOR=1e5: the serial update does not widen the momentum spread visibly")
    for mode, out in final.items():
        print(f"FACTOR=1e5 {mode}: final rms dp/p={np.std(out['p']) / B0:.16g}")
        # Same particles (all kept, same IDs) and the same six coordinates as the serial run.
        if len(out["p"]) != COUNT or not np.array_equal(out["particleID"], columns["particleID"]):
            sys.exit(f"FACTOR=1e5: {mode} does not hold the {COUNT} input particles")
        for name in ("x", "xp", "y", "yp", "t", "p"):
            if not np.allclose(out[name], reference[name], rtol=1e-12, atol=1e-18):
                sys.exit(f"FACTOR=1e5: {mode} column {name} differs from serial")
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
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
| Found (unset vertical growth-rate parameter) | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed (zero local-rate columns) | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

