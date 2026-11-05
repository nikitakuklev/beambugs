---
title: "OPT-04: Initial moments omit shared-dispersion covariance terms"
hide:
  - toc
---

# OPT-04 — Initial moments omit shared-dispersion covariance terms

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`setStartingMoments`, src/moments.c:681–703](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L681-L703)<br>[the `matched=0` branch of `moments_output`, src/moments.c:573–578](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L573-L578)<br>[`matched=0`: `emit`, `beta`, `alpha`, `eta`, `etap` specify the starting beam ellipses in all three planes, manual, doc/elegant.tex:5632–5633](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5632-L5633) |

With moments_output matched=0, equilibrium=0, explicit Twiss, dispersion and longitudinal inputs produce a covariance matrix Σ for (x, x′, y, y′, s, δ) missing eight terms, where s is longitudinal position and δ is relative momentum offset. For independent betatron motion, dispersion η requires cross-plane covariances ηᵢηⱼΣ₆₆ and transverse–longitudinal covariances ηᵢΣ₆₅. For longitudinal emittance ε_z = 1 μm, β_z = 4 m, α_z = 0.2, η_x = 0.1 m and η_y = −0.03 m, Σ₁₃ must be −7.800×10⁻¹⁰ m² and Σ₁₅ −2.000×10⁻⁸ m², but both are zero. Propagated covariance matrices inherit the error. Nonzero momentum spread with dispersion in both planes, or dispersion with longitudinal position–momentum correlation, triggers the error.

??? example "Reproducer: OPT-04.py"

    [Download OPT-04.py](OPT-04.py)

    ```python
    """OPT-04: Initial moments omit shared-dispersion covariance terms

    With moments_output matched=0, equilibrium=0, explicit Twiss, dispersion and longitudinal inputs
    produce a covariance matrix Σ for (x, x′, y, y′, s, δ) missing eight terms, where s is longitudinal
    position and δ is relative momentum offset. For independent betatron motion, dispersion η requires
    cross-plane covariances ηᵢηⱼΣ₆₆ and transverse–longitudinal covariances ηᵢΣ₆₅. For longitudinal
    emittance ε_z = 1 μm, β_z = 4 m, α_z = 0.2, η_x = 0.1 m and η_y = −0.03 m, Σ₁₃ must be −7.800×10⁻¹⁰
    m² and Σ₁₅ −2.000×10⁻⁸ m², but both are zero. Propagated covariance matrices inherit the error.
    Nonzero momentum spread with dispersion in both planes, or dispersion with longitudinal
    position–momentum correlation, triggers the error.

    The script starts a tilted quadrupole and a drift from those parameters in serial Elegant (and with
    `--mpi` in 3-rank Pelegant), and compares the written covariance matrix at the start and after each
    element with the one built from independent betatron and longitudinal blocks plus dispersion and
    propagated with Elegant's first-order matrix. Two controls isolate the cause: the same run with zero
    dispersion agrees everywhere, and propagating the incomplete start matrix that Elegant wrote
    reproduces its later rows.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    # Entrance parameters: (emittance m, beta m, alpha) per plane, dispersion (eta m, eta').
    PLANES = {"x": (2e-9, 3.0, 0.4), "y": (5e-10, 2.0, -0.3), "z": (1e-6, 4.0, 0.2)}
    DISPERSION = (0.1, 0.02, -0.03, 0.01)  # eta_x, eta'_x, eta_y, eta'_y
    ATOL_REL = 1e-9  # relative to sqrt(S_ii S_jj)
    COUPLING = [(1, 3), (1, 4), (2, 3), (2, 4), (1, 5), (2, 5), (3, 5), (4, 5)]  # 1-based indices
    LATTICE = "Q: QUAD,L=0.5,K1=0.6,TILT=0.23\nD: DRIF,L=1\nBL: LINE=(Q,D)\n"


    def commands(eta):
        x, y, z = (PLANES[p] for p in "xyz")
        return (
            "&run_setup lattice=lattice.lte, use_beamline=BL, p_central=1000, default_order=2 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&matrix_output SDDS_output=matrix.sdds, SDDS_output_order=1 &end\n"
            "&moments_output filename=moments.sdds, matched=0, equilibrium=0, radiation=0,\n"
            f" emit_x={x[0]!r}, beta_x={x[1]!r}, alpha_x={x[2]!r}, eta_x={eta[0]!r}, etap_x={eta[1]!r},\n"
            f" emit_y={y[0]!r}, beta_y={y[1]!r}, alpha_y={y[2]!r}, eta_y={eta[2]!r}, etap_y={eta[3]!r},\n"
            f" emit_z={z[0]!r}, beta_z={z[1]!r}, alpha_z={z[2]!r} &end\n"
            "&bunched_beam n_particles_per_bunch=12 &end\n"
            "&track &end\n"
        )


    def start_covariance(eta):
        """Independent-block covariance of (x, x', y, y', s, delta) plus dispersion.

        Each plane has the Twiss ellipse [[eps beta, -eps alpha], [-eps alpha, eps gamma]]. A particle
        has u_i = ub_i + eta_i*delta for the four transverse coordinates, with the betatron part ub
        independent of (s, delta), so u = A w with w = (ub, s, delta) and A = [[1, eta], [0, 1]].
        """
        sigma = np.zeros((6, 6))
        for k, plane in enumerate("xyz"):
            eps, beta, alpha = PLANES[plane]
            sigma[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = eps * np.array(
                [[beta, -alpha], [-alpha, (1 + alpha**2) / beta]]
            )
        a = np.eye(6)
        a[:4, 5] = eta
        return a @ sigma @ a.T


    def run_case(mode, eta):
        """Covariance matrices and first-order transfer matrices at _BEG_, Q and D."""
        out = run(
            {"lattice.lte": LATTICE, "run.ele": commands(eta)},
            mode,
            outputs=("moments.sdds", "matrix.sdds"),
        )
        moments, matrix = out["moments.sdds"], out["matrix.sdds"]
        sigmas, transfers = [], []
        for row in range(3):
            sigma = np.zeros((6, 6))
            for i in range(1, 7):
                sigma[i - 1, i - 1] = moments[0, f"s{i}"][row] ** 2  # s_i is the rms size
                for j in range(i + 1, 7):  # s_ij is the covariance
                    sigma[i - 1, j - 1] = sigma[j - 1, i - 1] = moments[0, f"s{i}{j}"][row]
            sigmas.append(sigma)
            transfers.append(
                np.array([[matrix[0, f"R{i}{j}"][row] for j in range(1, 7)] for i in range(1, 7)])
            )
        return sigmas, transfers


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.atleast_1d(values)) + "]"


    def mismatch(observed, expected):
        """Largest |difference| of any entry, relative to sqrt(expected_ii expected_jj)."""
        if not (np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))):
            sys.exit("Nonfinite covariance matrix: invalid run.")
        if not (np.all(np.isfinite(np.diag(expected))) and np.all(np.diag(expected) > 0)):
            sys.exit("Expected covariance has a nonpositive variance: invalid run.")
        scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
        return np.max(np.abs(observed - expected) / scale)


    def pick(sigma):
        return [sigma[i - 1, j - 1] for i, j in COUPLING]


    defect = False
    print("columns: S13, S14, S23, S24, S15, S25, S35, S45 (covariances of x, x', y, y', s)")
    for mode in modes():
        # Control 1: zero dispersion. The incomplete start matrix is then complete, so every row
        # equals R S0 R^T.
        zero = (0.0, 0.0, 0.0, 0.0)
        sigmas, transfers = run_case(mode, zero)
        start = start_covariance(zero)
        worst = max(mismatch(s, r @ start @ r.T) for s, r in zip(sigmas, transfers))
        print(f"{mode} zero dispersion: max relative mismatch over all rows and entries={worst:.3g}")
        if worst > ATOL_REL:
            sys.exit("The zero-dispersion control disagrees with R S0 R^T: comparison not valid.")

        sigmas, transfers = run_case(mode, DISPERSION)
        start = start_covariance(DISPERSION)
        # Eight required entrance covariances, e.g. S13 = eta_x eta_y S66 = -7.8e-10 and
        # S15 = eta_x S65 = -2e-8 (S66 = eps gamma = 2.6e-7, S65 = -eps alpha = -2e-7).
        observed = pick(sigmas[0])
        print(f"{mode} dispersive start: observed={fmt(observed)} expected={fmt(pick(start))}")
        bad = mismatch(sigmas[0], start) > ATOL_REL
        # Downstream rows against the full start matrix propagated with the first-order matrix.
        for name, sigma, r in zip(("_BEG_", "Q", "D"), sigmas, transfers):
            error = mismatch(sigma, r @ start @ r.T)
            print(f"{mode} dispersive {name}: max relative mismatch with R S0 R^T={error:.3g}")
            bad |= error > ATOL_REL
        # Control 2: the incomplete start matrix Elegant wrote propagates correctly, so the
        # propagation is not at fault.
        error = max(mismatch(s, r @ sigmas[0] @ r.T) for s, r in zip(sigmas, transfers))
        print(f"{mode} written start matrix propagated: max relative mismatch={error:.3g}")
        if error > ATOL_REL:
            sys.exit("The emitted start matrix does not propagate to the emitted rows: not valid.")
        defect |= bad
    print(f"comparison: relative tolerance {ATOL_REL:g}")
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
| Found | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

