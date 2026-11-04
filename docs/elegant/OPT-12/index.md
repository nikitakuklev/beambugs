---
title: "OPT-12: Unmatched chromatic derivatives use exit dispersion instead of entrance dispersion"
hide:
  - toc
---

# OPT-12 — Unmatched chromatic derivatives use exit dispersion instead of entrance dispersion

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`computeChromaticities`, chrom.c:839–844](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/chrom.c#L839-L844)<br>[`computeChromaticDerivRElem`, chrom.c:962–990](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/chrom.c#L962-L990)<br>[`compute_twiss_parameters`, twiss.cc:2332–2334](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L2332-L2334)<br>[`matched=0` starts the calculation from the given initial values, elegant.tex:8076–8078](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8076-L8078)<br>[initial values of the X plane, elegant.tex:8130–8131](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8130-L8131)<br>[chromatic derivatives of the beta functions, elegant.tex:8232–8233](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8232-L8233) |

With `twiss_output matched=0`, Elegant uses exit dispersion where the chain rule requires entrance dispersion to compute chromatic beta functions and phase advances. For a thin sextupole of integrated strength 1 m⁻² followed by a 1 m drift, take entrance beta functions βₓ = 3 m and βᵧ = 2 m, alpha αₓ = 0.4, horizontal dispersion ηₓ = 0 and slope ηₓ′ = 0.2. The analytic drift Jacobian and tracking give zero derivatives with respect to relative momentum offset δ, but the output gives dβₓ/dδ = −1.040 m, dβᵧ/dδ = 0.8000 m, dνₓ/dδ = 0.01231 and dνᵧ/dδ = −0.01273, where ν is phase advance in cycles. Users matching chromatic functions in dispersive transport lines are affected.

??? example "Reproducer: OPT-12.py"

    [Download OPT-12.py](OPT-12.py)

    ```python
    """OPT-12: Unmatched chromatic derivatives use exit dispersion instead of entrance dispersion

    With `twiss_output matched=0`, Elegant uses exit dispersion where the chain rule requires entrance
    dispersion to compute chromatic beta functions and phase advances. For a thin sextupole of
    integrated strength 1 m⁻² followed by a 1 m drift, take entrance beta functions βₓ = 3 m and
    βᵧ = 2 m, alpha αₓ = 0.4, horizontal dispersion ηₓ = 0 and slope ηₓ′ = 0.2. The analytic drift
    Jacobian and tracking give zero derivatives with respect to relative momentum offset δ, but the
    output gives dβₓ/dδ = −1.040 m, dβᵧ/dδ = 0.8000 m, dνₓ/dδ = 0.01231 and dνᵧ/dδ = −0.01273, where ν
    is phase advance in cycles. Users matching chromatic functions in dispersive transport lines are
    affected.

    The script transports the entrance optics through the sextupole and drift in serial Elegant (and
    with `--mpi` in 3-rank Pelegant), and compares the four written derivatives with those of the beta
    functions and phase advances reconstructed from tracked particle pairs around the dispersive
    reference orbit at δ = -0.01, 0 and 0.01. Two controls, a zero-strength sextupole and a zero
    entrance dispersion slope, remove the product of dispersion and quadratic focusing that the
    contraction needs; both must give zero.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    P0 = 1000.0  # reference momentum, units of m_e c
    ENTRANCE = {"x": (3.0, 0.4), "y": (2.0, 0.0)}  # (beta, alpha) at the entrance, in m
    ETA0 = 0.0  # entrance dispersion (m); the slope is varied between cases
    DRIFT = 1.0  # m
    DELTAS = (-0.01, 0.0, 0.01)
    PROBE = 1e-5  # half-step of the Jacobian probes
    COPIES = 2  # copies of each particle, so that each MPI worker tracks some
    ATOL = 1e-10


    def commands(etap):
        return (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, default_order=2,\n"
            " output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&twiss_output filename=twiss.sdds, concat_order=2, matched=0,\n"
            f" beta_x={ENTRANCE['x'][0]!r}, alpha_x={ENTRANCE['x'][1]!r},\n"
            f" beta_y={ENTRANCE['y'][0]!r}, alpha_y={ENTRANCE['y'][1]!r},\n"
            f" eta_x={ETA0!r}, etap_x={etap!r} &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )


    def probe_beam(etap):
        """Eight particles per delta, at +-PROBE in x, xp, y and yp around the reference orbit.

        The reference orbit of momentum P0*(1 + delta) starts at (eta*delta, eta'*delta) in x and xp,
        the entrance dispersion times delta, and on axis in y.
        """
        rows = []
        for delta in DELTAS:
            for axis in range(4):
                for sign in (1, -1):
                    coordinates = np.zeros(4)
                    coordinates[axis] = sign * PROBE
                    coordinates[:2] += (ETA0 * delta, etap * delta)
                    rows.append((*coordinates, P0 * (1 + delta)))
        x, xp, y, yp, p = np.repeat(np.array(rows), COPIES, axis=0).T
        return {"x": x, "xp": xp, "y": y, "yp": yp, "p": p, "particleID": np.arange(1, len(x) + 1)}


    def tracked_optics(out):
        """Exit beta function and phase advance (in cycles) of each plane at each delta.

        Symmetric differences about the reference orbit give the transfer matrix J at each delta.
        With the entrance beta0 and alpha0 and gamma0 = (1 + alpha0^2)/beta0,
        beta1 = J11^2 beta0 - 2 J11 J12 alpha0 + J12^2 gamma0 and the phase advance is
        atan2(J12, J11 beta0 - J12 alpha0) / (2 pi).
        """
        coordinates = np.column_stack([out[name] for name in ("x", "xp", "y", "yp")])
        if not np.array_equal(coordinates, np.repeat(coordinates[::COPIES], COPIES, axis=0)):
            sys.exit("Identical particles left the element differently.")
        # Index: delta, probed axis (x, xp, y, yp), sign (+, -), output coordinate.
        exit_ = coordinates[::COPIES].reshape(len(DELTAS), 4, 2, 4)
        jacobian = (exit_[:, :, 0] - exit_[:, :, 1]) / (2 * PROBE)  # [delta, probed axis, output]
        optics = {}
        for plane, (beta0, alpha0) in ENTRANCE.items():
            first = 0 if plane == "x" else 2  # index of the plane's position coordinate
            j11 = jacobian[:, first, first]  # d position_out / d position_in
            j12 = jacobian[:, first + 1, first]  # d position_out / d slope_in
            gamma0 = (1 + alpha0**2) / beta0
            beta = j11**2 * beta0 - 2 * j11 * j12 * alpha0 + j12**2 * gamma0
            phase = np.arctan2(j12, j11 * beta0 - j12 * alpha0) / (2 * np.pi)
            optics[plane] = (beta, phase)
        return optics


    def case(mode, knl, etap):
        """Written (dbetax/dp, dbetay/dp, dnux/dp, dnuy/dp), the tracked derivatives and exit betas."""
        lattice = f"S: MULT,ORDER=2,KNL={knl!r}\nD: DRIF,L={DRIFT!r}\nBL: LINE=(S,D)\n"
        result = run(
            {"lattice.lte": lattice, "run.ele": commands(etap)},
            mode,
            {"beam.sdds": probe_beam(etap)},
            outputs=("twiss.sdds", "out.sdds"),
        )
        twiss = result["twiss.sdds"]
        written = [float(twiss[name][0]) for name in ("dbetax/dp", "dbetay/dp", "dnux/dp", "dnuy/dp")]
        optics = tracked_optics(particles(result["out.sdds"]))
        span = DELTAS[2] - DELTAS[0]
        tracked = [(optics[p][k][2] - optics[p][k][0]) / span for k in (0, 1) for p in "xy"]
        return written, tracked, [optics[p][0][1] for p in "xy"]


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected disagree."""
        observed, expected = np.asarray(observed), np.asarray(expected)
        if not (np.isfinite(observed).all() and np.isfinite(expected).all()):
            sys.exit(f"{label}: nonfinite value, the comparison is not valid.")
        bad = bool(np.any(np.abs(observed - expected) > ATOL))
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    # The reference orbit of any momentum crosses the sextupole at x = eta*delta = 0, where a thin
    # sextupole kick (proportional to x^2 or x*y) and its derivatives (proportional to x) vanish. The
    # map is therefore the drift [[1, L], [0, 1]] in each plane for every delta, so the exit beta
    # function beta0 - 2 L alpha0 + L^2 gamma0 and the phase advance atan2(L, beta0 - L alpha0)/(2 pi)
    # do not depend on delta: all four derivatives are zero.
    EXPECTED_BETA = [
        b - 2 * DRIFT * a + DRIFT**2 * (1 + a**2) / b for b, a in (ENTRANCE["x"], ENTRANCE["y"])
    ]

    defect = False
    for mode in modes():
        for label, knl, etap in (
            ("control KNL=0", 0.0, 0.2),
            ("control etap_x=0", 1.0, 0.0),
            ("KNL=1, etap_x=0.2", 1.0, 0.2),
        ):
            written, tracked, exit_beta = case(mode, knl, etap)
            if compare(f"{mode} {label}: tracked exit beta x, y", exit_beta, EXPECTED_BETA):
                sys.exit("The tracked beta function disagrees with the analytic map: invalid oracle.")
            if compare(f"{mode} {label}: tracked dbeta/ddelta x, y, dnu/ddelta x, y", tracked, [0] * 4):
                sys.exit("The tracked derivatives are not zero: invalid oracle.")
            bad = compare(
                f"{mode} {label}: written dbetax/dp, dbetay/dp, dnux/dp, dnuy/dp", written, [0] * 4
            )
            if knl == 0.0 or etap == 0.0:
                if bad:
                    sys.exit("A control has nonzero derivatives: the comparison is not valid.")
                continue
            defect |= bad
            # Elegant contracts the second-order map T with the exit dispersion eta_exit = eta0 + L
            # eta0' instead of the entrance dispersion. For the combined x map on y = 0,
            # x1 = x + L x' - L KNL x^2 / 2 (T111 = -L KNL/2, R11 = 1, R12 = L),
            # dR11/ddelta = 2 T111 eta_exit; with dR12/ddelta = 0,
            # d beta1/d delta = 2 (dR11/ddelta) (beta0 - L alpha0) = -1.04 m.
            eta_exit = ETA0 + DRIFT * etap
            predicted = 2 * (-DRIFT * knl * eta_exit) * (ENTRANCE["x"][0] - DRIFT * ENTRANCE["x"][1])
            print(f"{mode} {label}: exit-dispersion contraction predicts dbetax/dp={predicted:.16g}")
    print(f"comparison: absolute tolerance {ATOL:g}")
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
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

