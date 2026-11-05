---
title: "COL-04: Nonlinear space charge ignores the horizontal and vertical plane switches"
hide:
  - toc
---

# COL-04 — Nonlinear space charge ignores the horizontal and vertical plane switches

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [applySCKick, insertSCeffects.cc:53–67](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L53-L67)<br>[unconditional kicks, insertSCeffects.cc:64–65](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L64-L65)<br>[nonlinearSCKick, insertSCeffects.cc:531–605](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L531-L605)<br>[switch storage, insertSCeffects.cc:124–125](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L124-L125)<br>[linear gating, insertSCeffects.cc:521–527](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L521-L527)<br>[switch description, elegant.tex:4485–4488](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4485-L4488)<br>[namelist defaults, insertSCeffects.nl:16–19](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.nl#L16-L19) |

With `insert_sceffects` set to `nonlinear=1`, SCMULT ignores the `horizontal` and `vertical` switches that enable space charge in each plane. For unsliced Gaussian bunches, successful nonlinear evaluations give the same kicks with either or both planes disabled; linear controls obey the switches. In the round-beam reproduction with 1 nC of electrons at 10 mₑc and 1 mm rms transverse sizes, a sampled particle receives −3.309×10⁻⁵ rad in each plane after a 0.5 m drift even with both switches off, instead of zero.

??? example "Reproducer: COL-04.py"

    [Download COL-04.py](COL-04.py)

    ```python
    """COL-04: Nonlinear space charge ignores the horizontal and vertical plane switches

    With `insert_sceffects` set to `nonlinear=1`, SCMULT ignores the `horizontal` and `vertical` switches that enable space charge in each plane. For unsliced Gaussian bunches, successful nonlinear evaluations give the same kicks with either or both planes disabled; linear controls obey the switches. In the round-beam reproduction with 1 nC of electrons at 10 mₑc and 1 mm rms transverse sizes, a sampled particle receives −3.309×10⁻⁵ rad in each plane after a 0.5 m drift even with both switches off, instead of zero.

    The script tracks 512 electrons through one drift of a closed four-bend ring with an SCMULT after
    it, with round and 2:1 elliptical beams and each plane selection, and reads each particle's slope
    kick. For every nonlinear run with a plane disabled it requires a finite, nonzero kick in each
    disabled component that equals the kick measured with both planes enabled for the same beam. The
    controls are the physical Gaussian field from the Poisson integral for the enabled components of
    the elliptical beam, the linear force with the same plane selections, and the all-enabled
    nonlinear run.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _scring import beam, compare, kick, levels, track

    SIGMA = 1e-3  # m, rms horizontal size
    TIMES = 1e-10 * levels(8)  # s, arrival times of unit-rms Gaussian quantile levels, sigma_t = 0.1 ns
    LENGTH = 0.5  # m, drift with the SCMULT
    RTOL = 1e-6  # the elliptical nonlinear field is accurate to 8e-8, see the Gaussian field entry

    ATOL = 1e-18  # rad

    # One entry per run: True when every disabled component of a nonlinear run keeps the full kick.
    leaks = []
    for mode in modes():
        reference = {}  # sy/sx -> kick measured with both planes enabled, nonlinear
        # The all-enabled nonlinear run of each beam comes first: it is the reference for the others.
        for ratio, nonlinear, planes in (
            (2.0, 0, (0, 0)),
            (2.0, 0, (1, 0)),
            (2.0, 1, (1, 1)),
            (2.0, 1, (0, 0)),
            (2.0, 1, (1, 0)),
            (2.0, 1, (0, 1)),
            (1.0, 1, (1, 1)),
            (1.0, 1, (0, 0)),
        ):
            particles_in = beam(SIGMA, ratio * SIGMA, TIMES)
            observed, leff = track(
                mode,
                LENGTH,
                particles_in,
                horizontal=planes[0],
                vertical=planes[1],
                nonlinear=nonlinear,
            )
            # The physical kick of an enabled plane, zero for a disabled plane. The enabled kick of a
            # round beam is not compared, since it has its own length defect: the round runs are
            # judged against the round kick measured with both planes enabled instead.
            enabled = np.array(planes, dtype=bool)
            expected = np.where(enabled, kick(particles_in, leff, nonlinear=bool(nonlinear)), 0.0)
            name = f"{mode} {'nonlinear' if nonlinear else 'linear'} sy/sx={ratio:g} h,v={planes[0]}{planes[1]}"
            bad = compare(name, observed, expected, RTOL)
            if nonlinear and planes == (1, 1):
                reference[ratio] = observed
            if not nonlinear or (ratio == 2.0 and planes == (1, 1)):
                if bad:
                    sys.exit("A control disagrees with the physical kick.")
            elif ratio == 2.0 and not np.allclose(
                observed[:, enabled], expected[:, enabled], rtol=RTOL, atol=ATOL
            ):
                sys.exit("An enabled component disagrees with the physical kick.")
            if nonlinear and planes != (1, 1):
                disabled = observed[:, ~enabled]
                leaks.append(
                    bool(
                        np.all(disabled != 0)
                        and np.allclose(disabled, reference[ratio][:, ~enabled], rtol=RTOL, atol=ATOL)
                    )
                )
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
    verdict(bool(leaks) and all(leaks))

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

??? example "Helper: _scring.py"

    [Download _scring.py](_scring.py)

    ```python
    """Shared setup for the SCMULT space-charge reproducers: a closed four-bend ring, a gridded beam and
    the Gaussian-beam kick computed from the Poisson integral, independently of Elegant's code.

    One turn around a ring of four 1 m sector bends (angle pi/2, K1 = -0.5 m^-2), each preceded by a
    drift of length L. `insert_sceffects` places one SCMULT after the first drift, and a WATCH right
    after it records every particle's slope kick before the first bend.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import particles, run
    from scipy.integrate import quad
    from scipy.special import ndtri

    # SDDS constants.h, as Elegant uses them.
    C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
    P = 10.0  # momentum, units of m_e c (electrons)
    GAMMA = np.sqrt(1 + P**2)
    BETA = P / GAMMA
    CHARGE = 1e-9  # C
    BEND_LENGTH = 1.0  # m, four bends make the ring


    def levels(n: int, uniform: bool = False) -> np.ndarray:
        """n symmetric levels of unit rms: Gaussian quantiles, or equally spaced for a uniform beam."""
        q = (np.arange(n) + 0.5) / n
        q = q - 0.5 if uniform else ndtri(q)
        return q / np.std(q)


    def beam(
        sx: float, sy: float, times: np.ndarray, n: int = 8, copies: int = 1
    ) -> dict[str, np.ndarray]:
        """Particles on the full grid of n levels in x and y and the given arrival times, zero slopes.

        The rms sizes are exactly sx and sy, which is what SCMULT reads from the particles.
        """
        q = levels(n)
        x, y, t = (a.ravel() for a in np.meshgrid(sx * q, sy * q, times, indexing="ij"))
        x, y, t = (np.tile(a, copies) for a in (x, y, t))
        return {"x": x, "y": y, "t": t, "p": np.full(len(x), P), "particleID": np.arange(1, len(x) + 1)}


    def circumference(length: float) -> float:
        return 4 * (BEND_LENGTH + length)


    def track(
        mode: str,
        length: float,
        particles_in: dict[str, np.ndarray],
        *,
        horizontal: int = 1,
        vertical: int = 1,
        nonlinear: int = 1,
        uniform: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slope kicks (xp, yp) of every particle after SCMULT, in input order, and the effective
        integrating lengths of the two planes.

        The effective length of a drift of constant beam size is L (beta_in + beta_out) / (2 beta_out),
        from the trapezoid weighting that SCMULT applies to the matched beta functions at the entrance
        and exit of the drift; the beta functions are read from the matched Twiss output.
        """
        lattice = (
            f"Q: CHARGE,TOTAL={CHARGE!r}\n"
            f"D: DRIF,L={length!r}\n"
            'W: WATCH,FILENAME="watch.sdds",MODE="coordinates"\n'
            f"A: DRIF,L={length!r}\n"
            f"B: SBEND,L={BEND_LENGTH!r},ANGLE={np.pi / 2!r},K1=-0.5\n"
            "BL: LINE=(Q,D,W,B,A,B,A,B,A,B)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f'&insert_sceffects name="D",skip=1,horizontal={horizontal},vertical={vertical},'
            f"nonlinear={nonlinear},uniform_distribution={uniform} &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            '&twiss_output filename="twiss.sdds",matched=1 &end\n'
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": particles_in},
            outputs=("watch.sdds", "twiss.sdds"),
        )
        twiss = result["twiss.sdds"]
        names = list(twiss[0, "ElementName"])
        if not names.index("D") < names.index("MYSC") < names.index("W"):
            raise RuntimeError(f"unexpected element order {names}")
        leff = np.array(
            [
                length * (b[0] + b[names.index("D")]) / (2 * b[names.index("MYSC")])
                for b in (twiss[0, "betax"], twiss[0, "betay"])
            ]
        )
        if not (np.all(np.isfinite(leff)) and np.all(leff > 0)):
            raise RuntimeError(f"{mode}: invalid effective lengths {leff}")
        out = particles(result["watch.sdds"])
        if not np.array_equal(out["particleID"], particles_in["particleID"]):
            raise RuntimeError(f"{mode}: particles lost before the kick")
        # The initial slopes are zero, so the slopes after SCMULT are the kicks.
        return np.column_stack([out["xp"], out["yp"]]), leff


    def poisson_field(x: np.ndarray, y: np.ndarray, sx: float, sy: float) -> np.ndarray:
        """Transverse field (Fx, Fy), in 1/m, of a Gaussian charge density with rms sizes sx and sy.

        Heat-kernel solution of the 2D Poisson equation, written with positive real integrands:
          Fx = x/2 int_0^inf exp[-x^2/(2(sx^2+t)) - y^2/(2(sy^2+t))] / [(sx^2+t)^(3/2) (sy^2+t)^(1/2)] dt
        and Fy with the planes exchanged. Round beams give Gauss's law,
        F = (1 - exp[-r^2/(2 sigma^2)]) (x, y)/r^2, and the origin gives the linear force,
        Fx = x/[sx (sx + sy)].
        """
        r2 = (sy / sx) ** 2  # t = sx^2 u makes the integrals dimensionless

        def integral(a: float, b: float, plane: int) -> float:
            def integrand(u: float) -> float:
                common = np.exp(-(a**2) / (2 * (1 + u)) - b**2 / (2 * (r2 + u)))
                common /= 2 * np.sqrt((1 + u) * (r2 + u))
                return common * (a / (1 + u) if plane == 0 else b / (r2 + u)) / sx

            return quad(integrand, 0, np.inf, epsabs=0, epsrel=1e-13)[0]

        pairs, inverse = np.unique(np.column_stack([x, y]), axis=0, return_inverse=True)
        unique = np.array(
            [[integral(px / sx, py / sx, 0), integral(px / sx, py / sx, 1)] for px, py in pairs]
        )
        return unique[inverse.reshape(-1)]


    def kick(
        particles_in: dict[str, np.ndarray],
        leff: np.ndarray,
        *,
        nonlinear: bool,
        uniform: bool = False,
    ) -> np.ndarray:
        """Physical slope kick (xp, yp) of every particle, shape (n, 2).

        dx' = K L lambda F, with K = 2 r_e / (beta^2 gamma^3) and L the effective length of the plane;
        K multiplies the particle number per metre lambda, not a charge density.
        beta^2 gamma^3 = P^2 sqrt(1 + P^2). The longitudinal density lambda = N f(z), in particles per
        metre with N = CHARGE/e, is Gaussian, f = exp(-z^2/(2 sz^2))/(sqrt(2 pi) sz), or uniform over the
        interval of the same rms, f = 1/(sqrt(12) sz). The field F is the Gaussian transverse field
        (nonlinear) or its linear part near the axis.
        """
        sx, sy = np.std(particles_in["x"]), np.std(particles_in["y"])
        x = particles_in["x"] - np.mean(particles_in["x"])
        y = particles_in["y"] - np.mean(particles_in["y"])
        z = BETA * C * (particles_in["t"] - np.mean(particles_in["t"]))
        sz = np.std(z)
        shape = np.full_like(z, 1 / (np.sqrt(12) * sz)) if uniform else _gauss(z, sz)
        strength = 2 * (CHARGE / QE) * RE / (P**2 * np.sqrt(1 + P**2))
        if nonlinear:
            field = poisson_field(x, y, sx, sy)
        else:
            field = np.column_stack([x / (sx * (sx + sy)), y / (sy * (sx + sy))])
        return strength * shape[:, None] * leff[None, :] * field


    def _gauss(z: np.ndarray, sz: float) -> np.ndarray:
        return np.exp(-(z**2) / (2 * sz**2)) / (np.sqrt(2 * np.pi) * sz)


    def fmt(values: np.ndarray) -> str:
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"


    def require_finite(label: str, *arrays: np.ndarray) -> None:
        """Fail the run when an array holds a NaN or infinity; NaN would otherwise pass as a mismatch."""
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise RuntimeError(f"{label}: nonfinite values")


    def compare(
        label: str, observed: np.ndarray, expected: np.ndarray, rtol: float, atol: float = 1e-18
    ) -> bool:
        """Print the first particle's (xp, yp) and return True when any particle disagrees.

        Arrays of different shape or with a nonfinite entry make the comparison invalid: the run fails
        instead of returning a verdict.
        """
        observed, expected = np.asarray(observed), np.asarray(expected)
        if observed.shape != expected.shape:
            raise RuntimeError(f"{label}: shapes {observed.shape} and {expected.shape} differ")
        require_finite(label, observed, expected)
        print(f"{label}: observed={fmt(observed[0])} expected={fmt(expected[0])}")
        return not np.allclose(observed, expected, rtol=rtol, atol=atol)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

