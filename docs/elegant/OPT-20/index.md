---
title: "OPT-20: Matrix-method tune shift with amplitude uses the wrong betatron slope phase"
hide:
  - toc
---

# OPT-20 — Matrix-method tune shift with amplitude uses the wrong betatron slope phase

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`computeTuneShiftWithAmplitudeM`, src/twiss.cc:4556–4561](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L4556-L4561)<br>[outputs are derivatives of the tunes with respect to A, doc/elegant.tex:8412–8413](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8412-L8413)<br>[concatenated multi-turn third-order matrix method appears reliable for many cases, doc/elegant.tex:8403–8404](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8403-L8404)<br>[deprecates the command, doc/elegant.tex:8389–8392](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8389-L8392)<br>[the matrix method does not work well with all lattices, doc/elegant.tex:8442–8446](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8442-L8446) |

Elegant's `tune_shift_with_amplitude` with `turns=0` averages the multi-turn third-order map over the betatron ellipse with the slope phase written as α sin θ + cos θ instead of sin θ + α cos θ, where α is the Twiss alpha at the start of the ring. The coefficients then depend on the starting point, and the cross terms dνx/dAy and dνy/dAx differ. For eight FODO cells with one thin octupole of K3L = 3 m⁻³, first-order perturbation theory gives dνx/dAx = 1.411, dνx/dAy = dνy/dAx = −7.538 and dνy/dAy = 10.07 m⁻¹; Elegant gives 3.658, −29.47, −19.54 and 39.35 m⁻¹ from one start and 1.416, −86.53, −7.561 and 115.5 m⁻¹ from another. The tracking method (`turns` > 0) agrees with theory.

??? example "Reproducer: OPT-20.py"

    [Download OPT-20.py](OPT-20.py)

    ```python
    """OPT-20: Matrix-method tune shift with amplitude uses the wrong betatron slope phase

    Elegant's `tune_shift_with_amplitude` with `turns=0` averages the multi-turn third-order map over
    the betatron ellipse with the slope phase written as α sin θ + cos θ instead of sin θ + α cos θ,
    where α is the Twiss alpha at the start of the ring. The coefficients then depend on the starting
    point, and the cross terms dνx/dAy and dνy/dAx differ. For eight FODO cells with one thin octupole
    of K3L = 3 m⁻³, first-order perturbation theory gives dνx/dAx = 1.411, dνx/dAy = dνy/dAx = −7.538
    and dνy/dAy = 10.07 m⁻¹; Elegant gives 3.658, −29.47, −19.54 and 39.35 m⁻¹ from one start and 1.416,
    −86.53, −7.561 and 115.5 m⁻¹ from another. The tracking method (`turns` > 0) agrees with theory.

    The script runs serial Elegant `tune_shift_with_amplitude turns=0` with `twiss_output
    concat_order=3` on eight FODO cells of first-order quadrupoles and one thin octupole, starting
    the same ring at the octupole and at a drift. The expected coefficients are the first-order
    thin-octupole values K3L/(32π)·[βx², −2βxβy, −2βxβy, βy²], with the beta functions at the
    octupole computed from analytic thick-quadrupole and drift matrices; Elegant's own tune and beta
    functions at the octupole are controls.
    """

    import sys

    import numpy as np
    from _elegant import run, verdict

    K3L = 3.0  # thin octupole strength, m^-3
    QF, QD, LQ = 1.1, -1.0, 0.25  # quadrupole K1 (m^-2) and length (m)
    LD, LE = 0.7, 0.9  # drift lengths, m
    LATTICE = f"""F: QUAD,L={LQ},K1={QF},ORDER=1
    G: QUAD,L={LQ},K1={QD},ORDER=1
    D: DRIF,L={LD}
    E: DRIF,L={LE}
    O: MULT,ORDER=3,KNL={K3L}
    C: LINE=(F,D,G,E)
    A: LINE=(O,8*C)
    B: LINE=(D,G,E,7*C,O,F)
    """
    NAMES = ["dnux/dAx", "dnux/dAy", "dnuy/dAx", "dnuy/dAy"]
    RTOL = 1e-3


    def quad(k, length):
        if k > 0:
            w = np.sqrt(k)
            return np.array(
                [
                    [np.cos(w * length), np.sin(w * length) / w],
                    [-w * np.sin(w * length), np.cos(w * length)],
                ]
            )
        w = np.sqrt(-k)
        return np.array(
            [
                [np.cosh(w * length), np.sinh(w * length) / w],
                [w * np.sinh(w * length), np.cosh(w * length)],
            ]
        )


    def drift(length):
        return np.array([[1.0, length], [0.0, 1.0]])


    def periodic_beta(m):
        """Periodic beta function and tune of a 2x2 one-cell matrix (8 cells per turn)."""
        cos_mu = (m[0, 0] + m[1, 1]) / 2
        sin_mu = np.sign(m[0, 1]) * np.sqrt(1 - cos_mu**2)
        mu = np.arctan2(sin_mu, cos_mu) % (2 * np.pi)
        return m[0, 1] / sin_mu, 8 * mu / (2 * np.pi)


    # The octupole sits at the cell start (entrance of F). The cell matrix in transport order is
    # E G D F (the rightmost acts first); y sees -K1.
    beta_x, nux = periodic_beta(drift(LE) @ quad(QD, LQ) @ drift(LD) @ quad(QF, LQ))
    beta_y, nuy = periodic_beta(drift(LE) @ quad(-QD, LQ) @ drift(LD) @ quad(-QF, LQ))
    # Thin octupole H = K3L (x^4 - 6 x^2 y^2 + y^4)/24 at s = 0, x = sqrt(2 Jx betax) cos(phi): the
    # phase average gives <H> = K3L (betax^2 Jx^2 - 4 betax betay Jx Jy + betay^2 Jy^2)/16, so
    # dnu/dJ = (1/2 pi) d2<H>/dJ dJ, and with A = 2J (Elegant's amplitude) the coefficients below.
    expected = (
        K3L
        / (32 * np.pi)
        * np.array([beta_x**2, -2 * beta_x * beta_y, -2 * beta_x * beta_y, beta_y**2])
    )


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in values) + "]"


    defect = False
    print(f"analytic: nux={nux:.10g} nuy={nuy:.10g} betax(O)={beta_x:.10g} m betay(O)={beta_y:.10g} m")
    for line in ("A", "B"):
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline={line}, p_central_mev=2400,"
            " default_order=3 &end\n"
            "&tune_shift_with_amplitude turns=0 &end\n"
            "&twiss_output filename=twiss.sdds, matched=1, output_at_each_step=1, concat_order=3 &end\n"
            "&run_control &end\n&bunched_beam &end\n&track &end\n"
        )
        table = run({"lattice.lte": LATTICE, "run.ele": commands}, outputs=("twiss.sdds",))[
            "twiss.sdds"
        ]
        names = list(table[0, "ElementName"])
        at = names.index("O")
        tunes = [float(table["nux"][0]), float(table["nuy"][0])]
        betas = [float(table[0, "betax"][at]), float(table[0, "betay"][at])]
        alphas = [float(table[0, "alphax"][0]), float(table[0, "alphay"][0])]
        print(
            f"start {line} control nux, nuy, betax(O), betay(O): observed={fmt(tunes + betas)}"
            f" expected={fmt([nux, nuy, beta_x, beta_y])}"
        )
        if not np.allclose(tunes + betas, [nux, nuy, beta_x, beta_y], rtol=1e-9):
            sys.exit("Elegant's linear optics disagree with the analytic matrices: invalid comparison.")
        observed = np.array([float(table[name][0]) for name in NAMES])
        if not np.all(observed):
            sys.exit("Elegant wrote zero tune-shift coefficients: the command did not run.")
        print(f"start {line} alphax, alphay at the start: {fmt(alphas)}")
        print(
            f"start {line} {', '.join(NAMES)} (1/m): observed={fmt(observed)} expected={fmt(expected)}"
        )
        print(f"start {line} observed/expected: {fmt(observed / expected)}")
        defect |= not np.allclose(observed, expected, rtol=RTOL)
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
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

