---
title: "COL-08: Broadband ZTRANSVERSE doubles its DC impedance"
hide:
  - toc
---

# COL-08 — Broadband ZTRANSVERSE doubles its DC impedance

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [DC coefficient, ztransverse.c:498–499](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ztransverse.c#L498-L499)<br>[interior coefficients, ztransverse.c:500–508](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ztransverse.c#L500-L508)<br>[impedance manual, ztransverse.tex:3–18](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ztransverse.tex#L3-L18)<br>[spectrum product, ztransverse.c:291–301](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ztransverse.c#L291-L301)<br>[real FFT packing, fftpackC.c:133–156](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/fftpack/fftpackC.c#L133-L156)<br>[file DC convention, ztransverse.c:575–581](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ztransverse.c#L575-L581)<br>[manual note, ztransverse.tex:33–37](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ztransverse.tex#L33-L37) |

Broadband ZTRANSVERSE sets its zero-frequency impedance to 2Rₛ/Q instead of the documented Rₛ/Q, where Rₛ is the shunt impedance and Q the resonator quality factor. This adds a constant transverse wake over the finite Fourier-transform window for beams with a net transverse offset. For 12 electrons at 50 mₑc, 2 nC total charge and 1 mm horizontal offset in adjacent 50 ps bins, with Rₛ = 1×10⁶ Ω/m, Q = 1, resonant frequency 1 GHz and 8192 bins, the leading group's kick is 6.960×10⁻⁶ rad instead of the independently summed 6.769×10⁻⁶ rad. The excess is 1.911×10⁻⁷ rad per particle and halves when the window doubles.

??? example "Reproducer: COL-08.py"

    [Download COL-08.py](COL-08.py)

    ```python
    """COL-08: Broadband ZTRANSVERSE doubles its DC impedance

    Broadband ZTRANSVERSE sets its zero-frequency impedance to 2Rₛ/Q instead of the documented Rₛ/Q, where Rₛ is the shunt impedance and Q the resonator quality factor. This adds a constant transverse wake over the finite Fourier-transform window for beams with a net transverse offset. For 12 electrons at 50 mₑc, 2 nC total charge and 1 mm horizontal offset in adjacent 50 ps bins, with Rₛ = 1×10⁶ Ω/m, Q = 1, resonant frequency 1 GHz and 8192 bins, the leading group's kick is 6.960×10⁻⁶ rad instead of the independently summed 6.769×10⁻⁶ rad. The excess is 1.911×10⁻⁷ rad per particle and halves when the window doubles.

    The script tracks the beam through the broadband element and through a table of the same impedance on the same frequency grid, and compares both with the kick of the analytic impedance, summed as a Fourier series without an FFT. The table is the control: it agrees with the series to rounding, so the kick difference of the broadband element isolates its DC coefficient. Adjacent bins with equal offsets cancel the Nyquist source moment, leaving the DC term as the only difference. The excess is compared with the kick of an extra Rs/Q coefficient, for windows of 8192 and 16384 bins.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _ztransverse import CHARGE, COUNT, DT, ELECTRON_EV, RS, P, Q, kicks, oracle, show

    ATOL = 1e-18  # rad; kicks are ~1e-5 rad, rounding of the sums ~1e-20 rad
    MIN_ERROR = 1e-12  # rad; the DC excess is ~1e-7 rad
    X = np.full(COUNT, 1e-3)  # m, equal offsets: the Nyquist source moment cancels

    defect = False
    for mode in modes():
        for n_bins in (8192, 16384):
            # Series kick of the analytic impedance (DC coefficient Rs/Q), the oracle.
            expected = oracle(X, n_bins)
            table = kicks(mode, X, n_bins, broadband=False)
            show(mode, f"N={n_bins} impedance table", table, expected)
            if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(table)):
                sys.exit("Nonfinite oracle or impedance-table kicks.")
            if not np.allclose(table, expected, rtol=0, atol=ATOL):
                sys.exit("The table control disagrees with the Fourier-series oracle.")
            broadband = kicks(mode, X, n_bins, broadband=True)
            if not np.all(np.isfinite(broadband)):
                sys.exit("Nonfinite broadband kicks.")
            show(mode, f"N={n_bins} broadband", broadband, expected)
            # A DC coefficient larger by Rs/Q adds the same slope to every particle:
            #   q (Rs/Q) sum(x) / (T p0 m c^2/e), T = N dt the window, q the macroparticle charge.
            predicted = CHARGE / COUNT * (RS / Q) * X.sum() / (n_bins * DT * P * ELECTRON_EV)
            excess = broadband - expected
            show(mode, f"N={n_bins} broadband excess", excess, np.full(2, predicted))
            defect |= bool(np.all(np.abs(excess - predicted) < ATOL) and np.all(excess > MIN_ERROR))
    print(f"tolerance {ATOL:g} rad; defect if the excess exceeds {MIN_ERROR:g} rad")
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

??? example "Helper: _ztransverse.py"

    [Download _ztransverse.py](_ztransverse.py)

    ```python
    """Shared setup for the broadband ZTRANSVERSE reproducers: decks, analytic impedance, Fourier oracle.

    The beam is twelve electrons (p = 50 m_e c, 2 nC) in two adjacent time bins of 50 ps, six particles
    each, so that the transverse source moment of each bin is six times its offset. `kicks` tracks it
    through a ZTRANSVERSE element, either the built-in broadband resonator or a table holding the
    documented impedance of the same resonator on the same frequency grid; `oracle` is the kick of the
    analytic impedance, evaluated as a finite Fourier series without Elegant's FFT.
    """

    from __future__ import annotations

    import numpy as np
    from _elegant import particles, run

    ELECTRON_EV = 0.51099906e6  # SDDS constants.h me_mev, in eV
    RS, Q, FR = 1e6, 1.0, 1e9  # shunt impedance Ohm/m, quality factor, resonant frequency Hz
    DT = 50e-12  # bin width, s
    CHARGE, COUNT, P = 2e-9, 12, 50.0
    TIMES = np.repeat([0.0, DT], COUNT // 2)  # s; group 1 in one bin, group 2 in the next


    def impedance(r):
        """Documented broadband impedance Z(r) = (-i/r) Rs / [1 + i Q (r - 1/r)], r = f/fr, Ohm/m.

        Z(0) = Rs/Q is the limit r -> 0 of the formula (the bracket tends to -iQ/r).
        """
        r = np.asarray(r, dtype=float)
        safe = np.where(r == 0, 1.0, r)
        z = (-1j / safe) * RS / (1 + 1j * Q * (safe - 1 / safe))
        return np.where(r == 0, RS / Q, z)


    def sdds_table(n_bins):
        """ASCII SDDS impedance table on the FFT frequency grid f_k = k/(N dt), k = 0..N/2.

        The loader doubles the DC row (it assumes a folded FFT of the impedance), so the table holds
        Rs/(2Q) there; it reads the real part alone at the Nyquist row.
        """
        k = np.arange(n_bins // 2 + 1)
        f = k / (n_bins * DT)
        z = impedance(f / FR)
        re = z.real.copy()
        re[0] /= 2
        head = (
            "SDDS1\n"
            "&column name=f, type=double, units=Hz &end\n"
            "&column name=re, type=double, units=Ohms/m &end\n"
            "&column name=im, type=double, units=Ohms/m &end\n"
            f"&data mode=ascii &end\n! page number 1\n{len(k)}\n"
        )
        rows = zip(f.tolist(), re.tolist(), z.imag.tolist())
        return head + "".join(f"{a!r} {b!r} {c!r}\n" for a, b, c in rows)


    def show(mode, label, observed, expected):
        """Print one case with 16 significant digits."""

        def fmt(values):
            return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"

        print(f"{mode} {label}: observed={fmt(observed)} expected={fmt(expected)}")


    def kicks(mode, x, n_bins, broadband):
        """Horizontal slope kicks of the two particle groups, from a run with offsets `x` (m)."""
        if broadband:
            element = (
                f"ZTRANSVERSE,BROAD_BAND=1,RS={RS!r},Q={Q!r},FREQ={FR!r},"
                f"N_BINS={n_bins},BIN_SIZE={DT!r},YFACTOR=0,SMOOTHING=0,INTERPOLATE=0"
            )
        else:
            element = (
                'ZTRANSVERSE,INPUTFILE="impedance.sdds",FREQCOLUMN="f",ZXREAL="re",ZXIMAG="im",'
                "YFACTOR=0,SMOOTHING=0,INTERPOLATE=0"
            )
        lattice = f"Q0: CHARGE,TOTAL={CHARGE!r}\nZ: {element}\nBL: LINE=(Q0,Z)\n"
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        files = {"lattice.lte": lattice, "run.ele": commands}
        if not broadband:
            files["impedance.sdds"] = sdds_table(n_bins)
        beam = {"x": x, "t": TIMES, "p": np.full(COUNT, P), "particleID": np.arange(1, COUNT + 1)}
        result = run(files, mode, {"beam.sdds": beam})
        if "adjust" in result["log"].lower():
            raise RuntimeError("ZTRANSVERSE changed the requested bin settings")
        out = particles(result["out.sdds"])
        if len(out["xp"]) != COUNT:
            raise RuntimeError(f"{mode}: particles lost")
        return out["xp"].reshape(2, -1).mean(axis=1)


    def oracle(x, n_bins, nyquist_re=None):
        """Kick of the analytic impedance, as slopes of the two groups.

        A real periodic signal of period T = N dt has the Green function (Fourier series)
            W[d] = [Z_0 + Re Z_{N/2} (-1)^d + 2 Re sum_{k=1}^{N/2-1} Z_k exp(2 pi i k d / N)] / T,
        with Z_k = Z(f_k), f_k = k/T, for a delay of d bins. The slope kick of a particle in bin i is
            dx' = q sum_j x_j W[b_i - b_j] / (p0 m c^2 / e),
        q the macroparticle charge in C, x_j the offsets (a source moment of the bins), p0 m c^2/e
        the beam momentum in volts. `nyquist_re` replaces Re Z_{N/2} (to evaluate a given endpoint).
        """
        period = n_bins * DT
        k = np.arange(1, n_bins // 2)
        z = impedance(np.arange(n_bins // 2 + 1) / period / FR)
        nyquist = z[-1].real if nyquist_re is None else nyquist_re
        q = CHARGE / COUNT
        source = x.reshape(2, -1).sum(axis=1)  # m, summed offset of each group
        bins = np.array([0, 1])
        delay = bins[:, None] - bins[None, :]  # b_i - b_j
        series = np.array(
            [
                z[0].real
                + nyquist * (-1.0) ** d
                + 2 * np.sum((z[k] * np.exp(2j * np.pi * k * d / n_bins)).real)
                for d in delay.ravel()
            ]
        ).reshape(2, 2)
        return q * (series @ source) / period / (P * ELECTRON_EV)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

