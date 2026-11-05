---
title: "COL-12: BEAMBEAM cancels the tracked particle's charge sign, so positrons are kicked like electrons"
hide:
  - toc
---

# COL-12 — BEAMBEAM cancels the tracked particle's charge sign, so positrons are kicked like electrons

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [Gaussian call, do_tracking.c:6316](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L6316)<br>[uniform call, do_tracking.c:6326](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L6326)<br>[parabolic call, do_tracking.c:6329](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L6329)<br>[species setup, elegant.c:3264–3283](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3264-L3283)<br>[BEAMBEAM manual, beambeam.tex:1–2](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/beambeam.tex#L1-L2)<br>[CHARGE parameter, track_data.c:3986](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3986) |

Tracking positrons through BEAMBEAM after `change_particle name=positron` gives them the same kick as electrons from a fixed opposing bunch, deflecting them in the wrong direction. Reversing the tracked charge at fixed mass and source must reverse the kick; an independent enclosed-charge calculation verifies the electron control and predicts the opposite positron kick. For a positive 0.7000 nC uniform opposing bunch of radius 1.700 mm, a positron with momentum 1700 mₑc at horizontal position x = 0.2300 mm receives a horizontal slope change Δx′ = −1.153×10⁻⁶ instead of +1.153×10⁻⁶.

??? example "Reproducer: COL-12.py"

    [Download COL-12.py](COL-12.py)

    ```python
    """COL-12: BEAMBEAM cancels the tracked particle's charge sign, so positrons are kicked like electrons

    Tracking positrons through BEAMBEAM after `change_particle name=positron` gives them the same kick
    as electrons from a fixed opposing bunch, deflecting them in the wrong direction. Reversing the
    tracked charge at fixed mass and source must reverse the kick; an independent enclosed-charge
    calculation verifies the electron control and predicts the opposite positron kick. For a positive
    0.7000 nC uniform opposing bunch of radius 1.700 mm, a positron with momentum 1700 mₑc at horizontal
    position x = 0.2300 mm receives a horizontal slope change Δx′ = −1.153×10⁻⁶ instead of +1.153×10⁻⁶.

    The script tracks 12 particles at x = ±0.23 mm through one BEAMBEAM element, as electrons and as
    positrons, for each distribution and for opposing charges of both signs. Expected kicks come from
    the enclosed-charge field of the manual's density, independent of Elegant's helper routines. The
    electron kick is the control that fixes the oracle's sign and magnitude; the positron kick must
    then be its negative.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    # SDDS constants.h, as Elegant uses them.
    C, QE, ME, RE = 299792458.0, 1.60217733e-19, 9.1093897e-31, 2.81794092e-15
    EPS0 = 1e7 / (4 * np.pi * C**2)
    P0 = 1700.0  # momentum, units of m_e c
    RADIUS, X0, SOURCE = 1.7e-3, 0.23e-3, 0.7e-9  # m, m, C
    BETA = P0 / np.hypot(P0, 1)
    N = 12
    RTOL = 4e-14


    def kick_magnitude(distribution):
        """|Delta x'| of a particle at |x| = X0 for an opposing bunch of charge +SOURCE.

        A thin slug with enclosed charge Q_enc gives the integrated field
        int E dz = Q_enc/(2 pi eps0 r). The enclosed fraction of the transverse density is
        u = r^2/a^2 (uniform), 2u - u^2 (parabolic) and 1 - exp(-u/2) (Gaussian, a = sigma).
        Delta x' = e int E dz/(m c^2 p0 beta); the Gaussian routine uses its ultrarelativistic
        form Delta x' = 2 N r_e (enclosed fraction)/(p0 x), with N = Q/e.
        """
        u = (X0 / RADIUS) ** 2
        if distribution == "gaussian":
            return 2 * RE * SOURCE / QE * -np.expm1(-u / 2) / (P0 * X0)
        enclosed = u if distribution == "uniform" else 2 * u - u**2
        integrated_field = SOURCE * enclosed / (2 * np.pi * EPS0 * X0)
        return QE * integrated_field / (ME * C**2 * P0 * BETA)


    def kicks(mode, distribution, species, polarity):
        """Slopes xp of the 12 particles after BEAMBEAM with opposing charge polarity * SOURCE."""
        lattice = (
            f"K: BEAMBEAM,DISTRIBUTION={distribution},CHARGE={polarity * SOURCE!r},"
            f"XSIZE={RADIUS!r},YSIZE={RADIUS!r}\n"
            "BL: LINE=(K)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&change_particle name={species} &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {"x": np.tile([X0, -X0], N // 2), "p": np.full(N, P0)}
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})
        out = result["out.sdds"]
        xp = np.asarray(out[0, "xp"])
        if len(xp) != N or not np.all(np.isfinite(xp)):
            sys.exit(f"{mode} {distribution} {species}: particles lost or non-finite kicks")
        return xp


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in values) + "]"


    def compare(label, observed, expected):
        """Print the first two particles' slopes; return whether any particle disagrees."""
        bad = not np.allclose(observed, expected, rtol=RTOL, atol=0)
        print(f"{label}: observed={fmt(observed[:2])} expected={fmt(expected[:2])}")
        return bad


    defect = False
    for mode in modes():
        for distribution in ("uniform", "parabolic", "gaussian"):
            for polarity in (1, -1):
                # Particle at +X0 is pulled toward an opposing charge of opposite sign: the kick
                # of a particle with charge q in the field of charge Q has the sign of q Q.
                # Electrons have q < 0, positrons q > 0.
                shape = np.tile([1.0, -1.0], N // 2)
                electron_expected = -polarity * kick_magnitude(distribution) * shape
                positron_expected = -electron_expected
                name = f"{mode} {distribution} source={polarity * SOURCE * 1e9:+.1f} nC"
                electron = kicks(mode, distribution, "electron", polarity)
                if compare(f"{name}, electron (control)", electron, electron_expected):
                    sys.exit("The electron kick disagrees with the enclosed-charge oracle.")
                positron = kicks(mode, distribution, "positron", polarity)
                defect |= compare(f"{name}, positron", positron, positron_expected)
                print(
                    f"{name}: positron kick equals electron kick: {np.array_equal(positron, electron)}"
                )
    print(f"comparison: rtol={RTOL:g}")
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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |

