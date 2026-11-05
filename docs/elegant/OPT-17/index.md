---
title: "OPT-17: CCBEND radiation integral I4 subtracts the sextupole-centering offset twice"
hide:
  - toc
---

# OPT-17 — CCBEND radiation integral I4 subtracts the sextupole-centering offset twice

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [shifts the particles by `dxOffset`, src/ccbend.c:530–531](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L530-L531)<br>[saves that x as `lastX`, src/ccbend.c:1240–1243](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L1240-L1243)<br>[`lastX` is flipped to the lab sign, src/ccbend.c:736–740](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L736-L740)<br>[`addCcbendRadiationIntegrals`, src/ccbend.c:1836–1837](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L1836-L1837)<br>[CCBEND multipole terms are defined in Cartesian coordinates, doc/ccbend.tex:3](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ccbend.tex#L3)<br>[`In` is the nth radiation integral, doc/elegant.tex:8262](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8262)<br>[`OPTIMIZE_DX` defaults to 1, src/track_data.c:3851](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3851) |

With a sextupole component K2, Elegant's CCBEND I4 radiation integral in `twiss_output` evaluates the gradient as K1 + K2(x − Δx), where x is already measured from the magnet axis that the default optimization (OPTIMIZE_DX) shifts by Δx; for a negative bend the offset enters with the opposite sign. A magnet and its mirror image (opposite angle and K2) therefore get different I4. In a 73.60 m, 3 GeV ring of 32 one-meter CCBENDs with K2 = 3 m⁻³ (Δx = 12.26 mm), Elegant gives a horizontal damping partition of 1.035 and a natural emittance of 351.5 nm, and for the mirrored ring 0.5606 and 648.8 nm. The two I4 differ by 0.5850 m⁻¹, close to 4K2·Δx·I1 = 0.5859 m⁻¹.

??? example "Reproducer: OPT-17.py"

    [Download OPT-17.py](OPT-17.py)

    ```python
    """OPT-17: CCBEND radiation integral I4 subtracts the sextupole-centering offset twice

    With a sextupole component K2, Elegant's CCBEND I4 radiation integral in `twiss_output` evaluates
    the gradient as K1 + K2(x − Δx), where x is already measured from the magnet axis that the default
    optimization (OPTIMIZE_DX) shifts by Δx; for a negative bend the offset enters with the opposite
    sign. A magnet and its mirror image (opposite angle and K2) therefore get different I4. In a 73.60
    m, 3 GeV ring of 32 one-meter CCBENDs with K2 = 3 m⁻³ (Δx = 12.26 mm), Elegant gives a horizontal
    damping partition of 1.035 and a natural emittance of 351.5 nm, and for the mirrored ring 0.5606 and
    648.8 nm. The two I4 differ by 0.5850 m⁻¹, close to 4K2·Δx·I1 = 0.5859 m⁻¹.

    The script runs serial Elegant `twiss_output` with radiation integrals on a CCBEND with K2 and on
    its mirror image (negative angle and K2, mirrored entrance dispersion), then on a ring of such
    CCBENDs and the mirrored ring. A mirrored magnet with mirrored dispersion has the same radiation
    integrals; the other integrals, the tunes and the optimized offset printed with VERBOSE=1 are equal
    in each pair and serve as controls. With OPTIMIZE_DX=0, which sets the offset to zero, the
    single-magnet pair must agree again; the script also compares the pair difference with
    4·K2·Δx·I1, the size the doubled subtraction predicts.
    """

    import re
    import sys

    import numpy as np
    from _elegant import run, verdict

    P_MEV = 3000.0  # reference momentum, MeV/c
    RTOL = 1e-10

    # Single magnet: entered with fixed optics; mirroring flips the sign of eta and eta'.
    SINGLE = {"L": 1.3, "ANGLE": 0.12, "K1": 0.05, "K2": 4.0}
    START = "beta_x=3.2, alpha_x=0.3, beta_y=2.8, alpha_y=-0.1"
    ETA0, ETAP0 = 0.4, 0.03

    # Ring: 16 FODO cells with two 1 m CCBENDs each, 2*pi/32 per bend, K2 = 3 m^-3, at 3 GeV.
    N_BENDS = 32
    RING = {"L": 1.0, "ANGLE": 2 * np.pi / N_BENDS, "K1": 0.0, "K2": 3.0}


    def bend(p, sign, optimize_dx):
        # Mirror image in x: the dipole field and the sextupole term change sign, K1 does not.
        return (
            f"B: CCBEND,L={p['L']!r},ANGLE={sign * p['ANGLE']!r},K1={p['K1']!r},"
            f"K2={sign * p['K2']!r},N_SLICES=256,OPTIMIZE_DX={optimize_dx},VERBOSE=1"
        )


    def twiss(p, sign, optimize_dx, ring):
        """Twiss parameters and the optimized offset dx of the (first) CCBEND."""
        if ring:
            lattice = (
                "F: QUAD,L=0.3,K1=2.4\nG: QUAD,L=0.3,K1=-2.4\nD: DRIF,L=0.5\nE: DRIF,L=0.5\n"
                f"{bend(p, sign, optimize_dx)}\nC: LINE=(F,D,B,E,G,D,B,E)\n"
                f"BL: LINE=({N_BENDS // 2}*C)\n"
            )
            matched = "matched=1"
        else:
            lattice = f"{bend(p, sign, optimize_dx)}\nBL: LINE=(B)\n"
            matched = f"matched=0, {START}, eta_x={sign * ETA0!r}, etap_x={sign * ETAP0!r}"
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central_mev={P_MEV!r} &end\n"
            f"&twiss_output filename=twiss.sdds, {matched}, radiation_integrals=1 &end\n"
        )
        result = run({"lattice.lte": lattice, "run.ele": commands}, outputs=("twiss.sdds",))
        found = re.findall(r"optimized: FSE=\S+, dx=(\S+),", result["log"])
        dx = float(found[0]) if found else 0.0
        table = result["twiss.sdds"]
        names = ["I1", "I2", "I3", "I4", "I5", "nux", "Jx", "ex0"]
        return {k: float(table[k][0]) for k in names}, dx


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in values) + "]"


    defect = False
    controls = ["I1", "I2", "I3", "I5", "nux"]
    for label, p, ring in (("single magnet", SINGLE, False), ("ring", RING, True)):
        pos, dx_pos = twiss(p, 1, 1, ring)
        neg, dx_neg = twiss(p, -1, 1, ring)
        print(f"{label} optimized offset dx (m): bend={dx_pos:.12g} mirror={dx_neg:.12g}")
        print(
            f"{label} control I1, I2, I3, I5, nux: bend={fmt([pos[k] for k in controls])}"
            f" mirror={fmt([neg[k] for k in controls])}"
        )
        same = all(np.isclose(pos[k], neg[k], rtol=RTOL, atol=1e-15) for k in controls)
        if not same or not np.isclose(dx_pos, dx_neg, rtol=RTOL) or dx_pos == 0:
            sys.exit(
                f"{label}: the mirror pair differs in geometry or linear optics: invalid comparison."
            )
        # The mirror image must have the same I4; the doubled subtraction gives the positive bend
        # K1 + K2 (x - dx) and the mirror K1 + K2 (x + dx), so the I4 pair differs by
        # 4 K2 dx sum(ds eta h cos(atan x')) ~ 4 K2 dx I1 (cos(atan x') is within 0.2% of 1 here).
        print(f"{label} I4 (1/m): bend={pos['I4']:.12g} mirror={neg['I4']:.12g}")
        print(f"{label} I4 mirror - bend: observed={neg['I4'] - pos['I4']:.10g} expected=0")
        print(f"{label} 4*K2*dx*I1 = {4 * p['K2'] * dx_pos * pos['I1']:.10g} 1/m")
        defect |= not np.isclose(pos["I4"], neg["I4"], rtol=RTOL, atol=1e-15)
        if ring:
            print(f"{label} Jx: bend={pos['Jx']:.10g} mirror={neg['Jx']:.10g}")
            print(f"{label} ex0 (m): bend={pos['ex0']:.10g} mirror={neg['ex0']:.10g}")
        else:
            # Control: OPTIMIZE_DX=0 leaves dx = 0, and the mirror pair agrees again.
            pos0, _ = twiss(p, 1, 0, ring)
            neg0, _ = twiss(p, -1, 0, ring)
            print(
                f"{label} control OPTIMIZE_DX=0 I4 (1/m): bend={pos0['I4']:.12g}"
                f" mirror={neg0['I4']:.12g}"
            )
            if not np.isclose(pos0["I4"], neg0["I4"], rtol=RTOL, atol=1e-15):
                sys.exit("The OPTIMIZE_DX=0 mirror pair differs: the offset is not the only cause.")
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
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Found (cause in source and executed verification) | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

