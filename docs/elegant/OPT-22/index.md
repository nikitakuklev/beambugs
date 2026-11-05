---
title: "OPT-22: Coupling integral reads a KSEXT's YAW as its vertical offset DY"
hide:
  - toc
---

# OPT-22 — Coupling integral reads a KSEXT's YAW as its vertical offset DY

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`propagate_twiss_parameters`, KSEXT case of the coupling sum, src/twiss.cc:873–876](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L873-L876)<br>[SEXT structure, src/track.h:1545–1549](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track.h#L1545-L1549)<br>[KSEXT structure, src/track.h:2329–2332](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track.h#L2329-L2332)<br>[the coupling computation includes vertical orbit in sextupoles and vertical sextupole displacement, doc/elegant.tex:8258–8261](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8258-L8261) |

Elegant's `twiss_output` coupling estimate (`couplingIntegral`, `emittanceRatio`) includes the skew gradient K2·(y − DY) of vertically displaced sextupoles, but for a KSEXT it reads DY through the SEXT data layout, where the KSEXT stores its YAW. A vertically displaced KSEXT therefore contributes no coupling, and a yawed one contributes as if displaced by its YAW in meters. In an eight-cell FODO ring with one 0.08 m sextupole of K2 = 4 m⁻³ displaced by DY = 0.7 mm, the single-source formula and a SEXT both give a coupling integral of 3.983×10⁻⁴; the equivalent KSEXT gives 0, and a KSEXT with YAW = 0.7 mrad and no displacement gives 3.983×10⁻⁴ instead of 0.

??? example "Reproducer: OPT-22.py"

    [Download OPT-22.py](OPT-22.py)

    ```python
    """OPT-22: Coupling integral reads a KSEXT's YAW as its vertical offset DY

    Elegant's `twiss_output` coupling estimate (`couplingIntegral`, `emittanceRatio`) includes the skew
    gradient K2·(y − DY) of vertically displaced sextupoles, but for a KSEXT it reads DY through the
    SEXT data layout, where the KSEXT stores its YAW. A vertically displaced KSEXT therefore contributes
    no coupling, and a yawed one contributes as if displaced by its YAW in meters. In an eight-cell FODO
    ring with one 0.08 m sextupole of K2 = 4 m⁻³ displaced by DY = 0.7 mm, the single-source formula and
    a SEXT both give a coupling integral of 3.983×10⁻⁴; the equivalent KSEXT gives 0, and a KSEXT with
    YAW = 0.7 mrad and no displacement gives 3.983×10⁻⁴ instead of 0.

    The script runs serial Elegant `twiss_output` on an eight-cell FODO ring with one 0.08 m sextupole of
    K2 = 4 m⁻³ and reads `couplingIntegral` and `emittanceRatio` for a SEXT with DY = 0.7 mm, a KSEXT
    with the same DY and a KSEXT with YAW = 0.7 mrad. The expected coupling integral of a single skew
    source is |K2·DY|·L·√(β̄x β̄y)/(2π), with β̄ the average of the beta functions at the sextupole's
    ends (the convention of Elegant's sum). Second-order matrices (`matrix_output`) show that the SEXT
    and the displaced KSEXT have the same x–y coupling term R13 and that the yawed KSEXT has none.
    """

    import sys

    import numpy as np
    from _elegant import run, verdict

    K2, LS, OFFSET = 4.0, 0.08, 7e-4


    def ring(element, misalignment):
        lattice = (
            "F: QUAD,L=0.25,K1=1.1\nG: QUAD,L=0.25,K1=-1\nD: DRIF,L=0.7\nE: DRIF,L=0.9\n"
            f"S: {element},L={LS},K2={K2},{misalignment}\nC: LINE=(F,D,G,E)\nBL: LINE=(4*C,S,4*C)\n"
        )
        commands = (
            "&run_setup lattice=lattice.lte, use_beamline=BL, p_central_mev=2400, default_order=2 &end\n"
            "&twiss_output filename=twiss.sdds, matched=1 &end\n"
            "&matrix_output SDDS_output=matrix.sdds, SDDS_output_order=2, full_matrix_only=1 &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands}, outputs=("twiss.sdds", "matrix.sdds")
        )
        twiss = result["twiss.sdds"]
        names = list(twiss[0, "ElementName"])
        i = names.index("S")
        # Average of the beta functions at the sextupole's entrance (row i-1) and exit (row i).
        beta = [float(np.mean(twiss[0, f"beta{p}"][i - 1 : i + 1])) for p in "xy"]
        values = [float(twiss[k][0]) for k in ("couplingIntegral", "emittanceRatio", "couplingDelta")]
        r13 = float(result["matrix.sdds"][0, "R13"][-1])
        return beta, values, r13


    defect = False
    cases = [
        ("SEXT DY=0.7 mm", "SEXT", f"DY={OFFSET!r}", True),
        ("KSEXT DY=0.7 mm", "KSEXT", f"DY={OFFSET!r}", True),
        ("KSEXT YAW=0.7 mrad", "KSEXT", f"YAW={OFFSET!r}", False),
    ]
    r13 = {}
    for label, element, misalignment, displaced in cases:
        beta, (integral, ratio, delta), r13[label] = ring(element, misalignment)
        # Skew gradient of a vertically displaced sextupole: K1r = K2 (y - DY) with y = 0 on axis.
        kappa = abs(K2 * OFFSET) * LS * np.sqrt(beta[0] * beta[1]) / (2 * np.pi) if displaced else 0.0
        # emittanceRatio = kappa^2 / (kappa^2 + couplingDelta^2), couplingDelta the tune distance.
        expected_ratio = kappa**2 / (kappa**2 + delta**2)
        print(
            f"{label} couplingIntegral, emittanceRatio: observed=[{integral:.10g}, {ratio:.10g}]"
            f" expected=[{kappa:.10g}, {expected_ratio:.10g}]; R13={r13[label]:.10g} m"
        )
        bad = not np.isclose(integral, kappa, rtol=1e-9, atol=1e-15)
        if element == "SEXT" and bad:
            sys.exit("The SEXT control disagrees with the single-source formula: invalid comparison.")
        defect |= bad
    if (
        not np.isclose(r13["SEXT DY=0.7 mm"], r13["KSEXT DY=0.7 mm"], rtol=1e-9)
        or r13["KSEXT YAW=0.7 mrad"] != 0
    ):
        sys.exit("The R13 control does not show the expected element equivalence: invalid comparison.")
    print("control: SEXT and displaced KSEXT have the same R13; the yawed KSEXT has R13 = 0")
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

