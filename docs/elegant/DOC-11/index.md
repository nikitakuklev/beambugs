---
title: "DOC-11: Manual orders tune_shift_with_amplitude after twiss_output, where Elegant rejects or ignores it"
hide:
  - toc
---

# DOC-11 — Manual orders tune_shift_with_amplitude after twiss_output, where Elegant rejects or ignores it

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [the `tune_shift_with_amplitude` section, doc/elegant.tex:8389–8395](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8389-L8395)<br>[`if (do_twiss_output) bombElegant("you must give tune_shift_with_amplitude before twiss_output", NULL);`, src/elegant.c:1754–1757](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L1754-L1757)<br>[src/elegant.c:1676–1689](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L1676-L1689)<br>[doc/elegant.tex:8241–8243](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8241-L8243) |

The Elegant manual says `tune_shift_with_amplitude` must follow `twiss_output`. Given in that order with Twiss output scheduled for each step (`output_at_each_step=1`), Elegant stops with "you must give tune_shift_with_amplitude before twiss_output"; after an immediate `twiss_output`, Elegant has already written the Twiss file, so no tune shift is computed and dnux/dAx and the other tune-shift columns stay 0. The working order is `tune_shift_with_amplitude` before `twiss_output`. For eight FODO cells with a thin octupole of integrated strength K₃L, that order gives dnux/dAx = 1.411 m⁻¹, within 0.02% of the first-order detuning K₃L βx²/(32π), while the documented order gives an error or 0.

??? example "Reproducer: DOC-11.py"

    [Download DOC-11.py](DOC-11.py)

    ```python
    """DOC-11: Manual orders tune_shift_with_amplitude after twiss_output, where Elegant rejects or ignores it

    The Elegant manual says `tune_shift_with_amplitude` must follow `twiss_output`. Given in that order
    with Twiss output scheduled for each step (`output_at_each_step=1`), Elegant stops with "you must
    give tune_shift_with_amplitude before twiss_output"; after an immediate `twiss_output`, Elegant has
    already written the Twiss file, so no tune shift is computed and dnux/dAx and the other tune-shift
    columns stay 0. The working order is `tune_shift_with_amplitude` before `twiss_output`. For eight
    FODO cells with a thin octupole of integrated strength K₃L, that order gives dnux/dAx = 1.411 m⁻¹,
    within 0.02% of the first-order detuning K₃L βx²/(32π), while the documented order gives an error or
    0.

    The script reads the sequence statement of the `tune_shift_with_amplitude` section from
    `doc/elegant.tex` of the Elegant source checkout named by `ELEGANT_SOURCE` (required) and exits if
    it is missing or unrecognized. It runs serial Elegant in tracking mode on eight FODO cells with one
    thin octupole, first with `tune_shift_with_amplitude` before a scheduled `twiss_output` (the
    control, compared with first-order octupole detuning), then in the documented order with scheduled
    and with immediate Twiss output, and compares their dnux/dAx, dnux/dAy, dnuy/dAx and dnuy/dAy with
    the control's.
    """

    import os
    import re
    import sys
    from pathlib import Path

    import numpy as np
    from _elegant import RunFailure, run, verdict

    K3L = 3.0  # m^-3, integrated octupole strength of the thin MULT at the start of the ring
    LATTICE = f"""F: QUAD,L=0.25,K1=1.1,ORDER=1
    G: QUAD,L=0.25,K1=-1.0,ORDER=1
    D: DRIF,L=0.7
    E: DRIF,L=0.9
    O: MULT,ORDER=3,KNL={K3L!r}
    C: LINE=(F,D,G,E)
    A: LINE=(O,8*C)
    """
    SETUP = "&run_setup lattice=lattice.lte, use_beamline=A, p_central_mev=2400, default_order=3 &end\n"
    TSWA = (
        "&tune_shift_with_amplitude turns=4096, x0=1e-6, y0=1e-6, x1=0.002, y1=0.002, grid_size=4,"
        " order=1, tune_output=tunes.sdds &end\n"
    )
    ACTION = "&run_control &end\n&bunched_beam &end\n&track &end\n"
    COLUMNS = ("dnux/dAx", "dnux/dAy", "dnuy/dAx", "dnuy/dAy")


    def twiss(scheduled):
        """twiss_output; output_at_each_step=1 schedules the computation at the next action."""
        return (
            "&twiss_output filename=twiss.sdds, matched=1, concat_order=3, "
            f"output_at_each_step={int(scheduled)} &end\n"
        )


    source = os.environ.get("ELEGANT_SOURCE")
    if not source:
        sys.exit("ELEGANT_SOURCE must name the Elegant source checkout.")
    manual = Path(source) / "doc" / "elegant.tex"
    if not manual.is_file():
        sys.exit(f"{manual} does not exist.")
    text = manual.read_text()
    section = text[text.index(r"\subsection{tune\_shift\_with\_amplitude") :]
    section = section[: section.index(r"\subsection", 1)]
    sequence = re.findall(r"\\item sequence: must (follow|precede) \\verb\|twiss_output\|\.", section)
    if len(sequence) != 1:
        sys.exit(
            f"The manual's sequence statement for tune_shift_with_amplitude is unrecognized: {sequence}"
        )
    documented_first = "twiss" if sequence[0] == "follow" else "tswa"
    print(f"manual: tune_shift_with_amplitude must {sequence[0]} twiss_output")


    def tswa_run(order, scheduled):
        """Derivatives written to the Twiss file, Twiss parameters at the start, and grid rows; None if
        Elegant stops with an error (then also the error line)."""
        body = TSWA + twiss(scheduled) if order == "tswa" else twiss(scheduled) + TSWA
        try:
            out = run({"lattice.lte": LATTICE, "run.ele": SETUP + body + ACTION},
                      outputs=("twiss.sdds", "tunes.sdds"))  # fmt: skip
        except RunFailure as failure:
            errors = [line for line in failure.log.splitlines() if "error" in line.lower()]
            return None, f"exit {failure.status}: {errors[-1] if errors else 'no error line'}"
        tw, tunes = out["twiss.sdds"], out["tunes.sdds"]
        par = {p.name: p.data[0] for p in tw.parameters}
        rows = len(tunes[0, "x"]) if tunes.n_pages else 0
        start = {k: float(tw[0, k][0]) for k in ("betax", "betay")}
        return dict(dnu={k: float(par[k]) for k in COLUMNS}, rows=rows, **start), "exit 0"


    # Control: the source order with scheduled Twiss output. First-order perturbation theory for the
    # thin octupole, H = K3L (x^4 - 6 x^2 y^2 + y^4)/24 (MULT kick -K3L/6 Re (x+iy)^3), averaged over
    # the betatron phases with x = sqrt(2 Jx bx) cos(phi), gives dnux/dJx = K3L bx^2/(16 pi),
    # dnux/dJy = dnuy/dJx = -K3L bx by/(8 pi), dnuy/dJy = K3L by^2/(16 pi). Elegant's amplitude is the
    # Courant-Snyder invariant A = 2J, so each dnu/dA is half of dnu/dJ.
    control, status = tswa_run("tswa", scheduled=True)
    if control is None:
        sys.exit(f"control: the source order failed ({status})")
    bx, by = control["betax"], control["betay"]
    theory = {
        "dnux/dAx": K3L * bx**2 / (32 * np.pi),
        "dnux/dAy": -K3L * bx * by / (16 * np.pi),
        "dnuy/dAx": -K3L * bx * by / (16 * np.pi),
        "dnuy/dAy": K3L * by**2 / (32 * np.pi),
    }
    print(f"control, tune_shift_with_amplitude before scheduled twiss_output: {status}, {control['rows']} "
          f"grid points; betax={bx:.10g} m, betay={by:.10g} m at the octupole")  # fmt: skip
    for k in COLUMNS:
        print(
            f"  {k}: observed={control['dnu'][k]:.6f} m^-1 first-order octupole theory={theory[k]:.6f} m^-1"
        )
    if not all(np.isclose(control["dnu"][k], theory[k], rtol=0.01, atol=0) for k in COLUMNS):
        sys.exit("control: the source-order tune shifts do not match the octupole detuning")

    defect = False
    for scheduled in (True, False):
        label = "scheduled" if scheduled else "immediate"
        result, status = tswa_run(documented_first, scheduled)
        if result is None:
            print(f"documented order, {label} twiss_output: observed={status} "
                  f"expected=exit 0 with dnux/dAx={control['dnu']['dnux/dAx']:.6f} m^-1")  # fmt: skip
            defect = True
            continue
        got = result["dnu"]
        print(f"documented order, {label} twiss_output: {status}, {result['rows']} grid points; "
              f"dnux/dAx observed={got['dnux/dAx']:.6f} expected={control['dnu']['dnux/dAx']:.6f} m^-1, "
              f"dnuy/dAy observed={got['dnuy/dAy']:.6f} expected={control['dnu']['dnuy/dAy']:.6f} m^-1")  # fmt: skip
        defect |= not all(np.isclose(got[k], control["dnu"][k], rtol=1e-6, atol=0) for k in COLUMNS)
    print(
        "comparison: control within 1% of first-order theory; the documented order must run and "
        "write the control's tune shifts (rtol=1e-6)"
    )
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
| Found (noted the manual's sequence line against the source guard while placing the command before twiss_output in its decks) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source reading, not executed for this claim |
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source reading, not executed for this claim |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

