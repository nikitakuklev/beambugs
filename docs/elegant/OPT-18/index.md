---
title: "OPT-18: Third-order concatenation multiplies spurious path-origin terms of tracking-derived maps by the accumulated path length"
hide:
  - toc
---

# OPT-18 — Third-order concatenation multiplies spurious path-origin terms of tracking-derived maps by the accumulated path length

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`concat_matrices`, src/concat_mat.c:56–81](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/concat_mat.c#L56-L81)<br>[`accumulate_matrices`, src/compute_matrices.c:261–263](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L261-L263)<br>[`remove_s_dependent_matrix_elements`, src/matrix.c:1271–1290](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/matrix.c#L1271-L1290)<br>[`determineMatrixHigherOrder` sees `trackingMatrixCleanUp`, src/analyze.c:1528–1531](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/analyze.c#L1528-L1531)<br>[default is 0, src/global_settings.nl:29](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/global_settings.nl#L29)<br>[s is the total equivalent distance, s = βct, doc/elegant.tex:364–378](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L364-L378)<br>[compaction coefficients are those of the path-length expansion, doc/elegant.tex:8254–8257](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8254-L8257) |

Elegant derives CCBEND maps by tracking, and those maps carry spurious terms that depend on the incoming path coordinate s. Third-order concatenation (`concat_order=3` in `twiss_output`, order-3 `matrix_output`) multiplies these terms by the path length accumulated before the element, so T566 and the second-order momentum compaction αc2 depend on where the element sits. In a 73.60 m, 3 GeV ring of 32 CCBENDs, Twiss gives αc2 = 0.1868 or 0.2117 depending on the starting point, while the path length of off-momentum closed orbits found by tracking gives 0.09547, which second-order concatenation reproduces. Two CCBENDs give T566 = −0.06916, −1.227 and −11.65 m after upstream drifts of 0, 40 and 400 m, while tracking gives −0.01417 m in every case.

??? example "Reproducer: OPT-18.py"

    [Download OPT-18.py](OPT-18.py)

    ```python
    """OPT-18: Third-order concatenation multiplies spurious path-origin terms of tracking-derived maps by the accumulated path length

    Elegant derives CCBEND maps by tracking, and those maps carry spurious terms that depend on the
    incoming path coordinate s. Third-order concatenation (`concat_order=3` in `twiss_output`, order-3
    `matrix_output`) multiplies these terms by the path length accumulated before the element, so T566
    and the second-order momentum compaction αc2 depend on where the element sits. In a 73.60 m, 3 GeV
    ring of 32 CCBENDs, Twiss gives αc2 = 0.1868 or 0.2117 depending on the starting point, while the
    path length of off-momentum closed orbits found by tracking gives 0.09547, which second-order
    concatenation reproduces. Two CCBENDs give T566 = −0.06916, −1.227 and −11.65 m after upstream
    drifts of 0, 40 and 400 m, while tracking gives −0.01417 m in every case.

    The script runs serial Elegant on a ring of 32 CCBENDs, whose maps Elegant derives by tracking.
    `twiss_output` gives αc2 at concat_order 3 from two starting points and at concat_order 2 as a
    control; the expected value is fitted to the one-turn path length of the off-momentum closed
    orbits, found by Newton iteration on single-turn tracking at nine momentum offsets. A second case
    puts a drift of 0, 40 or 400 m before two CCBENDs and compares the third-order T566 that
    `matrix_output` writes with T566 fitted to the tracked path length of 21 on-axis particles, after
    subtracting the drift length.
    """

    import sys

    import numpy as np
    from _elegant import particles, run, verdict

    P_MEV = 3000.0
    ME_MEV = 0.51099906  # electron rest energy in Elegant's constants.h
    P0 = P_MEV / ME_MEV  # reference momentum, units of m_e c
    C_LIGHT = 299792458.0

    N_BENDS = 32
    RING = f"""F: QUAD,L=0.3,K1=2.4
    G: QUAD,L=0.3,K1=-2.4
    D: DRIF,L=0.5
    E: DRIF,L=0.5
    B: CCBEND,L=1.0,ANGLE={2 * np.pi / N_BENDS!r},N_SLICES=32
    C: LINE=(F,D,B,E,G,D,B,E)
    R: LINE=({N_BENDS // 2}*C)
    Z: LINE=(G,D,B,E,{N_BENDS // 2 - 1}*C,F,D,B,E)
    """
    DELTAS = np.linspace(-0.004, 0.004, 9)
    STEP = 1e-6  # Newton Jacobian probe in x and xp


    def path_length(out):
        """Path length s = beta c t of each particle, ordered by particleID."""
        o = particles(out)
        beta = o["p"] / np.sqrt(1 + o["p"] ** 2)
        return o, beta * C_LIGHT * o["t"]


    def ring_twiss(line, order):
        commands = (
            f"&run_setup lattice=ring.lte, use_beamline={line}, p_central_mev={P_MEV!r},"
            f" default_order={order} &end\n"
            f"&twiss_output filename=twiss.sdds, matched=1, concat_order={order} &end\n"
        )
        table = run({"ring.lte": RING, "run.ele": commands}, outputs=("twiss.sdds",))["twiss.sdds"]
        return float(table["alphac"][0]), float(table["alphac2"][0])


    def tracked_compaction():
        """alphac and alphac2 from the one-turn path of the off-momentum closed orbits.

        For each delta, Newton iteration on (x, xp) with five particles (the orbit guess and +-STEP
        probes in x and xp) tracked one turn; y = yp = 0 by symmetry.
        """
        orbit = np.zeros((len(DELTAS), 2))
        commands = (
            f"&run_setup lattice=ring.lte, use_beamline=R, p_central_mev={P_MEV!r},"
            " output=out.sdds &end\n&run_control &end\n&sdds_beam input=beam.sdds &end\n&track &end\n"
        )
        probes = np.array([[0, 0], [STEP, 0], [-STEP, 0], [0, STEP], [0, -STEP]])
        for _ in range(10):
            start = (orbit[:, None, :] + probes[None, :, :]).reshape(-1, 2)
            delta = np.repeat(DELTAS, len(probes))
            beam = {
                "x": start[:, 0],
                "xp": start[:, 1],
                "p": P0 * (1 + delta),
                "particleID": np.arange(1, len(delta) + 1),
            }
            result = run({"ring.lte": RING, "run.ele": commands}, beams={"beam.sdds": beam})
            o, s = path_length(result["out.sdds"])
            end = np.c_[o["x"], o["xp"]].reshape(len(DELTAS), len(probes), 2)
            residual = end[:, 0] - orbit
            if np.abs(residual).max() < 1e-14:
                break
            for i in range(len(DELTAS)):
                jacobian = np.c_[
                    (end[i, 1] - end[i, 2]) / (2 * STEP), (end[i, 3] - end[i, 4]) / (2 * STEP)
                ] - np.eye(2)
                orbit[i] -= np.linalg.solve(jacobian, residual[i])
        else:
            sys.exit("Newton iteration for the off-momentum closed orbits did not converge.")
        s = s.reshape(len(DELTAS), len(probes))[:, 0]
        s0 = s[len(DELTAS) // 2]
        # Elegant's definition: s = s0 (1 + alphac delta + alphac2 delta^2 + alphac3 delta^3).
        fit = np.polynomial.polynomial.polyfit(DELTAS, (s - s0) / s0, 6)
        return fit[1], fit[2], np.abs(residual).max()


    defect = False
    alphac, alphac2, residual = tracked_compaction()
    print(
        f"ring tracked closed orbits: alphac={alphac:.10g} alphac2={alphac2:.10g} (residual {residual:.1e} m)"
    )
    a1, a2 = ring_twiss("R", 2)
    print(
        f"ring control concat_order=2: alphac={a1:.10g} alphac2: observed={a2:.10g} expected={alphac2:.10g}"
    )
    if abs(a2 - alphac2) > 1e-6 or abs(a1 - alphac) > 1e-9:
        sys.exit(
            "The second-order control disagrees with the tracked closed orbits: invalid comparison."
        )
    for line in ("R", "Z"):
        a1, a2 = ring_twiss(line, 3)
        print(
            f"ring start {line} concat_order=3: alphac={a1:.10g} alphac2: observed={a2:.10g}"
            f" expected={alphac2:.10g}"
        )
        defect |= abs(a2 - alphac2) > 1e-6

    # Beamline: drift L0, CCBEND, 0.8 m drift, CCBEND. An upstream drift adds L0 to every on-axis
    # particle's path and cannot change the momentum dependence of the path after it.
    LINE = "E: DRIF,L=0.8\nB: CCBEND,L=1.1,ANGLE=0.16,K1=0.1,K2=2,N_SLICES=32\nBL: LINE=(D,B,E,B)\n"
    offsets = np.linspace(-0.01, 0.01, 21)
    beam = {"p": P0 * (1 + offsets), "particleID": np.arange(1, len(offsets) + 1)}
    tracked = []
    for drift in (0.0, 40.0, 400.0):
        commands = (
            f"&run_setup lattice=line.lte, use_beamline=BL, p_central_mev={P_MEV!r},"
            " default_order=3, output=out.sdds &end\n"
            "&matrix_output SDDS_output=matrix.sdds, SDDS_output_order=3, full_matrix_only=1 &end\n"
            "&run_control &end\n&sdds_beam input=beam.sdds &end\n&track &end\n"
        )
        files = {"line.lte": f"D: DRIF,L={drift!r}\n{LINE}", "run.ele": commands}
        result = run(files, beams={"beam.sdds": beam}, outputs=("out.sdds", "matrix.sdds"))
        _, s = path_length(result["out.sdds"])
        # T566 is the delta^2 coefficient of the path length; a degree-6 fit over +-1% resolves it.
        t566 = np.polynomial.polynomial.polyfit(offsets, s - drift, 6)[2]
        tracked.append(t566)
        written = float(result["matrix.sdds"][0, "T566"][-1])
        print(f"line upstream drift={drift:g} m T566 (m): observed={written:.10g} expected={t566:.10g}")
        defect |= abs(written - t566) > 1e-6
    if np.ptp(tracked) > 1e-7:
        sys.exit("The tracked T566 depends on the upstream drift: invalid comparison.")
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

