---
title: "TRK-45: Element division repeats a one-sided RFCA end-focus lens at every piece"
hide:
  - toc
---

# TRK-45 — Element division repeats a one-sided RFCA end-focus lens at every piece

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`copy_element`, mad_parse.c:567–568](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/mad_parse.c#L567-L568)<br>[mad_parse.c:588–589](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/mad_parse.c#L588-L589)<br>[mad_parse.c:594–597](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/mad_parse.c#L594-L597)<br>[mad_parse.c:609–612](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/mad_parse.c#L609-L612)<br>[`trackRfCavityWithWakes`, simple_rfca.c:746–750](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L746-L750)<br>[simple_rfca.c:804–807](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L804-L807)<br>[simple_rfca.c:1019–1024](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1019-L1024)<br>[1072–1076](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1072-L1076)<br>[compute_matrices.c:1905–1907](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1905-L1907)<br>[`rf_cavity_matrix`, compute_matrices.c:2604–2610](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L2604-L2610)<br>[elegant.c:1106–1107](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L1106-L1107)<br>[`divide_elements` manual, elegant.tex:3005–3030](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L3005-L3030)<br>[parameter table, track_data.c:608–609](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L608-L609) |

Dividing an RFCA with divide_elements while only END1_FOCUS or only END2_FOCUS is enabled copies that end lens to every piece, adding a lens at each internal boundary inside the uniform accelerating field. A divided cavity must track like the undivided one with the same kicks, which focuses only at its physical end. For electrons entering a 0.7 m, 12 MV cavity at 20 MeV/c, split into four pieces with a static field, a particle entering at x = 1 mm parallel to the axis exits with entrance focusing at x = 0.4947 mm and slope x′ = −0.9116 mrad instead of 0.7651 mm and −0.2677 mrad. With exit focusing, x′ is 1.223 mrad instead of 0.2678 mrad. With both lenses or neither, the static-field result is unaffected.

??? example "Reproducer: TRK-45.py"

    [Download TRK-45.py](TRK-45.py)

    ```python
    """TRK-45: Element division repeats a one-sided RFCA end-focus lens at every piece

    Dividing an RFCA with divide_elements while only END1_FOCUS or only END2_FOCUS is enabled copies
    that end lens to every piece, adding a lens at each internal boundary inside the uniform
    accelerating field. A divided cavity must track like the undivided one with the same kicks, which
    focuses only at its physical end. For electrons entering a 0.7 m, 12 MV cavity at 20 MeV/c, split
    into four pieces with a static field, a particle entering at x = 1 mm parallel to the axis exits
    with entrance focusing at x = 0.4947 mm and slope x′ = −0.9116 mrad instead of 0.7651 mm and −0.2677
    mrad. With exit focusing, x′ is 1.223 mrad instead of 0.2678 mrad. With both lenses or neither, the
    static-field result is unaffected.

    The script tracks four electrons at 20 MeV/c through a 0.7 m, 12 MV RFCA with a static field
    (FREQ=0) and kick tracking, divided by divide_elements into two and four pieces of 16 kicks each.
    The expected result is the undivided cavity with the same total number of kicks, which has the
    identical kick sequence and only the physical end lenses; a Python model of Elegant's kick
    algorithm checks it independently, and the same model with the lens on every piece predicts the
    divided output. Divided cavities with no lens or with both lenses are the controls.
    """

    import numpy as np
    from _elegant import modes, particles, run, verdict

    ME_MEV = 0.51099906  # SDDS constants.h electron rest energy, as Elegant uses it
    LENGTH, VOLT = 0.7, 12e6  # m, V: a 17.14 MV/m gradient
    P_MEV = 20.0  # MeV/c, entrance momentum
    KICKS = 16  # RF kicks per divided piece
    X0 = np.array([1e-3, 5e-4, -5e-4, -1e-3])  # m; four particles so that Pelegant tracks them
    RTOL, ATOL = 2e-12, 1e-15  # rounding of the per-kick length and voltage


    def elegant(mode: str, pieces: int, end1: int, end2: int, divide: bool) -> np.ndarray:
        """Exit (x, x', p) of every particle, rows by particle ID.

        divide=True splits the cavity with divide_elements into `pieces` pieces of KICKS kicks each;
        divide=False tracks it undivided with pieces*KICKS kicks, the same kick sequence.
        """
        kicks = KICKS if divide else KICKS * pieces
        lattice = (
            f"R: RFCA,L={LENGTH!r},VOLT={VOLT!r},FREQ=0,PHASE=90,N_KICKS={kicks},"
            f"END1_FOCUS={end1},END2_FOCUS={end2}\n"
            "BL: LINE=(R)\n"
        )
        division = f"&divide_elements name=R, divisions={pieces} &end\n" if divide else ""
        commands = (
            division + "&run_setup lattice=lattice.lte, use_beamline=BL, "
            f"p_central_mev={P_MEV!r}, output=out.sdds &end\n"
            "&run_control &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {"x": X0, "p": np.full(len(X0), P_MEV / ME_MEV), "particleID": np.arange(1, 5)}
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})
        out = particles(result["out.sdds"])
        if len(out["x"]) != len(X0):
            raise SystemExit(f"{mode}: particles were lost")
        return np.column_stack([out["x"], out["xp"], out["p"]])


    def model(pieces: int, end1: int, end2: int, every_piece: bool) -> np.ndarray:
        """Elegant's RFCA kick tracking at FREQ=0 (simple_rfca.c), in Python.

        Each kick of length l = L/(pieces*KICKS) is: entrance lens on the first kick of a piece,
        drift l/2, energy gain dgamma = VOLT/(pieces*KICKS mc^2) at constant transverse momentum
        (add_to_particle_energy), drift l/2, exit lens on the last kick of a piece. The implemented
        lens law is x' -> x' - x/f with 1/f = dgamma/(2 gamma l) at the entrance (gamma before the
        gain) and -dgamma/(2 gamma l) at the exit (gamma after it). every_piece=False applies the
        lenses only at the cavity's physical ends; every_piece=True applies them on every piece.
        """
        n = pieces * KICKS
        step, dgamma = LENGTH / n, VOLT / (1e6 * ME_MEV) / n
        rows = []
        for x in X0:
            xp, p = 0.0, P_MEV / ME_MEV
            for k in range(n):
                gamma = np.hypot(p, 1)
                first = k % KICKS == 0 if every_piece else k == 0
                last = k % KICKS == KICKS - 1 if every_piece else k == n - 1
                if end1 and first:
                    xp -= x * dgamma / (2 * gamma * step)
                x += xp * step / 2
                pz = p / np.sqrt(1 + xp**2)
                gamma1 = gamma + dgamma
                xp *= pz / np.sqrt(pz**2 + gamma1**2 - gamma**2)
                p = np.sqrt(gamma1**2 - 1)
                x += xp * step / 2
                if end2 and last:
                    xp += x * dgamma / (2 * gamma1 * step)
            rows.append((x, xp, p))
        return np.array(rows)


    def close(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.allclose(a[:, :2], b[:, :2], rtol=RTOL, atol=ATOL)) and bool(
            np.allclose(a[:, 2], b[:, 2], rtol=2e-14, atol=0)
        )


    defect = True
    for mode in modes():
        for divisions in (2, 4):
            for end1, end2, label in (
                (1, 0, "entrance only"),
                (0, 1, "exit only"),
                (1, 1, "both (control)"),
                (0, 0, "neither (control)"),
            ):
                divided = elegant(mode, divisions, end1, end2, divide=True)
                undivided = elegant(mode, divisions, end1, end2, divide=False)
                physical = model(divisions, end1, end2, every_piece=False)
                repeated = model(divisions, end1, end2, every_piece=True)
                # The undivided cavity with the same kicks is the expected result; the Python model
                # with lenses only at the physical ends must reproduce it.
                if not close(undivided, physical):
                    raise SystemExit(f"{mode} {label}: the kick model does not match undivided Elegant")
                print(
                    f"{mode} {label}, {divisions} pieces, x0=1 mm: observed x={divided[0, 0]:.12g} m "
                    f"x'={divided[0, 1]:.12g}; expected (undivided, same kicks) x={undivided[0, 0]:.12g}"
                    f" m x'={undivided[0, 1]:.12g}; lens-on-every-piece model x={repeated[0, 0]:.12g}"
                    f" m x'={repeated[0, 1]:.12g}"
                )
                if not close(divided, repeated):
                    raise SystemExit(
                        f"{mode} {label}: divided output is not the lens-on-every-piece model"
                    )
                if end1 == end2:
                    # Static field, equal gradient: the copied exit lens of one piece and entrance lens
                    # of the next have equal and opposite strength at the same plane and cancel.
                    if not close(divided, undivided):
                        raise SystemExit(f"{mode} {label}: control fails")
                else:
                    defect &= not close(divided, undivided)
    print(
        f"controls: undivided Elegant equals the end-lens kick model within rtol {RTOL:g}; "
        "no-lens and both-lens divisions equal the undivided cavity"
    )
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
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

