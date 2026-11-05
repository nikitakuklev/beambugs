---
title: "IO-05: The manual's bunch-input example assigns particle IDs 2001–3000 to bunch 1 instead of bunch 2"
hide:
  - toc
---

# IO-05 — The manual's bunch-input example assigns particle IDs 2001–3000 to bunch 1 instead of bunch 2

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [bunch organization, doc/elegant.tex:498–506](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L498-L506)<br>[IDSlotsPerBunch, sdds_beam.c:1041–1047](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_beam.c#L1041-L1047)<br>[bunch assignment, sdds_beam.c:786–800](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_beam.c#L786-L800)<br>[bunch range, lrwake.c:59–89](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L59-L89)<br>[bunch selection, trwake.c:72–88](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L72-L88) |

The manual says that with `IDSlotsPerBunch` S the bunch of particle ID I is ⌊(I − 1)/S⌋, then illustrates this for S = 1000 with IDs 1 to 1000 in bunch 0 and IDs 2001 to 3000 in bunch 1. The formula, which the code implements, puts IDs 1001 to 2000 in bunch 1 and IDs 2001 to 3000 in bunch 2. A user who follows the example, leaving out IDs 1001 to 2000, and selects bunch 1 of a transverse wake gets no kick on any particle; in the test, the 1.252×10⁻⁷ rad kick that the example implies for particle 3000 appears only when bunch 2 is selected.

??? example "Reproducer: IO-05.py"

    [Download IO-05.py](IO-05.py)

    ```python
    """IO-05: The manual's bunch-input example assigns particle IDs 2001–3000 to bunch 1 instead of bunch 2

    The manual says that with `IDSlotsPerBunch` S the bunch of particle ID I is ⌊(I − 1)/S⌋, then illustrates this for S = 1000 with IDs 1 to 1000 in bunch 0 and IDs 2001 to 3000 in bunch 1. The formula, which the code implements, puts IDs 1001 to 2000 in bunch 1 and IDs 2001 to 3000 in bunch 2. A user who follows the example, leaving out IDs 1001 to 2000, and selects bunch 1 of a transverse wake gets no kick on any particle; in the test, the 1.252×10⁻⁷ rad kick that the example implies for particle 3000 appears only when bunch 2 is selected.

    The script tracks beams of 16 particles per ID group through a TRWAKE that selects one bunch with START_BUNCH and END_BUNCH, and compares every particle's slope kick with a causal pair sum over the particles whose integer ⌊(I − 1)/S⌋ equals the selected bunch. It runs IDs in bunches 0, 1 and 2 and IDs in bunches 0 and 2 only, as the example has them, selecting bunch 1 and bunch 2, and compares the second with the kicks the example's labelling implies. The control selects all bunches.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h: electron rest energy, eV.
    REST_EV = 510999.06
    P0 = 45.0  # momentum, units of m_e c
    SLOPE = 3e25  # V/C/m per s: Wx(tau) = SLOPE tau for tau >= 0, so that W(0) = 0
    S = 1000  # IDSlotsPerBunch
    X0 = 1e-3  # m, the same for every particle
    Q = 1e-12  # C per particle
    RTOL, ATOL = 3e-12, 1e-19
    # Sixteen IDs per group, at both ends of the interval; four arrival times of four particles each
    # per bunch, spaced 4 ps, bunches 1 ns apart.
    WITHIN = np.r_[np.arange(1, 9), np.arange(S - 7, S + 1)]
    OFFSETS = np.repeat(np.arange(4) * 4e-12, 4)

    COMMANDS = (
        "&global_settings echo_namelists=0 &end\n"
        f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P0!r},output=out.sdds &end\n"
        "&run_control n_steps=1,n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds,use_bunched_mode=1,track_pages_separately=0 &end\n"
        "&track &end\n"
    )
    # Causal wake on 257 points 2 ps apart; the arrival times lie on its grid.
    WAKE = (
        "SDDS1\n&column name=t, type=double, units=s &end\n"
        "&column name=Wx, type=double, units=V/C/m &end\n&data mode=ascii &end\n257\n"
        + "".join(f"{tau!r} {SLOPE * tau!r}\n" for tau in (float(n) * 2e-12 for n in range(257)))
    )


    def kicks(mode, groups, selected):
        """Slope kicks xp for the particles, ordered by particle ID, with IDs and bunch indices.

        Each group g holds IDs g*S + WITHIN and arrives at (g + 1) ns. Rows are ordered so that each
        MPI worker holds two of the four particles at every time of every group.
        """
        ids = np.concatenate([WITHIN + g * S for g in groups])
        times = np.concatenate([OFFSETS + (g + 1) * 1e-9 for g in groups])
        n = len(ids)
        half = np.arange(16).reshape(4, 4)[:, :2].ravel()
        other = np.arange(16).reshape(4, 4)[:, 2:].ravel()
        count = len(groups)
        order = np.r_[[half + 16 * j for j in range(count)], [other + 16 * j for j in range(count)]]
        order = order.ravel()
        beam = {
            "x": np.full(n, X0)[order],
            "t": times[order],
            "p": np.full(n, P0),
            "particleID": ids[order],
            "IDSlotsPerBunch": S,
        }
        lattice = (
            f"Q: CHARGE,TOTAL={n * Q!r}\n"
            'W: TRWAKE,INPUTFILE="wake.sdds",TCOLUMN="t",WXCOLUMN="Wx",N_BINS=0,SMOOTHING=0,'
            f"INTERPOLATE=0,BUNCHED_BEAM_MODE=1,START_BUNCH={selected},END_BUNCH={selected}\n"
            "BL: LINE=(Q,W)\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": COMMANDS, "wake.sdds": WAKE},
            mode,
            {"beam.sdds": beam},
        )
        out = particles(result["out.sdds"])
        if not np.array_equal(out["particleID"], np.sort(ids)):
            sys.exit(f"{mode}: particles lost")
        sorting = np.argsort(ids)
        return ids[sorting], times[sorting], out["xp"]


    def pair_sum(times, bunch, selected):
        """Slope kicks from a direct causal pair sum over particles of the same bunch.

        Particle i of the selected bunch receives dx'_i = Q sum_j W(t_i - t_j) x_j/(E0 P0) over the
        particles j of its bunch that arrive earlier (W(0) = 0), with E0 the electron rest energy in
        eV; other bunches, and a selection of -1 that means every bunch, are handled by the mask.
        """
        delay = np.maximum(times[:, None] - times[None, :], 0.0)
        delay *= bunch[:, None] == bunch[None, :]
        kick = Q * SLOPE * (delay @ np.full(len(times), X0)) / (REST_EV * P0)
        if selected >= 0:
            kick[bunch != selected] = 0.0
        return kick


    def manual_bunch(ids):
        """Bunch indices the manual's example gives: 1-1000 is bunch 0, 2001-3000 is bunch 1."""
        return np.where(ids <= S, 0, 1)


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"


    defect = False
    for mode in modes():
        for groups, selected in (
            ((0, 1, 2), 1),
            ((0, 1, 2), 2),
            ((0, 2), 1),
            ((0, 2), 2),
            ((0, 1, 2), -1),
        ):
            ids, times, observed = kicks(mode, groups, selected)
            # The formula floor((I - 1)/S) in integer arithmetic.
            bunch = (ids - 1) // S
            expected = pair_sum(times, bunch, selected)
            name = f"{mode} IDs in bunches {list(groups)}, selected bunch {selected}"
            tails = [i for i in range(len(ids)) if ids[i] % S == 0]  # last particle of each ID group
            print(
                f"{name}: last-in-group kicks at IDs {ids[tails].tolist()}: "
                f"observed={fmt(observed[tails])} expected={fmt(expected[tails])}"
            )
            if not np.allclose(observed, expected, rtol=RTOL, atol=ATOL):
                sys.exit("the kicks disagree with the formula: the comparison is invalid")
            if selected < 0:
                continue  # the control: every bunch is kicked, so the wake itself works
            # The example's labelling, applicable when the IDs are 1-1000 and 2001-3000.
            if groups == (0, 2):
                implied = pair_sum(times, manual_bunch(ids), selected)
                print(f"  kicks implied by the manual's example: {fmt(implied[tails])}")
                defect |= not np.allclose(observed, implied, rtol=RTOL, atol=ATOL)
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
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
| Found | `gpt-6-astra` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source reading, not executed for this claim |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

