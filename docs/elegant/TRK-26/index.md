---
title: "TRK-26: Negative CCBEND edge multipole signs are chosen from the body table's order labels"
hide:
  - toc
---

# TRK-26 — Negative CCBEND edge multipole signs are chosen from the body table's order labels

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`track_through_ccbend_impl`, ccbend.c:450–470](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L450-L470)<br>[entrance, ccbend.c:1017–1025](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L1017-L1025)<br>[exit, ccbend.c:1313–1322](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L1313-L1322)<br>[`computeTotalErrorMultipoleFields`, multipole.c:1684–1745](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1684-L1745)<br>[CCBEND multipole errors, ccbend.tex:65–70](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ccbend.tex#L65-L70)<br>[KQUAD multipole files, kquad.tex:11–35](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/kquad.tex#L11-L35)<br>[CCBEND multipole parameters, track_data.c:3832–3835](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3832-L3835)<br>[CCBEND dispatch, do_tracking.c:2145–2153](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2145-L2153) |

Negative-angle CCBEND magnets with independent entrance or exit multipole-error files choose edge sign changes from the body table's order labels. Relabelling zero-amplitude body rows therefore changes physical edge kicks; shorter body tables can also corrupt kicks, and runs without body tables terminate. With zero body-error amplitudes, the reference is a positive bend rotated by π with each order-n edge fraction multiplied by (−1)ⁿ. For a 0.4 m, −0.12 rad bend with edge orders 1 and 2 and zero body rows labelled 2 and 3, the vertical-slope error in y′ reaches 2.153×10⁻³.

??? example "Reproducer: TRK-26.py"

    [Download TRK-26.py](TRK-26.py)

    ```python
    """TRK-26: Negative CCBEND edge multipole signs are chosen from the body table's order labels

    Negative-angle CCBEND magnets with independent entrance or exit multipole-error files choose edge
    sign changes from the body table's order labels. Relabelling zero-amplitude body rows therefore
    changes physical edge kicks; shorter body tables can also corrupt kicks, and runs without body
    tables terminate. With zero body-error amplitudes, the reference is a positive bend rotated by π
    with each order-n edge fraction multiplied by (−1)ⁿ. For a 0.4 m, −0.12 rad bend with edge orders 1
    and 2 and zero body rows labelled 2 and 3, the vertical-slope error in y′ reaches 2.153×10⁻³.

    The script tracks a negative-angle CCBEND with an entrance or exit edge multipole table whose body
    table has all-zero rows labelled to match the edge orders or not, and compares the exit coordinates
    with the same magnet written as a positive bend rotated by π with each order-n edge fraction
    multiplied by (-1)ⁿ, which never enters the negative-angle branch. The controls are matching labels,
    padded zero body labels, zero edge strength, a positive bend and a negative bend without error
    files. It then repeats the comparison with a body table of one row and runs a negative bend with no
    body table, which terminates.
    """

    import sys
    from dataclasses import dataclass

    import numpy as np
    from _elegant import RunFailure, modes, particles, run, verdict

    ATOL = 3e-14  # transverse agreement of two tracks of the same physical field


    @dataclass
    class Setup:
        length: float
        angle: float  # magnitude, rad
        slices: int
        p0: float  # m_e c
        radius: float  # reference radius of the multipole tables, m
        orders: list[int]  # edge table orders
        normal: list[float]  # edge normal fractions of the dipole field at the reference radius
        skew: list[float]
        beam: dict[str, np.ndarray]


    # Two inputs. "mixed": edge orders 1 and 2 with normal and skew parts, body labels [1, 2] (aligned
    # with the edge) or [2, 3]. "odd": edge orders 1 and 3 (both odd), for body tables that are absent
    # or shorter than the edge table.
    SETUPS = {
        "mixed": Setup(
            length=0.4,
            angle=0.12,
            slices=64,
            p0=30.0,
            radius=0.025,
            orders=[1, 2],
            normal=[0.012, -0.007],
            skew=[0.005, 0.004],
            beam=dict(
                p=np.linspace(29.0, 31.0, 12),
                x=np.linspace(-0.012, 0.016, 12),
                y=np.linspace(0.009, -0.007, 12),
                xp=np.linspace(0.002, -0.001, 12),
                yp=np.linspace(-0.001, 0.002, 12),
            ),
        ),
        "odd": Setup(
            length=0.5,
            angle=0.1,
            slices=32,
            p0=40.0,
            radius=0.03,
            orders=[1, 3],
            normal=[0.004, 0.004],
            skew=[0.004 / 3, 0.004 / 3],
            beam=dict(
                p=np.full(16, 40.0),
                x=np.linspace(-0.006, 0.008, 16),
                y=np.linspace(0.004, -0.003, 16),
            ),
        ),
    }


    # Diagnostic that identifies the absent-body termination in each mode. Serial Elegant reports the
    # memory fault. Pelegant shows only an MPI abort from a worker (exit status 1); that is attributed
    # to the missing body table by controls that differ from the failing run only in that table.
    TERMINATION_TEXT = {"serial": "Terminated by SIGSEGV", "mpi": "MPI_ABORT was invoked on rank"}


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def table(radius, orders, normal, skew):
        """ASCII SDDS multipole table: field fractions relative to the dipole at the reference radius."""
        rows = "".join(f"{n} {a:.17g} {b:.17g}\n" for n, a, b in zip(orders, normal, skew, strict=True))
        return (
            "SDDS1\n&parameter name=referenceRadius,type=double,units=m &end\n"
            "&column name=order,type=long &end\n&column name=normal,type=double &end\n"
            "&column name=skew,type=double &end\n&data mode=ascii &end\n"
            f"{radius!r}\n{len(orders)}\n{rows}"
        )


    def track(mode, setup, edge, *, negative, body, strength=1.0):
        """Exit (x, xp, y, yp) per particle; the same physical bend written two ways.

        negative=True: ANGLE < 0 (the branch that converts edge signs). negative=False: ANGLE > 0
        with TILT = pi, the same magnet rotated by pi about the beam axis. A rotation by pi sends
        z = x + i y to -z, so the field of order n relative to the (now reversed) dipole acquires
        (-1)^n, for normal and skew parts alike; the fractions of the rotated input are multiplied
        by (-1)^n. body is the list of body-table orders (all amplitudes zero), or None for no file.
        """
        s = SETUPS[setup]
        sign = np.array([(-1.0) ** n for n in s.orders])
        parity = np.ones(len(sign)) if negative else sign
        normal = strength * parity * np.array(s.normal)
        skew = strength * parity * np.array(s.skew)
        names, files = [], {}
        if edge is not None:
            names.append(f'EDGE{edge}_MULTIPOLES="edge.sdds"')
            files["edge.sdds"] = table(s.radius, s.orders, normal, skew)
        if body is not None:
            files["body.sdds"] = table(s.radius, body, np.zeros(len(body)), np.zeros(len(body)))
            names.append('SYSTEMATIC_MULTIPOLES="body.sdds"')
        angle, tilt = (-s.angle, 0.0) if negative else (s.angle, np.pi)
        lattice = (
            f"B: CCBEND,L={s.length!r},ANGLE={angle!r},TILT={tilt!r},N_SLICES={s.slices},"
            "INTEGRATION_ORDER=4,OPTIMIZE_FSE=0,OPTIMIZE_DX=0,REFERENCE_CORRECTION=0,SYNCH_RAD=0,"
            "ISR=0,REFERENCE_ORDER=0,MIN_NORMAL_ORDER=-1,MAX_NORMAL_ORDER=-1,MIN_SKEW_ORDER=-1,"
            f"MAX_SKEW_ORDER=-1{''.join(',' + name for name in names)}\nBL: LINE=(B)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={s.p0!r}, output=out.sdds"
            " &end\n&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n&track &end\n"
        )
        n = len(s.beam["p"])
        beam = dict(s.beam, particleID=np.arange(1, n + 1))
        result = run({"lattice.lte": lattice, "run.ele": commands, **files}, mode, {"beam.sdds": beam})
        out = particles(result["out.sdds"])
        if len(out["p"]) != n:
            sys.exit(f"{mode}: {n - len(out['p'])} particles lost")
        observed = np.column_stack([out[c] for c in ("x", "xp", "y", "yp")])
        if not (np.all(np.isfinite(observed)) and np.all(np.isfinite(out["p"]))):
            sys.exit(f"{mode}: nonfinite coordinates or momentum")
        if not np.allclose(out["p"], beam["p"], rtol=0, atol=2e-13, equal_nan=False):
            sys.exit(f"{mode}: the momentum changed")
        return observed


    def compare(label, observed, expected):
        """Print the last particle's (x, xp, y, yp); return True when any coordinate differs."""
        worst = np.max(np.abs(observed - expected))
        print(
            f"{label}: observed={fmt(observed[-1])} expected={fmt(expected[-1])} "
            f"max|difference|={worst:.6g}"
        )
        return bool(worst > ATOL)


    def fails(mode, setup, edge):
        """True when the negative bend with no body table terminates instead of tracking."""
        try:
            track(mode, setup, edge, negative=True, body=None)
        except RunFailure as failure:
            if TERMINATION_TEXT[mode] not in failure.log:
                sys.exit(
                    f"{mode} {setup} edge {edge}, no body table: {failure} without the expected "
                    f"diagnostic {TERMINATION_TEXT[mode]!r}; inconclusive.\n{failure.log}"
                )
            print(
                f"{mode} {setup} edge {edge}, no body table: run failed with status {failure.status}: "
                f"{TERMINATION_TEXT[mode]}"
            )
            return True
        print(f"{mode} {setup} edge {edge}, no body table: tracked all particles")
        return False


    defect = []
    print(
        "columns: x (m), xp, y (m), yp of the last particle; expected from the pi-rotated positive bend"
    )
    for mode in modes():
        for edge in (1, 2):
            label = f"{mode} mixed edge {edge}"
            oracle = track(mode, "mixed", edge, negative=False, body=[1, 2])
            # Body rows are zero, so any labels give the same field; labels that match the edge orders
            # are the control, which must agree with the oracle.
            aligned = track(mode, "mixed", edge, negative=True, body=[1, 2])
            if compare(f"{label}, body labels [1, 2] (control)", aligned, oracle):
                sys.exit("The aligned-label control disagrees with the oracle: invalid comparison.")
            relabeled = track(mode, "mixed", edge, negative=True, body=[2, 3])
            defect.append(compare(f"{label}, body labels [2, 3]", relabeled, oracle))
        # Without edge errors the body labels cannot matter, and the positive bend ignores them.
        plain = track(mode, "mixed", 1, negative=True, body=[2, 3], strength=0.0)
        zero_edge = track(mode, "mixed", 1, negative=False, body=[2, 3], strength=0.0)
        if compare(
            f"{mode} mixed edge 1, zero edge strength, labels [2, 3] (control)", plain, zero_edge
        ):
            sys.exit("The zero-edge control differs: invalid comparison.")
        positive = [track(mode, "mixed", 1, negative=False, body=b) for b in ([1, 2], [2, 3])]
        if compare(f"{mode} positive bend, labels [1, 2] vs [2, 3] (control)", *positive):
            sys.exit("A positive bend depends on the body labels: invalid comparison.")
        # Body table shorter than the edge table, or absent: the sign loop reads past the body arrays.
        for edge in (1, 2):
            oracle = track(mode, "odd", edge, negative=False, body=None)
            # Control: zero body rows padded to the edge orders [1, 3] must agree with the oracle.
            padded = track(mode, "odd", edge, negative=True, body=[1, 3])
            if compare(f"{mode} odd edge {edge}, body labels [1, 3] (control)", padded, oracle):
                sys.exit("The padded-label control disagrees with the oracle: invalid comparison.")
            short = track(mode, "odd", edge, negative=True, body=[1])
            defect.append(compare(f"{mode} odd edge {edge}, one body row", short, oracle))
        # Control: without any error file the negative bend tracks, so the failure below needs the edge.
        bare = track(mode, "odd", None, negative=True, body=None)
        bare_oracle = track(mode, "odd", None, negative=False, body=None)
        if compare(f"{mode} odd, no error files (control)", bare, bare_oracle):
            sys.exit(
                "The error-free negative bend disagrees with the positive bend: invalid comparison."
            )
        defect.append(fails(mode, "odd", 1))
        # Control: the positive bend with edge errors needs no body table.
        track(mode, "odd", 1, negative=False, body=None)
    print(f"comparison: transverse coordinates agree to {ATOL:g}")
    verdict(all(defect))

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
| Found (edge parity read from the body order array; an edge-only table terminates the run) | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Found (shorter body table, source-only incidental observation) | `gpt-6-astra` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source reading, not executed for this claim |
| Confirmed (empty or shorter body table, run stops or edge kicks wrong) | `gpt-6.1-sol` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

