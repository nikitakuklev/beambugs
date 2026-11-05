---
title: "TRK-24: KQUAD and KSEXT edge multipole cutoffs select rows by the body file's order labels"
hide:
  - toc
---

# TRK-24 — KQUAD and KSEXT edge multipole cutoffs select rows by the body file's order labels

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`computeTotalErrorMultipoleFields`, multipole.c:1788–1795](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1788-L1795)<br>[row test, multipole.c:1791–1794](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1791-L1794)<br>[entrance edge consumer, multipole.c:1369–1379](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1369-L1379)<br>[exit edge consumer, multipole.c:1532–1542](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1532-L1542)<br>[KQUAD setup, multipole.c:818–846](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L818-L846)<br>[KSEXT setup, multipole.c:894–922](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L894-L922)<br>[edge arrays, multipole.c:1731–1744](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1731-L1744)<br>[KQUAD manual, kquad.tex:11](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/kquad.tex#L11)<br>[KSEXT manual, ksext.tex:11](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ksext.tex#L11)<br>[cutoff metadata, track_data.c:1279](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1279) |

When KQUAD or KSEXT magnets use separate body and edge multipole files, minimum and maximum normal/skew order cutoffs incorrectly select edge rows using the body file's order labels. Relabelling zero-amplitude body rows must leave every physical field unchanged, but changes the edge kicks. In a 0.17 m KQUAD with K₁ = −0.7 m⁻², edge orders 2 and 4 and minimum normal and skew orders of 3, reversing body labels from [2, 4] to [4, 2] changes horizontal slope x′ from −1.754×10⁻³ to −1.637×10⁻³. The largest vertical-slope error is 5.084×10⁻⁴; independent edge impulses and body integration establish the expected result.

??? example "Reproducer: TRK-24.py"

    [Download TRK-24.py](TRK-24.py)

    ```python
    """TRK-24: KQUAD and KSEXT edge multipole cutoffs select rows by the body file's order labels

    When KQUAD or KSEXT magnets use separate body and edge multipole files, minimum and maximum
    normal/skew order cutoffs incorrectly select edge rows using the body file's order labels.
    Relabelling zero-amplitude body rows must leave every physical field unchanged, but changes the edge
    kicks. In a 0.17 m KQUAD with K₁ = −0.7 m⁻², edge orders 2 and 4 and minimum normal and skew orders
    of 3, reversing body labels from [2, 4] to [4, 2] changes horizontal slope x′ from −1.754×10⁻³ to
    −1.637×10⁻³. The largest vertical-slope error is 5.084×10⁻⁴; independent edge impulses and body
    integration establish the expected result.

    The script tracks 12 electrons through a KQUAD or KSEXT with an edge file of orders 2 and 4 and a
    body file of zero amplitudes, with a minimum or a maximum order of 3, and varies only the body
    file's order labels. It compares each exit coordinate with an independent integration: the kept
    edge rows applied at entrance and exit around a straight magnet. The control is the body file with
    the edge file's own labels; every relabelling must give the same result, since zero rows add no
    field.
    """

    import sys
    from math import factorial

    import numpy as np
    from _elegant import modes, particles, run, verdict
    from _multipole import P0, RADIUS, beam, canonical, observable, straight_body, table_text

    LENGTH = 0.17  # m
    EDGE_ORDERS, EDGE_NORMAL, EDGE_SKEW = [2, 4], [0.08, -0.04], [0.03, 0.02]
    # Body files with zero amplitude in every row; only the order labels differ.
    BODY_LABELS = {
        "aligned [2,4]": [2, 4],
        "reversed [4,2]": [4, 2],
        "low [0,1]": [0, 1],
        "high [5,6]": [5, 6],
    }
    KINDS = {"KQUAD": (1, -0.7), "KSEXT": (2, -17.0)}  # element: (main order m, strength K_m)
    CUTOFFS = {  # cutoff text: (keeps order n if)
        "MIN_NORMAL_ORDER=3,MIN_SKEW_ORDER=3": lambda n: n >= 3,
        "MAX_NORMAL_ORDER=3,MAX_SKEW_ORDER=3": lambda n: n <= 3,
    }
    ATOL = 1e-13  # observed agreement with the oracle is ~1e-15; wrong row sets differ by >1e-7
    SPLIT = 1e-7
    P_ATOL = 2e-13  # no radiation, so the momentum is conserved


    def edge_oracle(initial, m, strength, keep):
        """Exit x, xp, y, yp for the edge rows of order in `keep`, from the independent integrator.

        A thin edge multipole of order n with fractions (a_n, b_n) of the main field at radius R
        gives the impulse (dqx, dqy) = (-Re C_e, Im C_e) with
        C_e = K_m L R^m/m! * sum_n (a_n - i b_n) ((x + iy)/R)^n, at entrance and at exit, with a
        straight magnet of the main field between.
        """

        def impulse(state):
            x, y, qx, qy = state
            c_e = np.zeros(len(x), dtype=complex)
            for n, a, b in zip(EDGE_ORDERS, EDGE_NORMAL, EDGE_SKEW, strict=True):
                if n in keep:
                    c_e += (
                        strength
                        * LENGTH
                        * RADIUS**m
                        / factorial(m)
                        * (a - 1j * b)
                        * ((x + 1j * y) / RADIUS) ** n
                    )
            return np.stack((x, y, qx - c_e.real, qy + c_e.imag))

        state = impulse(canonical(initial))
        state = straight_body(state, initial, m, strength, LENGTH, harmonics=())
        result = observable(impulse(state), initial)
        if not np.all(np.isfinite(result)):
            sys.exit("the independent edge-impulse oracle is not finite")
        return result


    def track(mode, kind, cutoff, labels, initial):
        m, strength = KINDS[kind]
        lattice = (
            f"E: {kind},L={LENGTH},K{m}={strength},N_SLICES=96,INTEGRATION_ORDER=4,{cutoff},"
            'SYSTEMATIC_MULTIPOLES="body.sdds",EDGE_MULTIPOLES="edge.sdds",SYNCH_RAD=0,ISR=0\n'
            "BL: LINE=(E)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0}, output=out.sdds, "
            "random_number_seed=753219 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        files = {
            "lattice.lte": lattice,
            "run.ele": commands,
            "body.sdds": table_text(labels, [0.0, 0.0], [0.0, 0.0]),
            "edge.sdds": table_text(EDGE_ORDERS, EDGE_NORMAL, EDGE_SKEW),
        }
        out = particles(run(files, mode, {"beam.sdds": initial})["out.sdds"])
        if len(out["p"]) != len(initial["p"]):
            sys.exit(f"{mode} {kind}: particles were lost")
        if not np.allclose(out["p"], initial["p"], rtol=0, atol=P_ATOL, equal_nan=False):
            sys.exit(f"{mode} {kind}: the momentum is not conserved")
        return np.column_stack((out["x"], out["xp"], out["y"], out["yp"]))


    def last(a):
        return "[" + ", ".join(repr(float(v)) for v in a[-1]) + "]"


    def compare(label, observed, expected):
        """Print the last particle's x, xp, y, yp and the largest difference over all particles.

        Nonfinite values make the comparison invalid: the run terminates instead of returning a NaN
        that every later `> ATOL` test would accept.
        """
        if not (np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))):
            sys.exit(f"{label}: a nonfinite value makes the comparison invalid")
        error = float(np.max(np.abs(observed - expected)))
        print(f"{label}: observed={last(observed)} expected={last(expected)} max|diff|={error:.4g}")
        return error


    defect = False
    initial = beam()
    for mode in modes():
        for kind, (m, strength) in KINDS.items():
            for cutoff, passes in CUTOFFS.items():
                # Correct selection tests the edge file's own order labels.
                keep = {n for n in EDGE_ORDERS if passes(n)}
                expected = edge_oracle(initial, m, strength, keep)
                for name, labels in BODY_LABELS.items():
                    observed = track(mode, kind, cutoff, labels, initial)
                    error = compare(f"{mode} {kind} {cutoff} body labels {name}", observed, expected)
                    # What the code does instead: row i of the edge file is kept when row i of the
                    # body file passes the cutoff.
                    by_body = {n for n, label in zip(EDGE_ORDERS, labels, strict=True) if passes(label)}
                    explained = edge_oracle(initial, m, strength, by_body)
                    if not np.allclose(observed, explained, rtol=0, atol=ATOL, equal_nan=False):
                        sys.exit(f"{mode} {kind} {name}: result matches neither row selection")
                    if name.startswith("aligned"):
                        if error > ATOL:
                            sys.exit(f"{mode} {kind} {cutoff}: the aligned control disagrees")
                    else:
                        defect |= error > SPLIT
    print(f"comparison: atol={ATOL:g} for agreement, {SPLIT:g} for disagreement")
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

??? example "Helper: _multipole.py"

    [Download _multipole.py](_multipole.py)

    ```python
    """Shared setup for the KQUAD and KSEXT multipole-table reproducers: a 12-electron beam, inline
    multipole tables, and an independent integrator for a straight magnet with steering fields.

    The integrator follows the particles through a straight hard-edge magnet in canonical transverse
    momenta q = p_perp/p0, solving the Lorentz equations with scipy, without any Elegant code.
    """

    from __future__ import annotations

    from math import factorial

    import numpy as np
    from scipy.integrate import solve_ivp

    P0 = 45.0  # reference momentum, units of m_e c
    N = 12  # particles, enough for both workers of a 3-rank run
    RADIUS = 0.027  # m, reference radius of the multipole tables


    def beam() -> dict[str, np.ndarray]:
        """Twelve electrons with transverse offsets and slopes and momenta 43 to 49 m_e c."""
        return {
            "x": np.linspace(-0.018, 0.021, N),
            "xp": np.linspace(0.003, -0.004, N),
            "y": np.linspace(0.008, -0.011, N),
            "yp": np.linspace(-0.002, 0.003, N),
            "t": np.zeros(N),
            "p": np.linspace(43.0, 49.0, N),
        }


    def table_text(orders, normal, skew=None) -> str:
        """ASCII SDDS multipole table: parameter referenceRadius, columns order, normal, (skew)."""
        columns = "&column name=order, type=long &end\n&column name=normal, type=double &end\n"
        if skew is not None:
            columns += "&column name=skew, type=double &end\n"
        rows = "".join(
            f"{n} {a!r}" + (f" {b!r}" if skew is not None else "") + "\n"
            for n, a, b in zip(orders, normal, skew if skew is not None else normal, strict=True)
        )
        return (
            "SDDS1\n"
            '&parameter name=referenceRadius, type=double, units="m" &end\n'
            + columns
            + "&data mode=ascii &end\n"
            + f"{RADIUS!r}\n{len(orders)}\n"
            + rows
        )


    def canonical(b: dict[str, np.ndarray]) -> np.ndarray:
        """Rows x, y, qx, qy with q = (p/p0) * slope / sqrt(1 + xp^2 + yp^2)."""
        ratio = b["p"] / P0
        qs = ratio / np.sqrt(1 + b["xp"] ** 2 + b["yp"] ** 2)
        return np.stack((b["x"], b["y"], qs * b["xp"], qs * b["yp"]))


    def observable(state: np.ndarray, b: dict[str, np.ndarray]) -> np.ndarray:
        """Columns x, xp, y, yp from canonical rows x, y, qx, qy."""
        x, y, qx, qy = state
        qs = np.sqrt((b["p"] / P0) ** 2 - qx**2 - qy**2)
        return np.column_stack((x, qx / qs, y, qy / qs))


    def straight_body(
        state: np.ndarray,
        b: dict[str, np.ndarray],
        main_order: int,
        strength: float,
        length: float,
        factor: float = 0.0,
        kick: tuple[float, float] = (0.0, 0.0),
        harmonics=((2, 0.07), (4, -0.025)),
    ) -> np.ndarray:
        """Canonical rows after a straight magnet of the given length (m).

        Complex field C = By + iBx divided by the reference rigidity, with z = x + iy:
        C = K_m z^m/m! - (H - iV)/L - factor/L sum_n a_n (H - i(-1)^(n/2) V) (z/R)^n,
        where K_m is `strength`, (H, V) = `kick` the steering angles spread over the length and a_n
        the normal steering fractions at order n (zero steering and zero factor leave the main field).
        The equations of motion are dx/ds = qx/qs, dy/ds = qy/qs, dqx/ds = -Re C, dqy/ds = Im C with
        qs = sqrt((p/p0)^2 - qx^2 - qy^2).
        """
        ratio = b["p"] / P0

        def rhs(_, flat):
            x, y, qx, qy = flat.reshape(4, N)
            z = x + 1j * y
            field = strength * z**main_order / factorial(main_order)
            field -= (kick[0] - 1j * kick[1]) / length
            for n, a in harmonics:
                coefficient = -factor * a * (kick[0] - 1j * (-1) ** (n // 2) * kick[1])
                field += coefficient / length * (z / RADIUS) ** n
            qs = np.sqrt(ratio**2 - qx**2 - qy**2)
            return np.stack((qx / qs, qy / qs, -field.real, field.imag)).ravel()

        sol = solve_ivp(rhs, (0, length), state.ravel(), method="DOP853", rtol=3e-13, atol=2e-15)
        if not sol.success:
            raise RuntimeError("straight-magnet integration failed")
        return sol.y[:, -1].reshape(4, N)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

