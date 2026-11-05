---
title: "OPT-16: CCBEND radiation integral I4 omits the rectangular-edge terms"
hide:
  - toc
---

# OPT-16 — CCBEND radiation integral I4 omits the rectangular-edge terms

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`addCcbendRadiationIntegrals`, slice sum, src/ccbend.c:1807–1847](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L1807-L1847)<br>[`incrementRadIntegrals`, CCBEND branch, src/twiss.cc:3091–3099](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3091-L3099)<br>[`etaK1_rhoAve`, src/twiss.cc:3278](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3278)<br>[`I4 = I2/rho*etaAve - 2*length*etaK1_rhoAve`, src/twiss.cc:3329](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3329)<br>[CCBEND description, doc/ccbend.tex:1–4](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ccbend.tex#L1-L4)<br>[`In` is the nth radiation integral, doc/elegant.tex:8262](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8262) |

With the default hard-edge fringe model, Elegant's `twiss_output` omits the rectangular-face contributions to CCBEND radiation integral I₄. For a uniform, unyawed bend they are −h² tan(θ/2)(η₁ + η₂), where h is curvature, θ is bend angle and η₁, η₂ are face dispersions. A 1.3 m, 0.12 rad bend entered with η = 0.4 m and η′ = 0.03 gives 4.562×10⁻⁴ m⁻¹ instead of −1.331×10⁻⁵ m⁻¹. In a 73.60 m, 3 GeV ring, the horizontal damping partition is 0.8707 instead of 1.003 and the natural emittance is 441.7 nm instead of 383.2 nm, 15.26% high.

??? example "Reproducer: OPT-16.py"

    [Download OPT-16.py](OPT-16.py)

    ```python
    """OPT-16: CCBEND radiation integral I4 omits the rectangular-edge terms

    With the default hard-edge fringe model, Elegant's `twiss_output` omits the rectangular-face
    contributions to CCBEND radiation integral I₄. For a uniform, unyawed bend they are −h²
    tan(θ/2)(η₁ + η₂), where h is curvature, θ is bend angle and η₁, η₂ are face dispersions. A 1.3
    m, 0.12 rad bend entered with η = 0.4 m and η′ = 0.03 gives 4.562×10⁻⁴ m⁻¹ instead of
    −1.331×10⁻⁵ m⁻¹. In a 73.60 m, 3 GeV ring, the horizontal damping partition is 0.8707 instead of
    1.003 and the natural emittance is 441.7 nm instead of 383.2 nm, 15.26% high.

    The script runs serial Elegant `twiss_output` with radiation integrals on one constant-field
    CCBEND and on the equivalent rectangular CSBEND (edge angles of half the bend angle), whose I4 is
    Elegant's analytic sector-bend formula including the edge terms; a third case moves the CSBEND's
    entrance-edge dispersion kick into the starting dispersion slope, which keeps the body's dispersion
    and removes both edge terms. It then compares the horizontal damping partition and natural emittance
    of a ring of CCBENDs with those of the same ring built from the equivalent CSBENDs. The horizontal
    tune, I2 and the single-magnet exit dispersion are controls showing that the two magnets have the
    same linear optics.
    """

    import sys

    import numpy as np
    from _elegant import run, verdict

    P_MEV = 3000.0  # reference momentum, MeV/c

    # Single magnet: 1.3 m long, bending 0.12 rad, entered with fixed optics and dispersion.
    L1, THETA1 = 1.3, 0.12
    START = "beta_x=3.2, alpha_x=0.3, beta_y=2.8, alpha_y=-0.1"
    ETA0, ETAP0 = 0.4, 0.03  # entrance dispersion (m) and slope

    # Ring: 16 FODO cells with two 1 m bends each, 2*pi/32 per bend, at 3 GeV.
    N_BENDS = 32
    L_RING, THETA_RING = 1.0, 2 * np.pi / N_BENDS


    def ccbend(length, angle):
        return f"B: CCBEND,L={length!r},ANGLE={angle!r},N_SLICES=128"


    def csbend(length, angle, e1, e2):
        # Rectangular magnet: each pole face is rotated by half the bend angle relative to the sector.
        return (
            f"B: CSBEND,L={length!r},ANGLE={angle!r},E1={e1!r},E2={e2!r},N_SLICES=128,"
            "EXPAND_HAMILTONIAN=0"
        )


    def twiss(bend, line, start):
        """Twiss parameters and columns of one run; `start` is empty for a periodic solution."""
        if start:
            lattice = f"{bend}\nBL: LINE=(B)\n"
            matched = f"matched=0, {start}"
        else:
            lattice = (
                "F: QUAD,L=0.3,K1=2.4\nG: QUAD,L=0.3,K1=-2.4\nD: DRIF,L=0.5\nE: DRIF,L=0.5\n"
                f"{bend}\nC: LINE=(F,D,B,E,G,D,B,E)\nBL: LINE=({N_BENDS // 2}*C)\n"
            )
            matched = "matched=1"
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central_mev={P_MEV!r} &end\n"
            f"&twiss_output filename=twiss.sdds, {matched}, radiation_integrals=1 &end\n"
        )
        return run({"lattice.lte": lattice, "run.ele": commands}, outputs=("twiss.sdds",))["twiss.sdds"]


    def par(table, name):
        return float(table[name][0])


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.atleast_1d(values)) + "]"


    defect = False
    h = THETA1 / L1
    start = f"{START}, eta_x={ETA0!r}, etap_x={ETAP0!r}"
    cc = twiss(ccbend(L1, THETA1), "BL", start)
    cs = twiss(csbend(L1, THETA1, THETA1 / 2, THETA1 / 2), "BL", start)
    # Body only: the entrance edge kicks the dispersion slope by eta0*h*tan(E1); starting a sector
    # body with that slope gives the same body dispersion without either edge term of I4.
    body_start = f"{START}, eta_x={ETA0!r}, etap_x={float(ETAP0 + ETA0 * h * np.tan(THETA1 / 2))!r}"
    body = twiss(csbend(L1, THETA1, 0.0, 0.0), "BL", body_start)

    eta_cc, eta_cs = float(cc[0, "etax"][-1]), float(cs[0, "etax"][-1])
    print(f"single magnet control, exit dispersion (m): CCBEND={eta_cc:.12g} CSBEND={eta_cs:.12g}")
    if abs(eta_cc - eta_cs) > 1e-9:
        sys.exit("The CCBEND and the rectangular CSBEND differ in linear optics: invalid comparison.")
    # The edge terms of I4 (CSBEND formula in twiss.cc): -h^2 (eta_1 tan E1 + eta_2 tan E2), with
    # eta_1, eta_2 the dispersion at the entrance and exit faces and E1 = E2 = theta/2 here.
    edge = -(h**2) * np.tan(THETA1 / 2) * (ETA0 + eta_cs)
    print(
        f"single magnet CSBEND edge terms: with edges - body only = {par(cs, 'I4') - par(body, 'I4'):.10g}"
        f" 1/m, formula -h^2 tan(theta/2) (eta1 + eta2) = {edge:.10g} 1/m"
    )
    # For K1 = K2 = 0 the CCBEND slice sum gives exactly I4 = h^2 I1: the body term alone.
    print(
        f"single magnet CCBEND: I4={par(cc, 'I4'):.10g} 1/m, h^2*I1={h**2 * par(cc, 'I1'):.10g} 1/m,"
        f" CSBEND body only={par(body, 'I4'):.10g} 1/m"
    )
    mismatch = abs(par(cc, "I4") - par(cs, "I4"))
    print(f"single magnet I4 (1/m): observed={par(cc, 'I4'):.10g} expected={par(cs, 'I4'):.10g}")
    # The CCBEND body sum has a persistent chord-measure/frame bias, plus finite-slice
    # endpoint terms. Its uniform-field limit is the sector-body value times L/chord.
    # Here the body bias is about 6e-4 relative; adding the faces alone leaves about
    # 2.06% error in the small net I4. The missing-face contribution is much larger.
    defect |= mismatch > 0.5 * abs(edge)

    ring_cc = twiss(ccbend(L_RING, THETA_RING), "BL", "")
    ring_cs = twiss(csbend(L_RING, THETA_RING, THETA_RING / 2, THETA_RING / 2), "BL", "")
    controls = ["nux", "I2"]
    print(
        f"ring control nux, I2: CCBEND={fmt([par(ring_cc, k) for k in controls])}"
        f" CSBEND={fmt([par(ring_cs, k) for k in controls])}"
    )
    if abs(par(ring_cc, "nux") - par(ring_cs, "nux")) > 1e-8:
        sys.exit("The CCBEND and CSBEND rings differ in horizontal tune: invalid comparison.")
    names = ["I4", "Jx", "ex0"]
    observed = [par(ring_cc, k) for k in names]
    expected = [par(ring_cs, k) for k in names]
    print(f"ring I4 (1/m), Jx, ex0 (m): observed={fmt(observed)} expected={fmt(expected)}")
    print(f"ring ex0 observed/expected = {observed[2] / expected[2]:.6g}")
    defect |= abs(observed[1] - expected[1]) > 0.01
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

