---
title: "OPT-26: moments_output omits the radiation of KSEXT and SEXT sextupoles"
hide:
  - toc
---

# OPT-26 — moments_output omits the radiation of KSEXT and SEXT sextupoles

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`determineRadiationMatrix`, src/analyze.c:1767](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/analyze.c#L1767)<br>[the KSEXT branch, src/analyze.c:1949–1962](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/analyze.c#L1949-L1962)<br>[the SEXT branch, src/analyze.c:1963–1973](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/analyze.c#L1963-L1973)<br>[`determineRadiationMatrix1`, src/analyze.c:2357–2358](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/analyze.c#L2357-L2358)<br>[propagates the moments including coupling from rotated elements or offset sextupoles, with or without synchrotron radiation effects in dipoles, quadrupoles, and sextupoles, doc/elegant.tex:5644–5651](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5644-L5651)<br>[KSEXT and other kick elements need SYNCH_RAD=1 and ISR=1, doc/elegant.tex:5608–5612](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5608-L5612)<br>[suggest setting them on KSEXT when a closed orbit is present, doc/elegant.tex:5695–5700](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5695-L5700) |

Elegant's `moments_output` with `radiation=1` leaves out the synchrotron radiation of KSEXT and SEXT sextupoles, even with SYNCH_RAD=1 and ISR=1 on the KSEXT as the manual requires: energy loss, damping and quantum diffusion are all zero. Tracking the same KSEXT radiates. For a 0.2 m sextupole with K₂ = 100 m⁻³ displaced by 5 mm at p = 6000 mₑc, the radiation integrals along the orbit give a fractional momentum change of −1.277×10⁻¹⁰ (0.3914 eV), a damping term 1 − R₆₆ = 2.545×10⁻¹⁰ and a growth of the momentum-deviation variance ⟨δ²⟩ of 4.417×10⁻¹⁸; tracking gives the same loss, `moments_output` zero for all three. Loss and diffusion grow as the fourth and sixth power of the offset; a displaced KQUAD radiates correctly.

??? example "Reproducer: OPT-26.py"

    [Download OPT-26.py](OPT-26.py)

    ```python
    """OPT-26: moments_output omits the radiation of KSEXT and SEXT sextupoles

    Elegant's `moments_output` with `radiation=1` leaves out the synchrotron radiation of KSEXT and SEXT
    sextupoles, even with SYNCH_RAD=1 and ISR=1 on the KSEXT as the manual requires: energy loss,
    damping and quantum diffusion are all zero. Tracking the same KSEXT radiates. For a 0.2 m sextupole
    with K₂ = 100 m⁻³ displaced by 5 mm at p = 6000 mₑc, the radiation integrals along the orbit give a
    fractional momentum change of −1.277×10⁻¹⁰ (0.3914 eV), a damping term 1 − R₆₆ = 2.545×10⁻¹⁰ and a
    growth of the momentum-deviation variance ⟨δ²⟩ of 4.417×10⁻¹⁸; tracking gives the same loss,
    `moments_output` zero for all three. Loss and diffusion grow as the fourth and sixth power of the
    offset; a displaced KQUAD radiates correctly.

    The script runs moments_output with radiation=1 through one 0.2 m sextupole of K₂ = 100 m⁻³
    displaced by DX = 5 mm, as a KSEXT with SYNCH_RAD=1, ISR=1 and as a SEXT, at p = 6000 mₑc, and
    reads the element's centroid change C6, R66 and diffusion D66. The expected values come from a
    SciPy integration of the paraxial orbit through the displaced sextupole field and the classical
    loss and Sands diffusion integrals along it. Two controls validate the setup: a KQUAD with
    K₁ = 0.5 m⁻² displaced by 2.5 mm must match the same integrals in moments_output, and KSEXT
    tracking with SYNCH_RAD=1 must match the sextupole's expected loss.
    """

    import sys

    import numpy as np
    from _elegant import particles, run, verdict
    from scipy.integrate import solve_ivp

    P0 = 6000.0  # reference momentum beta*gamma (3.066 GeV electrons)
    GAMMA = np.hypot(1.0, P0)
    LENGTH, SLICES = 0.2, 200
    RE = 2.8179403205e-15  # classical electron radius (m), CODATA 2022
    LAMBDA_C = 3.8615926744e-13  # reduced Compton wavelength of the electron (m), CODATA 2022
    DDELTA = 1e-3  # momentum offset of the finite-difference damping estimate
    # moments_output and tracking slice the element (200 slices); the KQUAD control agrees with the
    # continuous integrals to 1.2e-5 (D66) and the KSEXT tracking loss to 5.5e-7.
    RTOL = 1e-4
    SEXTUPOLE = {"K2": 100.0, "DX": 0.005}
    QUADRUPOLE = {"K1": 0.5, "DX": 0.0025}


    def curvature(kind, strength, x):
        """Field curvature h = B/(B rho)_ref in m^-1 seen at x by an element displaced by dx.

        Normal sextupole B_y/(B rho) = K2/2 (x - dx)^2 and quadrupole K1 (x - dx), on the midplane.
        """
        u = x - strength["DX"]
        return strength["K2"] / 2 * u * u if kind == "sext" else strength["K1"] * u


    def radiation(kind, strength, delta=0.0):
        """Mean change of delta and its variance growth for a particle entering on axis at delta.

        Paraxial orbit x'' = -h/(1 + delta). Classical loss: d(delta)/ds = -(2/3) r_e gamma0^3
        (1 + delta)^2 h^2 (the particle's curvature is h/(1 + delta) and its gamma is
        gamma0 (1 + delta)). Sands quantum diffusion of delta at delta = 0:
        d<delta^2>/ds = (55/(24 sqrt 3)) r_e lambda_c gamma0^5 |h|^3.
        """

        def rhs(_, z):
            h = curvature(kind, strength, z[0])
            return [z[1], -h / (1 + delta), h * h, abs(h) ** 3]

        sol = solve_ivp(rhs, (0, LENGTH), [0, 0, 0, 0], method="DOP853", rtol=1e-12, atol=1e-20)
        i2, i3 = sol.y[2, -1], sol.y[3, -1]
        loss = -2 / 3 * RE * GAMMA**3 * (1 + delta) ** 2 * i2
        return loss, 55 / (24 * np.sqrt(3)) * RE * LAMBDA_C * GAMMA**5 * i3


    def expected(kind, strength):
        """Expected C6, 1 - R66 and D66; R66 by a central difference of the delta change."""
        c6, d66 = radiation(kind, strength)
        plus, _ = radiation(kind, strength, DDELTA)
        minus, _ = radiation(kind, strength, -DDELTA)
        return np.array([c6, -(plus - minus) / (2 * DDELTA), d66])


    def element(kind, strength, isr):
        if kind == "sext" and isr is None:
            return f"E: SEXT,L={LENGTH},K2={strength['K2']},DX={strength['DX']}\n"
        name, k = ("KSEXT", "K2") if kind == "sext" else ("KQUAD", "K1")
        return (
            f"E: {name},L={LENGTH},{k}={strength[k]},DX={strength['DX']},N_SLICES={SLICES},"
            f"SYNCH_RAD=1,ISR={isr}\n"
        )


    def moments(kind, strength, isr=1):
        """C6, 1 - R66 and D66 of the element from moments_output matrix_output with radiation=1."""
        # Only matrix output: writing moments along a one-element beamline crashes (a separate defect).
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, default_order=2 &end\n"
            "&run_control &end\n"
            "&twiss_output matched=0, beta_x=10, beta_y=10 &end\n"
            "&moments_output matrix_output=out.mat, matched=0, equilibrium=0, radiation=1,"
            " beta_x=10, beta_y=10, beta_z=10, emit_x=1e-16, emit_y=1e-16, emit_z=1e-16,"
            f" n_slices={SLICES} &end\n"
            "&bunched_beam &end\n&track &end\n"
        )
        files = {"lattice.lte": element(kind, strength, isr) + "BL: LINE=(E)\n", "run.ele": commands}
        mat = run(files, outputs=("out.mat",))["out.mat"]
        row = {n: np.asarray(mat[0, n], float) for n in ("C6", "R66", "D66")}
        if len(row["C6"]) != 1:
            sys.exit("expected one matrix row for the one-element beamline: invalid")
        return np.array([row["C6"][0], 1 - row["R66"][0], row["D66"][0]])


    def tracked_loss(strength):
        """delta change of one on-axis electron tracked through the KSEXT with SYNCH_RAD=1, ISR=0."""
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds &end\n"
            "&run_control &end\n&sdds_beam input=beam.sdds &end\n&track &end\n"
        )
        files = {"lattice.lte": element("sext", strength, 0) + "BL: LINE=(E)\n", "run.ele": commands}
        beam = {"x": np.zeros(1), "p": np.full(1, P0)}
        out = particles(run(files, beams={"beam.sdds": beam})["out.sdds"])
        return out["p"][0] / P0 - 1


    def fmt(values):
        return "[" + ", ".join(f"{v:.7g}" for v in np.atleast_1d(values)) + "]"


    def close(observed, reference):
        return bool(np.all(np.abs(observed - reference) <= RTOL * np.abs(reference)))


    print(f"p0={P0:g} m_e c, L={LENGTH} m, {SLICES} slices; columns C6, 1-R66, D66")
    quad_expected = expected("quad", QUADRUPOLE)
    quad_observed = moments("quad", QUADRUPOLE)
    print(
        f"control KQUAD K1=0.5 DX=2.5 mm moments: observed={fmt(quad_observed)} "
        f"expected={fmt(quad_expected)}"
    )
    if not close(quad_observed, quad_expected):
        sys.exit("The KQUAD control disagrees with the radiation integrals: invalid.")
    sext_expected = expected("sext", SEXTUPOLE)
    tracked = tracked_loss(SEXTUPOLE)
    print(
        f"control KSEXT K2=100 DX=5 mm tracking SYNCH_RAD=1: delta change observed={tracked:.7g} "
        f"expected={sext_expected[0]:.7g}"
    )
    if not close(np.array([tracked]), sext_expected[:1]):
        sys.exit("KSEXT tracking disagrees with the sextupole loss integral: invalid.")
    print(f"expected sextupole loss {-sext_expected[0] * GAMMA * 0.51099895069e6:.4g} eV")
    defect = []
    for label, isr in (("KSEXT SYNCH_RAD=1 ISR=1", 1), ("SEXT", None)):
        observed = moments("sext", SEXTUPOLE, isr)
        print(f"{label} K2=100 DX=5 mm moments: observed={fmt(observed)} expected={fmt(sext_expected)}")
        # The defect: every radiation entry is exactly zero, so the element radiates nothing.
        defect.append(bool(np.all(observed == 0)) and not close(observed, sext_expected))
    print(f"comparison: relative tolerance {RTOL:g}")
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
| Found (noted, source (KSEXT moments radiation zero in its steering-kick decks, traced to the length check)) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Found (source tracing of the KSEXT and SEXT branches) | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source reading, not executed for this claim |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

