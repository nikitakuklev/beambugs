---
title: "SCT-23: moments_output IBS rotates the coupled diffusion tensor with transposed eigenvectors"
hide:
  - toc
---

# SCT-23 — moments_output IBS rotates the coupled diffusion tensor with transposed eigenvectors

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant master (`0355dd6`) |
| Source | [`diagonalization_matrix_3x3`, moments.c:1227–1335](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L1227-L1335)<br>[moments.c:1536–1540](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L1536-L1540)<br>[moments.c:1603–1607](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L1603-L1607)<br>[moments.c:1612–1636](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L1612-L1636)<br>[`accumulateRadiationMatrices`, compute_matrices.c:416–438](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L416-L438)<br>[`ibs_iterations`, elegant.tex:5617–5624](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5617-L5624)<br>[elegant.tex:5644–5648](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5644-L5648) |

With ibs_iterations, moments_output rotates each element's intrabeam-scattering (IBS) diffusion out of the principal axes of the beam-frame momentum covariance with the transposed eigenvector matrix, so a momentum ellipsoid tilted by φ gets diffusion tilted by −φ. A ROTATE pair around a drift changes no physics, and the Kubo–Oide diffusion turns with the beam, so the equilibrium must not change. In a 524.8 m ring at 1.5 GeV/c with 1 nC, wrapping every drift in ±45° ROTATE pairs changes the IBS mode emittance e₂ from 21.00 to 20.77 pm (−1.086%) and e₁ by +0.1601%. Without rotated frames this weakly coupled ring is off by under 10⁻⁵, and beams with a diagonal momentum covariance are unaffected.

??? example "Reproducer: SCT-23.py"

    [Download SCT-23.py](SCT-23.py)

    ```python
    """SCT-23: moments_output IBS rotates the coupled diffusion tensor with transposed eigenvectors

    With ibs_iterations, moments_output rotates each element's intrabeam-scattering (IBS) diffusion out
    of the principal axes of the beam-frame momentum covariance with the transposed eigenvector matrix,
    so a momentum ellipsoid tilted by φ gets diffusion tilted by −φ. A ROTATE pair around a drift
    changes no physics, and the Kubo–Oide diffusion turns with the beam, so the equilibrium must not
    change. In a 524.8 m ring at 1.5 GeV/c with 1 nC, wrapping every drift in ±45° ROTATE pairs changes
    the IBS mode emittance e₂ from 21.00 to 20.77 pm (−1.086%) and e₁ by +0.1601%. Without rotated
    frames this weakly coupled ring is off by under 10⁻⁵, and beams with a diagonal momentum covariance
    are unaffected.

    The script runs moments_output with radiation and IBS (automatic Coulomb logarithm, 30 iterations)
    on a 128-cell, 524.8 m FODO ring at 1.5 GeV/c with a 1 nC bunch and a 1 mrad quadrupole roll, in
    three equivalent descriptions: plain, and with every drift wrapped in a ROTATE pair of ±45° and of
    ±90°. Since a ROTATE pair around a drift is the identity map, every description must give the plain
    ring's equilibrium mode emittances and rms sizes at the ring end, outside the rotated frames. The
    controls are the radiation-only emittances of all descriptions and the ±90° wrap, which the
    transposition leaves exact because the eigenvectors of a beam turned by 90° are still the
    coordinate axes.
    """

    import numpy as np
    from _elegant import run, verdict

    P_MEV, CHARGE, VOLT, CELLS = 1500.0, 1e-9, 1e6, 128  # MeV/c, C, V, FODO cells
    ITERATIONS = 30  # IBS iterations; 60 change the IBS emittances by less than 2e-7 relative


    def lattice(phase, wrap):
        """128 FODO cells (QF rolled by 1 mrad), two SBENDs per cell closing 2 pi; with `wrap` (rad)
        every drift is DW = (ROTATE +wrap, drift, ROTATE -wrap), which leaves the optics unchanged."""
        drift = "D"
        rotate = ""
        if wrap is not None:
            drift = "DW"
            rotate = f"RP: ROTATE,TILT={wrap!r}\nRM: ROTATE,TILT={-wrap!r}\nDW: LINE=(RP,D,RM)\n"
        return (
            f"Q: CHARGE,TOTAL={CHARGE!r}\n"
            "D: DRIF,L=0.4\n"
            "QF: QUAD,L=0.25,K1=0.8,TILT=0.001\n"
            "QD: QUAD,L=0.25,K1=-0.7\n"
            f"B: SBEN,L=1,ANGLE={np.pi / CELLS!r}\n"
            f"RF: RFCA,VOLT={VOLT!r},FREQ=500e6,PHASE={phase!r}\n"
            f"{rotate}"
            f"CELL: LINE=(QF,{drift},B,{drift},QD,{drift},B,{drift})\n"
            f"RING: LINE=(Q,{CELLS}*CELL,RF)\n"
        )


    def moments(phase, wrap, iterations):
        """Mode emittances (e1, e2, e3) and rms sizes (x, y, s, delta) at the end of the ring, outside
        every ROTATE pair, from the last moments_output page, and the twiss parameters."""
        commands = (
            f'&run_setup lattice="ring.lte", use_beamline="RING", p_central_mev={P_MEV!r}, '
            "default_order=2 &end\n"
            '&twiss_output filename="ring.twi", matched=1, radiation_integrals=1 &end\n'
            '&moments_output filename="ring.mom", radiation=1, equilibrium=1, '
            f"ibs_iterations={iterations} &end\n"
        )
        out = run(
            {"ring.lte": lattice(phase, wrap), "run.ele": commands}, outputs=("ring.mom", "ring.twi")
        )
        mom, twi = out["ring.mom"], out["ring.twi"]
        par = {p.name: p.data for p in mom.parameters}
        e = np.array([par[f"e{i}"][-1] for i in (1, 2, 3)], dtype=float)
        last = mom.n_pages - 1
        sizes = np.array([mom[last, f"s{i}"][-1] for i in (1, 3, 5, 6)], dtype=float)
        return e, sizes, {p.name: p.data[0] for p in twi.parameters}


    # The RF phase balancing the radiation loss U0 above transition is 180 deg - asin(U0/V).
    _, _, tw = moments(180.0, None, 0)
    u0 = float(tw["U0"]) * 1e6  # eV per turn
    phase = float(180 - np.degrees(np.arcsin(u0 / VOLT)))
    print(f"ring: U0={u0 / 1e3:.6f} keV/turn, RF phase {phase:.10f} deg")

    cases = {
        "plain": None,
        "drifts wrapped in +-45 deg": np.pi / 4,
        "drifts wrapped in +-90 deg": np.pi / 2,
    }
    radiation, ibs, sizes = {}, {}, {}
    for name, wrap in cases.items():
        radiation[name] = moments(phase, wrap, 0)[0]
        ibs[name], sizes[name], _ = moments(phase, wrap, ITERATIONS)
        print(
            f"{name}: radiation-only (e1, e2, e3) = {np.array2string(radiation[name], precision=10)} m"
        )
        print(f"  IBS (e1, e2, e3) = {np.array2string(ibs[name], precision=10)} m; rms (x, y, s, delta) "
              f"at the ring end = {np.array2string(sizes[name], precision=10)}")  # fmt: skip

    # A ROTATE pair around a drift is the identity map, and the Kubo-Oide diffusion of a Gaussian beam
    # is covariant under a rotation R of the coordinates (K(R F R^T) = R K(F) R^T for the beam-frame
    # momentum covariance F), so every description has the plain ring's equilibrium moments.
    plain = ibs["plain"]
    for name in cases:
        if not np.allclose(radiation[name], radiation["plain"], rtol=1e-6, atol=0):
            raise SystemExit(
                f"control: radiation-only emittances of '{name}' differ from the plain ring's"
            )
    if not np.all(plain > radiation["plain"]):
        raise SystemExit("control: IBS does not increase the plain ring's emittances")
    control = ibs["drifts wrapped in +-90 deg"]
    if not (
        np.allclose(control, plain, rtol=1e-6, atol=0)
        and np.allclose(sizes["drifts wrapped in +-90 deg"], sizes["plain"], rtol=1e-6, atol=0)
    ):
        raise SystemExit("control: the +-90 deg wrap changes the IBS emittances")
    observed = ibs["drifts wrapped in +-45 deg"]
    rel = (observed / plain - 1) * 100
    for i in range(3):
        print(f"+-45 deg wrap, IBS e{i + 1}: observed={observed[i]:.6e} m expected={plain[i]:.6e} m "
              f"(plain ring, frame invariance; {rel[i]:+.4g}%)")  # fmt: skip
    size_rel = (sizes["drifts wrapped in +-45 deg"] / sizes["plain"] - 1) * 100
    for i, name in enumerate(("x", "y", "s", "delta")):
        print(
            f"+-45 deg wrap, IBS rms {name} at the ring end: observed/expected - 1 = {size_rel[i]:+.4g}%"
        )
    print("comparison: radiation-only emittances and the +-90 deg control rtol=1e-6; defect needs the "
          "+-45 deg IBS e2 and rms y to differ from the plain ring's by more than 0.1%")  # fmt: skip
    verdict(bool(abs(rel[1]) > 0.1 and abs(size_rel[1]) > 0.1))

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
| Found | `gpt-6-astra` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

