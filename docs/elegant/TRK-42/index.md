---
title: "TRK-42: RFCA END1_FOCUS and END2_FOCUS edge lenses omit the 1/β² factor"
hide:
  - toc
---

# TRK-42 — RFCA END1_FOCUS and END2_FOCUS edge lenses omit the 1/β² factor

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`trackRfCavityWithWakes`, simple_rfca.c:745–768](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L745-L768)<br>[simple_rfca.c:804–807](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L804-L807)<br>[simple_rfca.c:1019–1024](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1019-L1024)<br>[1072–1076](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L1072-L1076)<br>[`rf_cavity_matrix`, compute_matrices.c:2604–2610](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L2604-L2610)<br>[parameter table, track_data.c:608–609](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L608-L609)<br>[RFCA manual, rfca.tex](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/rfca.tex) |

RFCA's END1_FOCUS and END2_FOCUS edge lenses, in tracking and in the transfer matrix, have inverse focal length γ′/(2γ), where γ′ is the energy gain per length in units of mc². Integrating the radial field −(r/2)∂E_z/∂z of a hard field edge over the transit time dz/(βc) gives γ′/(2β²γ) at the local β and γ, so both lenses are too weak by the factor β². For electrons entering a 0.7 m cavity that adds 12 MeV at 1.5 MeV kinetic energy, the entrance lens is 4.262 m⁻¹ instead of 4.556 m⁻¹, 6.457% weak; at 5 MeV it is 0.8598% weak and at 20 MeV 0.06207% weak.

??? example "Reproducer: TRK-42.py"

    [Download TRK-42.py](TRK-42.py)

    ```python
    """TRK-42: RFCA END1_FOCUS and END2_FOCUS edge lenses omit the 1/β² factor

    RFCA's END1_FOCUS and END2_FOCUS edge lenses, in tracking and in the transfer matrix, have inverse
    focal length γ′/(2γ), where γ′ is the energy gain per length in units of mc². Integrating the radial
    field −(r/2)∂E_z/∂z of a hard field edge over the transit time dz/(βc) gives γ′/(2β²γ) at the local
    β and γ, so both lenses are too weak by the factor β². For electrons entering a 0.7 m cavity that
    adds 12 MeV at 1.5 MeV kinetic energy, the entrance lens is 4.262 m⁻¹ instead of 4.556 m⁻¹, 6.457%
    weak; at 5 MeV it is 0.8598% weak and at 20 MeV 0.06207% weak.

    The script sends electrons of 1.5, 5 and 20 MeV kinetic energy through a 0.7 m RFCA that adds 12 MeV
    in a static field (FREQ=0), with each edge lens alone and with none, in matrix-method tracking
    (N_KICKS = 0), one-kick tracking and Elegant's own transfer matrix. The lens strengths follow from
    the tracked and matrix maps and are compared with the finite-β hard-edge impulse. The control is the
    cavity without edge lenses, whose map must conserve transverse momentum.
    """

    import numpy as np
    from _elegant import modes, verdict
    from _rfca_lens import LENGTH, ME_MEV, VOLT, momentum, transfer

    KINETIC = (1.5, 5.0, 20.0)  # MeV
    KICKS = (0, 1)
    RTOL = 1e-9  # relative; the maps are exact up to rounding


    def lenses(no_edge, edge1, edge2):
        """Entrance and exit inverse focal lengths (positive = focusing) from three 2x2 maps.

        With the body map B, END1 alone gives B @ [[1, 0], [-k1, 1]], so R21 = B21 - k1 B22;
        END2 alone gives [[1, 0], [-k2, 1]] @ B, so R21 = B21 - k2 B11.
        """
        k1 = (no_edge[1, 0] - edge1[1, 0]) / no_edge[1, 1]
        k2 = (no_edge[1, 0] - edge2[1, 0]) / no_edge[0, 0]
        return float(k1), float(k2)


    defect = False
    for mode in modes():
        for kinetic in KINETIC:
            maps = {
                (e1, e2, n): transfer(mode, kinetic, n, e1, e2)
                for n in KICKS
                for e1, e2 in ((0, 0), (1, 0), (0, 1))
            }
            ref = maps[(0, 0, 0)]
            gamma_i, gamma_f = ref["gamma_in"], ref["gamma_out"]
            beta2_i, beta2_f = 1 - 1 / gamma_i**2, 1 - 1 / gamma_f**2
            gradient = VOLT / (1e6 * ME_MEV) / LENGTH  # gamma' = dgamma/dz, 1/m
            # Hard edge of an axisymmetric field: E_r = -(r/2) dE_z/dz, no magnetic field for FREQ=0.
            # dp_x/dt = q E_r with dt = dz/(beta c) integrates over the edge to
            # dp_x = -q x E_z/(2 beta c), and dx' = dp_x/p_z with p_z = beta gamma m c gives
            # dx' = -x gamma'/(2 beta^2 gamma) at the entrance (E_z switches on) and
            # dx' = +x gamma'/(2 beta^2 gamma) at the exit, each with the local beta and gamma.
            k1_phys = gradient / (2 * beta2_i * gamma_i)
            k2_phys = -gradient / (2 * beta2_f * gamma_f)
            # Control: without edges p_x is conserved, so R21 = 0 and R22 = p_in/p_out.
            p_ratio = momentum(kinetic) / np.sqrt(gamma_f**2 - 1)
            for n in KICKS:
                for source in ("tracked", "matrix"):
                    if source == "matrix" and n:
                        continue  # the matrix does not depend on N_KICKS
                    body = maps[(0, 0, n)][source]
                    label = f"{mode} {kinetic:g} MeV {'matrix_output' if source == 'matrix' else f'tracking N_KICKS={n}'}"
                    if abs(body[1, 0]) > RTOL * gradient or abs(body[1, 1] - p_ratio) > RTOL:
                        raise SystemExit(
                            f"{label}: control fails: the edge-free map does not conserve p_x"
                        )
                    k1, k2 = lenses(body, maps[(1, 0, n)][source], maps[(0, 1, n)][source])
                    e1, e2 = k1 / k1_phys - 1, k2 / k2_phys - 1
                    print(
                        f"{label}: entrance 1/f observed={k1:.12g} expected={k1_phys:.12g} /m "
                        f"(rel {e1:+.6e}, -1/gamma_in^2 = {-1 / gamma_i**2:+.6e}); exit 1/f "
                        f"observed={k2:.12g} expected={k2_phys:.12g} /m (rel {e2:+.6e})"
                    )
                    defect |= abs(e1) > 1e-6 and kinetic < 10
    print(f"control: edge-free maps give R21 = 0 and R22 = p_in/p_out within {RTOL:g}")
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

??? example "Helper: _rfca_lens.py"

    [Download _rfca_lens.py](_rfca_lens.py)

    ```python
    """Transverse transfer of a static-field RFCA: tracked 2x2 map and Elegant's analytic matrix.

    The cavity is RFCA with FREQ=0 and PHASE=90, so every particle gains exactly VOLT and the field is a
    uniform accelerating gradient VOLT/L between hard edges, with no body focusing (the default model).
    Transverse momentum is then conserved inside the cavity, and only the END1_FOCUS and END2_FOCUS
    edge lenses focus. Energies are kinetic, momenta beta*gamma in units of m_e c.
    """

    from __future__ import annotations

    import sys

    import numpy as np
    from _elegant import particles, run

    # SDDS constants.h, as Elegant uses them.
    ME_MEV = 0.51099906
    LENGTH, VOLT = 0.7, 12e6  # m, V: a 17.14 MV/m gradient
    EPS = 1e-7  # m and rad: offsets and slopes that measure the linear map


    def momentum(kinetic_mev: float) -> float:
        gamma = 1 + kinetic_mev / ME_MEV
        return float(np.sqrt(gamma**2 - 1))


    def transfer(mode: str, kinetic_mev: float, kicks: int, end1: int, end2: int) -> dict:
        """Tracked map R (2x2, horizontal) and the matrix_output map M of the same element.

        Particles at x = +-EPS and at x' = +-EPS give the columns of the tracked map by central
        differences; tracking is linear in them, so the differences are exact up to rounding.
        """
        p0 = momentum(kinetic_mev)
        lattice = (
            f"R: RFCA,L={LENGTH!r},VOLT={VOLT!r},FREQ=0,PHASE=90,N_KICKS={kicks},"
            f"END1_FOCUS={end1},END2_FOCUS={end2},CHANGE_P0=0,T_REFERENCE=0\n"
            "BL: LINE=(R)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, "
            "output=out.sdds, default_order=1 &end\n"
            "&matrix_output SDDS_output=matrix.sdds, individual_matrices=1 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {
            "x": np.array([EPS, -EPS, 0.0, 0.0]),
            "xp": np.array([0.0, 0.0, EPS, -EPS]),
            "p": np.full(4, p0),
            "particleID": np.arange(1, 5),
        }
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": beam},
            outputs=("out.sdds", "matrix.sdds"),
        )
        out = particles(result["out.sdds"])
        if len(out["x"]) != 4:
            sys.exit(f"{mode}: particles were lost")
        x, xp = out["x"], out["xp"]
        tracked = np.array(
            [
                [(x[0] - x[1]) / (2 * EPS), (x[2] - x[3]) / (2 * EPS)],
                [(xp[0] - xp[1]) / (2 * EPS), (xp[2] - xp[3]) / (2 * EPS)],
            ]
        )
        matrix = result["matrix.sdds"]
        names = [str(n) for n in matrix[0, "ElementName"]]
        row = names.index("R")
        analytic = np.array([[float(matrix[0, f"R{i}{j}"][row]) for j in (1, 2)] for i in (1, 2)])
        gamma_in = 1 + kinetic_mev / ME_MEV
        gamma_out = float(np.hypot(out["p"][0], 1))
        if abs(gamma_out - gamma_in - VOLT / (1e6 * ME_MEV)) > 1e-12 * gamma_out:
            sys.exit(f"{mode}: the energy gain is not VOLT")
        return {"tracked": tracked, "matrix": analytic, "gamma_in": gamma_in, "gamma_out": gamma_out}

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

