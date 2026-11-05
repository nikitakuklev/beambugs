---
title: "OPT-23: moments_output mode emittances normalize the (s, δ) pair with the transverse sign"
hide:
  - toc
---

# OPT-23 — moments_output mode emittances normalize the (s, δ) pair with the transverse sign

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`NormalizeEigenvectors`, src/moments.c:1123–1136](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L1123-L1136)<br>[`computeNaturalEmittances`, src/moments.c:1054–1055](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L1054-L1055)<br>[`e1`, `e2`, `e3`, emittances of modes 1 to 3, src/moments.c:111–113](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L111-L113)<br>[changing the time sign for symplecticity, src/coupled_twiss.c:381–385](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/coupled_twiss.c#L381-L385)<br>[canonical variables are (x, q_x, y, q_y, −s, δ), doc/elegant.tex:987](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L987)<br>[the equilibrium moments, doc/elegant.tex:5613](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5613) |

Elegant's `moments_output` normalizes its normal modes with the same symplectic sign for all coordinate pairs, although the canonical longitudinal pair is (−s, δ), where s is path length and δ is relative momentum deviation. With transverse–longitudinal mixing, this gives wrong mode emittances; without intrabeam scattering, the covariance still solves the exported linear equilibrium equation. In a 2.4 GeV, 70 m ring with a 1.5 MV cavity, the covariance's paraxial eigen-emittances are 5.448 and 51.45 μm, but Elegant reports 5.418 and 51.17 μm (−0.5564% and −0.5555%). Changing the ring's starting point gives a −0.7020% error in the first mode.

??? example "Reproducer: OPT-23.py"

    [Download OPT-23.py](OPT-23.py)

    ```python
    """OPT-23: moments_output mode emittances normalize the (s, δ) pair with the transverse sign

    Elegant's `moments_output` normalizes its normal modes with the same symplectic sign for all
    coordinate pairs, although the canonical longitudinal pair is (−s, δ), where s is path length
    and δ is relative momentum deviation. With transverse–longitudinal mixing, this gives wrong mode
    emittances; without intrabeam scattering, the covariance still solves the exported linear
    equilibrium equation. In a 2.4 GeV, 70 m ring with a 1.5 MV cavity, the covariance's paraxial
    eigen-emittances are 5.448 and 51.45 μm, but Elegant reports 5.418 and 51.17 μm (−0.5564% and
    −0.5555%). Changing the ring's starting point gives a −0.7020% error in the first mode.

    The script runs serial Elegant `moments_output` (matched, equilibrium, radiation) on a 2.4 GeV,
    70 m ring of 20 FODO cells with radiating CSBENDs and a 1.5 MV RF cavity, from two starting points
    of the same ring, with `matrix_output` writing the one-turn matrix M and diffusion matrix D. It
    solves Σ = MΣMᵀ + D independently with SciPy, checks it against Elegant's equilibrium covariance,
    and takes the expected mode emittances as the moduli of the eigenvalues of ΣS with
    S = diag(J, J, −J), J = [[0, 1], [−1, 0]]: the minus sign makes (−s, δ) the canonical pair. It
    compares these with the reported e1 and e3 (e2 is zero in this flat ring), and recomputes the
    mode emittances from the eigenvectors of M normalized with diag(J, J, J), as the code does, and with
    diag(J, J, −J).
    """

    import sys

    import numpy as np
    from _elegant import run, verdict
    from scipy.linalg import block_diag, solve_discrete_lyapunov

    N_CELLS = 20
    LATTICE = f"""F: QUAD,L=0.3,K1=0.9
    G: QUAD,L=0.3,K1=-0.85
    D: DRIF,L=0.65
    E: DRIF,L=0.4
    B: CSBEND,L=1.2,ANGLE={2 * np.pi / N_CELLS!r},N_SLICES=40,EXPAND_HAMILTONIAN=0,SYNCH_RAD=1,ISR=1
    C: LINE=(F,D,B,E,G,D)
    RF: RFCA,VOLT=1500000,FREQ={80 * 299792458 / 70!r},PHASE=170,CHANGE_P0=0
    A: LINE=(RF,{N_CELLS}*C)
    Z: LINE=(F,D,B,E,G,D,RF,{N_CELLS - 1}*C)
    """
    J = np.array([[0.0, 1.0], [-1.0, 0.0]])
    RTOL = 1e-5  # the expected values are exact eigen-emittances of a covariance matched to ~1e-13


    def mode_emittances(m, sigma, s):
        """Emittance of each oscillation mode from the left eigenvectors v of the one-turn matrix:
        (v^H sigma v) / |Im(v^H s v)|, the normalization moments_output uses with its own s."""
        values, vectors = np.linalg.eig(m.T)
        found = []
        for k in np.where(values.imag > 0)[0]:
            v = vectors[:, k]
            found.append((v.conj() @ sigma @ v).real / abs((v.conj() @ s @ v).imag))
        return np.sort(found)


    def eigen_emittances(sigma, s):
        """Mode emittances: moduli of the eigenvalues +-i eps_k of sigma @ s, in increasing order."""
        values = np.linalg.eigvals(sigma @ s)
        return np.sort(np.abs(values.imag))[::2]


    defect = False
    for line in ("A", "Z"):
        commands = (
            f"&run_setup lattice=ring.lte, use_beamline={line}, p_central_mev=2400, default_order=2 &end\n"
            "&run_control &end\n"
            "&twiss_output filename=twiss.sdds, matched=1, radiation_integrals=1, output_at_each_step=1 &end\n"
            "&rf_setup harmonic=80, total_voltage=1500000, set_for_each_step=1 &end\n"
            "&closed_orbit output=orbit.sdds, fixed_length=2 &end\n"
            "&moments_output filename=moments.sdds, matrix_output=matrices.sdds, matched=1,"
            " equilibrium=1, radiation=1, n_slices=20, output_at_each_step=1 &end\n"
            "&bunched_beam &end\n&track &end\n"
        )
        result = run(
            {"ring.lte": LATTICE, "run.ele": commands}, outputs=("moments.sdds", "matrices.sdds")
        )
        moments, matrices = result["moments.sdds"], result["matrices.sdds"]
        sigma = np.empty((6, 6))
        for i in range(6):
            for j in range(i, 6):
                value = moments[0, f"s{i + 1}"][0] ** 2 if i == j else moments[0, f"s{i + 1}{j + 1}"][0]
                sigma[i, j] = sigma[j, i] = value
        m = np.array([[matrices[0, f"R{i}{j}"][0] for j in range(1, 7)] for i in range(1, 7)])
        d = np.array([[matrices[0, f"D{i}{j}"][0] for j in range(1, 7)] for i in range(1, 7)])
        solved = solve_discrete_lyapunov(m, d)
        agreement = np.abs(solved - sigma).max() / np.abs(solved).max()
        print(
            f"start {line} control: SciPy equilibrium covariance vs Elegant's, max relative deviation {agreement:.2e}"
        )
        if agreement > 1e-9:
            sys.exit(
                "Elegant's equilibrium covariance disagrees with the Lyapunov solution: invalid comparison."
            )
        canonical = eigen_emittances(solved, block_diag(J, J, -J))
        same_sign = mode_emittances(m, solved, block_diag(J, J, J))
        flipped = mode_emittances(m, solved, block_diag(J, J, -J))
        # Ordered by size, the modes are (vertical ~0, horizontal, longitudinal) here.
        expected = [canonical[1], canonical[2]]
        observed = [float(moments["e1"][0]), float(moments["e3"][0])]
        print(
            f"start {line} e2 (m): reported={float(moments['e2'][0]):.3g} expected={canonical[0]:.3g}"
        )
        print(
            f"start {line} e1, e3 (m): observed=[{observed[0]:.10g}, {observed[1]:.10g}]"
            f" expected=[{expected[0]:.10g}, {expected[1]:.10g}]"
        )
        print(
            f"start {line} relative error e1, e3: {observed[0] / expected[0] - 1:+.4%}, {observed[1] / expected[1] - 1:+.4%}"
        )
        print(
            f"start {line} eigenvector normalization with diag(J, J, J): e1, e3 = [{same_sign[1]:.10g},"
            f" {same_sign[2]:.10g}]; with diag(J, J, -J): [{flipped[1]:.10g}, {flipped[2]:.10g}]"
        )
        defect |= not np.allclose(observed, expected, rtol=RTOL, atol=0)
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
| Confirmed (e1 and e3; e2 is zero in the confirming ring) | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

