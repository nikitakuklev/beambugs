---
title: "OPT-31: Resonance driving terms omit fractional strength errors applied by tracking"
hide:
  - toc
---

# OPT-31 — Resonance driving terms omit fractional strength errors applied by tracking

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`computeDrivingTerms`, twiss.cc:5229–5270](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5229-L5270)<br>[`quad_param`, track_data.c:376–376](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L376-L376)<br>[`sext_param`, track_data.c:485–485](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L485-L485)<br>[`octu_param`, track_data.c:499–499](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L499-L499)<br>[`sbend_param`, track_data.c:420–422](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L420-L422)<br>[`kquse_param`, track_data.c:3420–3421](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3420-L3421)<br>[`multipole_tracking2`, multipole.c:791–795](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L791-L795)<br>[`multipole_tracking2`, multipole.c:858–862](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L858-L862)<br>[`multipole_tracking2`, multipole.c:930–934](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L930-L934)<br>[`multipole_tracking2`, multipole.c:979–1005](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L979-L1005)<br>[`compute_matrix`, compute_matrices.c:1482–1509](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L1482-L1509)<br>[`track_through_csbend`, csbend.c:1077–1081](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1077-L1081)<br>[`track_through_ccbend_impl`, ccbend.c:349–358](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L349-L358)<br>[`track_through_csbendCSR`, csbend.c:2846–2853](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2846-L2853)<br>[`computeDrivingTerms`, twiss.cc:5275–5287](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5275-L5287)<br>[`computeDrivingTerms`, twiss.cc:5329–5355](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5329-L5355)<br>[`computeDrivingTerms`, twiss.cc:5410–5425](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5410-L5425) |

With `compute_driving_terms` enabled, `twiss_output` uses nominal multipole strengths even when tracking applies fractional strength errors. Equivalent applied fields must give identical driving terms, as independent linear-optics formulas and tracked trajectories establish. For a 0.1200 m sextupole of strength K₂ = 4.000 m⁻³ and fractional strength error 0.001000, the real part of the horizontal sextupole driving term h₂₁₀₀₀ is 0.6324 instead of 0.6330, an underestimate of 0.09990%; a strength error of 0.02000 gives a 1.961% underestimate in isolated sextupole and octupole coefficients. Tracking agrees exactly between the equivalent-field cases.

??? example "Reproducer: OPT-31.py"

    [Download OPT-31.py](OPT-31.py)

    ```python
    """OPT-31: Resonance driving terms omit fractional strength errors applied by tracking

    With `compute_driving_terms` enabled, `twiss_output` uses nominal multipole strengths even when tracking applies fractional strength errors. Equivalent applied fields must give identical driving terms, as independent linear-optics formulas and tracked trajectories establish. For a 0.1200 m sextupole of strength K₂ = 4.000 m⁻³ and fractional strength error 0.001000, the real part of the horizontal sextupole driving term h₂₁₀₀₀ is 0.6324 instead of 0.6330, an underestimate of 0.09990%; a strength error of 0.02000 gives a 1.961% underestimate in isolated sextupole and octupole coefficients. Tracking agrees exactly between the equivalent-field cases.

    The script runs serial Elegant on a matched periodic cell with one test magnet and three fixed particles, comparing fractional-error and explicitly scaled-strength decks. Exact drift/quadrupole matrices provide an independent sextupole and octupole oracle, including zero-error and zero-field controls. Bend pairs keep their dipole field and geometry fixed; CSRCSBEND is checked with its actual endpoint optics because it lacks separate dipole compensation.
    """

    import numpy as np
    from _elegant import particles, run, verdict


    def case(magnet):
        lattice = (
            "QF: QUAD,L=0.2,K1=1.4\nQD: QUAD,L=0.2,K1=-1.4\n"
            f"D: DRIF,L=0.37\nM: {magnet}\nE: DRIF,L=0.86\n"
            "BL: LINE=(QF,D,M,D,QD,E)\n"
        )
        command = """&run_setup lattice="line.lte",use_beamline=BL,
     p_central=1200,default_order=2,output="final.sdds" &end
    &run_control n_steps=1,n_passes=1 &end
    &twiss_output filename="twiss.sdds",matched=1,compute_driving_terms=1,
     leading_order_driving_terms_only=1 &end
    &sdds_beam input="input.sdds" &end
    &track &end
    """
        beam = {
            "x": [0.0011, -0.0008, 0],
            "xp": [0.00013, -0.00011, 0],
            "y": [-0.0006, 0.0009, 0],
            "yp": [0.00021, -0.00017, 0],
            "t": [0, 2e-12, -2e-12],
            "p": [1200, 1200.12, 1200],
            "particleID": np.arange(1, 4),
        }
        result = run(
            {"line.lte": lattice, "run.ele": command},
            beams={"input.sdds": beam},
            outputs=("twiss.sdds", "final.sdds"),
        )
        twiss = result["twiss.sdds"]
        columns = {c.name: np.asarray(c.data[0]) for c in twiss.columns}
        parameters = {p.name: p.data[0] for p in twiss.parameters}
        final = particles(result["final.sdds"])
        assert len(final["p"]) == 3, "Unexpected particle loss"
        return columns, parameters, final


    def h(parameters, term):
        value = complex(parameters["Re" + term], parameters["Im" + term])
        assert np.isfinite(value), "Nonfinite driving term"
        return value


    def same_tracking(a, b):
        # Particle IDs, rather than row order, identify trajectories.
        gap = max(float(np.max(np.abs(a[k] - b[k]))) for k in ("x", "xp", "y", "yp", "t", "p"))
        assert gap == 0, f"Equivalent-field tracking control failed: {gap}"
        return gap


    def analytic_optics():
        def matrix(length, k):
            if k == 0:
                return np.array([[1, length], [0, 1]])
            r = np.sqrt(abs(k))
            c = np.cos(r * length) if k > 0 else np.cosh(r * length)
            s = np.sin(r * length) / r if k > 0 else np.sinh(r * length) / r
            return np.array([[c, s], [-k * s, c]])

        # The centered sextupole/octupole is a drift at linear order. The periodic
        # covariance satisfies M Sigma M^T = Sigma; propagate it to both magnet ends.
        matrices = [
            matrix(l, k)
            for l, k in [(0.2, 1.4), (0.37, 0), (0.12, 0), (0.37, 0), (0.2, -1.4), (0.86, 0)]
        ]
        turn = np.eye(2)
        for m in matrices:
            turn = m @ turn
        mu = np.arccos(np.trace(turn) / 2)
        beta0 = turn[0, 1] / np.sin(mu)
        alpha0 = (turn[0, 0] - turn[1, 1]) / (2 * np.sin(mu))
        sigma0 = np.array([[beta0, -alpha0], [-alpha0, (1 + alpha0**2) / beta0]])
        cumulative = np.eye(2)
        ends = []
        for m in matrices[:3]:
            cumulative = m @ cumulative
            sigma = cumulative @ sigma0 @ cumulative.T
            psi = np.arctan2(cumulative[0, 1], cumulative[0, 0] * beta0 - cumulative[0, 1] * alpha0)
            ends.append((sigma[0, 0], psi))
        return np.mean(ends[1:], axis=0)


    def main():
        defects = []
        for typ, order, strength, field, extra in [
            ("QUAD", 1, 0.08, "FSE", ""),
            ("KQUAD", 1, 0.08, "FSE", ",N_SLICES=12"),
            ("SEXT", 2, 4.0, "FSE", ""),
            ("KSEXT", 2, 4.0, "FSE", ",N_SLICES=12"),
            ("OCTU", 3, 80.0, "FSE", ""),
            ("KOCT", 3, 80.0, "FSE", ",N_SLICES=12"),
            ("KQUSE", 1, 0.08, "FSE1", ",N_SLICES=12"),
            ("KQUSE", 2, 4.0, "FSE2", ",N_SLICES=12"),
            ("SBEN", 2, 4.0, "FSE", ",ANGLE=.006,FSE_DIPOLE=-0.02"),
            ("CSBEND", 2, 4.0, "FSE", ",ANGLE=.006,FSE_DIPOLE=-0.02,N_SLICES=12"),
            (
                "CCBEND",
                2,
                4.0,
                "FSE",
                ",ANGLE=.006,FSE_DIPOLE=-0.02,N_SLICES=12,OPTIMIZE_FSE=0,OPTIMIZE_DX=0",
            ),
        ]:
            frac = 0.02
            ca, pa, fa = case(f"{typ},L=.12,K{order}={strength},{field}={frac}{extra}")
            # FSE_DIPOLE=-FSE preserves the dipole field without changing ANGLE or
            # survey geometry. Only the explicit-strength control removes compensation.
            cb, pb, fb = case(
                f"{typ},L=.12,K{order}={strength * (1 + frac):.17g}"
                + extra.replace(",FSE_DIPOLE=-0.02", "")
            )
            gap = same_tracking(fa, fb)
            for key in ("betax", "betay", "psix", "psiy"):
                np.testing.assert_allclose(ca[key], cb[key], rtol=0, atol=1e-10)
            term = {1: "h11001", 2: "h21000", 3: "h22000"}[order]
            observed, expected = h(pa, term), h(pb, term)
            defects.append(not np.isclose(observed, expected, rtol=1e-11, atol=1e-13))
            missing_factor = (
                "not isolated" if order == 1 else str(abs(observed / expected - 1 / (1 + frac)) < 1e-11)
            )
            print(
                f"serial {typ} K{order} {field}=0.02: {term} observed={observed:.15g} "
                f"expected={expected:.15g} (identical applied field); track_max={gap:g}; "
                f"missing-factor signature={missing_factor}"
            )

        beta, psi = analytic_optics()
        # Expanding K2(x^3-3xy^2)/6 and K3(x^4-6x^2y^2+y^4)/24 in the
        # diagnostic's complex normalization gives h21000=K2 L beta^(3/2)e^(i psi)/16
        # and h22000=K3 L beta^2/64. Use the same mean-endpoint convention as the sum.
        base = 4 * 0.12 * beta**1.5 * np.exp(1j * psi) / 16
        _, p, _ = case("KSEXT,L=.12,K2=4,N_SLICES=12")
        np.testing.assert_allclose(h(p, "h21000"), base, rtol=2e-13, atol=0)
        print(
            f"serial analytic control: beta={beta:.15g} m phase={psi:.15g} rad; "
            f"h21000 observed={h(p, 'h21000'):.15g} expected={base:.15g} (exact linear matrices)"
        )
        _, p, _ = case("KOCT,L=.12,K3=80,FSE=.02,N_SLICES=12")
        expected_oct = 80 * 0.12 * 1.02 * beta**2 / 64
        defects.append(not np.isclose(h(p, "h22000"), expected_oct, rtol=2e-13, atol=0))
        print(
            f"serial analytic octupole: h22000 observed={h(p, 'h22000'):.15g} "
            f"expected={expected_oct:.15g} (effective K3 L beta^2/64)"
        )
        for frac in (0.001, -1.0):
            _, p, final = case(f"KSEXT,L=.12,K2=4,FSE={frac},N_SLICES=12")
            _, control, other = case(f"KSEXT,L=.12,K2={4 * (1 + frac):.17g},N_SLICES=12")
            gap = same_tracking(final, other)
            expected = (1 + frac) * base
            np.testing.assert_allclose(h(control, "h21000"), expected, rtol=2e-13, atol=1e-14)
            defects.append(not np.isclose(h(p, "h21000"), expected, rtol=2e-13, atol=1e-14))
            print(
                f"serial KSEXT FSE={frac}: h21000 observed={h(p, 'h21000'):.15g} "
                f"expected={expected:.15g} (effective K2 analytic formula); track_max={gap:g}"
            )
        for frac in (0.0, 0.001, 0.02):
            c, p, _ = case(
                f"CSRCSBEND,L=.12,ANGLE=.006,K2=4,FSE={frac},N_SLICES=12,"
                "BINS=32,CSR=0,ISR=0,SYNCH_RAD=0"
            )
            (i,) = np.flatnonzero(c["ElementName"] == "M")
            b, phase = [(c[k][i - 1] + c[k][i]) / 2 for k in ("betax", "psix")]
            expected = 4 * 0.12 * (1 + frac) * b**1.5 * np.exp(1j * phase) / 16
            if frac == 0:
                np.testing.assert_allclose(h(p, "h21000"), expected, rtol=2e-13, atol=0)
            else:
                defects.append(not np.isclose(h(p, "h21000"), expected, rtol=2e-13, atol=0))
            print(
                f"serial CSRCSBEND FSE={frac}: h21000 observed={h(p, 'h21000'):.15g} "
                f"expected={expected:.15g} (effective K2 with native endpoint optics)"
            )
        verdict(all(defects))


    if __name__ == "__main__":
        main()

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

