---
title: "COL-50: CSRCSBEND BIN_ONCE wake output corrupts retained histograms and accesses past their ends"
hide:
  - toc
---

# COL-50 — CSRCSBEND BIN_ONCE wake output corrupts retained histograms and accesses past their ends

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant master (`0355dd6`) |
| Source | [`track_through_csbendCSR`, csbend.c:3686–3695](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3686-L3695)<br>[`track_through_csbendCSR`, csbend.c:3712–3716](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3712-L3716)<br>[`track_through_csbendCSR`, csbend.c:2973–2979](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2973-L2979)<br>[`csrcsbend_param`, track_data.c:2058–2069](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2058-L2069)<br>[`track_through_csbendCSR`, csbend.c:3442–3449](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3442-L3449)<br>[`track_through_csbendCSR`, csbend.c:3613–3617](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3613-L3617) |

With `BIN_ONCE=1` and a wake `OUTPUT_FILE`, CSRCSBEND leaves its retained histogram scaled after writing each page and attempts to restore elements one past both histogram arrays. Later output pages should retain the initial density and derivative, but for a 100.0 pC bunch represented by 4096 particles, the second page integrates to 7.319×10⁻⁴ pC and the fourth to 3.921×10⁻¹⁴ pC. The derivative grows by a factor of 5.321×10⁷ per additional page instead of remaining unchanged, and Valgrind confirms invalid reads and writes. The tested tracking kicks are unchanged by wake output.

??? example "Reproducer: COL-50.py"

    [Download COL-50.py](COL-50.py)

    ```python
    """COL-50: CSRCSBEND BIN_ONCE wake output corrupts retained histograms and accesses past their ends

    With `BIN_ONCE=1` and a wake `OUTPUT_FILE`, CSRCSBEND leaves its retained histogram scaled after writing each page and attempts to restore elements one past both histogram arrays. Later output pages should retain the initial density and derivative, but for a 100.0 pC bunch represented by 4096 particles, the second page integrates to 7.319×10⁻⁴ pC and the fourth to 3.921×10⁻¹⁴ pC. The derivative grows by a factor of 5.321×10⁷ per additional page instead of remaining unchanged, and Valgrind confirms invalid reads and writes. The tested tracking kicks are unchanged by wake output.

    The script tracks deterministic Gaussian quantiles through a four-slice bend and compares all-page, last-only, interval-two and disabled wake output, with a rebinned control. It integrates the reported current, checks the repeated scaling against the macroparticle charge and bin width, and compares final coordinates. It also runs the retained and rebinned cases under Valgrind, which must be on PATH with Elegant and RPN_DEFNS set.
    """

    import re

    import numpy as np
    from _csr import C, page, track, valgrind
    from _elegant import particles, verdict
    from scipy.special import ndtri

    N, CHARGE, P0 = 4096, 100e-12, 5000.0
    RTOL = 3e-14
    BEAM = {
        "t": 1e-12 * ndtri((np.arange(N) + 0.5) / N),
        "p": np.full(N, P0),
        "particleID": np.arange(1, N + 1),
    }


    def elements(once: int, last: int = 0, output: bool = True, interval: int = 1) -> str:
        wake = (
            f',OUTPUT_FILE="wake.sdds",OUTPUT_LAST_WAKE_ONLY={last},OUTPUT_INTERVAL={interval}'
            if output
            else ""
        )
        return (
            f"Q: CHARGE,TOTAL={CHARGE:.17g}\n"
            "B: CSRCSBEND,L=0.8,ANGLE=0.08,N_SLICES=4,BINS=64,"
            f"BIN_ONCE={once},STEADY_STATE=1,SG_HALFWIDTH=1,SG_ORDER=1,"
            f"SYNCH_RAD=0,ISR=0{wake}\n"
        )


    def memory_check(label: str, lattice: str) -> tuple[int, int]:
        status, log = valgrind(lattice, "Q,B", BEAM, P0)
        if status:
            raise RuntimeError(f"Valgrind/Elegant exited {status}:\n{log[-4000:]}")
        match = re.search(r"ERROR SUMMARY: (\d+) errors from (\d+) contexts", log)
        if match is None:
            raise RuntimeError("Valgrind did not produce its error summary")
        errors, contexts = map(int, match.groups())
        if errors and label == "BIN_ONCE=1":
            # Require the reported accesses to be precisely beyond the two 64-double arrays.
            # Function names and allocation size are stable if a later build moves line numbers.
            accesses = re.findall(
                r"Invalid (?:read|write) of size 8\n[^\n]*track_through_csbendCSR[^\n]*"
                r"(?:(?!Invalid |ERROR SUMMARY:)[\s\S])*?0 bytes after a block of size 512 alloc'd",
                log,
            )
            if len(accesses) != 4 or (errors, contexts) != (16, 4):
                raise RuntimeError(f"Unexpected memory errors:\n{log}")
        print(f"serial Valgrind {label}: errors={errors} contexts={contexts}; expected=0 errors")
        return errors, contexts


    def main() -> None:
        results = {}
        for name, once, last, output, interval in (
            ("all", 1, 0, True, 1),
            ("last", 1, 1, True, 1),
            ("off", 1, 0, False, 1),
            ("rebin", 0, 0, True, 1),
            ("interval2", 1, 0, True, 2),
        ):
            result = track(
                elements(once, last, output, interval),
                "Q,B",
                BEAM,
                P0,
                outputs=("out.sdds", "wake.sdds") if output else ("out.sdds",),
            )
            final = particles(result["out.sdds"])
            if not np.array_equal(final["particleID"], BEAM["particleID"]):
                raise RuntimeError("Unexpected particle loss")
            values: dict = {"final": np.column_stack([final[k] for k in ("x", "xp", "y", "yp", "t", "p")])}
            if output:
                wake = result["wake.sdds"]
                pages = [page(wake, i) for i in range(wake.n_pages)]
                dct = pages[0][1]["BinSize"]
                density = np.array([columns["LinearDensity"] for columns, _ in pages])
                derivative = np.array([columns["LinearDensityDeriv"] for columns, _ in pages])
                # LinearDensity is current (C/s); each bin spans dt = dct/c, so sum(I dt) = Q.
                charges = np.sum(density, axis=1) * dct / C
                values.update(density=density, derivative=derivative, dct=dct, charge=charges)
                print(
                    f"serial {name}: integrated_charge_C={charges.tolist()}; expected={CHARGE:.16g} per page"
                )
            results[name] = values

        all_pages, last = results["all"], results["last"]
        if (
            len(all_pages["charge"]) != 4
            or len(last["charge"]) != 1
            or len(results["interval2"]["charge"]) != 2
        ):
            raise RuntimeError("Unexpected number of wake output pages")
        qfactor = CHARGE / N * C
        dfactor = CHARGE / N * C**2 / all_pages["dct"]
        print(
            f"serial extra scaling per page: density={qfactor:.16g} derivative={dfactor:.16g}; expected=1 for both"
        )
        np.testing.assert_allclose(all_pages["charge"][0], CHARGE, rtol=RTOL, atol=0)
        # Rebinning evolves the distribution and filtering introduces a small integration residual.
        # It must conserve charge to 3e-8 relative, as opposed to losing five orders per output.
        np.testing.assert_allclose(results["rebin"]["charge"], CHARGE, rtol=3e-8, atol=0)
        corrupted = []
        for key, factor in (("density", qfactor), ("derivative", dfactor)):
            data = all_pages[key]
            reference = np.broadcast_to(data[0], data.shape)
            changed = not np.allclose(data, reference, rtol=RTOL, atol=0)
            corrupted.append(changed)
            # A missing restore leaves physical-unit values in internal arrays. Each subsequent
            # output multiplies them again by q_macro*c (density) or q_macro*c^2/dct (derivative).
            expected = data[0][None, :] * (factor if changed else 1.0) ** np.arange(4)[:, None]
            np.testing.assert_allclose(data, expected, rtol=RTOL, atol=0)
            np.testing.assert_allclose(last[key][0], data[0], rtol=RTOL, atol=0)
            interval_expected = data[0][None, :] * (factor if changed else 1.0) ** np.arange(2)[:, None]
            np.testing.assert_allclose(results["interval2"][key], interval_expected, rtol=RTOL, atol=0)
            print(
                f"serial {key}: retained pages unchanged={not changed}; expected=True; last-only matches first page=True"
            )
        print(
            f"serial peak derivative_C_per_s2={np.max(abs(all_pages['derivative']), axis=1).tolist()}; expected={np.max(abs(all_pages['derivative'][0])):.16g} per page"
        )
        for name in ("last", "off", "interval2"):
            difference = np.max(abs(results[name]["final"] - all_pages["final"]))
            print(
                f"serial final coordinates all vs {name}: max_abs_difference={difference:.16g}; expected=0"
            )
            if not np.array_equal(results[name]["final"], all_pages["final"]):
                raise RuntimeError("Wake output changed tracking; the unchanged-kick control failed")
        errors, _ = memory_check("BIN_ONCE=1", elements(1))
        control_errors, _ = memory_check("BIN_ONCE=0", elements(0))
        if control_errors:
            raise RuntimeError("Rebinning control has memory errors")
        verdict(any(corrupted) or errors > 0)


    if __name__ == "__main__":
        main()

    ```

??? example "Helper: _csr.py"

    [Download _csr.py](_csr.py)

    ```python
    """Shared setup for the CSRCSBEND and CSRDRIFT reproducers: Elegant's physical constants,
    deterministic electron beams, one-run tracking with the wake files Elegant exports, the analytic
    steady-state CSR wake of a Gaussian bunch, and a serial Valgrind runner for the memory defects.

    Lengths are in m, momenta p in units of m_e c, and the longitudinal coordinate z = c t of a
    particle grows toward the tail of the bunch, as in Elegant's wake output column `s`.
    """

    from __future__ import annotations

    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import numpy as np
    from _elegant import RunFailure, particles, run, write_beam
    from scipy.integrate import quad
    from scipy.special import ndtri

    # SDDS constants.h, the values Elegant computes with: the comparisons below are then not limited
    # by revisions of the CODATA constants.
    C = 2.99792458e8
    E_CHARGE = 1.60217733e-19
    M_E = 9.1093897e-31
    EPSILON_0 = 1e7 / (4 * np.pi * C**2)
    R_E = E_CHARGE**2 / (4 * np.pi * EPSILON_0 * M_E * C**2)
    TIMEOUT = 600  # seconds per Valgrind run


    def gaussian_beam(count: int, sigma_ct: float, p0: float, chirp: float = 0.0, x0: float = 0.0):
        """Gaussian quantiles in z = c t with rms sigma_ct, all at p0 (1 + chirp z) and x = x0.

        Quantiles (i + 1/2)/count place the particles deterministically and symmetrically about z = 0.
        """
        z = sigma_ct * ndtri((np.arange(count) + 0.5) / count) if count > 1 else np.zeros(1)
        return {
            "x": np.full(count, x0),
            "t": z / C,
            "p": p0 * (1 + chirp * z),
            "particleID": np.arange(1, count + 1),
        }


    def commands(p0: float, steps: int = 1, passes: int = 1, seed: int = 0) -> str:
        """Command file: track beam.sdds through BL, final coordinates in out.sdds."""
        random = f",random_number_seed={seed}" if seed else ""
        return (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={p0!r},output=out.sdds{random} &end\n"
            f"&run_control n_steps={steps},n_passes={passes} &end\n"
            "&sdds_beam input=beam.sdds,reuse_bunch=1 &end\n"
            "&track &end\n"
        )


    def track(elements: str, line: str, beam, p0: float, outputs=("out.sdds",), **kw):
        """Track `beam` through BL: LINE=(line) defined after `elements`; returns run()'s dict."""
        lattice = elements.rstrip("\n") + f"\nBL: LINE=({line})\n"
        return run(
            {"lattice.lte": lattice, "run.ele": commands(p0, **kw)},
            beams={"beam.sdds": beam},
            outputs=outputs,
        )


    def status(elements: str, line: str, beam, p0: float, **kw) -> int:
        """Exit status of a plain serial run (0 on success): the memory defects end in a signal."""
        try:
            track(elements, line, beam, p0, outputs=(), **kw)
        except RunFailure as failure:
            return failure.status
        return 0


    def page(sdds, index: int = 0) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Columns and parameters of one page of an SDDS file."""
        columns = {c.name: np.asarray(sdds[index, c.name]) for c in sdds.columns}
        parameters = {p.name: p.data[index] for p in sdds.parameters}
        return columns, parameters


    def final(result: dict) -> dict[str, np.ndarray]:
        """Final particle coordinates of a run, ordered by particle ID."""
        return particles(result["out.sdds"])


    def steady_state_wake(z: np.ndarray, sigma: float, k: float) -> np.ndarray:
        """Steady-state CSR energy change of a Gaussian bunch, per particle, in units of m_e c^2.

        Saldin, Schneidmiller and Yurkov (NIM A 398, 1997) and Stupakov and Emma (LCLS-TN-01-12, 2001):
            Delta gamma(z) = k int_0^inf lambda'(z + u) u^(-1/3) du,
        lambda the line density normalized to 1 and k = 2 N r_e L / (3 rho^2)^(1/3) for N electrons
        in a bend of length L and radius rho. With z = a sigma and u = sigma t^3 the integrable
        endpoint singularity disappears:
            Delta gamma = k sigma^(-4/3) int_0^inf 3 t lambda_1'(a + t^3) dt,
        lambda_1(a) = exp(-a^2/2)/sqrt(2 pi), lambda_1'(a) = -a lambda_1(a); the integrand vanishes
        beyond a + t^3 = 12.
        """

        def one(a: float) -> float:
            upper = max(0.0, 12.0 - a) ** (1 / 3)
            integrand = lambda t: -3 * t * (a + t**3) * np.exp(-0.5 * (a + t**3) ** 2)
            return quad(integrand, 0, upper, epsabs=1e-13, epsrel=1e-12, limit=200)[0]

        values = np.array([one(a) for a in np.asarray(z) / sigma])
        return k * sigma ** (-4 / 3) * values / np.sqrt(2 * np.pi)


    def valgrind(elements: str, line: str, beam, p0: float, **kw) -> tuple[int, str]:
        """Serial Elegant under `valgrind --track-origins=yes`: (exit status, Valgrind and run log)."""
        if shutil.which("valgrind") is None or shutil.which("elegant") is None:
            sys.exit("valgrind and elegant must be on PATH.")
        if "RPN_DEFNS" not in os.environ:
            sys.exit("Set RPN_DEFNS to the defns.rpn file of SDDS (see the elegant installation).")
        env = dict(os.environ, OMP_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="elegant-repro-") as tmp:
            work = Path(tmp)
            (work / "lattice.lte").write_text(elements.rstrip("\n") + f"\nBL: LINE=({line})\n")
            (work / "run.ele").write_text(commands(p0, **kw))
            write_beam(work / "beam.sdds", beam)
            try:
                done = subprocess.run(
                    ["valgrind", "--error-limit=no", "--track-origins=yes", "elegant", "run.ele"],
                    cwd=work,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                sys.exit(f"valgrind elegant exceeded {TIMEOUT} s")
        return done.returncode, done.stdout + done.stderr

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
| Found (source observation of the missing restoration loop; no execution for this claim) | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

