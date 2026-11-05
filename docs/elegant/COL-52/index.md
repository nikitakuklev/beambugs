---
title: "COL-52: CSRCSBEND BIN_ONCE wake output overwrites retained kick components"
hide:
  - toc
---

# COL-52 — CSRCSBEND BIN_ONCE wake output overwrites retained kick components

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant master (`0355dd6`) |
| Source | [`csrcsbend_param`, track_data.c:2060–2069](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2060-L2069)<br>[`track_through_csbendCSR`, csbend.c:2973–2979](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2973-L2979)<br>[`track_through_csbendCSR`, csbend.c:3545–3551](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3545-L3551)<br>[`track_through_csbendCSR`, csbend.c:3697–3701](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3697-L3701)<br>[`track_through_csbendCSR`, csbend.c:3717–3725](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3717-L3725)<br>[`track_through_csbendCSR`, csbend.c:3442–3449](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3442-L3449)<br>[`track_through_csbendCSR`, csbend.c:3613–3617](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3613-L3617) |

With `IGF=0`, `BIN_ONCE=1` and repeated wake output, CSRCSBEND overwrites its retained kick-component arrays while preparing the first output page. Later `DeltaGammaT1` and `DeltaGammaT2` columns contain the preceding page's longitudinal positions and wake per metre, rather than components whose sum equals the reported change in Lorentz factor Δγ. For a 250.0 pC, 2.000 ps rms bunch in a 1.200 m steady-state bend, the peak component-sum residual is 0.007825 instead of zero, or 227.5% of peak |Δγ|; the transient case gives 219.0%. First-page, last-only and rebinning controls satisfy the sum identity, and changing the output schedule leaves the total wake and tracked coordinates unchanged.

??? example "Reproducer: COL-52.py"

    [Download COL-52.py](COL-52.py)

    ```python
    """COL-52: CSRCSBEND BIN_ONCE wake output overwrites retained kick components

    With `IGF=0`, `BIN_ONCE=1` and repeated wake output, CSRCSBEND overwrites its retained kick-component arrays while preparing the first output page. Later `DeltaGammaT1` and `DeltaGammaT2` columns contain the preceding page's longitudinal positions and wake per metre, rather than components whose sum equals the reported change in Lorentz factor Δγ. For a 250.0 pC, 2.000 ps rms bunch in a 1.200 m steady-state bend, the peak component-sum residual is 0.007825 instead of zero, or 227.5% of peak |Δγ|; the transient case gives 219.0%. First-page, last-only and rebinning controls satisfy the sum identity, and changing the output schedule leaves the total wake and tracked coordinates unchanged.

    The script tracks deterministic Gaussian time quantiles through steady-state and transient bends in serial Elegant. It compares component sums and retained arrays across all-page, last-only, interval-two and disabled output, then checks rebinning at every slice. The expected components come from the first page and independent last-only output, without using the corrupted density columns.
    """

    import numpy as np
    from _elegant import particles, run, verdict
    from scipy.special import ndtri

    COMPONENTS = ("DeltaGamma", "DeltaGammaT1", "DeltaGammaT2")


    def columns(wake, page=0):
        data = {c.name: np.asarray(c.data[page]) for c in wake.columns}
        assert all(np.all(np.isfinite(c)) for c in data.values()), "Nonfinite wake output"
        return data


    def case(mode, schedule, bin_once=1):
        n = 8192
        beam = {
            "t": 2e-12 * ndtri((np.arange(n) + 0.5) / n),
            "p": np.full(n, 1500.0),
            "particleID": np.arange(1, n + 1),
        }
        output = ',OUTPUT_FILE="wake.sdds"'
        if schedule == "last":
            output += ",OUTPUT_LAST_WAKE_ONLY=1"
        elif schedule == "interval2":
            output += ",OUTPUT_INTERVAL=2"
        elif schedule == "off":
            output = ""
        lattice = (
            "Q: CHARGE,TOTAL=2.5e-10\n"
            f"B: CSRCSBEND,L=1.2,ANGLE={0.12 if mode == 'steady' else 0.3},"
            f"N_SLICES=4,BINS=96,BIN_ONCE={bin_once},STEADY_STATE={int(mode == 'steady')},"
            "IGF=0,SG_HALFWIDTH=1,SG_ORDER=1,SYNCH_RAD=0,ISR=0,TRAPAZOID_INTEGRATION=1"
            + output
            + "\nBL: LINE=(Q,B)\n"
        )
        command = """&run_setup lattice="line.lte",use_beamline=BL,
     p_central=1500,output="final.sdds" &end
    &run_control n_steps=1,n_passes=1 &end
    &sdds_beam input="input.sdds" &end
    &track &end
    """
        outputs = ("final.sdds",) if schedule == "off" else ("final.sdds", "wake.sdds")
        result = run(
            {"line.lte": lattice, "run.ele": command}, beams={"input.sdds": beam}, outputs=outputs
        )
        final = particles(result["final.sdds"])
        assert len(final["p"]) == n, "Unexpected particle loss"
        return result.get("wake.sdds"), final


    def residual(c):
        # With no wake filter, the IGF=0 branch defines dGamma=T1+T2. Steady
        # state sets T2=0. BIN_ONCE retains these components between slices.
        return float(np.max(np.abs(c["DeltaGammaT1"] + c["DeltaGammaT2"] - c["DeltaGamma"])))


    def check(mode):
        wake, final = case(mode, "all")
        assert wake.n_pages == 4
        first = columns(wake)
        d = first["DeltaGamma"]
        peak = float(np.max(np.abs(d)))
        assert peak > 1e-4
        assert residual(first) < peak * 2e-14, "First-page sum control failed"
        if mode == "steady":
            assert np.count_nonzero(first["DeltaGammaT2"]) == 0
        print(
            f"serial {mode} first page: peak_abs_DeltaGamma={peak:.15g}; "
            f"sum_residual={residual(first):.15g} expected=0 (DeltaGamma=T1+T2)"
        )
        defects = []
        for page in range(1, 4):
            c, prev = columns(wake, page), columns(wake, page - 1)
            np.testing.assert_array_equal(c["DeltaGamma"], d)
            overwritten = np.array_equal(c["DeltaGammaT1"], prev["s"]) and np.array_equal(
                c["DeltaGammaT2"], prev["GammaDeriv"]
            )
            error = residual(c)
            defects.append(error > peak * 2e-14 and overwritten)
            print(
                f"serial {mode} page {page + 1}: sum_residual={error:.15g} expected=0; "
                f"residual/peak={error / peak:.15g}; previous scratch arrays={overwritten}"
            )
        for schedule in ("last", "interval2", "off"):
            other, out = case(mode, schedule)
            equal = all(np.array_equal(final[k], out[k]) for k in ("x", "xp", "y", "yp", "t", "p"))
            assert equal, "Output schedule changed tracking"
            print(f"serial {mode} all vs {schedule}: final coordinates identical={equal} expected=True")
            if schedule == "last":
                assert other.n_pages == 1
                c = columns(other)
                for key in COMPONENTS:
                    np.testing.assert_array_equal(c[key], first[key])
                print(
                    f"serial {mode} last-only: retained components equal first page=True expected=True; "
                    f"sum_residual={residual(c):g} expected=0"
                )
            elif schedule == "interval2":
                assert other.n_pages == 2
                c, prev = columns(other, 1), columns(other)
                for key in COMPONENTS:
                    np.testing.assert_array_equal(prev[key], first[key])
                np.testing.assert_array_equal(c["DeltaGamma"], d)
                overwritten = np.array_equal(c["DeltaGammaT1"], prev["s"]) and np.array_equal(
                    c["DeltaGammaT2"], prev["GammaDeriv"]
                )
                defects.append(residual(c) > peak * 2e-14 and overwritten)
                print(
                    f"serial {mode} interval2 second page: sum_residual={residual(c):.15g} "
                    f"expected=0; previous scratch arrays={overwritten}"
                )
        rebin, _ = case(mode, "all", bin_once=0)
        assert rebin.n_pages == 4
        error = max(residual(columns(rebin, i)) for i in range(4))
        assert error < 1e-14, "Rebinning sum control failed"
        print(f"serial {mode} BIN_ONCE=0: max_sum_residual={error:.15g} expected=0")
        c = columns(wake, 1)
        row = int(np.argmax(np.abs(d)))
        print(
            f"serial {mode} page 2 row {row}: T1 observed={c['DeltaGammaT1'][row]:.15g} "
            f"expected={first['DeltaGammaT1'][row]:.15g}; T2 observed={c['DeltaGammaT2'][row]:.15g} "
            f"expected={first['DeltaGammaT2'][row]:.15g} (retained first/last-only components)"
        )
        return all(defects)


    if __name__ == "__main__":
        results = [check(mode) for mode in ("steady", "transient")]
        verdict(all(results))

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
| Found (serial T1/T2 buffer-reuse observation; MPI runs in this session concerned other entries) | `gpt-6-astra` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

