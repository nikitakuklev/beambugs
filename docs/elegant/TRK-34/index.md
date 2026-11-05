---
title: "TRK-34: CSBEND with EXPAND_HAMILTONIAN=1 scales its quantum excitation by the integration order"
hide:
  - toc
---

# TRK-34 — CSBEND with EXPAND_HAMILTONIAN=1 scales its quantum excitation by the integration order

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`integrate_csbend_ordn_expanded`, csbend.c:2205–2224](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2205-L2224)<br>[csbend.c:2104–2119](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2104-L2119)<br>[csbend.c:1950–1952](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1950-L1952)<br>[`determineRadiationMatrix`, analyze.c:1775–1783](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/analyze.c#L1775-L1783)<br>[manual, csbend.tex:105–110](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L105-L110)<br>[parameter table, track_data.c:1977](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1977) |

A CSBEND with EXPAND_HAMILTONIAN=1 and ISR=1 gives every radiation kick the step length s/3, although integration orders 2, 4 and 6 have 1, 3 and 7 kicks per step of length s. The energy-spread variance σδ² it adds in tracking is therefore 1/3 of the correct value at order 2 and 7/3 at order 6, and moments_output, which always integrates at order 6, applies 7/3 at every order. For a 0.9 m, 0.07 rad bend at 2.555 GeV, moments_output gives σδ² = 4.446×10⁻¹² instead of the Sands quantum-diffusion value 1.905×10⁻¹², which the exact Hamiltonian reproduces. Order 4 is correct in tracking.

??? example "Reproducer: TRK-34.py"

    [Download TRK-34.py](TRK-34.py)

    ```python
    """TRK-34: CSBEND with EXPAND_HAMILTONIAN=1 scales its quantum excitation by the integration order

    A CSBEND with EXPAND_HAMILTONIAN=1 and ISR=1 gives every radiation kick the step length s/3,
    although integration orders 2, 4 and 6 have 1, 3 and 7 kicks per step of length s. The energy-spread
    variance σδ² it adds in tracking is therefore 1/3 of the correct value at order 2 and 7/3 at order
    6, and moments_output, which always integrates at order 6, applies 7/3 at every order. For a 0.9 m,
    0.07 rad bend at 2.555 GeV, moments_output gives σδ² = 4.446×10⁻¹² instead of the Sands
    quantum-diffusion value 1.905×10⁻¹², which the exact Hamiltonian reproduces. Order 4 is correct in
    tracking.

    The script tracks 160000 identical on-axis electrons through a pure dipole CSBEND with ISR=1,
    using the exact and the expanded Hamiltonian at integration orders 2, 4 and 6 with the same random
    seed, and compares the energy-spread variances with each other and with the quantum-diffusion
    integral. It also compares the energy-spread growth that moments_output computes for both
    Hamiltonians with that integral. The exact-Hamiltonian runs are the control.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    P0 = 5000.0  # reference momentum in m_e c (2.555 GeV electrons)
    LENGTH, ANGLE, SLICES = 0.9, 0.07, 12
    N = 160000  # identical particles; the sample variance has a relative standard error of 0.35%
    RE = 2.8179403205e-15  # classical electron radius (m), CODATA 2022
    LAMBDA_C = 3.8615926744e-13  # reduced Compton wavelength of the electron (m), CODATA 2022
    # Paired runs draw the same Gaussian numbers, so their variance ratio is the ratio of ISR step
    # lengths up to the small classical-loss and orbit differences (observed within 5e-8).
    RATIO_TOL = 1e-6
    STAT_TOL = 0.02  # sample variance against the integral: about 6 standard errors
    MOMENTS_TOL = 1e-4  # deterministic; the exact Hamiltonian agrees to about 2e-6


    def lattice(order, expand):
        return (
            f"B: CSBEND,L={LENGTH!r},ANGLE={ANGLE!r},N_SLICES={SLICES},INTEGRATION_ORDER={order},"
            f"EXPAND_HAMILTONIAN={expand},EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,SYNCH_RAD=1,ISR=1\n"
            # The marker keeps moments_output off its one-element-beamline crash.
            "END: MARK\nBL: LINE=(B,END)\n"
        )


    SETTINGS = "&global_settings SR_gaussian_limit=10 &end\n"  # no Gaussian truncation at 3 sigma


    def tracked_variance(mode, order, expand):
        commands = SETTINGS + (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds,"
            " random_number_seed=7654321 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        beam = {"p": np.full(N, P0), "particleID": np.arange(1, N + 1)}
        files = {"lattice.lte": lattice(order, expand), "run.ele": commands}
        out = particles(run(files, mode, {"beam.sdds": beam})["out.sdds"])
        if len(out["p"]) != N:
            sys.exit(f"{mode}: particles were lost")
        return float(np.var(out["p"] / P0, ddof=1))


    def moments_variance(mode, expand):
        commands = SETTINGS + (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r} &end\n"
            "&moments_output filename=out.mom, matched=0, equilibrium=0, radiation=1,"
            " beta_x=2, beta_y=3, beta_z=1 &end\n"
            "&run_control &end\n&bunched_beam &end\n&track &end\n"
        )
        files = {"lattice.lte": lattice(4, expand), "run.ele": commands}
        s6 = np.asarray(run(files, mode, outputs=("out.mom",))["out.mom"][0, "s6"])
        return float(s6[-1] ** 2 - s6[0] ** 2)


    def show(label, observed, expected):
        print(
            f"{label}: observed={observed:.10g} expected={expected:.10g} observed/expected={observed / expected:.9g}"
        )


    # Sands quantum diffusion through a uniform bend of curvature h = ANGLE/L:
    # sigma_delta^2 = (55/(24 sqrt 3)) r_e lambda_c gamma^5 |h|^3 L.
    gamma = np.hypot(1.0, P0)
    analytic = 55 / (24 * np.sqrt(3)) * RE * LAMBDA_C * gamma**5 * (ANGLE / LENGTH) ** 3 * LENGTH
    # Kicks per integration step: 1, 3 and 7 for orders 2, 4 and 6. The exact integrator gives each kick
    # the ISR length s/(kicks); the expanded one uses s/3, so its variance is kicks/3 times larger.
    KICKS = {2: 1, 4: 3, 6: 7}

    defect = []
    print(f"p0={P0:g} m_e c, L={LENGTH} m, ANGLE={ANGLE} rad, {SLICES} slices, {N} particles")
    print(f"analytic sigma_delta^2 = {analytic:.10g}")
    for mode in modes():
        for order, kicks in KICKS.items():
            exact = tracked_variance(mode, order, 0)
            expanded = tracked_variance(mode, order, 1)
            show(f"{mode} order {order} exact Hamiltonian tracking (control)", exact, analytic)
            if abs(exact / analytic - 1) > STAT_TOL:
                sys.exit(
                    "The exact-Hamiltonian control disagrees with the diffusion integral: invalid."
                )
            print(f"{mode} order {order} expanded tracking: variance {expanded:.10g}")
            ratio = expanded / exact
            print(
                f"{mode} order {order} expanded/exact variance: observed={ratio:.10g} expected=1"
                f" (ISR step factor {kicks}/3 = {kicks / 3:.10g})"
            )
            if abs(ratio - kicks / 3) > RATIO_TOL:
                sys.exit("The variance ratio is not the ISR step-length factor: the cause differs.")
            if order != 4:  # order 4 has the correct factor in tracking; it is a control there
                defect.append(abs(ratio - 1) > RATIO_TOL)
        exact = moments_variance(mode, 0)
        show(f"{mode} moments_output exact Hamiltonian (control)", exact, analytic)
        if abs(exact / analytic - 1) > MOMENTS_TOL:
            sys.exit("The exact-Hamiltonian moments control disagrees with the integral: invalid.")
        expanded = moments_variance(mode, 1)
        show(f"{mode} moments_output EXPAND_HAMILTONIAN=1 (INTEGRATION_ORDER=4)", expanded, analytic)
        defect.append(abs(expanded / analytic - 1) > MOMENTS_TOL)
    print(f"comparison: ratio tolerance {RATIO_TOL:g}, moments relative tolerance {MOMENTS_TOL:g}")
    verdict(all(defect))

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

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Found (moments_output) | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

