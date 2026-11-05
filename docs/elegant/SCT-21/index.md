---
title: "SCT-21: moments_output IBS uses a negative particle count for positrons and protons"
hide:
  - toc
---

# SCT-21 — moments_output IBS uses a negative particle count for positrons and protons

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [Coulomb log, moments.c:1551–1582](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L1551-L1582)<br>[moments.c:1592–1597](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L1592-L1597)<br>[elegant.c:3264–3291](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3264-L3291)<br>[charge table, elegant.c:3244–3245](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3244-L3245)<br>[elegant.c:3292–3298](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3292-L3298)<br>[do_tracking.c:894–897](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L894-L897)<br>[elegant.tex:5548–5550](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5548-L5550)<br>[`ibs_iterations`, elegant.tex:5617–5624](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5617-L5624)<br>[change_particle, elegant.tex:1497–1517](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L1497-L1517) |

After change_particle selects positrons or protons, moments_output with ibs_iterations divides the bunch charge by Elegant's internal particle charge, negative for these species, so its intrabeam-scattering (IBS) calculation uses a negative particle count. With the default automatic Coulomb logarithm ln Λ, the beam density is negative, and ln Λ and every moment are NaN from the first IBS update, without warning. A prescribed ln Λ reverses the IBS diffusion. By charge conjugation, a positron ring with the RF phase shifted by 180° must reproduce the electron result: in a 524.8 m ring at 1.5 GeV/c with 1 nC and ln Λ = 10, IBS raises the mode emittance e₁ from 6.604 to 8.742 nm for electrons, but positrons get 6.599 nm.

??? example "Reproducer: SCT-21.py"

    [Download SCT-21.py](SCT-21.py)

    ```python
    """SCT-21: moments_output IBS uses a negative particle count for positrons and protons

    After change_particle selects positrons or protons, moments_output with ibs_iterations divides the
    bunch charge by Elegant's internal particle charge, negative for these species, so its
    intrabeam-scattering (IBS) calculation uses a negative particle count. With the default automatic
    Coulomb logarithm ln Λ, the beam density is negative, and ln Λ and every moment are NaN from the
    first IBS update, without warning. A prescribed ln Λ reverses the IBS diffusion. By charge
    conjugation, a positron ring with the RF phase shifted by 180° must reproduce the electron result:
    in a 524.8 m ring at 1.5 GeV/c with 1 nC and ln Λ = 10, IBS raises the mode emittance e₁ from 6.604
    to 8.742 nm for electrons, but positrons get 6.599 nm.

    The script runs moments_output with radiation and IBS on a 128-cell, 524.8 m FODO ring at 1.5 GeV/c
    with a 1 nC bunch, for the electron and for the positron with the RF phase shifted by 180°, with two
    and with thirty IBS iterations, each with the automatic and with a prescribed (ln Λ = 10) Coulomb
    logarithm. It evaluates Kubo and Oide's Coulomb logarithm independently from the radiation-only
    moments, positive for either charge, and takes the electron's results as the positron's expected
    values by charge conjugation. The electron is the control: its logarithm must equal the independent
    value to 10⁻¹² and IBS must raise its emittances.
    """

    import numpy as np
    from _elegant import run, verdict

    # SDDS constants.h and the static species table of elegant.c
    C, E_CHARGE, ME, ME_MEV = 2.99792458e8, 1.60217733e-19, 9.1093897e-31, 0.51099906
    RE = 2.81794092e-15  # m, electron classical radius re_mks
    P_MEV, CHARGE, VOLT, CELLS = 1500.0, 1e-9, 1e6, 128  # MeV/c, C, V, FODO cells
    CLOG = 10.0  # prescribed Coulomb logarithm of the fixed-log runs


    def lattice(phase):
        """128 weakly coupled FODO cells (QF rolled by 10 mrad), two SBENDs per cell closing 2 pi."""
        return (
            f"Q: CHARGE,TOTAL={CHARGE!r}\n"
            "D: DRIF,L=0.4\n"
            "QF: QUAD,L=0.25,K1=0.8,TILT=0.01\n"
            "QD: QUAD,L=0.25,K1=-0.7\n"
            f"B: SBEN,L=1,ANGLE={np.pi / CELLS!r}\n"
            f"RF: RFCA,VOLT={VOLT!r},FREQ=500e6,PHASE={phase!r}\n"
            "CELL: LINE=(QF,D,B,D,QD,D,B,D)\n"
            f"RING: LINE=(Q,{CELLS}*CELL,RF)\n"
        )


    def moments(particle, phase, iterations, clog):
        """moments_output parameters of every page (radiation, then IBS iterations), the columns of the
        first and the last page, and the twiss parameters."""
        commands = (
            f'&change_particle name="{particle}" &end\n'
            f'&run_setup lattice="ring.lte", use_beamline="RING", p_central_mev={P_MEV!r}, '
            "default_order=2 &end\n"
            '&twiss_output filename="ring.twi", matched=1, radiation_integrals=1 &end\n'
            '&moments_output filename="ring.mom", radiation=1, equilibrium=1, '
            f"ibs_iterations={iterations}, ibs_output_iterations=1, ibs_coulomb_log={clog!r} &end\n"
        )
        out = run({"ring.lte": lattice(phase), "run.ele": commands}, outputs=("ring.mom", "ring.twi"))
        mom, twi = out["ring.mom"], out["ring.twi"]
        par = {p.name: np.asarray(p.data) for p in mom.parameters}
        first, last = (
            {c.name: np.asarray(mom[k, c.name]) for c in mom.columns} for k in (0, mom.n_pages - 1)
        )
        tw = {p.name: p.data[0] for p in twi.parameters}
        return par, first, last, tw


    def sigma_matrix(col, row):
        """Lab-frame second moments in (x, x', y, y', s, delta) at the end of the element in `row`."""
        s = np.zeros((6, 6))
        for i in range(6):
            for j in range(i, 6):
                s[i, j] = s[j, i] = (
                    col[f"s{i + 1}"][row] ** 2 if i == j else col[f"s{i + 1}{j + 1}"][row]
                )
        return s


    def coulomb_log(sigma, beta_gamma, tau_lab):
        """Kubo-Oide Coulomb log ln(bmax/bmin) of one element for a singly charged particle of mass me.

        Lab moments are converted to momenta with p0 = me c beta*gamma and boosted to the beam frame
        (z -> gamma z, p_z -> p0 delta/gamma). The conditional momentum covariance at fixed position
        (Schur complement) has trace U and the rms velocity is sqrt(U)/me. The particle count N = Q/e is
        positive for either sign of charge, so the beam-frame density n = N/(8 sqrt(pi^3 det Sigma_x))
        is positive; bmax = min(spatial rms sizes, n^(-1/3)); bmin1 = 1/sqrt(pi n v tau) with tau the
        largest damping time in the beam frame; bmin2 = sqrt(2) re (me c)^2/U, with the sqrt(2) of
        moments.c.
        """
        gamma = np.hypot(beta_gamma, 1.0)
        p0 = ME * C * beta_gamma
        t = np.diag([1, p0, 1, p0, gamma, p0 / gamma])
        cov = t @ sigma @ t
        xx, pp = [0, 2, 4], [1, 3, 5]
        cxx, cpp, cpx = cov[np.ix_(xx, xx)], cov[np.ix_(pp, pp)], cov[np.ix_(pp, xx)]
        trace = np.trace(cpp - cpx @ np.linalg.solve(cxx, cpx.T))
        # The boosted bunch length exceeds the vertical size by ~10^5: take the small eigenvalues from
        # the inverse matrix and the middle one from the determinant, each to full relative precision.
        largest = np.linalg.eigvalsh(cxx)[-1]
        smallest = 1 / np.linalg.eigvalsh(np.linalg.inv(cxx))[-1]
        eig = np.array([smallest, np.linalg.det(cxx) / (smallest * largest), largest])
        density = CHARGE / E_CHARGE / (8 * np.sqrt(np.pi**3 * np.prod(eig)))  # m^-3
        bmax = min(np.sqrt(eig).min(), density ** (-1 / 3))
        bmin1 = 1 / np.sqrt(np.pi * density * (np.sqrt(trace) / ME) * tau_lab / gamma)
        bmin2 = np.sqrt(2) * RE * (ME * C) ** 2 / trace
        return np.log(bmax / max(bmin1, bmin2)), density


    # The RF phase balancing the radiation loss U0 above transition is 180 deg - asin(U0/V) for the
    # electron. RFCA multiplies its energy gain by the species' charge sign, so the positron needs the
    # phase shifted by 180 deg; U0 depends only on the bends and the energy, not on the charge sign.
    _, _, _, tw = moments("electron", 180.0, 0, 0.0)
    u0 = float(tw["U0"]) * 1e6  # eV per turn
    phase_e = float(180 - np.degrees(np.arcsin(u0 / VOLT)))
    tau = max(float(tw[k]) for k in ("taux", "tauy", "taudelta"))
    beta_gamma = P_MEV / ME_MEV  # elegant's p_central from p_central_mev
    print(f"ring: U0={u0 / 1e3:.6f} keV/turn, electron RF phase {phase_e:.10f} deg, "
          f"largest damping time {tau * 1e3:.4f} ms")  # fmt: skip

    results = {}
    for species, phase in (("electron", phase_e), ("positron", phase_e - 180)):
        auto2 = moments(species, phase, 2, 0.0)  # page 0: radiation-only moments, log of first update
        fixed2 = moments(species, phase, 2, CLOG)  # page 1: moments after one full-charge IBS update
        auto30 = moments(species, phase, 30, 0.0)  # last page: IBS equilibrium, automatic log
        fixed30 = moments(species, phase, 30, CLOG)
        par, col, _, _ = auto2
        row = list(col["ElementName"]).index("QF")  # first element with length
        radiation = np.array([par[f"e{i}"][0] for i in (1, 2, 3)])
        expected_log, density = coulomb_log(sigma_matrix(col, row), beta_gamma, tau)
        observed_log = float(col["CoulombLog"][row])
        update = np.array([fixed2[0][f"e{i}"][1] for i in (1, 2, 3)]) / radiation - 1
        eq_auto = np.array([auto30[0][f"e{i}"][-1] for i in (1, 2, 3)])
        eq_fixed = np.array([fixed30[0][f"e{i}"][-1] for i in (1, 2, 3)])
        sizes = np.array([auto30[2][f"s{i}"] for i in range(1, 7)])
        results[species] = dict(radiation=radiation, log=(observed_log, expected_log), update=update,
                                eq_auto=eq_auto, eq_fixed=eq_fixed, nan=int(np.isnan(sizes).sum()))  # fmt: skip
        print(f"{species}: radiation-only (e1, e2, e3) = {np.array2string(radiation, precision=10)} m; "
              f"independent beam-frame density at QF#1 {density:.6e} m^-3")  # fmt: skip
        print(f"  automatic Coulomb log of the first IBS update at QF#1: observed={observed_log:.8f} "
              f"expected={expected_log:.8f}")  # fmt: skip
        print(f"  first IBS update with ln(Lambda)={CLOG:g}, relative change of (e1, e2, e3): "
              f"{np.array2string(update, precision=6)}")  # fmt: skip
        print(f"  30 iterations, automatic log: (e1, e2, e3) = {np.array2string(eq_auto, precision=6)} m;"
              f" NaN rms sizes {results[species]['nan']} of {sizes.size}")  # fmt: skip
        print(f"  30 iterations, ln(Lambda)={CLOG:g}: (e1, e2, e3) = "
              f"{np.array2string(eq_fixed, precision=6)} m")  # fmt: skip

    # Charge conjugation: at the same mass and normalized optics, with the RF phase shifted by 180 deg,
    # radiation damping and quantum excitation (both quadratic in the charge) and the Coulomb
    # scattering rate (quadratic in the charge, linear in the particle count) equal the electron's, so
    # the positron must reproduce the electron's radiation-only moments, Coulomb log and IBS results.
    el, po = results["electron"], results["positron"]
    if not np.isclose(*el["log"], rtol=1e-12, atol=0):
        raise SystemExit("control: electron Coulomb log differs from the independent value")
    if not (np.all(el["update"] > 0) and np.all(el["eq_auto"] > el["radiation"]) and el["nan"] == 0):
        raise SystemExit("control: electron IBS does not increase the emittances")
    if not np.allclose(po["radiation"], el["radiation"], rtol=1e-6, atol=0):
        raise SystemExit("control: positron radiation-only emittances differ from the electron's")
    rel = (po["eq_fixed"] / el["eq_fixed"] - 1) * 100
    print(f"positron automatic Coulomb log: observed={po['log'][0]:.8f} expected={po['log'][1]:.8f}")
    print(f"positron IBS equilibrium e1, automatic log: observed={po['eq_auto'][0]:.6e} m "
          f"expected={el['eq_auto'][0]:.6e} m (electron, charge conjugation)")  # fmt: skip
    print(f"positron IBS equilibrium e1, ln(Lambda)={CLOG:g}: observed={po['eq_fixed'][0]:.6e} m "
          f"expected={el['eq_fixed'][0]:.6e} m ({rel[0]:+.3f}%; radiation-only "
          f"{po['radiation'][0]:.6e} m)")  # fmt: skip
    defect = bool(
        np.isnan(po["log"][0])
        and po["nan"] > 0
        and np.allclose(po["update"], -el["update"], rtol=1e-3, atol=0)
        and np.all(po["eq_fixed"] < po["radiation"])
    )
    print("comparison: electron log rtol=1e-12; radiation-only emittances rtol=1e-6; defect needs a NaN "
          "automatic log and NaN moments, a first update equal to minus the electron's at rtol=1e-3 and "
          "a fixed-log equilibrium below the radiation-only emittances")  # fmt: skip
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

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

