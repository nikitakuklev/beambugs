---
title: "OPT-28: rf_setup sets the electron's synchronous phase for positrons and protons"
hide:
  - toc
---

# OPT-28 — rf_setup sets the electron's synchronous phase for positrons and protons

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [the synchronous phase as 180° − arcsin(1/q) above transition and arcsin(1/q) below, with q the overvoltage, src/twiss.cc:5736–5741](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5736-L5741)<br>[writes `phase + phase_offset` into every selected RFCA and MODRF, src/twiss.cc:5744–5765](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5744-L5765)<br>[RFCA tracking divides its voltage by `particleRelSign`, src/simple_rfca.c:518–519](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/simple_rfca.c#L518-L519)<br>[MODRF tracking, src/mod_rfca.c:181](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/mod_rfca.c#L181)<br>[the RFCA matrix, src/compute_matrices.c:2485](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L2485)<br>[`particleRelSign = -1` for the positron, src/elegant.c:3273–3281](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3273-L3281)<br>[for the proton and every positively charged custom species, src/elegant.c:3306](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3306)<br>[synchrotron tune, bunch length and phase written to the `filename` output and stored as `Sz0` and `St0`, src/twiss.cc:5773–5793](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5773-L5793)<br>[set up rf cavity frequency, phase, and voltage for a storage ring, doc/elegant.tex:6756](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L6756)<br>[the energy loss per turn from the radiation integrals, doc/elegant.tex:6785–6786](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L6785-L6786)<br>[the frequency and phase are computed for this voltage, doc/elegant.tex:6877–6878](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L6877-L6878)<br>[offset added to the synchronous phase computed by this command, doc/elegant.tex:6883–6885](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L6883-L6885)<br>[has had limited testing, doc/elegant.tex:1497–1500](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L1497-L1500) |

After `change_particle` selects a positron or a proton, `rf_setup` gives the RFCA and MODRF cavities the electron's synchronous phase, although these cavities reverse the energy gain of positively charged particles. The positron's cavity then removes the radiation loss per turn U₀ instead of replacing it, and the synchrotron motion of either species grows exponentially. By charge conjugation the correct phase is the electron's plus 180°, which `phase_offset=180` applies. In a 98 m ring at 2.044 GeV with U₀ = 202.1 keV and 1 MV, positrons get 168.3° instead of 348.3°, lose 404.2 keV in the first turn, and the one-turn longitudinal map trace is 2.094 (amplitude growth 1.357 per turn) instead of the stable 1.906.

??? example "Reproducer: OPT-28.py"

    [Download OPT-28.py](OPT-28.py)

    ```python
    """OPT-28: rf_setup sets the electron's synchronous phase for positrons and protons

    After `change_particle` selects a positron or a proton, `rf_setup` gives the RFCA and MODRF cavities
    the electron's synchronous phase, although these cavities reverse the energy gain of positively
    charged particles. The positron's cavity then removes the radiation loss per turn U₀ instead of
    replacing it, and the synchrotron motion of either species grows exponentially. By charge
    conjugation the correct phase is the electron's plus 180°, which `phase_offset=180` applies. In a 98
    m ring at 2.044 GeV with U₀ = 202.1 keV and 1 MV, positrons get 168.3° instead of 348.3°, lose 404.2
    keV in the first turn, and the one-turn longitudinal map trace is 2.094 (amplitude growth 1.357 per
    turn) instead of the stable 1.906.

    The script builds a 98 m ring of 20 FODO cells with 40 CSBEND sector bends and one thin cavity, runs
    twiss_output with radiation integrals and rf_setup (harmonic 176, 1 MV) for the electron, the
    positron with an RFCA and with a MODRF, and a proton at p = 10 mₚc, and tracks one turn. It compares
    the applied phase, the tracked one-turn (t, δ) map and the one-turn energy change with radiation
    against an independent hard-edge calculation of U₀, the momentum compaction, the synchronous phase
    and the linear map of arc and thin cavity. The electron and the positive species with
    phase_offset=180 are the controls: they must get the correct phase and a stable map, and the shifted
    positron must reproduce the electron exactly.
    """

    import re

    import numpy as np
    from _elegant import particles, run, verdict

    # SDDS constants.h and elegant.c's species table, as Elegant uses them
    C, E_CHARGE, ME = 2.99792458e8, 1.60217733e-19, 9.1093897e-31
    MP = 1.6726485e-27  # kg, proton mass in elegant.c
    CELLS, LQ, K1, LD, LB = 20, 0.25, 1.0, 0.5, 1.2  # FODO cells; quad, drift, bend lengths (m)
    THETA = np.pi / CELLS  # two sector bends per cell close 2 pi
    HARMONIC, VOLT = 176, 1.0e6  # rf_setup harmonic and total voltage (V)
    CIRCUMFERENCE = CELLS * 2 * (LQ + 2 * LD + LB)
    # name: (p_central = beta*gamma, rest energy in eV, charge sign sigma relative to the electron).
    # RFCA and MODRF divide their voltage by particleRelSign (simple_rfca.c, mod_rfca.c), so the
    # energy gain is sigma V sin(phi).
    SPECIES = {
        "electron": (4000.0, ME * C**2 / E_CHARGE, 1),
        "positron": (4000.0, ME * C**2 / E_CHARGE, -1),
        "proton": (10.0, MP * C**2 / E_CHARGE, -1),
    }
    # (species, cavity element, rf_setup phase_offset); the electron and the offset cases are controls
    CASES = [
        ("electron", "RFCA", 0),
        ("positron", "RFCA", 0),
        ("positron", "MODRF", 0),
        ("positron", "RFCA", 180),
        ("proton", "RFCA", 0),
        ("proton", "RFCA", 180),
    ]
    DT, DDELTA = 1e-12, 1e-6  # s and fractional momentum: finite-difference steps of the tracked map
    PHASE_TOL, TRACE_RTOL, OPTICS_RTOL = 1e-6, 1e-6, 1e-6


    def lattice(cavity, synch_rad):
        """20 FODO cells of CSBEND sector bends (no edge angles, no fringe) and one thin cavity."""
        return (
            f"QF: QUAD,L={LQ},K1={K1}\nQD: QUAD,L={LQ},K1={-K1}\nD: DRIF,L={LD}\n"
            f"B: CSBEND,L={LB},ANGLE={THETA!r},N_KICKS=40,SYNCH_RAD={synch_rad}\n"
            f"RF: {cavity},L=0,VOLT=0,FREQ=1e8,PHASE=0\n"
            f"CELL: LINE=(QF,D,B,D,QD,D,B,D)\nRING: LINE=({CELLS}*CELL,RF)\n"
        )


    def optics():
        """Linear momentum compaction alpha_c, I2 and the periodic dispersion (D, D') at the ring start
        from hard-edge 3x3 (x, x', delta) matrices, independent of Elegant. alpha_c is the integral of
        D/rho over the sector bends divided by C, with D(s) = D0 cos(s/rho) + D0' rho sin(s/rho)
        + rho (1 - cos(s/rho)) in a bend entered with (D0, D0')."""
        rho = LB / THETA

        def quad(k):
            w = np.sqrt(abs(k))
            if k > 0:
                m = [[np.cos(w * LQ), np.sin(w * LQ) / w], [-w * np.sin(w * LQ), np.cos(w * LQ)]]
            else:
                m = [[np.cosh(w * LQ), np.sinh(w * LQ) / w], [w * np.sinh(w * LQ), np.cosh(w * LQ)]]
            out = np.eye(3)
            out[:2, :2] = m
            return out

        drift = np.array([[1, LD, 0], [0, 1, 0], [0, 0, 1.0]])
        c, s = np.cos(THETA), np.sin(THETA)
        bend = np.array([[c, rho * s, rho * (1 - c)], [-s / rho, c, s], [0, 0, 1.0]])
        sequence = [quad(K1), drift, bend, drift, quad(-K1), drift, bend, drift]
        cell = np.eye(3)
        for m in sequence:
            cell = m @ cell
        # periodic dispersion at the cell start: (1 - M) d = M[:2, 2]
        d = np.linalg.solve(np.eye(2) - cell[:2, :2], cell[:2, 2])
        vec, integral = np.array([d[0], d[1], 1.0]), 0.0
        for m in sequence:
            if m is bend:
                integral += vec[0] * s + vec[1] * rho * (1 - c) + (LB - rho * s)
            vec = m @ vec
        # I2 = sum of L/rho^2 = 2 pi/rho for one bending radius
        return CELLS * integral / CIRCUMFERENCE, 2 * np.pi / rho, d


    def elegant_case(species, cavity, phase_offset, beam, synch_rad):
        """twiss_output and rf_setup on the ring, then one turn of `beam`. Returns the phase rf_setup
        applied (from its log line), the .rf and twiss parameters and the tracked particles."""
        commands = (
            f"&change_particle name={species} &end\n"
            f"&run_setup lattice=ring.lte, use_beamline=RING, p_central={SPECIES[species][0]!r},"
            " output=out.sdds, default_order=2 &end\n"
            "&run_control n_passes=1 &end\n"
            "&twiss_output filename=ring.twi, matched=1, radiation_integrals=1 &end\n"
            f"&rf_setup filename=ring.rf, harmonic={HARMONIC}, total_voltage={VOLT!r},"
            f" phase_offset={phase_offset} &end\n"
            "&sdds_beam input=beam.sdds &end\n&track &end\n"
        )
        out = run(
            {"ring.lte": lattice(cavity, synch_rad), "run.ele": commands},
            beams={"beam.sdds": beam},
            outputs=("out.sdds", "ring.twi", "ring.rf"),
        )
        applied = [float(v) for v in re.findall(r"Set phase of RF to (\S+) deg", out["log"])]
        if len(applied) != 1:
            raise SystemExit("expected one 'Set phase of RF' line in the log: invalid")
        twi = {p.name: p.data[0] for p in out["ring.twi"].parameters}
        rf = {p.name: p.data[0] for p in out["ring.rf"].parameters}
        return applied[0], rf, twi, particles(out["out.sdds"])


    def turn_map(species, cavity, phase_offset, dispersion):
        """Tracked one-turn (t, delta) map about the nominal particle, radiation off: central
        differences of five particles (nominal, +-DT, +-DDELTA), symmetric so that the cavity's
        default mean-time fiducial is the nominal particle's arrival. The momentum-offset particles
        start on the periodic dispersion orbit (x, x') = (D, D') delta, so that their path length is
        alpha_c C delta and the 2x2 map is the arc's followed by the cavity's."""
        p0 = SPECIES[species][0]
        offsets = np.array([0, 0, 0, DDELTA, -DDELTA])
        beam = {"particleID": np.arange(1, 6), "t": np.array([0, DT, -DT, 0, 0]),
                "p": p0 * (1 + offsets), "x": dispersion[0] * offsets, "xp": dispersion[1] * offsets}  # fmt: skip
        applied, rf, twi, out = elegant_case(species, cavity, phase_offset, beam, synch_rad=0)
        t, delta = out["t"], out["p"] / p0 - 1
        m = np.array(
            [
                [(t[1] - t[2]) / (2 * DT), (t[3] - t[4]) / (2 * DDELTA)],
                [(delta[1] - delta[2]) / (2 * DT), (delta[3] - delta[4]) / (2 * DDELTA)],
            ]
        )
        return applied, rf, twi, m


    def turn_energy_change(species, cavity, phase_offset):
        """Energy change (eV) over one turn of a particle starting on axis at p_central, with
        deterministic synchrotron radiation in the bends (SYNCH_RAD=1, no ISR)."""
        p0, rest, _ = SPECIES[species]
        beam = {"particleID": np.arange(1, 2), "p": np.full(1, p0)}
        out = elegant_case(species, cavity, phase_offset, beam, synch_rad=1)[3]
        return (np.hypot(out["p"][0], 1) - np.hypot(p0, 1)) * rest


    alpha_c, i2, dispersion = optics()
    print(f"ring: C={CIRCUMFERENCE:g} m, 40 sector bends, independent alpha_c={alpha_c:.10g}, "
          f"I2={i2:.10g} m^-1; rf_setup harmonic={HARMONIC}, total_voltage={VOLT:g} V")  # fmt: skip
    results, controls_ok = {}, True
    for species, cavity, offset in CASES:
        p0, rest, sigma = SPECIES[species]
        gamma = np.hypot(p0, 1)
        beta = p0 / gamma
        # Radiation loss per turn U0 = (2/3) r gamma^4 m c^2 I2 with the species' classical radius
        # r = e^2/(4 pi eps0 m c^2) = 1e-7 e^2/m (constants.h's epsilon_o = 1e7/(4 pi c^2)).
        u0 = 2 / 3 * (1e-7 * E_CHARGE**2 / (rest * E_CHARGE / C**2)) * gamma**4 * rest * i2
        eta = alpha_c - 1 / gamma**2  # slip factor, positive: above transition
        t0 = CIRCUMFERENCE / (beta * C)
        omega = 2 * np.pi * HARMONIC / t0
        # Correct synchronous phase: sigma V sin(phi) = U0 replaces the radiation loss, and
        # eta sigma cos(phi) < 0 gives late particles (larger t, larger phase), which have higher
        # momentum above transition, less energy: phi = 180 - asin(U0/V) for sigma = +1, and that
        # plus 180 for sigma = -1 (charge conjugation), modulo 360.
        phi_e = 180 - np.degrees(np.arcsin(u0 / VOLT))
        phi = (phi_e if sigma > 0 else phi_e + 180) % 360
        applied, rf, twi, m = turn_map(species, cavity, offset, dispersion)
        # One-turn trace of the linear map: arc t -> t + T0 eta delta, then the thin cavity
        # delta -> delta + k t, k = sigma V omega cos(phase)/(beta^2 gamma m c^2), since
        # d delta = d gamma/(beta^2 gamma); trace = 2 + T0 eta k, stable for |trace| < 2.
        factor = t0 * eta * sigma * VOLT * omega / (beta**2 * gamma * rest)
        trace = {phase: 2 + factor * np.cos(np.radians(phase)) for phase in (phi, applied)}
        observed = np.trace(m)
        growth = max(abs(np.linalg.eigvals(m)))  # largest one-turn eigenvalue modulus
        label = f"{species} {cavity}" + (f" phase_offset={offset}" if offset else "")
        print(f"{label}: U0 Elegant={float(twi['U0']) * 1e6:.10g} eV independent={u0:.10g} eV")
        print(f"  applied phase: observed={applied:.9f} deg expected={phi:.9f} deg; "
              f".rf nuSynch={float(rf['nuSynch']):.6g}")  # fmt: skip
        print(f"  one-turn (t, delta) trace: observed={observed:.8f} expected={trace[phi]:.8f} "
              f"(linear map at the applied phase {trace[applied]:.8f}); largest |eigenvalue| "
              f"{growth:.6f}")  # fmt: skip
        controls_ok &= bool(
            np.isclose(float(twi["U0"]) * 1e6, u0, rtol=OPTICS_RTOL)
            and np.isclose(float(twi["alphac"]), alpha_c, rtol=OPTICS_RTOL)
            and np.isclose(observed - 2, trace[applied] - 2, rtol=TRACE_RTOL)
        )
        gain = None
        if species != "proton":
            gain = turn_energy_change(species, cavity, offset)
            # With the correct phase the cavity's sigma V sin(phi) = U0 replaces the loss along the
            # trajectory, so the turn changes the energy by much less than U0.
            print(f"  energy change over one turn with radiation: observed={gain:.3f} eV; cavity "
                  f"gain expected={sigma * VOLT * np.sin(np.radians(phi)):.3f} eV, at the applied "
                  f"phase {sigma * VOLT * np.sin(np.radians(applied)):.3f} eV")  # fmt: skip
        results[label] = dict(applied=applied, phi=phi, trace=observed, growth=growth, gain=gain, u0=u0)

    if not controls_ok:
        raise SystemExit("control: Elegant's U0, alphac or tracked map disagrees with the oracle")
    el = results["electron RFCA"]
    if not (abs(el["applied"] - el["phi"]) < PHASE_TOL and abs(el["trace"]) < 2
            and abs(el["gain"]) < 0.01 * el["u0"]):  # fmt: skip
        raise SystemExit("control: electron phase, stability or energy balance wrong: invalid")
    for label in ("positron RFCA phase_offset=180", "proton RFCA phase_offset=180"):
        if not (abs(results[label]["applied"] - results[label]["phi"]) < PHASE_TOL
                and abs(results[label]["trace"]) < 2):  # fmt: skip
            raise SystemExit(f"control: {label} is not the stable correct setup: invalid")
    po180 = results["positron RFCA phase_offset=180"]
    if not (np.isclose(po180["trace"], el["trace"], rtol=1e-12) and po180["gain"] == el["gain"]):
        raise SystemExit("control: positron with phase_offset=180 does not reproduce the electron")
    # The defect: the positive species get the electron's phase (for the proton 180 deg less a
    # negligible asin(U0/V)), so the positron loses about 2 U0 per turn and the longitudinal motion of
    # every positive species grows (trace > 2).
    wrong = ("positron RFCA", "positron MODRF", "proton RFCA")
    defect = all(
        abs(results[k]["applied"] - (el["applied"] if k.startswith("positron") else 180)) < PHASE_TOL
        and results[k]["trace"] > 2
        for k in wrong
    ) and all(results[k]["gain"] < -1.9 * el["u0"] for k in wrong[:2])
    print(f"comparison: phases to {PHASE_TOL:g} deg; U0 and alphac rtol {OPTICS_RTOL:g}; "
          f"trace - 2 rtol {TRACE_RTOL:g} against the linear map at the applied phase")  # fmt: skip
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
| Found (noted while writing SCT-21; traced the missing species sign in the rf_setup phase) | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source reading, not executed for this claim |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

