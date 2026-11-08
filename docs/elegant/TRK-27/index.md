---
title: "TRK-27: CSBEND with EXPAND_HAMILTONIAN=1 does not transport the spin through the bend body"
hide:
  - toc
---

# TRK-27 — CSBEND with EXPAND_HAMILTONIAN=1 does not transport the spin through the bend body

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`track_through_csbend`, csbend.c:1496–1502](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1496-L1502)<br>[expanded integrator, csbend.c:2064–2255](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2064-L2255)<br>[spin update in the exact integrator, csbend.c:1930–1935](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1930-L1935)<br>[EXPAND_HAMILTONIAN, track_data.c:1977](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L1977)<br>[spin_tracking, elegant.tex:7291–7292](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7291-L7292)<br>[CSBEND among spin-aware elements, elegant.tex:11191–11193](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L11191-L11193) |

With spin tracking enabled, a CSBEND using EXPAND_HAMILTONIAN=1 omits spin transport through the magnet body, although the exact-Hamiltonian branch applies the Thomas–BMT equation. For a reference electron with initially longitudinal spin and momentum 450 mₑc through a 0.8 m bend with fringes disabled and angle θ = 0.18 rad, the correct spin rotation in the exit frame is −aγθ = −0.09393 rad, where a is the electron's anomalous magnetic moment and γ its Lorentz factor. The exact-Hamiltonian control agrees, while the expanded branch returns zero rotation without a warning. Fringe rotations are separate.

Run `python TRK-27.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-27.py"

    [Download TRK-27.py](TRK-27.py)

    ```python
    """TRK-27: CSBEND with EXPAND_HAMILTONIAN=1 does not transport the spin through the bend body

    With spin tracking enabled, a CSBEND using EXPAND_HAMILTONIAN=1 omits spin transport through the
    magnet body, although the exact-Hamiltonian branch applies the Thomas–BMT equation. For a reference
    electron with initially longitudinal spin and momentum 450 mₑc through a 0.8 m bend with fringes
    disabled and angle θ = 0.18 rad, the correct spin rotation in the exit frame is −aγθ = −0.09393 rad,
    where a is the electron's anomalous magnetic moment and γ its Lorentz factor. The exact-Hamiltonian
    control agrees, while the expanded branch returns zero rotation without a warning. Fringe rotations are separate.

    The script tracks 12 identical on-reference electrons with spin (0, 0, 1) through a pure CSBEND with
    edge effects off, for the exact and the expanded Hamiltonian, at two momenta and two bend angles of
    both signs, and compares the final spin with the analytic precession of a spin along the reference
    orbit. The exact-Hamiltonian run of each case is the control.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    A_E = 0.00115965218128  # electron anomalous magnetic moment, particleAnomalousMagneticMoment
    LENGTH, SLICES = 0.8, 256
    N = 12  # identical particles, so that every MPI worker tracks some
    # Absolute tolerance on dimensionless sx and sz; the fourth-order exact integration is within
    # about 2e-14 here.
    TOL = 1e-9
    CASES = ((450.0, 0.18), (900.0, 0.18), (450.0, -0.09))  # (p0 in m_e c, bend angle in rad)


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def spin_after(mode, p0, angle, expand):
        """Final spin (sx, sz) of the reference electrons through a pure dipole body."""
        lattice = (
            f"B: CSBEND,L={LENGTH!r},ANGLE={angle!r},N_SLICES={SLICES},INTEGRATION_ORDER=4,"
            f"EXPAND_HAMILTONIAN={expand},EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,SYNCH_RAD=0,ISR=0,"
            "REFERENCE_CORRECTION=0\nBL: LINE=(B)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, output=out.sdds,"
            " spin_tracking=1 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        zero = np.zeros(N)
        beam = {"p": np.full(N, p0), "spx": zero, "spy": zero, "spz": np.ones(N)}
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})
        out = particles(result["out.sdds"])
        if len(out["p"]) != N:
            sys.exit(f"{mode}: {N - len(out['p'])} particles lost")
        if not all(np.all(np.isfinite(out[k])) for k in ("p", "spx", "spy", "spz")):
            sys.exit(f"{mode}: nonfinite particle momentum or spin in the output: invalid run")
        if np.max(np.abs(out["spx"] - out["spx"][0])) > 0 or np.max(np.abs(out["spy"])) > 1e-15:
            sys.exit(f"{mode}: identical reference particles leave with different spins")
        return np.array([out["spx"][0], out["spz"][0]]), result["log"]


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected differ by more than TOL."""
        if not (np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))):
            sys.exit(f"{label}: nonfinite comparison operand: invalid run")
        bad = bool(np.any(np.abs(observed - expected) > TOL))
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    defect = []
    print("columns: sx, sz after the bend; expected from the Thomas-BMT precession")
    for mode in modes():
        for p0, angle in CASES:
            # On the reference orbit the field is perpendicular to the velocity, so Thomas-BMT rotates
            # the spin by -(1 + a*gamma)*theta in the laboratory frame, and transforming to the exit
            # reference frame adds +theta to this signed laboratory spin angle: the sum is
            # -a*gamma*theta (gamma = sqrt(1 + p0^2) for p0 in m_e c).
            theta = -A_E * np.sqrt(1 + p0 * p0) * angle
            expected = np.array([np.sin(theta), np.cos(theta)])
            label = f"{mode} p0={p0:g} angle={angle}"
            exact, _ = spin_after(mode, p0, angle, 0)
            if compare(f"{label} exact Hamiltonian (control)", exact, expected):
                sys.exit("The exact-Hamiltonian control disagrees with the precession: invalid.")
            expanded, log = spin_after(mode, p0, angle, 1)
            defect.append(compare(f"{label} EXPAND_HAMILTONIAN=1", expanded, expected))
            print(
                f"{label}: expected angle {theta:.16g} rad; warnings in expanded log: {'warn' in log.lower()}"
            )
    print(f"comparison: abs tolerance {TOL:g} on sx and sz")
    verdict(all(defect))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-03</span> | master (`0355dd6aca12`), source reading, not executed for this claim |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-03</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |

