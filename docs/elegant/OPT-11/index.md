---
title: "OPT-11: MULT pole-tip optics use electron rigidity for other species"
hide:
  - toc
---

# OPT-11 — MULT pole-tip optics use electron rigidity for other species

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`mult_matrix`, rigidity `H = P me c/e` and the pole-tip conversion, compute_matrices.c:3251–3254](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L3251-L3254)<br>[`particleCharge/(particleMass c Po)`, multipole.c:531–535](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L531-L535)<br>[`change_particle` sets the species mass and signed charge, elegant.c:3244–3290](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/elegant.c#L3244-L3290)<br>[K_nL = n! B_tip L/(rⁿ Bρ), doc/mult.tex:5–15](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/mult.tex#L5-L15)<br>[`BORE` and `BTIPL` parameters, track_data.c:508–509](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L508-L509)<br>[`change_particle`, which must precede `run_setup` and has had limited testing, doc/elegant.tex:1493–1518](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L1493-L1518) |

For `MULT` quadrupoles specified by pole-tip fields, analytic optics use electron rigidity even when `change_particle` selects positrons or protons. For a 0.4 m magnet with bore radius 0.02 m, integrated pole-tip field 2.000×10⁻⁵ T·m and dimensionless momentum p/(mc) = 10, the physical horizontal slope response R₂₁ is −0.05844 m⁻¹ for electrons, +0.05890 m⁻¹ for positrons and +3.195×10⁻⁵ m⁻¹ for protons; the analytic matrix gives −0.05844 m⁻¹ for all three. The positron and proton focusing signs are reversed, and the proton |R₂₁| is 1829 times too large. Direct tracking and explicit integrated strengths agree with the physical quadrupole matrix.

Run `python OPT-11.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: OPT-11.py"

    [Download OPT-11.py](OPT-11.py)

    ```python
    """OPT-11: MULT pole-tip optics use electron rigidity for other species

    For `MULT` quadrupoles specified by pole-tip fields, analytic optics use electron rigidity even when
    `change_particle` selects positrons or protons. For a 0.4 m magnet with bore radius 0.02 m,
    integrated pole-tip field 2.000×10⁻⁵ T·m and dimensionless momentum p/(mc) = 10, the physical
    horizontal slope response R₂₁ is −0.05844 m⁻¹ for electrons, +0.05890 m⁻¹ for positrons and
    +3.195×10⁻⁵ m⁻¹ for protons; the analytic matrix gives −0.05844 m⁻¹ for all three. The positron and
    proton focusing signs are reversed, and the proton |R₂₁| is 1829 times too large. Direct tracking and
    explicit integrated strengths agree with the physical quadrupole matrix.

    The script builds a 0.4 m `MULT` quadrupole specified by its pole-tip field for an electron, a
    positron and a proton beam in serial Elegant (and with `--mpi` in 3-rank Pelegant). It compares the
    analytic R21 written by `matrix_output` and the R21 measured by tracking small-amplitude particles
    with the transport of the physical field and rigidity. Controls: the same magnet with the physically
    correct integrated strength `KNL` given explicitly, and the convergence of the tracking error with
    the number of slices, show that the matrix code and the integrator are sound.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # Constants of SDDS constants.h and the particle table of elegant.c.
    C, QE, ME, MP = 299792458.0, 1.60217733e-19, 9.1093897e-31, 1.6726485e-27
    LENGTH, BORE, BTIPL, P0 = 0.4, 0.02, 2e-5, 10.0  # m, m, T m, p/(m c) of the species
    CHARGE = {"electron": -QE, "positron": QE, "proton": QE}  # physical charge (C)
    MASS = {"electron": ME, "positron": ME, "proton": MP}  # kg
    X0 = np.array([-6.0, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]) * 1e-7  # amplitudes (m)
    ANALYTIC_ATOL = 1e-12  # explicit-KNL matrix against the oracle: rounding
    TRACK_ATOL = 2e-9  # tracking at 1024 slices: second-order integrator error is about 2e-10


    def expected_knl(species):
        """Physical integrated quadrupole strength in Elegant's sign convention (focusing > 0).

        The pole-tip field gradient is BTIPL/(BORE L) per unit length, so K1 L = (BTIPL/BORE)/(B rho)
        with rigidity B rho = m c p0/|q|, where p0 = p/(m c) is the dimensionless momentum of the
        selected species. Elegant's focusing sign is that of an electron: positive for q < 0.
        """
        rigidity = MASS[species] * C * P0 / abs(CHARGE[species])
        return -np.sign(CHARGE[species]) * (BTIPL / BORE) / rigidity


    def expected_r21(knl):
        """R21 of a hard-edge quadrupole for small amplitudes: x'' = -k x with k = KNL/L."""
        k = knl / LENGTH
        root = np.sqrt(abs(k))
        return -root * np.sin(root * LENGTH) if k > 0 else root * np.sinh(root * LENGTH)


    def case(mode, species, strength, slices):
        """(analytic R21, tracked R21) of the MULT with the given strength specification."""
        lattice = (
            f"E: MULT,L={LENGTH!r},N_SLICES={slices},ORDER=1,MATRIX_ORDER=1,{strength}\nBL: LINE=(E)\n"
        )
        commands = (
            f"&change_particle name={species} &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds,\n"
            " default_order=2 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&matrix_output SDDS_output=matrix.sdds &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        beam = {
            "x": X0,
            "p": np.full(len(X0), P0),
            "particleID": np.arange(1, len(X0) + 1),
        }
        out = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            beams={"beam.sdds": beam},
            outputs=("matrix.sdds", "out.sdds"),
        )
        analytic = out["matrix.sdds"][0, "R21"][-1]
        final = particles(out["out.sdds"])
        # Least-squares response of the output slope to the (symmetric) input amplitudes.
        tracked = np.sum(X0 * final["xp"]) / np.sum(X0**2)
        return analytic, tracked


    def compare(label, observed, expected, atol):
        """Print one case; return True when observed and expected disagree."""
        if not (np.isfinite(observed) and np.isfinite(expected)):
            sys.exit(f"{label}: nonfinite value, the comparison is not valid.")
        bad = abs(observed - expected) > atol
        print(f"{label}: observed=[{observed:.12g}] expected=[{expected:.12g}]")
        return bad


    defect = False
    print("R21 (1/m): analytic matrix from matrix_output, and tracked response at 1024 slices")
    for mode in modes():
        errors = {}
        for species in ("electron", "positron", "proton"):
            knl = expected_knl(species)
            oracle = expected_r21(knl)
            # Control: the same strength given as KNL explicitly. The analytic matrix is right, and
            # tracking agrees with the oracle to the integrator's error.
            analytic, tracked = case(mode, species, f"KNL={float(knl)!r}", 1024)
            if compare(f"{mode} {species} KNL analytic", analytic, oracle, ANALYTIC_ATOL) or compare(
                f"{mode} {species} KNL tracked", tracked, oracle, TRACK_ATOL
            ):
                sys.exit(
                    "The explicit-KNL control disagrees with the oracle: the comparison is not valid."
                )
            # Pole-tip specification: tracking uses the species rigidity, the analytic matrix does not.
            analytic, tracked = case(mode, species, f"BORE={BORE!r},BTIPL={BTIPL!r}", 1024)
            if compare(f"{mode} {species} pole-tip tracked", tracked, oracle, TRACK_ATOL):
                sys.exit("Pole-tip tracking disagrees with the oracle: the comparison is not valid.")
            defect |= compare(f"{mode} {species} pole-tip analytic", analytic, oracle, ANALYTIC_ATOL)
            if species == "positron":
                errors[1024] = abs(tracked - oracle)
        # Control: the tracking error is the second-order integration error, which shrinks fourfold
        # when the slice count doubles; it cannot explain the analytic discrepancy of order 0.1.
        _, coarse = case(mode, "positron", f"BORE={BORE!r},BTIPL={BTIPL!r}", 512)
        ratio = abs(coarse - expected_r21(expected_knl("positron"))) / errors[1024]
        print(
            f"{mode} positron tracking error, 512 slices / 1024 slices: observed={ratio:.6g} expected=4"
        )
        if not 3.9 < ratio < 4.1:
            sys.exit("The tracking error does not converge as the square of the slice count.")
    print(f"comparison: analytic tolerance {ANALYTIC_ATOL:g}, tracking tolerance {TRACK_ATOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

