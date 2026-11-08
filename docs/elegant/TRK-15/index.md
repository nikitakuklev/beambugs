---
title: "TRK-15: GKICKMAP applies classical radiation loss with SYNCH_RAD disabled"
hide:
  - toc
---

# TRK-15 — GKICKMAP applies classical radiation loss with SYNCH_RAD disabled

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [`trackKickMap`, kickmap.c:92–99](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/kickmap.c#L92-L99)<br>[`radCoef` and `isrCoef`, kickmap.c:60–61](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/kickmap.c#L60-L61)<br>[parameter dictionary, track_data.c:3469–3470](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3469-L3470)<br>[UKICKMAP coefficients, ukickmap.c:56–59](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ukickmap.c#L56-L59)<br>[GKICKMAP manual, gkickmap.tex:1–22](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/gkickmap.tex#L1-L22) |

Selecting ISR = 1 with SYNCH_RAD = 0 in GKICKMAP also applies classical radiation loss, despite the switches independently selecting excitation and mean loss. In a one-slice, 0.6 m map at reference momentum p₀ = 4200 mₑc, with entrance slope x′ = 0.004 and horizontal slope kick 0.025, the unwanted relative-momentum loss is 1.450×10⁻⁷. A separate small-angle radiation calculation agrees with the classical-only control within 10⁻¹⁶, while the radiation-free control preserves momentum. With paired random seeds, the ISR-only and both-enabled runs give identical particle momenta instead of differing by that loss.

Run `python TRK-15.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-15.py"

    [Download TRK-15.py](TRK-15.py)

    ```python
    """TRK-15: GKICKMAP applies classical radiation loss with SYNCH_RAD disabled

    Selecting ISR = 1 with SYNCH_RAD = 0 in GKICKMAP also applies classical radiation loss, despite the
    switches independently selecting excitation and mean loss. In a one-slice, 0.6 m map at reference
    momentum p₀ = 4200 mₑc, with entrance slope x′ = 0.004 and horizontal slope kick 0.025, the
    unwanted relative-momentum loss is 1.450×10⁻⁷. A separate small-angle radiation calculation agrees
    with the classical-only control within 10⁻¹⁶, while the radiation-free control preserves momentum.
    With paired random seeds, the ISR-only and both-enabled runs give identical particle momenta
    instead of differing by that loss.

    The script tracks 24 identical electrons (p0 = 4200 mₑc, entrance slope 0.004) through one slice of
    a 0.6 m GKICKMAP whose map is a constant kick of 0.025, for the four combinations of the
    SYNCH_RAD and ISR switches, all with the same random seed. It checks the classical-only case
    against a paraxial one-slice radiation model (curvature κ = KICK/L, path weighted by the post-kick
    slope; not an exact constant-curvature arc), and the no-radiation case against zero loss; these
    controls validate the oracle. It then compares the case with only ISR against the case with both
    switches, which must differ by exactly that classical loss.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h, as Elegant uses them; eps_0 = 1/(mu_0 c^2) with mu_0 = 4 pi 1e-7.
    C, QE, ME = 299792458.0, 1.60217733e-19, 9.1093897e-31
    EPS0 = 1e7 / (4 * np.pi * C**2)
    RE = QE**2 / (4 * np.pi * EPS0 * ME * C**2)  # classical electron radius, m
    P0, LENGTH, SLOPE, KICK = 4200.0, 0.6, 0.004, 0.025  # m_e c, m, rad, rad
    COPIES = 24
    SEED = 24681357
    ATOL = 2e-16  # delta is recovered from p/p0 - 1, which rounds at about 1e-16

    # Constant kick (xpFactor) on a 5 x 5 grid, y outer and x inner as the manual documents.
    GRID = [-0.05, -0.025, 0.0, 0.025, 0.05]
    MAP = (
        "SDDS1\n"
        '&column name=x, type=double, units="m" &end\n'
        '&column name=y, type=double, units="m" &end\n'
        "&column name=xpFactor, type=double &end\n"
        "&column name=ypFactor, type=double &end\n"
        "&data mode=ascii &end\n"
        f"{len(GRID) ** 2}\n" + "".join(f"{x!r} {y!r} {KICK!r} 0\n" for y in GRID for x in GRID)
    )


    def classical_loss():
        """Fractional momentum loss of one pass in the paraxial one-slice radiation model.

        The slope change KICK over L is taken as a curvature kappa = KICK/L, a small-angle
        approximation (a slope change is not exactly an angular bend). An electron radiates classically
        d(delta) = -(2/3) r_e p0^3 (1 + delta)^2 kappa^2 ds per path length ds (p0 in units of
        m_e c; the power (2/3) r_e m c^3 gamma^4 kappa^2 divided by the energy gamma m c^2). The path
        weight is L sqrt(1 + xp^2) with the slope xp = SLOPE + KICK after the kick, as for one slice
        in the code, not the exact circular path integral.
        """
        kappa = KICK / LENGTH
        return 2 / 3 * RE * P0**3 * kappa**2 * LENGTH * np.sqrt(1 + (SLOPE + KICK) ** 2)


    def momentum_change(mode, synch_rad, isr):
        """delta = p/p0 - 1 of each particle, ordered by particle ID."""
        lattice = (
            f'K: GKICKMAP, L={LENGTH!r}, N_KICKS=1, INPUT_FILE="map.sdds", '
            f"SYNCH_RAD={synch_rad}, ISR={isr}\n"
            "BL: LINE=(K)\n"
        )
        commands = (
            "&change_particle name=electron &end\n"
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
            f"output=out.sdds, random_number_seed={SEED} &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track soft_failure=0 &end\n"
        )
        beam = {"p": np.full(COPIES, P0), "xp": np.full(COPIES, SLOPE)}
        result = run(
            {"lattice.lte": lattice, "run.ele": commands, "map.sdds": MAP}, mode, {"beam.sdds": beam}
        )
        out = particles(result["out.sdds"])
        if len(out["p"]) != COPIES:
            sys.exit(f"{mode}: {COPIES - len(out['p'])} particles lost")
        if not np.all(np.isfinite(out["p"])):
            sys.exit(f"{mode} SYNCH_RAD={synch_rad} ISR={isr}: nonfinite output momenta")
        return out["p"] / P0 - 1


    def compare(label, observed, expected):
        """Print the mean over the particles; return True when every particle agrees."""
        print(f"{label}: observed=[{float(np.mean(observed))!r}] expected=[{float(expected)!r}]")
        return bool(np.allclose(observed, expected, rtol=0, atol=ATOL))


    loss = classical_loss()
    defect = False
    print("mean fractional momentum change delta of 24 electrons")
    for mode in modes():
        delta = {(sr, isr): momentum_change(mode, sr, isr) for sr in (0, 1) for isr in (0, 1)}
        if not compare(f"{mode} SYNCH_RAD=0 ISR=0", delta[0, 0], 0.0):
            sys.exit("The radiation-free control loses momentum: the comparison is not valid.")
        if not compare(f"{mode} SYNCH_RAD=1 ISR=0", delta[1, 0], -loss):
            sys.exit("The classical-only control disagrees with the curvature oracle: not valid.")
        # With one slice the random draw does not depend on the classical switch, so the particles
        # of the ISR-only run (where no classical loss should occur) sit exactly `loss` above those
        # of the run with both switches.
        print(f"{mode} SYNCH_RAD=0 ISR=1: mean delta={float(np.mean(delta[0, 1]))!r}")
        print(f"{mode} SYNCH_RAD=1 ISR=1: mean delta={float(np.mean(delta[1, 1]))!r}")
        if compare(f"{mode} ISR only minus both, paired", delta[0, 1] - delta[1, 1], loss):
            continue  # the ISR-only run lacks the classical loss: correct
        # Otherwise the two runs must be identical, the recorded unwanted loss; anything else is a
        # failed run, not a verdict.
        if not np.allclose(delta[0, 1], delta[1, 1], rtol=0, atol=ATOL):
            sys.exit(f"{mode}: the ISR-only run matches neither the paired prediction nor the both run")
        defect = True
    print(f"comparison: atol={ATOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |

