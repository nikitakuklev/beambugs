---
title: "COL-20: MPI SCMULT mixes bunch statistics when a rank holds no particles of a bunch"
hide:
  - toc
---

# COL-20 — MPI SCMULT mixes bunch statistics when a rank holds no particles of a bunch

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [trackThroughSCMULT skip, insertSCeffects.cc:253–261](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L253-L261)<br>[initializeSCMULT skip, insertSCeffects.cc:680–687](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L680-L687)<br>[accumulateSCMULT skip, insertSCeffects.cc:810–818](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L810-L818)<br>[global bunch range, lrwake.c:74–84](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L74-L84)<br>[local bunch counts, lrwake.c:149–166](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L149-L166)<br>[charge reduction, insertSCeffects.cc:289–300](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L289-L300)<br>[computeRmsCoordinate, insertSCeffects.cc:874–930](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L874-L930)<br>[initializeSCMULT, do_tracking.c:804–813](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L804-L813)<br>[accumulateSCMULT, do_tracking.c:2964–2975](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2964-L2975)<br>[trackThroughSCMULT, do_tracking.c:2188–2191](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2188-L2191)<br>[bunch-mode input, elegant.tex:481–506](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L481-L506)<br>[linear force, scmult.tex:12–22](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/scmult.tex#L12-L22)<br>[row partition, SDDS_MPI_binary.c:2235–2252](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/SDDSlib/SDDS_MPI_binary.c#L2235-L2252) |

When two bunches are distributed one per tracking rank, linear unsliced SCMULT space-charge tracking under MPI (Pelegant) merges their statistics. Reordering the same input rows changes the kicks; serial tracking and MPI with both bunches on both ranks agree with the manual's separate-bunch force. For two 30.00 pC bunches 0.4000 ns apart, the first particle's outward horizontal slope-kick magnitude is 2.130×10⁻⁷ rather than 4.630×10⁻⁷, 53.99% too small; at 0.8000 ns it is 73.49% too small.

Run `python COL-20.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: COL-20.py"

    [Download COL-20.py](COL-20.py)

    ```python
    """COL-20: MPI SCMULT mixes bunch statistics when a rank holds no particles of a bunch

    When two bunches are distributed one per tracking rank, linear unsliced SCMULT space-charge
    tracking under MPI (Pelegant) merges their statistics. Reordering the same input rows changes the
    kicks; serial tracking and MPI with both bunches on both ranks agree with the manual's
    separate-bunch force. For two 30.00 pC bunches 0.4000 ns apart, the first particle's outward
    horizontal slope-kick magnitude is 2.130×10⁻⁷ rather than 4.630×10⁻⁷, 53.99% too small; at
    0.8000 ns it is 73.49% too small.

    Setup: always runs 3-rank Pelegant (Open MPI `mpirun` on `PATH`) beside serial Elegant, because the
    defect occurs only under MPI.

    The script tracks two identical 24-particle bunches through one SCMULT of a closed four-sector ring,
    once with the input rows interleaved, so that every tracking rank holds half of each bunch, and once
    with the rows grouped, so that each of two tracking ranks holds one whole bunch. It compares the
    first-pass slope kicks with the manual's linear Gaussian force evaluated for each bunch alone; a
    second prediction treats both bunches as one merged distribution. Serial runs with either row order
    and the interleaved MPI run are the controls.
    """

    import itertools
    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h, as Elegant uses them.
    C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
    P = 23.0  # electron momentum, units of m_e c
    BETA = P / np.hypot(1, P)
    SX, SY = 0.9e-3, 1.6e-3  # m, rms sizes of each bunch (population standard deviation)
    TIMES = np.array([-5, -3, -1, 1, 3, 5]) * 10e-12  # s, arrival times within a bunch
    # A cold bunch: every combination of x = +-SX, y = +-SY and the arrival times, zero slopes.
    X, Y, T = np.array(list(itertools.product([-SX, SX], [-SY, SY], TIMES))).T
    N = len(X)
    CHARGE = 30e-12  # C per bunch
    STRAIGHT, BEND_LENGTH = 1.0, 1.2  # m
    H = np.pi / (2 * BEND_LENGTH)  # curvature of a 90-degree sector bend
    GAPS = (0.4e-9, 0.8e-9)  # s between the bunches
    RTOL, ATOL = 3e-12, 2e-18


    def matched_beta_change() -> float:
        """Relative change of the matched beta function across the straight, from matrices.

        One cell is the straight followed by a sector bend with K1 = -h^2/2, which focuses equally in
        both planes with k = h/sqrt(2). The periodic Courant-Snyder parameters follow from the one-cell
        matrix M: sin(mu) = sqrt(1 - (tr M/2)^2), beta0 = M12/sin(mu), alpha0 = (M11 - M22)/(2 sin(mu)),
        gamma0 = -M21/sin(mu); a drift of length s transports beta to beta0 - 2 alpha0 s + gamma0 s^2.
        """
        k = H / np.sqrt(2)
        phase = k * BEND_LENGTH
        bend = np.array([[np.cos(phase), np.sin(phase) / k], [-k * np.sin(phase), np.cos(phase)]])
        cell = bend @ np.array([[1, STRAIGHT], [0, 1]])
        sine = np.sqrt(1 - (np.trace(cell) / 2) ** 2)
        beta0 = cell[0, 1] / sine
        alpha0 = (cell[0, 0] - cell[1, 1]) / (2 * sine)
        gamma0 = -cell[1, 0] / sine
        return (-2 * alpha0 * STRAIGHT + gamma0 * STRAIGHT**2) / beta0


    def kicks(mode, layout, gap):
        """Slope kicks (xp, yp) after SCMULT of the 2N particles, ordered by particle ID."""
        t = np.concatenate([T, T + gap])
        ids = np.arange(1, 2 * N + 1)
        # Interleaved rows give every tracking rank half of each bunch; grouped rows give the first
        # rank bunch 1 and the second rank bunch 2 (the three-rank run has two tracking ranks).
        order = np.arange(2 * N) if layout == "grouped" else np.arange(2 * N).reshape(2, N).T.ravel()
        beam = {
            "x": np.tile(X, 2)[order],
            "y": np.tile(Y, 2)[order],
            "t": t[order],
            "p": np.full(2 * N, P),
            "particleID": ids[order],
            "IDSlotsPerBunch": N,  # SDDS parameter: bunch b holds IDs b*N+1 ... (b+1)*N
        }
        lattice = (
            f"Q: CHARGE,TOTAL={2 * CHARGE!r}\n"
            f"DS: DRIF,L={STRAIGHT!r}\n"
            'W: WATCH,FILENAME="exit.sdds",MODE="coordinates"\n'
            f"DA: DRIF,L={STRAIGHT!r}\n"
            f"B: SBEND,L={BEND_LENGTH!r},ANGLE={np.pi / 2!r},K1={-(H**2) / 2!r}\n"
            "BL: LINE=(Q,DS,W,B,DA,B,DA,B,DA,B)\n"
        )
        # insert_sceffects places one SCMULT after DS; it integrates the optics over DS alone.
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            '&insert_sceffects name="DS",skip=1,horizontal=1,vertical=1,nonlinear=0 &end\n'
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},default_order=1,"
            "output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&twiss_output matched=1 &end\n"
            "&sdds_beam input=beam.sdds,use_bunched_mode=1,track_pages_separately=0 &end\n"
            "&track &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": beam},
            outputs=("exit.sdds",),
        )
        out = particles(result["exit.sdds"])
        if not np.array_equal(out["particleID"], ids):
            sys.exit(f"{mode}: particles lost before the kick")
        # Initial slopes are zero, so the slopes after SCMULT are the kicks.
        return np.column_stack([out["xp"], out["yp"]])


    def linear_kick(x, y, t, charge):
        """Manual's linear Gaussian force integrated over the straight, for one distribution.

        dx' = K L exp(-z^2/2 sz^2)/(sqrt(2 pi) sz) * x/[sx (sx + sy)], K = 2 N r_e/(gamma^3 beta^2),
        gamma^3 beta^2 = p^2 sqrt(1 + p^2), z = beta c (t - mean t); y has x/[sx (sx + sy)] replaced by
        y/[sy (sx + sy)]. The beta function is the same at both ends of the straight, so SCMULT's
        trapezoidal optical weight, divided by the beta at the kick, leaves exactly L.
        """
        z = BETA * C * (t - t.mean())
        sz = np.std(z)
        sx, sy = np.std(x), np.std(y)
        strength = 2 * (charge / QE) * RE / (P**2 * np.sqrt(1 + P**2))
        line = strength * STRAIGHT * np.exp(-(z**2) / (2 * sz**2)) / (np.sqrt(2 * np.pi) * sz)
        return np.column_stack([line * x / (sx * (sx + sy)), line * y / (sy * (sx + sy))])


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"


    def compare(label, observed, expected):
        """Print the first particle's (xp, yp) kick; True when all particles agree."""
        print(f"{label}: observed={fmt(observed[0])} expected={fmt(expected[0])}")
        return bool(np.allclose(observed, expected, rtol=RTOL, atol=ATOL))


    if abs(matched_beta_change()) > 1e-12:
        sys.exit(
            "The beta functions at the straight's endpoints differ; "
            "the expected kick needs the trapezoidal optical weight."
        )

    defect = False
    print("first-particle kicks (xp, yp)")
    for mode in ("serial", *modes(mpi_only=True)):
        for gap in GAPS:
            for layout in ("interleaved", "grouped"):
                label = f"{mode} {layout} rows, bunches {gap * 1e9:g} ns apart"
                observed = kicks(mode, layout, gap)
                # Each bunch alone, then both bunches treated as one merged distribution.
                x, y, t = np.tile(X, 2), np.tile(Y, 2), np.concatenate([T, T + gap])
                alone = np.vstack([linear_kick(X, Y, T, CHARGE), linear_kick(X, Y, T + gap, CHARGE)])
                merged = linear_kick(x, y, t, 2 * CHARGE)
                agrees = compare(label, observed, alone)
                if mode == "mpi" and layout == "grouped":
                    # Cause: each rank's moments are those of a different bunch's sums merged.
                    print(f"{label}: merged-distribution prediction={fmt(merged[0])}")
                    follows_merge = np.allclose(observed, merged, rtol=RTOL, atol=ATOL)
                    if not agrees and not follows_merge:
                        sys.exit(f"{label}: kicks follow neither the bunches alone nor their merger")
                    defect |= not agrees
                    print(
                        f"{label}: first-particle kick is {1 - observed[0, 0] / alone[0, 0]:.4%} too weak"
                    )
                elif not agrees:
                    sys.exit(f"{label}: control disagrees with the force; the comparison is invalid")
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-29</span> | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-03</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial control and 3-rank Pelegant |

