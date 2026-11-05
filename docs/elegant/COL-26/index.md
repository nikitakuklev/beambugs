---
title: "COL-26: MPI TRWAKE combines different bunches when ranks skip locally empty bunches"
hide:
  - toc
---

# COL-26 — MPI TRWAKE combines different bunches when ranks skip locally empty bunches

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [local-empty skip, trwake.c:86–88](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L86-L88)<br>[global extrema, trwake.c:118–121](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L118-L121)<br>[length check, trwake.c:122–128](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L122-L128)<br>[binned check, trwake.c:171–176](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L171-L176)<br>[histogram sum, trwake.c:206–213](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L206-L213)<br>[global bunch range, lrwake.c:74–84](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L74-L84)<br>[local bunch counts, lrwake.c:149–166](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lrwake.c#L149-L166)<br>[bunch-mode input, elegant.tex:481–506](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L481-L506)<br>[bunched mode, trwake.tex:59–62](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/trwake.tex#L59-L62)<br>[BUNCHED_BEAM_MODE, track_data.c:2612](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2612)<br>[TRWAKE caller, do_tracking.c:2483–2486](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2483-L2486)<br>[CHARGE, do_tracking.c:1441–1457](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L1441-L1457)<br>[row partition, SDDS_MPI_binary.c:2235–2252](https://github.com/rtsoliday/SDDS/blob/a3f730a14c49ddd4f4d6347b42d60d34aa7cadc6/SDDSlib/SDDS_MPI_binary.c#L2235-L2252) |

With automatic bins and bunch mode, TRWAKE tracking under MPI (Pelegant) combines different bunches when each of two tracking ranks holds one whole bunch. Skipping locally empty bunches pairs reductions for different bunches; the correct same-bunch causal sum is recovered by serial tracking and by distributing both bunches across both ranks. For two 16.00 pC bunches 80.00 ps apart, the second bunch's last particle receives a horizontal slope kick of 1.364×10⁻⁶ instead of 1.670×10⁻⁷. At 800.0 ps separation, the grouped MPI run aborts even though each bunch individually fits within the 512.0 ps wake table.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)

??? example "Reproducer: COL-26.py"

    [Download COL-26.py](COL-26.py)

    ```python
    """COL-26: MPI TRWAKE combines different bunches when ranks skip locally empty bunches

    With automatic bins and bunch mode, TRWAKE tracking under MPI (Pelegant) combines different bunches
    when each of two tracking ranks holds one whole bunch. Skipping locally empty bunches pairs
    reductions for different bunches; the correct same-bunch causal sum is recovered by serial tracking
    and by distributing both bunches across both ranks. For two 16.00 pC bunches 80.00 ps apart, the
    second bunch's last particle receives a horizontal slope kick of 1.364×10⁻⁶ instead of 1.670×10⁻⁷.
    At 800.0 ps separation, the grouped MPI run aborts even though each bunch individually fits within
    the 512.0 ps wake table.

    The script tracks two 16-particle bunches through one dipole TRWAKE with a linear causal wake and
    automatic bins, with the bunches 80 ps, 160 ps and 800 ps apart. The correct kick is a direct causal
    pair sum over the particles of the same bunch. In one input row order both tracking ranks hold part
    of each bunch; in the other each of the two ranks holds one whole bunch. A second prediction sums over both bunches; serial runs and the first row order are the controls,
    and the 800 ps separation exceeds the wake table although each bunch separately fits within it.
    """

    import sys

    import numpy as np
    from _elegant import RunFailure, modes, particles, run, verdict

    REST_EV = 510999.06  # SDDS constants.h: electron rest energy, eV
    P = 36.0  # electron momentum, units of m_e c
    SLOPE = 2e25  # V/C/m per s; Wx(tau) = SLOPE tau for tau >= 0, Wy = Wx/2, W(0) = 0
    CHARGE = 16e-12  # C per bunch
    SPACING, POINTS = 2e-12, 257  # wake table: 512 ps long
    OFFSETS = np.repeat([0, 4, 8, 12], 4) * 1e-12  # s, arrival times within a bunch
    N = len(OFFSETS)  # particles per bunch
    X = np.repeat([0.8e-3, 1.6e-3], N)  # m; the second bunch is displaced twice as far
    Y = -X / 2
    RTOL, ATOL = 3e-12, 1e-19


    def wake_table():
        """ASCII SDDS table of Wx, Wy on POINTS points of spacing SPACING."""
        rows = "".join(
            f"{tau!r} {SLOPE * tau!r} {0.5 * SLOPE * tau!r}\n"
            for tau in (float(n) * SPACING for n in range(POINTS))
        )
        return (
            "SDDS1\n"
            "&column name=time, units=s, type=double &end\n"
            "&column name=wx, units=V/C/m, type=double &end\n"
            "&column name=wy, units=V/C/m, type=double &end\n"
            "&data mode=ascii, no_row_counts=1 &end\n" + rows
        )


    def times(gap):
        return np.concatenate([OFFSETS + 1e-9, OFFSETS + 1e-9 + gap])


    def kicks(mode, layout, gap):
        """Slope kicks (xp, yp) of the 2N particles, ordered by particle ID."""
        ids = np.arange(1, 2 * N + 1)
        if layout == "grouped":
            order = np.arange(2 * N)  # first rank: bunch 1, second rank: bunch 2
        else:
            # Both ranks get two of the four particles at each time of each bunch.
            half = np.arange(N).reshape(4, 4)[:, :2].ravel()
            other = np.arange(N).reshape(4, 4)[:, 2:].ravel()
            order = np.r_[half, half + N, other, other + N]
        beam = {
            "x": X[order],
            "y": Y[order],
            "t": times(gap)[order],
            "p": np.full(2 * N, P),
            "particleID": ids[order],
            "IDSlotsPerBunch": N,
        }
        lattice = (
            f"Q: CHARGE,TOTAL={2 * CHARGE!r}\n"
            'W: TRWAKE,INPUTFILE="wake.sdds",TCOLUMN="time",WXCOLUMN="wx",WYCOLUMN="wy",'
            "N_BINS=0,SMOOTHING=0,INTERPOLATE=0,BUNCHED_BEAM_MODE=1\n"
            "BL: LINE=(Q,W)\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},output=out.sdds &end\n"
            "&run_control n_steps=1,n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds,use_bunched_mode=1,track_pages_separately=0 &end\n"
            "&track &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands, "wake.sdds": wake_table()},
            mode,
            {"beam.sdds": beam},
        )
        out = particles(result["out.sdds"])
        if not np.array_equal(out["particleID"], ids):
            sys.exit(f"{mode}: particles lost")
        # The wake changes only the slopes, which start at zero.
        return np.column_stack([out["xp"], out["yp"]])


    def pair_sum(gap, bunched):
        """Slope kicks (xp, yp) of the 2N particles from a direct causal pair sum.

        Particle i receives dq'_i = (Q/N) sum_j q_j W_q(t_i - t_j)/(E0 p) for q = x, y, summed over the
        particles j that arrive no later than i (W(0) = 0), with Q/N the macroparticle charge, q_j the
        source's offset and E0 the electron rest energy in eV. Bunch mode restricts j to the particles
        of i's own bunch. The times lie on the wake table's grid, so no interpolation enters.
        """
        t = times(gap)
        delay = np.maximum(t[:, None] - t[None, :], 0.0)
        if bunched:
            bunch = np.arange(2 * N) // N
            delay = delay * (bunch[:, None] == bunch[None, :])
        scale = (CHARGE / N) * SLOPE / (REST_EV * P)
        return scale * np.column_stack([delay @ X, 0.5 * (delay @ Y)])


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values) + 0.0) + "]"


    def compare(label, observed, expected):
        """Print the kicks of the second bunch's head and tail; True when all particles agree."""
        for index in (N, 2 * N - 1):
            print(
                f"{label}, particle {index + 1}: "
                f"observed={fmt(observed[index])} expected={fmt(expected[index])}"
            )
        return bool(np.allclose(observed, expected, rtol=RTOL, atol=ATOL))


    TABLE = (POINTS - 1) * SPACING  # s, length of the wake table
    # Printed by Elegant when the global time extrema span more than the table; in MPI runs the
    # message is intermittently lost, so the exit status is the primary evidence.
    LONG_BEAM = "The beam is longer than the transverse wake function"

    defect = False
    mixing_validated = False  # a grouped MPI run followed the merged-bunch prediction
    for mode in ("serial", *modes(mpi_only=True)):
        for gap in (80e-12, 160e-12, 800e-12):
            expected = pair_sum(gap, True)
            merged = pair_sum(gap, False)
            for layout in ("interleaved", "grouped"):
                label = f"{mode} {layout} rows, bunches {gap * 1e12:g} ps apart"
                faulty = mode == "mpi" and layout == "grouped"
                try:
                    observed = kicks(mode, layout, gap)
                except RunFailure as error:
                    # The two bunches span 812 ps in all, more than the 512 ps wake table, although
                    # each is 12 ps long and a bunch never needs the delay between bunches. Only exit
                    # status 1 with a validated mixing result or the table-length diagnostic counts.
                    if not (
                        faulty
                        and gap > TABLE
                        and error.status == 1
                        and (mixing_validated or LONG_BEAM in error.log)
                    ):
                        raise
                    print(f"{label}: run exited with status 1; expected status 0")
                    defect = True
                    continue
                agrees = compare(label, observed, expected)
                if not faulty:
                    if not agrees:
                        sys.exit(f"{label}: control disagrees with the pair sum; invalid comparison")
                    continue
                if gap > TABLE:
                    # The merged-bunch prediction needs wake delays beyond the table: not evaluated.
                    if not agrees:
                        sys.exit(f"{label}: the run completed with kicks that differ from the pair sum")
                    continue
                # Cause: the ranks' histograms of different bunches are added, which is the pair sum
                # over both bunches.
                print(f"{label}, particle {2 * N}: merged-bunch prediction={fmt(merged[2 * N - 1])}")
                follows_merge = np.allclose(observed, merged, rtol=RTOL, atol=ATOL)
                if not agrees and not follows_merge:
                    sys.exit(f"{label}: kicks follow neither the pair sum nor the merged bunches")
                mixing_validated |= not agrees
                if gap == 80e-12:
                    print(
                        f"{label}: tail kick is {observed[2 * N - 1, 0] / expected[2 * N - 1, 0]:.3f} "
                        "times the correct value"
                    )
                defect |= not agrees
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source reading, not executed for this claim |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-03 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial control and 3-rank Pelegant |

