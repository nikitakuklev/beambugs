---
title: "COL-18: SCMULT size-filter history is shared between bunches"
hide:
  - toc
---

# COL-18 — SCMULT size-filter history is shared between bunches

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`, head build) |
| Source | [bunch loop, insertSCeffects.cc:316–323](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L316-L323)<br>[lastSigma, track.h:3458](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track.h#L3458)<br>[averaging_factor, elegant.tex:4492–4503](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4492-L4503) |

Nonlinear SCMULT bunch-train tracking with size-averaging factor 0 < f < 1 uses one shared size history per element: after the first turn, each bunch filters against history last updated by another bunch instead of its own previous turn. In this independent-bunch model, changing another bunch must not change the kick; an isolated bunch checked against an independent Gaussian-field integral supplies the control. At momentum 17 mₑc, charge 0.2000 pC per bunch and f = 0.4000, making a second bunch 2.500 times wider changes the first bunch's second-turn horizontal slope kick on its first particle from 2.510×10⁻⁹ to 4.532×10⁻⁹; making it 2.500 times longer changes it to 2.733×10⁻⁹.

Run `python COL-18.py` with [_elegant.py](../helpers/_elegant.py), [_scmult_ring.py](../helpers/_scmult_ring.py) and [_scring.py](../helpers/_scring.py) in the same directory.

??? example "Reproducer: COL-18.py"

    [Download COL-18.py](COL-18.py)

    ```python
    """COL-18: SCMULT size-filter history is shared between bunches

    Nonlinear SCMULT bunch-train tracking with size-averaging factor 0 < f < 1 uses one shared size
    history per element: after the first turn, each bunch filters against history last updated by
    another bunch instead of its own previous turn. In this independent-bunch model, changing another
    bunch must not change the kick; an isolated bunch checked against an independent Gaussian-field
    integral supplies the control. At momentum 17 mₑc, charge 0.2000 pC per bunch and f = 0.4000, making
    a second bunch 2.500 times wider changes the first bunch's second-turn horizontal slope kick on its
    first particle from 2.510×10⁻⁹ to 4.532×10⁻⁹; making it 2.500 times longer changes it to
    2.733×10⁻⁹.

    The script tracks two turns of a four-sector ring with SCMULT, first one matched bunch alone and
    then the same bunch followed by a second bunch that is wider or longer, and reads the first bunch's
    slope kick on the second turn. It compares that kick with the isolated bunch's kick, with a
    Poisson-integral Gaussian-field oracle that filters each bunch against its own history, and with
    the same oracle filtering against the other bunch's size. Controls are f = 1.000, which bypasses
    filtering, and an identical second bunch. A reversed bunch order is also run.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _scmult_ring import BETA, QE, RE, STRAIGHT, C, P, fmt, optics, rms, track, train
    from _scring import poisson_field

    TIMES = np.array([-3, -2, -1, 1, 2, 3]) * 17e-12  # s, arrival times within a bunch
    CHARGE = 2e-13  # C per bunch
    N = 96  # particles per bunch: 16 transverse phase combinations at 6 arrival times
    ATOL = 5e-18  # slope; the controls agree exactly
    RTOL_ORACLE, ATOL_ORACLE = 5e-7, 2e-16  # native Gaussian-force kernel accuracy


    def oracle(entrance, exit_, sigma_hat):
        """Slope kick (xp, yp) of every particle of one bunch, with filtered sizes `sigma_hat`.

        SCMULT's nonlinear kick is dx' = K lambda(z) (D/beta_kick) sigma_x (sigma_x + sigma_y) F_x(x, y),
        with F the Gaussian transverse field of rms sizes sigma_hat (normalized to x/[sx (sx + sy)]
        near the axis; here the Poisson integral, independent of Elegant's error-function kernel),
        K lambda = sqrt(2/pi) r_e N_e/(p^2 gamma sz) exp(-z^2/(2 sz^2)), N_e = Q/e, and D the
        trapezoid of beta/[sigma_x (sigma_x + sigma_y)] over the drift from the true (unfiltered)
        entrance and exit moments. The matched beta is the same at both ends of the full straight.
        """
        before, after = rms(entrance)[:2], rms(exit_)[:2]
        beta0, _, beta1 = optics(STRAIGHT)
        weight = STRAIGHT / 2 * (beta0 / (before * before.sum()) + beta1 / (after * after.sum()))
        sx, sy, sz = sigma_hat
        xy = np.column_stack([exit_["x"], exit_["y"]])
        centered = xy - xy.mean(axis=0)
        field = poisson_field(centered[:, 0], centered[:, 1], sx, sy)
        z = BETA * C * (exit_["t"] - exit_["t"].mean())
        strength = np.sqrt(2 / np.pi) * RE * (CHARGE / QE) / (P**2 * np.hypot(P, 1))
        line = strength / sz * np.exp(-(z**2) / (2 * sz**2))
        return line[:, None] * weight / beta1 * np.array([sx, sy]) * (sx + sy) * field


    def second_turn_kick(entrance, exit_, rows):
        """Slope change (xp, yp) of the particles `rows` during the second turn's SCMULT kick."""
        return np.column_stack([exit_[1][key][rows] - entrance[1][key][rows] for key in ("xp", "yp")])


    def pages(page, rows):
        return {key: value[rows] for key, value in page.items()}


    defect = False
    first, other = slice(0, N), slice(N, 2 * N)
    for mode in modes():
        for factor in (0.4, 0.7, 1.0):
            single = train(((1, 1),), TIMES, matched=True)
            s_in, s_out = track(mode, single, 1, charge=CHARGE, passes=2, nonlinear=1, factor=factor)
            expected = second_turn_kick(s_in, s_out, slice(None))
            # Filtered size for the isolated bunch: its own size on this turn and on the last turn.
            own = factor * rms(s_out[1]) + (1 - factor) * rms(s_out[0])
            own_oracle = oracle(s_in[1], s_out[1], own)
            if not np.allclose(expected, own_oracle, rtol=RTOL_ORACLE, atol=ATOL_ORACLE):
                sys.exit(f"{mode} f={factor}: the isolated kick disagrees with the field oracle")
            for label, specs in (
                ("identical second bunch", ((1, 1), (1, 1))),
                ("second bunch 2.5 times wider", ((1, 1), (2.5, 1))),
                ("second bunch 2.5 times longer", ((1, 1), (1, 2.5))),
            ):
                columns = train(specs, TIMES, matched=True)
                e_in, e_out = track(
                    mode, columns, 2, charge=CHARGE, passes=2, nonlinear=1, factor=factor
                )
                # Neither bunch can affect the other's coordinates before the second-turn kick.
                for key in ("x", "xp", "y", "yp"):
                    if not np.allclose(e_in[1][key][first], s_in[1][key], rtol=0, atol=2e-17):
                        sys.exit(f"{mode} f={factor} {label}: the first bunch's entrance changed")
                observed = second_turn_kick(e_in, e_out, first)
                # The shared history holds the second bunch's size from the last turn.
                shared = factor * rms(e_out[1], first) + (1 - factor) * rms(e_out[0], other)
                shared_oracle = oracle(pages(e_in[1], first), pages(e_out[1], first), shared)
                bad = not np.allclose(observed, expected, rtol=0, atol=ATOL)
                # The shared-history oracle diagnoses a deviation from the isolated kick; a bunch
                # that matches the isolated kick is accepted (a corrected own-history filter).
                if bad and not np.allclose(observed, shared_oracle, rtol=RTOL_ORACLE, atol=ATOL_ORACLE):
                    sys.exit(f"{mode} f={factor} {label}: shared-history prediction fails")
                print(
                    f"{mode} f={factor} {label}: observed={fmt(observed[0])} expected={fmt(expected[0])} "
                    f"shared-history prediction={fmt(shared_oracle[0])} "
                    f"max difference={np.max(np.abs(observed - expected)):.6g}"
                )
                if factor == 1.0 or specs[1] == (1, 1):
                    if bad:
                        sys.exit(f"{mode} f={factor} {label}: control differs from the isolated bunch")
                else:
                    defect |= bad
        # Reversed bunch order: the wide bunch has the first IDs and the unit bunch the last.
        factor = 0.4
        single = train(((1, 1),), TIMES, matched=True)
        s_in, s_out = track(mode, single, 1, charge=CHARGE, passes=2, nonlinear=1, factor=factor)
        expected = second_turn_kick(s_in, s_out, slice(None))
        columns = train(((2.5, 1), (1, 1)), TIMES, matched=True, reverse=True)
        e_in, e_out = track(mode, columns, 2, charge=CHARGE, passes=2, nonlinear=1, factor=factor)
        # The unit bunch's incoming coordinates must equal the isolated run's before the kick.
        for key in ("x", "xp", "y", "yp"):
            if not np.allclose(e_in[1][key][other], s_in[1][key], rtol=0, atol=2e-17):
                sys.exit(f"{mode} reversed order: the unit bunch's entrance changed")
        observed = second_turn_kick(e_in, e_out, other)
        # The first-ID bunch reads the last turn's last bunch; the last-ID bunch then reads that
        # bunch's filtered size.
        wide = factor * rms(e_out[1], first) + (1 - factor) * rms(e_out[0], other)
        shared = factor * rms(e_out[1], other) + (1 - factor) * wide
        shared_oracle = oracle(pages(e_in[1], other), pages(e_out[1], other), shared)
        bad = not np.allclose(observed, expected, rtol=0, atol=ATOL)
        if bad and not np.allclose(observed, shared_oracle, rtol=RTOL_ORACLE, atol=ATOL_ORACLE):
            sys.exit(f"{mode} reversed order: shared-history prediction fails")
        print(
            f"{mode} f={factor} reversed bunch order, 2.5 times wider first bunch: "
            f"observed={fmt(observed[0])} expected={fmt(expected[0])} "
            f"shared-history prediction={fmt(shared_oracle[0])} "
            f"max difference={np.max(np.abs(observed - expected)):.6g}"
        )
        defect |= bad
    print(f"comparison: atol={ATOL:g} against the isolated bunch; field oracle rtol={RTOL_ORACLE:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

