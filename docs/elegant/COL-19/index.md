---
title: "COL-19: Empty SCMULT slice centroids break translation symmetry"
hide:
  - toc
---

# COL-19 — Empty SCMULT slice centroids break translation symmetry

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [calloc, insertSCeffects.cc:386–390](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L386-L390)<br>[occupied-bin guard, insertSCeffects.cc:411](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L411)<br>[interpolation, insertSCeffects.cc:464–466](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L464-L466)<br>[slice parameters, elegant.tex:4504–4511](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4504-L4511)<br>[SDDS loading, sdds_beam.c:446](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_beam.c#L446)<br>[drift, compute_matrices.c:614](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/compute_matrices.c#L614)<br>[`computeTimeCoordinatesOnly`, trwake.c:539–549](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L539-L549) |

Linear SCMULT tracking with time slicing and slice interpolation leaves empty slices' transverse centroids at zero instead of using the whole-bunch centroids specified by the manual. Mixing those values with occupied slices breaks translation symmetry: a common transverse displacement changes the self-kicks, whose mean must remain zero for the symmetric test bunch. For 20 electrons carrying 0.2000 nC in 100.0 ps slices with one empty neighbor, a displacement of (1.700, −2.200) mm produces a mean horizontal slope of 5.597×10⁻⁶ instead of zero. Extrapolation can also reverse the erroneous mean.

Run `python COL-19.py` with [_elegant.py](../helpers/_elegant.py) and [_scmult_ring.py](../helpers/_scmult_ring.py) in the same directory.

??? example "Reproducer: COL-19.py"

    [Download COL-19.py](COL-19.py)

    ```python
    """COL-19: Empty SCMULT slice centroids break translation symmetry

    Linear SCMULT tracking with time slicing and slice interpolation leaves empty slices' transverse
    centroids at zero instead of using the whole-bunch centroids specified by the manual. Mixing those
    values with occupied slices breaks translation symmetry: a common transverse displacement changes
    the self-kicks, whose mean must remain zero for the symmetric test bunch. For 20 electrons carrying
    0.2000 nC in 100.0 ps slices with one empty neighbor, a displacement of (1.700, −2.200) mm produces a
    mean horizontal slope of 5.597×10⁻⁶ instead of zero. Extrapolation can also reverse the erroneous mean.

    The script tracks one turn of a four-sector ring with SCMULT and a cold, transversely symmetric
    bunch, centered and shifted, and compares the shifted kicks and their mean with the centered ones,
    which translation symmetry requires to be equal. It reconstructs Elegant's slice grid to predict the
    spurious mean from the empty neighbor's interpolation weight. Controls are slice widths that leave
    no empty bin, interpolation off and no slicing; an interior-gap bunch shows the effect with
    interpolation fractions reaching 1.5.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _scmult_ring import QE, RE, STRAIGHT, SX, SY, C, P, fmt, track, train

    CHARGE = 2e-10  # C
    SHIFT = (1.7e-3, -2.2e-3)  # m, common transverse translation
    ATOL, RTOL = 2e-17, 3e-12  # slope; the translated and centered kicks of a control agree to ~1e-20

    # Twenty particles: the four transverse corners at five arrival times, or six for the gap bunch.
    TIMES = np.array([-2, -1, 0, 1, 2]) * 9e-12
    GAP_TIMES = np.array([-250, -240, -230, 230, 240, 250]) * 1e-12


    def slice_grid(times, duration):
        """Reconstruct SCMULT's slices for a cold bunch: counts, interpolation fractions, mean factor.

        The arrival time SCMULT uses is not `times`: the SDDS loader stores s = t c p/gamma, the cold
        drift adds the straight's length, and SCMULT recomputes t = s gamma/(c p). Rounding then
        decides the slice of particles exactly on a slice boundary (the earliest particles always are:
        tmin is half a slice before them). Slice j is centered on tmin + j duration; a particle in it
        is interpolated between slice j and j+1 (j-1 and j if it is before the center), with
        fraction = offset from the lower center / duration.
        The mean factor is mean(q (1 - c)) over particles, where q is the interpolated charge fraction
        and c the interpolated centroid factor, 0 for an empty slice and 1 for an occupied one.
        """
        gamma = np.sqrt(P * P + 1)
        times = np.repeat(times, 4)  # four transverse corners per arrival time
        s = times * C * P / gamma + STRAIGHT
        t = s * gamma / (C * P)
        tmin, tmax = t.min() - duration / 2, t.max() + duration / 2
        n_slices = int((tmax - tmin) / duration + 1)
        bins = ((t - tmin) / duration + 0.5).astype(int)
        counts = np.bincount(bins, minlength=n_slices)
        if len(counts) != n_slices:
            raise RuntimeError("particle beyond the last slice")
        lower = bins.copy()
        offset = t - (tmin + lower * duration)
        before = ((offset < 0) & (lower > 0)) | (lower == n_slices - 1)
        lower[before] -= 1
        offset[before] += duration
        fraction = offset / duration
        occupied = (counts != 0).astype(float)
        centroid = (1 - fraction) * occupied[lower] + fraction * occupied[lower + 1]
        charge = ((1 - fraction) * counts[lower] + fraction * counts[lower + 1]) / len(t)
        return counts, fraction, float(np.mean(charge * (1 - centroid)))


    def mean_kick(mode, times, duration, interpolate, shift):
        """Slope kicks (xp, yp) of every particle for a bunch translated by `shift`."""
        columns = train(((1, 1),), times, shift=shift)
        _, exit_ = track(mode, columns, 1, charge=CHARGE, duration=duration, interpolate=interpolate)
        # The bunch starts with zero slopes and the straight is a drift, so the exit slopes are kicks.
        return np.column_stack([exit_[0]["xp"], exit_[0]["yp"]])


    def predicted_mean(duration, shift, mean_factor):
        """Mean slope from the interpolated empty-slice centroid.

        For a cold beam translated by a, the interpolated centroid is c a instead of a, and the
        interpolated slice charge fraction is q. The linear kick of SCMULT's uniform slices
          dx' = K q Q L (x - c a)/(c_light Delta_t sx (sx + sy)),   K = 2 r_e/(e p^2 gamma),
        averaged over the symmetric corners (mean x - a = 0) leaves K Q L a mean(q (1 - c))/(c_light
        Delta_t sx (sx + sy)), with K = sqrt(2 pi) c1 and c1 = sqrt(2/pi) r_e/(e p^2 gamma). The optical
        weight reduces to L because beta is equal at both ends of the full straight.
        """
        c1 = np.sqrt(2 / np.pi) * RE / QE / (P**2 * np.hypot(P, 1))
        scale = c1 * CHARGE / (duration * C) * np.sqrt(2 * np.pi) * STRAIGHT * mean_factor
        return scale * np.array(shift) / (np.array([SX, SY]) * (SX + SY))


    CASES = (
        # label, arrival times, slice duration, interpolation, shifts
        ("100 ps slices, interpolation", TIMES, 100e-12, 1, (SHIFT, (-3.4e-3, 4.4e-3))),
        ("160 ps slices, interpolation", TIMES, 160e-12, 1, (SHIFT,)),
        ("120 ps slices, interpolation (control)", TIMES, 120e-12, 1, (SHIFT,)),
        ("180 ps slices, interpolation (control)", TIMES, 180e-12, 1, (SHIFT,)),
        ("100 ps slices, no interpolation (control)", TIMES, 100e-12, 0, (SHIFT,)),
        ("no slicing (control)", TIMES, 0.0, 0, (SHIFT,)),
        (
            "interior gap, 100 ps slices, interpolation",
            GAP_TIMES,
            100e-12,
            1,
            (SHIFT, (-3.4e-3, 4.4e-3)),
        ),
        ("interior gap, 100 ps slices, no interpolation (control)", GAP_TIMES, 100e-12, 0, (SHIFT,)),
    )

    defect = False
    for mode in modes():
        for label, times, duration, interpolate, shifts in CASES:
            centered = mean_kick(mode, times, duration, interpolate, (0.0, 0.0))
            # The symmetric centered bunch must have zero mean kick; this is the symmetry control.
            if not np.allclose(centered.mean(axis=0), 0, rtol=0, atol=ATOL):
                sys.exit(f"{mode} {label}: the centered bunch's mean kick is not zero")
            means = []
            for shift in shifts:
                kicks = mean_kick(mode, times, duration, interpolate, shift)
                mean = kicks.mean(axis=0)
                # Expected: translation symmetry preserves every kick, so the mean stays that of the
                # centered bunch (zero by symmetry).
                bad = not np.allclose(kicks, centered, rtol=0, atol=ATOL)
                counts = np.array([])
                if duration:
                    counts, fraction, mean_factor = slice_grid(times, duration)
                print(
                    f"{mode} {label}, shift {fmt(np.array(shift) * 1e3)} mm, "
                    f"particles per slice {counts.astype(int).tolist()}: "
                    f"observed mean={fmt(mean)} expected={fmt(centered.mean(axis=0))}"
                )
                if not (duration and interpolate and (counts == 0).any()):
                    if bad:
                        sys.exit(f"{mode} {label}: control breaks translation symmetry")
                    continue
                if not bad:
                    continue  # translation symmetry holds: a valid NOT REPRODUCED outcome
                # Cause: the empty slice's zero centroid, weighted by the interpolation fractions.
                prediction = predicted_mean(duration, shift, mean_factor)
                print(
                    f"{mode} {label}: empty-slice prediction={fmt(prediction)}, "
                    f"interpolation fraction range={fraction.min():.4g}..{fraction.max():.4g}"
                )
                if not np.allclose(mean, prediction, rtol=RTOL, atol=ATOL):
                    sys.exit(f"{mode} {label}: the empty-centroid prediction disagrees")
                means.append(mean)
                defect = True
            if len(means) == 2 and not np.allclose(means[1], -2 * means[0], rtol=RTOL, atol=ATOL):
                sys.exit(f"{mode} {label}: the spurious mean does not reverse and double")
    print(f"comparison: kicks atol={ATOL:g}; prediction rtol={RTOL:g}, atol={ATOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

