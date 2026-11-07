---
title: "COL-07: Loss-map periodic aperture cells have negative lengths"
hide:
  - toc
---

# COL-07 — Loss-map periodic aperture cells have negative lengths

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xcoll release [v0.12.5](https://github.com/xsuite/xcoll/tree/v0.12.5) (`64e13f5`) |
| Source | [`LossMap._create_aperture_pos_dict`, lossmap.py:863–888](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/lossmap.py#L863-L888)<br>[loss-density normalization](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/plot.py#L215-L230) |

For a non-interpolated loss map (`interpolation=False`), `LossMap` bounds each aperture cell by the midpoints to its neighbours without wrapping the first and last neighbours around the ring. Those two cells get negative lengths and displaced centres, whereas periodic midpoint cells must have positive lengths summing to the circumference. For apertures at 1, 4 and 9 m in a 12 m ring, the lengths are −2.5, 4 and −1.5 m instead of 3.5, 4 and 4.5 m, summing to 0 instead of 12 m, and the end-cell centres are 3.75 and 5.75 m instead of 0.75 and 8.75 m. Non-interpolated ring loss maps report wrong loss densities and positions in the boundary cells; interior cells are correct.

Run `python COL-07.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: COL-07.py"

    [Download COL-07.py](COL-07.py)

    ```python
    """COL-07: Loss-map periodic aperture cells have negative lengths

    For a non-interpolated loss map (`interpolation=False`), `LossMap` bounds each aperture cell by the
    midpoints to its neighbours without wrapping the first and last neighbours around the ring. Those
    two cells get negative lengths and displaced centres, whereas periodic midpoint cells must have
    positive lengths summing to the circumference. For apertures at 1, 4 and 9 m in a 12 m ring, the
    lengths are −2.5, 4 and −1.5 m instead of 3.5, 4 and 4.5 m, summing to 0 instead of 12 m, and the
    end-cell centres are 3.75 and 5.75 m instead of 0.75 and 8.75 m. Non-interpolated ring loss maps
    report wrong loss densities and positions in the boundary cells; interior cells are correct.

    The script tracks three 2 GeV/c protons into the three apertures, checks where they were lost, and
    builds the loss map with `correct_aperture_absorption=False` and no material or scattering engine.
    It compares the cell lengths and centres with periodic midpoint cells built from the positive cyclic
    gaps between apertures.
    """

    from _prebuilt import CTX, identity, kernels, np, report, xc, xt

    identity()
    line = xt.Line(
        elements=[
            xt.Drift(length=1),
            xt.LimitRect(min_x=-0.004, max_x=0.004),
            xt.Drift(length=3),
            xt.LimitRect(min_x=-0.002, max_x=0.002),
            xt.Drift(length=5),
            xt.LimitRect(min_x=-0.0007, max_x=0.0007),
            xt.Drift(length=3),
        ],
        element_names=["d0", "ap0", "d1", "ap1", "d2", "ap2", "d3"],
    )
    line.particle_ref = xt.Particles(p0c=2e9, pdg_id=2212)
    line.build_tracker(_context=CTX)
    p = xt.Particles(_context=CTX, p0c=2e9, pdg_id=2212, x=[0.005, 0.003, 0.001])
    line.track(p)
    order = np.argsort(p.particle_id)
    print(
        "TRACKED states=",
        np.asarray(p.state)[order],
        "at_element=",
        np.asarray(p.at_element)[order],
        "s=",
        np.asarray(p.s)[order],
    )
    np.testing.assert_array_equal(np.asarray(p.state)[order], [0, 0, 0])
    np.testing.assert_array_equal(np.asarray(p.at_element)[order], [1, 3, 5])
    np.testing.assert_array_equal(np.asarray(p.s)[order], [1, 4, 9])


    def lossmap(part, reversed=False):
        return xc.LossMap(
            line=line,
            part=part,
            line_is_reversed=reversed,
            interpolation=False,
            correct_aperture_absorption=False,
            verbose=False,
        )


    forward = lossmap(p.copy())
    # A periodic Voronoi cell has a left half-gap and a right half-gap. This
    # construction uses positive cyclic differences, without the source's neighbours.
    L = 12.0
    aperture_s = np.array([1.0, 4.0, 9.0])
    forward_gaps = np.mod(np.roll(aperture_s, -1) - aperture_s, L)
    backward_gaps = np.roll(forward_gaps, 1)
    expected_lengths = (backward_gaps + forward_gaps) / 2
    expected_centres = np.mod(aperture_s + (forward_gaps - backward_gaps) / 4, L)
    data = forward.aperture_losses
    assert list(data["name"]) == ["ap0", "ap1", "ap2"]
    print(
        "LENGTHS observed=",
        data["length"],
        "expected=",
        expected_lengths,
        "sum=",
        data["length"].sum(),
        "circumference=",
        L,
    )
    print("CENTRES observed=", data["s"], "expected=", expected_centres)
    np.testing.assert_array_equal(expected_lengths, [3.5, 4.0, 4.5])
    assert expected_lengths.sum() == L

    kernels()
    report(
        {
            "aperture lengths": (data["length"], expected_lengths),
            "aperture centres": (data["s"], expected_centres),
        },
        rtol=0,
        atol=1e-13,
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xcoll 0.12.5 (`64e13f596c87`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xcoll 0.12.5 (`64e13f596c87`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |

