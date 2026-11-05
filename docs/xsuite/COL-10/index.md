---
title: "COL-10: Noninterpolated loss maps discard coordinate reversal"
hide:
  - toc
---

# COL-10 — Noninterpolated loss maps discard coordinate reversal

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xcoll main (`64e13f5`) |
| Source | [`LossMap._create_aperture_pos_dict`, lossmap.py:881–885](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/lossmap.py#L881-L885) |

For a reversed line (`line_is_reversed=True`), a non-interpolated loss map computes the reflected aperture position L − s, with L the circumference, and then overwrites it with the unreflected expression, so the flag has no effect on aperture positions. The intended reflection is explicit in the code. An interior aperture at 4 m, with neighbour midpoints at 2.5 and 6.5 m, has centre 4.5 m, which must become 7.5 m in a reversed 12 m ring; the reversed map reports 4.5 m.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _prebuilt.py](../helpers/_prebuilt.py) · [View source](../helpers/index.md#_prebuilt)
- [Download _common.py](../helpers/_common.py) · [View source](../helpers/index.md#_common)

??? example "Reproducer: COL-10.py"

    [Download COL-10.py](COL-10.py)

    ```python
    """COL-10: Noninterpolated loss maps discard coordinate reversal

    For a reversed line (`line_is_reversed=True`), a non-interpolated loss map computes the reflected
    aperture position L − s, with L the circumference, and then overwrites it with the unreflected
    expression, so the flag has no effect on aperture positions. The intended reflection is explicit in
    the code. An interior aperture at 4 m, with neighbour midpoints at 2.5 and 6.5 m, has centre 4.5 m,
    which must become 7.5 m in a reversed 12 m ring; the reversed map reports 4.5 m.

    The script tracks three protons into apertures at 1, 4 and 9 m and builds forward and reversed loss
    maps from these losses. It checks the forward centre of the interior aperture `ap1`, whose cell does
    not cross the ring boundary, and compares its reversed centre, looked up by name, with L minus that
    centre. It requires the installed prebuilt kernels (compilation is forbidden).
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
    reverse = lossmap(p.copy(), reversed=True)
    data = forward.aperture_losses
    L = line.get_length()
    # Compare the interior aperture independently of the periodic-boundary defect: its midpoint and
    # length do not cross the periodic boundary, so its expected centre is correct.
    revdata = reverse.aperture_losses
    rev_by_name = dict(zip(revdata["name"], revdata["s"]))
    print(
        "INTERIOR ap1 forward=",
        data["s"][1],
        "reverse=",
        rev_by_name["ap1"],
        "expected_reverse=",
        L - data["s"][1],
    )
    assert data["s"][1] == 4.5
    assert L - data["s"][1] == 7.5

    kernels()
    report({"reversed interior centre": (rev_by_name["ap1"], L - data["s"][1])}, rtol=0, atol=0)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xcoll 0.12.5 (`64e13f596c87`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xcoll 0.12.5 (`64e13f596c87`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |

