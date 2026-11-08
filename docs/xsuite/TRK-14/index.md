---
title: "TRK-14: Elens residual kicks fail forward/backward tracking"
hide:
  - toc
---

# TRK-14 — Elens residual kicks fail forward/backward tracking

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Source | [`Elens_track_local_particle`, elens.h:137–147](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/elens.h#L137-L147)<br>[`Elens` class advertises `has_backtrack=True`](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elens.py#L47) |

Backtracking through an `Elens` repeats its hollow-core residual kicks instead of undoing them, so a forward-then-backward round trip doubles the residual momentum. The inverse of a constant momentum kick has the opposite sign, and an explicit inverse element with both residual kicks negated restores the initial state exactly. For a 1.4 m, 3 A, 12 kV lens with radii 1 and 4 mm and residual normalized kicks (2.3, −4.1) × 10⁻⁷, a 1 GeV/c proton in the core returns with normalized transverse momenta (`px`, `py`) = (4.600, −8.200) × 10⁻⁷ instead of zero. The annular electron-lens force round-trips correctly.

Run `python TRK-14.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-14.py"

    [Download TRK-14.py](TRK-14.py)

    ```python
    """TRK-14: Elens residual kicks fail forward/backward tracking

    Backtracking through an `Elens` repeats its hollow-core residual kicks instead of undoing them, so a
    forward-then-backward round trip doubles the residual momentum. The inverse of a constant momentum
    kick has the opposite sign, and an explicit inverse element with both residual kicks negated
    restores the initial state exactly. For a 1.4 m, 3 A, 12 kV lens with radii 1 and 4 mm and residual
    normalized kicks (2.3, −4.1) × 10⁻⁷, a 1 GeV/c proton in the core returns with normalized transverse
    momenta (`px`, `py`) = (4.600, −8.200) × 10⁻⁷ instead of zero. The annular electron-lens force
    round-trips correctly.

    The script tracks two 1 GeV/c protons, one inside the core and one in the annulus, forward through
    the lens and back with `Line.track(backtrack=True)`, and expects their transverse momenta to return
    to exactly zero with positions and momentum deviation unchanged. As controls, the annular proton
    must round-trip, and an explicit inverse with negated length and residual kicks must restore both
    protons. It runs serially on the installed prebuilt CPU kernels with kernel compilation forbidden.
    """

    from _prebuilt import CTX, identity, kernels, np, particles, report, xt

    identity()
    cases = {}
    kx, ky = 2.3e-7, -4.1e-7
    lens_length = 1.4
    lens = xt.Elens(
        _context=CTX,
        inner_radius=0.001,
        outer_radius=0.004,
        current=3.0,
        voltage=12000.0,
        elens_length=lens_length,
        residual_kick_x=kx,
        residual_kick_y=ky,
    )
    assert lens.has_backtrack is True
    for mode in ("line_backtrack", "explicit_inverse_control"):
        p = particles(x=[0.0002, 0.002], y=[-0.0001, 0.0003], delta=[0.0, 0.0])
        initial = np.stack([np.array(getattr(p, f)) for f in ("x", "px", "y", "py", "zeta", "delta")])
        if mode == "explicit_inverse_control":
            inverse = lens.copy()
            inverse.elens_length = -lens_length
            inverse.residual_kick_x = -kx
            inverse.residual_kick_y = -ky
            lens.track(p)
            after_forward = (p.px.copy(), p.py.copy())
            inverse.track(p)
        else:
            line = xt.Line(elements=[lens])
            line.build_tracker(_context=CTX)
            line.track(p)
            after_forward = (p.px.copy(), p.py.copy())
            line.track(p, backtrack=True)
        final = np.stack([np.array(getattr(p, f)) for f in ("x", "px", "y", "py", "zeta", "delta")])
        print(mode, "FORWARD px=", after_forward[0], "py=", after_forward[1])
        print(mode, "ROUNDTRIP px=", p.px, "py=", p.py, "expected=[0,0]")
        if mode == "line_backtrack":
            np.testing.assert_array_equal([p.px[1], p.py[1]], [0.0, 0.0])
            cases["core round trip px,py"] = ([p.px[0], p.py[0]], [0.0, 0.0])
        else:
            np.testing.assert_array_equal(p.px, [0.0, 0.0])
            np.testing.assert_array_equal(p.py, [0.0, 0.0])
        np.testing.assert_array_equal(final[[0, 2, 4, 5]], initial[[0, 2, 4, 5]])
    kernels()
    report(cases, rtol=4e-15, atol=0)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, serial |

