---
title: "SCT-07: Smooth IBS ignores the plane switches DO_X, DO_Y and DO_Z"
hide:
  - toc
---

# SCT-07 — Smooth IBS ignores the plane switches DO_X, DO_Y and DO_Z

| | |
|---|---|
| Code | [Elegant](../index.md) · Scattering and pressure |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [random-method plane gates, ibs_tracking.c:479–489](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibs_tracking.c#L479-L489)<br>[smooth update, ibs_tracking.c:513–516](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ibs_tracking.c#L513-L516)<br>[IBSCATTER parameters, track_data.c:2512–2515](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2512-L2515) |

With the smooth method (SMOOTH=1), an IBSCATTER element applies the intrabeam-scattering growth to all three planes whatever the plane switches DO_X, DO_Y and DO_Z say. With all three switches at 0 the beam should leave the element unchanged, as it does for FACTOR=0, and with one switch at 1 only that plane should change. In a closed ring of 16 identical bends and 256 particles, every switch setting gives the same output as the fully enabled element: the horizontal variance grows by 1.707% and the rms momentum deviation by 2.000%, where the all-off case should show 0%.

Run `python SCT-07.py` with [_elegant.py](../helpers/_elegant.py) and [_ibs_ring.py](../helpers/_ibs_ring.py) in the same directory.

??? example "Reproducer: SCT-07.py"

    [Download SCT-07.py](SCT-07.py)

    ```python
    """SCT-07: Smooth IBS ignores the plane switches DO_X, DO_Y and DO_Z

    With the smooth method (SMOOTH=1), an IBSCATTER element applies the intrabeam-scattering growth to
    all three planes whatever the plane switches DO_X, DO_Y and DO_Z say. With all three switches at 0
    the beam should leave the element unchanged, as it does for FACTOR=0, and with one switch at 1 only
    that plane should change. In a closed ring of 16 identical bends and 256 particles, every switch
    setting gives the same output as the fully enabled element: the horizontal variance grows by 1.707%
    and the rms momentum deviation by 2.000%, where the all-off case should show 0%.

    The script tracks a 256-particle beam once around a ring of sixteen identical sector bends, with
    constant matched beta functions, ending in an IBSCATTER with the smooth method and a growth-rate
    factor that makes the largest applied change 2%. It compares the beam's variances and momentum
    spread with those of a run with FACTOR=0, for all planes switched off and for each plane alone. A
    plane that is off must leave its spread equal to the FACTOR=0 value, and a plane that is on must
    change it exactly as in the run with all three planes on, which is the control.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _ibs_ring import FACTOR, beam, spreads, track

    RTOL = 1e-12  # relative; serial runs agree exactly and MPI reductions differ at about 1e-16
    MEAN_DELTA = 1e-4


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.ravel(values)) + "]"


    def compare(label, observed, expected):
        bad = not np.allclose(observed, expected, rtol=RTOL, atol=0)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    def checked(spread, positive=False):
        """The spreads of a run; exits when they are not finite (and, for a reference, positive)."""
        if not np.all(np.isfinite(spread)) or (positive and not np.all(spread > 0)):
            sys.exit(f"invalid spreads {fmt(spread)}: the comparison is not valid")
        return spread


    columns = beam(MEAN_DELTA)
    defect = False
    print("spread ratios to the FACTOR=0 run: var(x), var(x'), var(y), var(y'), rms(p)")
    for mode in modes():
        reference = checked(spreads(track(mode, columns, 0.0)), positive=True)
        full = checked(spreads(track(mode, columns, FACTOR))) / reference
        # The control must show a visible update in x and in p, or the comparison is empty.
        if full[0] < 1.01 or full[4] < 1.01:
            sys.exit(f"{mode}: the all-planes update is not visible: the comparison is not valid")
        for label, planes in (
            ("all planes off", (0, 0, 0)),
            ("x only", (1, 0, 0)),
            ("y only", (0, 1, 0)),
            ("z only", (0, 0, 1)),
        ):
            observed = checked(spreads(track(mode, columns, FACTOR, planes))) / reference
            # The smooth update of a plane does not depend on the other planes' switches, so a plane
            # that is on changes its spread as in the all-planes run, and a plane that is off keeps
            # the FACTOR=0 value (ratio 1). Columns: x and x' follow DO_X, y and y' DO_Y, rms(p) DO_Z.
            on = np.array([planes[0], planes[0], planes[1], planes[1], planes[2]], dtype=bool)
            defect |= compare(f"{mode} {label}", observed, np.where(on, full, 1.0))
    print(f"comparison: rtol={RTOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source reading, not executed for this claim |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

