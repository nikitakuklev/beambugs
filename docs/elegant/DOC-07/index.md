---
title: "DOC-07: WAKE manual says ALLOW_LONG_BEAM ignores the beam's far part, but Elegant keeps the whole beam"
hide:
  - toc
---

# DOC-07 — WAKE manual says ALLOW_LONG_BEAM ignores the beam's far part, but Elegant keeps the whole beam

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Source | [WAKE manual, wake.tex:34–38](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/wake.tex#L34-L38)<br>[long-beam check, wake.c:154–168](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L154-L168)<br>[wake.c:176–181](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L176-L181)<br>[bin range, wake.c:200–207](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L200-L207)<br>[wake.c:514–529](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L514-L529)<br>[track_data.c:2540](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2540) |

The WAKE manual says that when the beam is longer than the wake data and ALLOW_LONG_BEAM is nonzero, Elegant warns and ignores the part of the beam furthest from the mean arrival time in computing the wake. Elegant warns but bins and kicks the whole beam; the trimming is commented out in the source. For a 0.8 nC beam of 90 electrons at t = 0 and satellite groups of 10 at 50 ps and 51 ps, with 20 ps of wake data, the manual's algorithm keeps only the main group and gives no loss, whereas the 51 ps group loses 2182 eV, the whole-beam pair sum.

Run `python DOC-07.py` with [_elegant.py](../helpers/_elegant.py) and [_wake.py](../helpers/_wake.py) in the same directory.

??? example "Reproducer: DOC-07.py"

    [Download DOC-07.py](DOC-07.py)

    ```python
    """DOC-07: WAKE manual says ALLOW_LONG_BEAM ignores the beam's far part, but Elegant keeps the whole beam

    The WAKE manual says that when the beam is longer than the wake data and ALLOW_LONG_BEAM is nonzero, Elegant warns and ignores the part of the beam furthest from the mean arrival time in computing the wake. Elegant warns but bins and kicks the whole beam; the trimming is commented out in the source. For a 0.8 nC beam of 90 electrons at t = 0 and satellite groups of 10 at 50 ps and 51 ps, with 20 ps of wake data, the manual's algorithm keeps only the main group and gives no loss, whereas the 51 ps group loses 2182 eV, the whole-beam pair sum.

    The script tracks 90 electrons at t = 0 and two satellite groups of 10 at 50 ps and 51 ps through
    a WAKE whose table covers only 20 ps (W(0) = 0, W = 3e13 V/C for 0 < t <= 20 ps), with
    ALLOW_LONG_BEAM=1 and, as the control, with ALLOW_LONG_BEAM=0. It compares the losses with the
    manual's algorithm, which trims the beam to the 20 ps nearest its mean arrival time and so
    ignores both satellite groups, and with the direct pair sum over the whole beam.
    """

    import numpy as np
    from _elegant import verdict
    from _wake import RunFailure, beam, loss, table, track

    P0, CHARGE, W_PLUS = 2000.0, 0.8e-9, 3e13  # m_e c, C, V/C
    grid = np.arange(201) * 0.1e-12  # s; the wake data last 20 ps
    wake = np.where(grid > 0, W_PLUS, 0.0)  # W(0) = 0 keeps the own-bin term out of the comparison
    TABLE = table({"t": grid, "W": wake}, {"t": "s", "W": "V/C"})
    TIMES = np.r_[np.zeros(90), np.full(10, 50e-12), np.full(10, 51e-12)]
    Q = CHARGE / len(TIMES)  # macroparticle charge

    # Manual (doc/wake.tex, step 2): the beam spans 51 ps > 20 ps, and "that part of the beam that is
    # furthest from t_mean is ignored". t_mean = (10*50 + 10*51)/110 = 9.18 ps, so the far end is the
    # tail: keeping the 20 ps from t_min retains the 90 particles at t = 0 and ignores both satellite
    # groups; with W(0) = 0 nobody then loses energy.
    expected_manual = np.zeros(len(TIMES))
    # Whole-beam pair sum: a particle loses Q W(t_i - t_j) for each particle j ahead of it within the
    # 20 ps of data; only the 51 ps group has such sources (the 50 ps group, 1 ps ahead).
    delay = TIMES[:, None] - TIMES[None, :]
    pair_sum = Q * np.where((delay > 0) & (delay <= grid[-1]), W_PLUS, 0.0).sum(axis=1)

    element = 'W1: WAKE,INPUTFILE="wake.sdds",TCOLUMN="t",WCOLUMN="W"'
    try:
        track(
            f"Q: CHARGE,TOTAL={CHARGE!r}\n{element},ALLOW_LONG_BEAM=0",
            "Q,W1",
            beam(TIMES, P0),
            P0,
            {"wake.sdds": TABLE},
        )
        raise SystemExit("ALLOW_LONG_BEAM=0 accepted a beam longer than the wake data")
    except RunFailure as failure:
        print(f"serial ALLOW_LONG_BEAM=0 (control): fatal error, exit status {failure.status}")

    out, log = track(
        f"Q: CHARGE,TOTAL={CHARGE!r}\n{element},ALLOW_LONG_BEAM=1",
        "Q,W1",
        beam(TIMES, P0),
        P0,
        {"wake.sdds": TABLE},
    )
    observed = loss(out, P0)
    groups = [slice(0, 90), slice(90, 100), slice(100, 110)]
    summary = ", ".join(f"{observed[g].mean():.10g}" for g in groups)
    print(
        f"serial ALLOW_LONG_BEAM=1: loss of the groups at 0, 50, 51 ps observed=[{summary}] eV "
        f"expected (manual, trimmed)=[0, 0, 0] eV; whole-beam pair sum=[0, 0, {pair_sum[-1]:.10g}] eV; "
        f"long-beam warning printed: {'longer than the longitudinal wake' in log}"
    )
    if not np.allclose(observed, pair_sum, rtol=1e-9, atol=1e-6):
        raise SystemExit("the losses differ from the whole-beam pair sum")
    verdict(bool(np.max(np.abs(observed - expected_manual)) > 1.0))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed<br><small>WAKE documentation mismatch</small> | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

