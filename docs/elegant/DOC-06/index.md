---
title: "DOC-06: WAKE and TRWAKE manuals say ACAUSAL_ALLOWED=0 enables acausal wakes; Elegant requires 1"
hide:
  - toc
---

# DOC-06 — WAKE and TRWAKE manuals say ACAUSAL_ALLOWED=0 enables acausal wakes; Elegant requires 1

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [WAKE manual, wake.tex:13–16](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/wake.tex#L13-L16)<br>[TRWAKE manual, trwake.tex:17–20](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/trwake.tex#L17-L20)<br>[WAKE check, wake.c:488–492](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/wake.c#L488-L492)<br>[TRWAKE check, trwake.c:514–517](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L514-L517)<br>[WAKE, track_data.c:2545](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2545)<br>[TRWAKE, track_data.c:2615](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2615)<br>[release notes, elegant.tex:12959–12960](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L12959-L12960) |

The WAKE and TRWAKE manual pages state that acausal wakes, tabulated from negative times, are supported provided the user sets ACAUSAL_ALLOWED=0. Elegant does the opposite: with ACAUSAL_ALLOWED=0, the default, a table that does not start at t = 0 stops the run with an error asking for ACAUSAL_ALLOWED=1, as the parameter descriptions and the release note that introduced the feature say. With ACAUSAL_ALLOWED=1 both elements apply the acausal wake as tabulated: for two electron groups 3 ps apart and a wake that is nonzero only at negative times, the leading group loses 12.00 keV in WAKE and receives a 3.914×10⁻⁷ rad kick in TRWAKE, equal to the direct pair sums.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)
- [Download _wake.py](../helpers/_wake.py) · [View source](../helpers/index.md#_wake)

??? example "Reproducer: DOC-06.py"

    [Download DOC-06.py](DOC-06.py)

    ```python
    """DOC-06: WAKE and TRWAKE manuals say ACAUSAL_ALLOWED=0 enables acausal wakes; Elegant requires 1

    The WAKE and TRWAKE manual pages state that acausal wakes, tabulated from negative times, are supported provided the user sets ACAUSAL_ALLOWED=0. Elegant does the opposite: with ACAUSAL_ALLOWED=0, the default, a table that does not start at t = 0 stops the run with an error asking for ACAUSAL_ALLOWED=1, as the parameter descriptions and the release note that introduced the feature say. With ACAUSAL_ALLOWED=1 both elements apply the acausal wake as tabulated: for two electron groups 3 ps apart and a wake that is nonzero only at negative times, the leading group loses 12.00 keV in WAKE and receives a 3.914×10⁻⁷ rad kick in TRWAKE, equal to the direct pair sums.

    The script writes a wake table on -4 ps <= t <= 4 ps in steps of 0.1 ps whose wake is nonzero only
    at negative times, and tracks two groups of electrons 3 ps apart through a WAKE and through a
    TRWAKE reading it, first with ACAUSAL_ALLOWED=0 as the manual instructs and then with
    ACAUSAL_ALLOWED=1. It checks that the documented setting stops the run with Elegant's request for
    ACAUSAL_ALLOWED=1, and that the other setting applies the acausal wake to the leading group with
    the amplitude of the direct pair sum, which is the control.
    """

    import numpy as np
    from _elegant import verdict
    from _wake import ME_EV, RunFailure, beam, loss, table, track

    P0, CHARGE = 2000.0, 0.8e-9  # m_e c, C
    W_LONG, W_TRANS, X_TRAIL = 3e13, 1e15, 1e-3  # V/C, V/C/m, m
    grid = np.round(np.arange(-40, 41) * 0.1e-12, 20)  # s; includes t = 0 exactly
    acausal = np.where(grid < 0, 1.0, 0.0)  # wake only for sources behind the probe
    TABLE = table(
        {"t": grid, "W": W_LONG * acausal, "Wx": W_TRANS * acausal, "Wy": 0 * acausal},
        {"t": "s", "W": "V/C", "Wx": "V/C/m", "Wy": "V/C/m"},
    )
    # Five particles lead at t = 0 and five trail at t = 3 ps, the trailers offset by x = 1 mm.
    TIMES = np.r_[np.zeros(5), np.full(5, 3e-12)]
    BEAM = beam(TIMES, P0)
    BEAM["x"][5:] = X_TRAIL
    ELEMENTS = {
        "WAKE": 'WAKE,INPUTFILE="wake.sdds",TCOLUMN="t",WCOLUMN="W"',
        "TRWAKE": 'TRWAKE,INPUTFILE="wake.sdds",TCOLUMN="t",WXCOLUMN="Wx",WYCOLUMN="Wy"',
    }
    # Pair sums: the table vanishes for t >= 0, so only the leaders feel the trailers, which are
    # 3 ps behind them (wake argument -3 ps). Each leader receives the trailers' charge Q/2 times
    # W(-3 ps): a loss Q/2 W_LONG in eV, and a slope Q/2 W_TRANS x_trail/(p m_e c^2) from TRWAKE.
    EXPECTED = {
        "WAKE": np.r_[np.full(5, CHARGE / 2 * W_LONG), np.zeros(5)],
        "TRWAKE": np.r_[np.full(5, CHARGE / 2 * W_TRANS * X_TRAIL / (P0 * ME_EV)), np.zeros(5)],
    }
    RTOL = 1e-9
    # What each setting should do: the manual's prose says 0 enables acausal wakes; the parameter
    # table (track_data.c) says "If non-zero, then an acausal wake is allowed".
    DOCUMENTED = {0: "acausal wake applied (manual prose)", 1: "acausal wake applied (parameter table)"}


    def response(kind: str, out: dict) -> np.ndarray:
        """Energy loss (eV) after WAKE, horizontal slope after TRWAKE."""
        return loss(out, P0) if kind == "WAKE" else out["xp"]


    defect = True
    for kind, element in ELEMENTS.items():
        for flag in (0, 1):
            try:
                out, _ = track(
                    f"Q: CHARGE,TOTAL={CHARGE!r}\nW1: {element},ACAUSAL_ALLOWED={flag}",
                    "Q,W1",
                    BEAM,
                    P0,
                    {"wake.sdds": TABLE},
                )
            except RunFailure as failure:
                asks = "set ACAUSAL_ALLOWED=1" in failure.log
                print(
                    f"serial {kind} ACAUSAL_ALLOWED={flag}: observed=fatal error (exit status "
                    f"{failure.status}, asks to set ACAUSAL_ALLOWED=1: {asks}) expected={DOCUMENTED[flag]}"
                )
                defect &= flag == 0 and asks
                continue
            observed = response(kind, out)
            match = np.allclose(observed, EXPECTED[kind], rtol=RTOL, atol=RTOL * EXPECTED[kind].max())
            print(
                f"serial {kind} ACAUSAL_ALLOWED={flag}: observed leaders={observed[0]:.10g}, "
                f"trailers={observed[-1]:.3g}; pair sum leaders={EXPECTED[kind][0]:.10g}, trailers=0; "
                f"expected={DOCUMENTED[flag]}"
            )
            if flag == 1 and not match:
                raise SystemExit(f"{kind}: ACAUSAL_ALLOWED=1 does not apply the acausal pair sum")
            defect &= flag == 1
    print("defect if the manual's ACAUSAL_ALLOWED=0 is rejected and ACAUSAL_ALLOWED=1 runs")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

