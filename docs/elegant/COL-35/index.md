---
title: "COL-35: CSRCSBEND applies quantum excitation to a single particle only when ISR1PART=0"
hide:
  - toc
---

# COL-35 — CSRCSBEND applies quantum excitation to a single particle only when ISR1PART=0

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Source | [CSRCSBEND condition, csbend.c:2870–2875](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2870-L2875)<br>[CSBEND condition, csbend.c:1155–1159](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1155-L1159)<br>[ISR1PART, track_data.c:2083](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2083)<br>[CSBEND manual, csbend.tex:108–110](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L108-L110)<br>[CSRCSBEND manual, csrcsbend.tex:25–28](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csrcsbend.tex#L25-L28) |

CSRCSBEND inverts its ISR1PART switch: for a single tracked particle with ISR=1, quantum excitation is applied with ISR1PART=0 and omitted with the default ISR1PART=1. The parameter description includes it for a single-particle beam only if ISR=1 and ISR1PART=1, and CSBEND follows that rule. With a fixed seed, one on-axis electron at p = 10⁴ mₑc in a 0.8 m, 0.12 rad bend keeps its relative momentum deviation δ = 0 with ISR1PART=1 and, with ISR1PART=0, receives exactly the δ = 1.040×10⁻⁵ that CSBEND applies with ISR1PART=1; the expected rms excitation is 1.972×10⁻⁵. Serial runs of more than one particle always receive the excitation and are unaffected.

Run `python COL-35.py` with [_csr.py](../helpers/_csr.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: COL-35.py"

    [Download COL-35.py](COL-35.py)

    ```python
    """COL-35: CSRCSBEND applies quantum excitation to a single particle only when ISR1PART=0

    CSRCSBEND inverts its ISR1PART switch: for a single tracked particle with ISR=1, quantum excitation is applied with ISR1PART=0 and omitted with the default ISR1PART=1. The parameter description includes it for a single-particle beam only if ISR=1 and ISR1PART=1, and CSBEND follows that rule. With a fixed seed, one on-axis electron at p = 10⁴ mₑc in a 0.8 m, 0.12 rad bend keeps its relative momentum deviation δ = 0 with ISR1PART=1 and, with ISR1PART=0, receives exactly the δ = 1.040×10⁻⁵ that CSBEND applies with ISR1PART=1; the expected rms excitation is 1.972×10⁻⁵. Serial runs of more than one particle always receive the excitation and are unaffected.

    The script tracks one on-axis electron at p = 10⁴ mₑc through a 0.8 m, 0.12 rad bend with ISR=1,
    SYNCH_RAD=0, CSR=0, 24 slices and a fixed random seed, as CSRCSBEND and as CSBEND, with
    ISR1PART=1 and 0. Expected is the parameter's documented rule, which CSBEND follows: a nonzero
    quantum-excitation momentum change with ISR1PART=1 and none with ISR1PART=0. CSBEND is the
    control, and the swapped CSRCSBEND value is compared with it.
    """

    import numpy as np
    from _csr import R_E, final, gaussian_beam, track
    from _elegant import verdict

    P0, SEED = 1e4, 81731
    BEND = "L=0.8,ANGLE=0.12,N_SLICES=24,ISR=1,SYNCH_RAD=0"
    BEAM = gaussian_beam(1, 0.0, P0)  # one particle on the reference orbit


    def delta(kind: str, isr1part: int) -> float:
        extra = ",CSR=0,BINS=64" if kind == "CSRCSBEND" else ""
        out = final(track(f"B: {kind},{BEND},ISR1PART={isr1part}{extra}", "B", BEAM, P0, seed=SEED))
        return float(out["p"][0] / P0 - 1)


    observed = {(k, f): delta(k, f) for k in ("CSBEND", "CSRCSBEND") for f in (1, 0)}
    # With SYNCH_RAD=0 and the particle on axis, only quantum excitation changes the momentum. The
    # parameter text of both elements: "Include ISR for single-particle beam only if ISR=1 and
    # ISR1PART=1". CSBEND (control) must give a nonzero excitation with ISR1PART=1 and none with 0.
    control = observed[("CSBEND", 1)] != 0 and observed[("CSBEND", 0)] == 0
    for (kind, flag), value in observed.items():
        expected = "nonzero" if flag == 1 else "0"
        print(f"serial {kind} ISR1PART={flag}: delta observed={value:.13g} expected={expected}")
    # Size of the excitation at stake: in a bend of radius rho, d<delta^2>/ds =
    # r_e^2 55/(24 sqrt 3) gamma^5/(alpha rho^3) (Sands), so over L = 0.8 m with rho = L/angle and
    # gamma ~ p the rms is r_e sqrt(55/(24 sqrt 3) p^5 L/(alpha rho^3)), with 1/alpha = 137.0359895.
    RHO = 0.8 / 0.12
    rms = R_E * np.sqrt(55 / (24 * np.sqrt(3)) * P0**5 * 137.0359895 * 0.8 / RHO**3)
    print(f"rms quantum excitation of delta through the bend with ISR applied: {rms:.4e}")
    if not control:
        raise SystemExit("The CSBEND control does not follow the ISR1PART rule.")
    # The same seed gives CSRCSBEND the excitation CSBEND applies, but for the opposite setting.
    swapped = (
        observed[("CSRCSBEND", 1)] == 0
        and observed[("CSRCSBEND", 0)] != 0
        and np.isclose(observed[("CSRCSBEND", 0)], observed[("CSBEND", 1)], rtol=1e-12, atol=0)
    )
    print(f"serial CSRCSBEND ISR1PART=0 equals CSBEND ISR1PART=1: {swapped}")
    verdict(observed[("CSRCSBEND", 1)] == 0 or observed[("CSRCSBEND", 0)] != 0)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source reading, not executed for this claim |
| Found<br><small>serial single-particle tracking</small> | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed<br><small>serial single-particle tracking</small> | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

