---
title: "COL-03: wakeconvolution.w_t_rf has the wrong functional form"
hide:
  - toc
---

# COL-03 — `wakeconvolution.w_t_rf` has the wrong functional form

| | |
|---|---|
| Code | [ImpactX](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Source | [`WakeConvolution.cpp#L36-L49`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/particles/wakefields/WakeConvolution.cpp#L36-L49)<br>[`WakeConvolution.H#L56-L66`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/particles/wakefields/WakeConvolution.H#L56-L66) |

The Python helper `impactx.wakeconvolution.w_t_rf` uses the bracket `u·exp(−u)`, with `u = sqrt(s/s0)`, where the cited Bane model has `1 − (1+u)·exp(−u)`; its documentation also calls it a resistive-wall wake, although it is the short-range dipole wake of a periodic accelerating structure, and states units of V/m/pC/mm although it returns SI. Bane (SLAC-PUB-9663, 2003, Eqs. 15–16) gives the correct form, which rises linearly in `s`, whereas the implemented one starts as `sqrt(s)`. For an NLC-like cell (`a = 4.92 mm`, `g = 6.89 mm`, `L = 8.75 mm`) the wake is 19.34 times too large at `s/s0 = 0.01` and 1.392 times at `s/s0 = 1`. Built-in tracking does not call this function; users who call it directly are affected.

## Reproducer

Bane, SLAC-PUB-9663 (2003): W\_perp(s) = 4 Z0 c s0 / (pi a^4) \* \[1 - (1 + u) exp(-u)\], u = sqrt(s/s0), s0 = 0.169 a^1.79 g^0.38 / L^1.17. Its short-range limit is 2 Z0 c s / (pi a^4) (linear in s).

??? example "COL-03.py"

    [Download COL-03.py](COL-03.py)

    ```python
    """COL-03: wakeconvolution.w_t_rf has the wrong functional form.

    Bane, SLAC-PUB-9663 (2003): W_perp(s) = 4 Z0 c s0 / (pi a^4) * [1 - (1 + u) exp(-u)], u = sqrt(s/s0),
    s0 = 0.169 a^1.79 g^0.38 / L^1.17. Its short-range limit is 2 Z0 c s / (pi a^4) (linear in s).
    """
    import numpy as np
    from impactx import wakeconvolution

    Z0 = 377.0  # value used by ImpactX
    c = 299792458.0
    a, g, L = 4.92e-3, 6.89e-3, 8.75e-3  # NLC-like cell, inside the fit's range
    s0 = 0.169 * a**1.79 * g**0.38 / L**1.17
    bad = False
    for x in (0.01, 1.0):
        s = x * s0
        u = np.sqrt(x)
        expected = 4 * Z0 * c * s0 / (np.pi * a**4) * (1 - (1 + u) * np.exp(-u))
        observed = wakeconvolution.w_t_rf(s, a, g, L)
        print(f"s/s0 = {x:g}: observed {observed:.4g}, expected {expected:.4g}, ratio {observed / expected:.4g}")
        bad |= abs(observed / expected - 1) > 1e-6
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `claude-opus-5-5` |  | 2026-10-27 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

