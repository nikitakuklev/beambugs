---
title: "COL-37: CSRCSBEND with LINEARIZE=1 mixes arrival time with path length, applies ETILT twice and drops SYNCH_RAD"
hide:
  - toc
---

# COL-37 — CSRCSBEND with LINEARIZE=1 mixes arrival time with path length, applies ETILT twice and drops SYNCH_RAD

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Source | [conversion to ct, csbend.c:3062–3064](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3062-L3064)<br>[matrix branch, csbend.c:3164–3165](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3164-L3165)<br>[conversion back, csbend.c:3833–3836](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3833-L3836)<br>[section matrix, csbend.c:2879–2888](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2879-L2888)<br>[tilt, csbend.c:2891](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2891)<br>[ETILT offset, csbend.c:2909–2910](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2909-L2910)<br>[integrator branch, csbend.c:3166–3221](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3166-L3221)<br>[parameters, track_data.c:2052–2053](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2052-L2053)<br>[CSBEND manual, csbend.tex:107](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L107)<br>[errors, csbend.tex:124–131](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L124-L131) |

CSRCSBEND with LINEARIZE=1 replaces the integrator by a linear matrix, but applies the matrix's path-length row to the arrival time c t, includes ETILT both in the matrix and in the element's misalignment transform, and skips the classical radiation requested by SYNCH_RAD=1. In a 0.8 m, 0.12 rad bend, an on-axis electron at p = 8 mₑc arrives at c t = 0.8000 m instead of the bend length over β = v/c, 0.8062 m; with ETILT = 2 mrad at 1200 mₑc, y and y′ are 2.001 and 2.002 times those of the rotated circular orbit; and the classical relative momentum loss −5.843×10⁻⁸ becomes zero. With LINEARIZE=0 all three agree with these expectations within 10⁻⁶.

Run `python COL-37.py` with [_csr.py](../helpers/_csr.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: COL-37.py"

    [Download COL-37.py](COL-37.py)

    ```python
    """COL-37: CSRCSBEND with LINEARIZE=1 mixes arrival time with path length, applies ETILT twice and drops SYNCH_RAD

    CSRCSBEND with LINEARIZE=1 replaces the integrator by a linear matrix, but applies the matrix's
    path-length row to the arrival time c t, includes ETILT both in the matrix and in the element's
    misalignment transform, and skips the classical radiation requested by SYNCH_RAD=1. In a 0.8 m, 0.12
    rad bend, an on-axis electron at p = 8 mₑc arrives at c t = 0.8000 m instead of the bend length over
    β = v/c, 0.8062 m; with ETILT = 2 mrad at 1200 mₑc, y and y′ are 2.001 and 2.002 times those of the
    rotated circular orbit; and the classical relative momentum loss −5.843×10⁻⁸ becomes zero. With
    LINEARIZE=0 all three agree with these expectations within 10⁻⁶.

    The script tracks single electrons through a 0.8 m, 0.12 rad CSRCSBEND (CSR=0) with LINEARIZE=1
    and, as the control, with LINEARIZE=0 (the symplectic integrator). It compares the arrival time of
    an on-axis electron at p = 8 mₑc with L/β; y and y′ of an on-axis electron at p = 1200 mₑc with
    ETILT = 2 mrad with the intersection of the rotated circular orbit and the design exit plane; and
    the momentum of an on-axis electron at p = 1200 mₑc with SYNCH_RAD=1 with the integrated
    classical loss.
    """

    import numpy as np
    from _csr import R_E, C, final, gaussian_beam, track
    from _elegant import verdict
    from scipy.optimize import brentq

    L, ANGLE = 0.8, 0.12
    RHO = L / ANGLE
    RTOL = 1e-6  # the LINEARIZE=0 control agrees within 2e-7 relative (integration error)


    def run_bend(p0: float, linearize: int, params: str) -> dict:
        bend = f"B: CSRCSBEND,L={L},ANGLE={ANGLE},N_SLICES=64,CSR=0,BINS=32,LINEARIZE={linearize}"
        out = final(track(f"{bend}{params}", "B", gaussian_beam(1, 0.0, p0), p0))
        return {name: float(values[0]) for name, values in out.items()}


    # Arrival time: the on-axis particle travels exactly L, so c t = L/beta, beta = p/sqrt(1+p^2).
    P_LOW = 8.0
    CT_EXPECTED = L * np.sqrt(1 + P_LOW**2) / P_LOW

    # ETILT: the bend's field is rotated by ETILT about the entrance axis, so the circular orbit of
    # radius rho lies in a plane rotated by ETILT; it ends where it crosses the design exit plane
    # (through rho (1 - cos A, 0, sin A) with normal (sin A, 0, cos A), coordinates (x, y, s)).
    ETILT = 0.002
    EXIT_POINT = RHO * np.array([1 - np.cos(ANGLE), 0.0, np.sin(ANGLE)])
    EXIT_NORMAL = np.array([np.sin(ANGLE), 0.0, np.cos(ANGLE)])


    def orbit(a: float) -> np.ndarray:
        return RHO * np.array(
            [(1 - np.cos(a)) * np.cos(ETILT), -(1 - np.cos(a)) * np.sin(ETILT), np.sin(a)]
        )


    a_exit = brentq(
        lambda a: np.dot(orbit(a) - EXIT_POINT, EXIT_NORMAL), ANGLE - 0.01, ANGLE + 0.01, xtol=1e-16
    )
    tangent = np.array(
        [np.sin(a_exit) * np.cos(ETILT), -np.sin(a_exit) * np.sin(ETILT), np.cos(a_exit)]
    )
    Y_EXPECTED = orbit(a_exit)[1]  # the y axis is perpendicular to the design bend plane
    YP_EXPECTED = tangent[1] / np.dot(tangent, EXIT_NORMAL)  # y' = dy/ds = P_y/P_s

    # Classical radiation of an on-axis electron in a uniform field: the momentum p0 (1 + delta)
    # loses d(delta)/ds = -a (1 + delta)^2 with a = 2 r_e p0^3 / (3 rho^2) (Larmor power in a fixed
    # field scales with gamma^2), so delta(L) = 1/(1 + a L) - 1.
    P_HIGH = 1200.0
    A_SR = 2 * R_E * P_HIGH**3 / (3 * RHO**2)
    DELTA_EXPECTED = 1 / (1 + A_SR * L) - 1

    defect = False
    for linearize in (0, 1):
        tag = f"serial LINEARIZE={linearize}"
        t = run_bend(P_LOW, linearize, "")
        tilt = run_bend(P_HIGH, linearize, f",ETILT={ETILT}")
        sr = run_bend(P_HIGH, linearize, ",SYNCH_RAD=1")
        cases = [
            ("p=8 on axis: c t (m)", C * t["t"], CT_EXPECTED),
            (f"ETILT={ETILT}: y (m)", tilt["y"], Y_EXPECTED),
            (f"ETILT={ETILT}: y'", tilt["yp"], YP_EXPECTED),
            ("SYNCH_RAD=1: delta", sr["p"] / P_HIGH - 1, DELTA_EXPECTED),
        ]
        for label, observed, expected in cases:
            good = np.isclose(observed, expected, rtol=RTOL, atol=0)
            print(
                f"{tag} {label}: observed={observed:.13g} expected={expected:.13g} ratio={observed / expected:.7g}"
            )
            if linearize == 0 and not good:
                raise SystemExit(f"{tag} {label}: the control disagrees with the expectation")
            defect |= linearize == 1 and not good
    print(f"comparison rtol={RTOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-medium` |  | 2026-11-04 | master (`7514db36f764`), source reading, not executed for this claim |
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

