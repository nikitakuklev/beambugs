---
title: "TRK-55: CSBEND and CSRCSBEND curved drifts lose position accuracy at weak curvature"
hide:
  - toc
---

# TRK-55 — CSBEND and CSRCSBEND curved drifts lose position accuracy at weak curvature

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [src/csbend.c:1883–1908](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1883-L1908)<br>[src/csbend.c:1780–1821](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1780-L1821)<br>[src/csbend.c:3176–3179](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L3176-L3179)<br>[CSBEND contract](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L1-L24)<br>[src/csbend.c:1493–1505](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1493-L1505)<br>[src/csbend.c:1019–1022](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1019-L1022)<br>[src/csbend.c:873–878](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L873-L878) |

CSBEND's exact-Hamiltonian tracker loses position accuracy in weak bends because its curved drift subtracts nearly equal quantities. A 0.4 m, 1 µrad bend with 64 fourth-order slices gives a 3.485 nm horizontal error for a 100 µrad ray with momentum 1% above a 3 GeV/c reference. Stable Hamiltonian flows and independent continuous integration separate this error from finite-step integration error. Orders two and six and CSRCSBEND share the loss; expanded-Hamiltonian CSBEND avoids this subtraction.

Run `python TRK-55.py` with [_precision.py](../helpers/_precision.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-55.py"

    [Download TRK-55.py](TRK-55.py)

    ```python
    """TRK-55: CSBEND and CSRCSBEND curved drifts lose position accuracy at weak curvature

    CSBEND's exact-Hamiltonian tracker loses position accuracy in weak bends because its curved drift subtracts nearly equal quantities. A 0.4 m, 1 µrad bend with 64 fourth-order slices gives a 3.485 nm horizontal error for a 100 µrad ray with momentum 1% above a 3 GeV/c reference. Stable Hamiltonian flows and independent continuous integration separate this error from finite-step integration error. Orders two and six and CSRCSBEND share the loss; expanded-Hamiltonian CSBEND avoids this subtraction.

    Track a fixed off-momentum planar ray through weak CSBEND and CSRCSBEND elements and compare with stable 70-digit split Hamiltonian flows using each integration order's own coefficients. Integrate the unsplit Hamiltonian independently to separate arithmetic error from integration truncation, and include zero-angle, ordinary-bend and expanded-Hamiltonian controls.
    """

    from _precision import LENGTH, PIN, D, bend_split, mp, native, position_line, require_close, verdict


    def continuous(angle):
        h = mp.mpf(angle) / mp.mpf(LENGTH)
        q0 = D * mp.mpf(0.0001) / mp.sqrt(1 + mp.mpf(0.0001) ** 2)

        # Unsplit Frenet Hamiltonian: H=-(1+h*x)*sqrt(D²-q²)+h*x+h²*x²/2.
        def rhs(_s, state):
            x, q, _path = state
            ps = mp.sqrt(D * D - q * q)
            return [(1 + h * x) * q / ps, h * ps - h * (1 + h * x), (1 + h * x) * D / ps]

        return mp.odefun(rhs, 0, (mp.mpf(0.001), q0, mp.mpf(0)), tol=mp.mpf("1e-48"), degree=18)(
            LENGTH
        )[0]


    def main():
        defect = False
        unsplit = continuous(1e-6)
        for order in (2, 4, 6):
            split = bend_split(1e-6, 64, LENGTH, 0.001, 0.0001, D, order)[0]
            print(
                f"oracle/order={order}: stable x={float(split):.17g} expected={float(unsplit):.17g} m; "
                f"split-minus-continuous={float(split - unsplit):+.6e} m (unsplit Hamiltonian Taylor integration)"
            )
            if abs(split - unsplit) > mp.mpf("1e-19"):
                raise RuntimeError("Weak-bend truncation exceeds the arithmetic comparison budget")
        for kind, angle, slices, order, expand in (
            ("CSBEND", 0.0, 1, 4, 0),
            ("CSBEND", 0.01, 64, 4, 0),
            ("CSBEND", 1e-6, 1, 4, 0),
            ("CSBEND", 1e-6, 16, 4, 0),
            ("CSBEND", 1e-6, 64, 4, 0),
            ("CSBEND", -1e-6, 64, 4, 0),
            ("CSBEND", 1e-6, 64, 2, 0),
            ("CSBEND", 1e-6, 64, 6, 0),
            ("CSRCSBEND", 1e-6, 64, 4, 0),
            ("CSBEND", 1e-6, 64, 4, 1),
        ):
            extra = (
                ",CSR=0,BINS=16"
                if kind == "CSRCSBEND"
                else f",EXPAND_HAMILTONIAN={expand},REFERENCE_CORRECTION=0"
            )
            spec = (
                f"{kind},L={LENGTH},ANGLE={angle:.17g},N_SLICES={slices},INTEGRATION_ORDER={order},"
                "EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,ISR=0,SYNCH_RAD=0" + extra
            )
            out, _ = native(spec, [{"x": 0.001, "xp": 0.0001, "p": PIN}])
            expected = (
                bend_split(angle, slices, LENGTH, 0.001, 0.0001, D, order)[0]
                if angle
                else mp.mpf(0.001) + mp.mpf(LENGTH) * mp.mpf(0.0001)
            )
            method = "70-digit same-order stable split flow"
            if not angle:
                method = "exact straight drift"
            elif expand:
                method = "exact-flow comparison; expanded-model approximation"
            error = position_line(
                f"{kind}/angle={angle:+.0e}/slices={slices}/order={order}/expanded={expand}",
                out["x"][0],
                expected,
                method,
            )
            if angle == 0 or abs(angle) == 0.01 or expand:
                require_close(out["x"][0], expected, 2e-13)
            else:
                defect |= error > 1e-12
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial |

