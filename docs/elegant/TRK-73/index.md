---
title: "TRK-73: CCBEND ignores DX_DY_SIGN, so the documented legacy DX and DY signs at negative angle cannot be restored"
hide:
  - toc
---

# TRK-73 — CCBEND ignores DX_DY_SIGN, so the documented legacy DX and DY signs at negative angle cannot be restored

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/track_data.c:3857](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3857-L3857)<br>[src/track.h:2737](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track.h#L2737-L2737)<br>[doc/elegant.tex:12326–12339](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L12326-L12339)<br>[src/ccbend.c:365–377](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L365-L377)<br>[src/ccbend.c:492–506](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L492-L506)<br>[src/ccbend.c:526–529](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L526-L529)<br>[src/ccbend.c:639–642](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L639-L642)<br>[doc/ccbend.tex:89–100](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ccbend.tex#L89-L100) |

For a negative-angle CCBEND, DX_DY_SIGN=−1 does not restore the documented earlier signs of DX and DY: the parameter is accepted but never read during tracking. With default fitting, a 0.6000 m, −0.04000 rad bend with quadrupole gradient K₁=0.2000 m⁻² and DY=1.000 mm sends an on-axis ray to y=−35.35 µm and y′=−118.5 µrad instead of +35.35 µm and +118.5 µrad. Reversing the supplied offset establishes the legacy result independently. Serial and three-rank tracking agree across all five alignment methods. The default positive setting retains the consistent displacement convention.

Run `python TRK-73.py` with [_elegant.py](../helpers/_elegant.py) and [_ccbend.py](../helpers/_ccbend.py) in the same directory.

??? example "Reproducer: TRK-73.py"

    [Download TRK-73.py](TRK-73.py)

    ```python
    """TRK-73: CCBEND ignores DX_DY_SIGN, so the documented legacy DX and DY signs at negative angle cannot be restored

    For a negative-angle CCBEND, DX_DY_SIGN=−1 does not restore the documented earlier signs of DX and DY: the parameter is accepted but never read during tracking. With default fitting, a 0.6000 m, −0.04000 rad bend with quadrupole gradient K₁=0.2000 m⁻² and DY=1.000 mm sends an on-axis ray to y=−35.35 µm and y′=−118.5 µrad instead of +35.35 µm and +118.5 µrad. Reversing the supplied offset establishes the legacy result independently. Serial and three-rank tracking agree across all five alignment methods. The default positive setting retains the consistent displacement convention.

    The script tracks three fixed rays through one 0.6 m, ±0.04 rad CCBEND with K1 = 0.2 m^-2 and a 1 mm DX or DY, for MALIGN_METHOD 0 to 4 with fitting and edge kicks off, and once with the element's default fitting and edges. The documented legacy convention reverses DX and DY for ANGLE < 0, so DX_DY_SIGN = -1 must reproduce the default setting with the offset negated; for MALIGN_METHOD 4 the expectation is also computed by the shared rigid-face Lorentz integration with the magnet displaced by the negated offset in laboratory axes. Positive-angle and zero-offset runs are controls, and --mpi repeats the fixed-fit cases in three-rank Pelegant.
    """

    import _elegant as native
    import numpy as np
    from _ccbend import ANGLE, DEFAULT, K1, RAYS, Magnet, close, error, track
    from _elegant import verdict

    OFFSET = 1e-3  # DX or DY, m
    SIGNS = (1, 0, -1)  # DX_DY_SIGN values; 1 is the default
    ORACLE_TOL = 1e-15  # native method-4 transport against the independent integration
    DEFECT_MIN = 1e-9  # a discrepancy this large is not rounding
    AXES = (("DX", 0), ("DY", 1))  # parameter and index into the laboratory offset vector


    def run(mode, angle, axis, offset, sign, **params):
        return track(RAYS, mode=mode, ANGLE=angle, **{axis: offset, "DX_DY_SIGN": sign}, **params)[0]


    def main():
        modes = native.modes()
        failures = []
        legacy_runs = {}
        for mode in modes:
            for method in range(5):
                for axis, index in AXES:
                    fixture = dict(MALIGN_METHOD=method)
                    # Legacy convention (dictionary and release note): for ANGLE < 0 the sign of DX and
                    # DY is reversed, so F_{-1}(D) = F_{+1}(-D); for ANGLE > 0 F_{-1}(D) = F_{+1}(D).
                    negative = {s: run(mode, -ANGLE, axis, OFFSET, s, **fixture) for s in SIGNS}
                    expected = run(mode, -ANGLE, axis, -OFFSET, 1, **fixture)
                    legacy = error(negative[-1], expected)
                    switch = max(error(negative[s], negative[1]) for s in (0, -1))
                    positive = close(
                        run(mode, ANGLE, axis, OFFSET, -1, **fixture),
                        run(mode, ANGLE, axis, OFFSET, 1, **fixture),
                        0.0, f"{mode} method {method} {axis} positive-angle control",
                    )  # fmt: skip
                    failures.append(legacy > DEFECT_MIN)
                    legacy_runs[mode, method, axis] = negative[-1]
                    print(
                        f"{mode} MALIGN_METHOD={method} {axis}=1e-3 ANGLE=-0.04 DX_DY_SIGN=-1: max error "
                        f"vs DX_DY_SIGN=+1 with {axis}=-1e-3={legacy!r} expected=0 (legacy sign "
                        f"reversal); DX_DY_SIGN=0,-1 vs +1={switch:.4g}; control ANGLE=+0.04 "
                        f"DX_DY_SIGN=-1 vs +1={positive:.4g} expected=0"
                    )
                    if method == 4 and mode == "serial":
                        # Independent expectation: the magnet displaced rigidly in laboratory axes.
                        d = np.zeros(3)
                        d[index] = OFFSET
                        modern = close(
                            negative[1], Magnet(angle=-ANGLE, k1=K1, d=d).forward(RAYS), ORACLE_TOL,
                            f"method 4 {axis} DX_DY_SIGN=+1 against the integration",
                        )  # fmt: skip
                        oracle = error(negative[-1], Magnet(angle=-ANGLE, k1=K1, d=-d).forward(RAYS))
                        failures.append(oracle > DEFECT_MIN)
                        print(
                            f"serial MALIGN_METHOD=4 {axis}=1e-3 ANGLE=-0.04 DX_DY_SIGN=-1: max error vs "
                            f"laboratory integration with {axis}=-1e-3={oracle!r} expected=0; control "
                            f"DX_DY_SIGN=+1 vs integration with {axis}=+1e-3={modern:.4g}"
                        )
            zero = close(
                run(mode, -ANGLE, "DX", 0.0, -1, MALIGN_METHOD=4),
                run(mode, -ANGLE, "DX", 0.0, 1, MALIGN_METHOD=4),
                0.0, f"{mode} zero-offset control",
            )  # fmt: skip
            print(f"control {mode} ANGLE=-0.04 zero offset: DX_DY_SIGN=-1 vs +1={zero:.4g} expected=0")
        if "mpi" in modes:
            spread = max(
                error(legacy_runs["mpi", m, a], legacy_runs["serial", m, a])
                for m in range(5)
                for a, _ in AXES
            )
            print(
                f"control three-rank vs serial DX_DY_SIGN=-1 outputs, all methods: max difference={spread:.4g}"
            )

        # Default fitting, edge model, reference correction, alignment method and integration order.
        defaults = dict.fromkeys(
            ("OPTIMIZE_FSE", "OPTIMIZE_DX", "REFERENCE_CORRECTION", "FRINGEMODEL", "EDGE_ORDER"),
            DEFAULT,
        )
        defaults["INTEGRATION_ORDER"] = DEFAULT
        for axis, index in AXES:
            observed = run("serial", -ANGLE, axis, OFFSET, -1, **defaults)
            same = error(observed, run("serial", -ANGLE, axis, OFFSET, 1, **defaults))
            expected = run("serial", -ANGLE, axis, -OFFSET, 1, **defaults)
            failures.append(error(observed, expected) > DEFECT_MIN)
            position, slope = 2 * index, 2 * index + 1  # x, x' or y, y' of the on-axis ray
            print(
                f"default fit {axis}=1e-3 ANGLE=-0.04 DX_DY_SIGN=-1 on-axis ray: "
                f"{'xy'[index]}={observed[0, position]:.10e} {'xy'[index]}'={observed[0, slope]:.10e} "
                f"expected {expected[0, position]:.10e} {expected[0, slope]:.10e} "
                f"(DX_DY_SIGN=+1 with {axis}=-1e-3); DX_DY_SIGN=-1 vs +1={same:.4g}"
            )
        verdict(all(failures))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-06</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-07</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial and three-rank MPI, one thread |
| Confirmed<br><small class="bb-scope">rigid-face laboratory-offset expectation</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial and three-rank MPI, one thread |

