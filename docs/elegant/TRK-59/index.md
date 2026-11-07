---
title: "TRK-59: TAPERAPE uses a polar angle as an ellipse parameter and accepts particles outside the wall"
hide:
  - toc
---

# TRK-59 — TAPERAPE uses a polar angle as an ellipse parameter and accepts particles outside the wall

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/limit_amplitudes.c:1699–1720](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1699-L1720)<br>[src/limit_amplitudes.c:1728–1735](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1728-L1735)<br>[src/limit_amplitudes.c:1752–1777](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1752-L1777)<br>[doc/elegant.tex:12784–12785](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L12784-L12785)<br>[src/track_data.c:3923–3935](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3923-L3935)<br>[src/do_tracking.c:1611–1621](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L1611-L1621)<br>[src/do_tracking.c:2778–2787](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2778-L2787) |

TAPERAPE with unequal semi-axes and default exponents uses a particle's polar angle as an ellipse parameter, making the accepted cross-section too large away from the principal axes. With semi-axes 12.00 and 6.000 mm, the aperture extends 25.00% too far along a diagonal ray. A particle with x′=y′=0.02000 is lost at 0.3354 m instead of the analytic 0.2683 m, and survives a 0.3000 m element that should intercept it. The implicit ellipse equation and an equivalent ECOL establish the error. Refining RESOLUTION does not remove it; circular and principal-axis controls agree.

Run `python TRK-59.py` with [_loss_geometry.py](../helpers/_loss_geometry.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-59.py"

    [Download TRK-59.py](TRK-59.py)

    ```python
    """TRK-59: TAPERAPE uses a polar angle as an ellipse parameter and accepts particles outside the wall

    TAPERAPE with unequal semi-axes and default exponents uses a particle's polar angle as an ellipse parameter, making the accepted cross-section too large away from the principal axes. With semi-axes 12.00 and 6.000 mm, the aperture extends 25.00% too far along a diagonal ray. A particle with x′=y′=0.02000 is lost at 0.3354 m instead of the analytic 0.2683 m, and survives a 0.3000 m element that should intercept it. The implicit ellipse equation and an equivalent ECOL establish the error. Refining RESOLUTION does not remove it; circular and principal-axis controls agree.

    Compare native wall crossings with the implicit ellipse equation, including a true shrinking taper, and compare entrance acceptance with an equivalent ECOL. Check resolution refinement, circular and principal-axis controls, common placement and STICKY.
    """

    import math

    from _loss_geometry import close, fault, ids, track

    # isort: split
    from _elegant import verdict
    from scipy.optimize import brentq

    A, B, V = 0.012, 0.006, 0.02


    def aperture(length, a=A, b=B, resolution=None, extra=""):
        resolution_text = "" if resolution is None else f",RESOLUTION={resolution}"
        return (
            f"A:TAPERAPE,L={length},ASTART={a},AEND={a},BSTART={b},BEND={b}"
            f"{resolution_text}{extra}\nBL:LINE=(A)\n"
        )


    def acceptance(lost, out):
        if out["particleID"].astype(int).tolist() == [2]:
            ids(lost, out)
            return False
        ids(lost, out, [], [1, 2])
        return True


    def main():
        defect = False
        # Substitution of x=x's, y=y's into x²/a²+y²/b²=1 gives this root.
        expected_s = 1 / math.sqrt((V / A) ** 2 + (V / B) ** 2)
        # At polar angle pi/4, the implemented radial boundary has r²=(a²+b²)/2.
        faulty_s = math.sqrt((A * A + B * B) / 4) / V
        for length in (0.3, 0.5):
            for resolution in (1e-6, 1e-13):
                lost, out, _ = track(
                    aperture(length, resolution=None if resolution == 1e-6 else resolution),
                    [[0, V, 0, V], [0, 0, 0, 0]],
                    p0=2000,
                    global_coords=False,
                )
                if length == 0.3:
                    mismatch = acceptance(lost, out)
                    ids(lost, out, [] if mismatch else [1], [1, 2] if mismatch else [2])
                    defect |= mismatch
                    if not mismatch:
                        close(lost["s"], [expected_s], resolution)
                else:
                    ids(lost, out)
                    defect |= fault(lost["s"][0], expected_s, faulty_s, resolution)
                print(
                    f"diagonal/L={length}/resolution={resolution:g}: lost={lost['particleID'].astype(int).tolist()} "
                    f"survivors={out['particleID'].astype(int).tolist()} s={lost['s'].tolist()} "
                    f"expected lost=[1] s={expected_s:.16g} m (implicit ellipse); "
                    f"radial excess={100 * (faulty_s / expected_s - 1):.12g}%"
                )
        for kind in ("TAPERAPE", "ECOL"):
            for length in (0, 0.4):
                lattice = (
                    aperture(length)
                    if kind == "TAPERAPE"
                    else f"A:ECOL,L={length},X_MAX={A},Y_MAX={B}\nBL:LINE=(A)\n"
                )
                lost, out, _ = track(
                    lattice, [[0.006, 0, 0.006, 0], [0.002, 0, 0.001, 0]], p0=2000, global_coords=False
                )
                if kind == "TAPERAPE":
                    mismatch = acceptance(lost, out)
                    defect |= mismatch
                    ids(lost, out, [] if mismatch else [1], [1, 2] if mismatch else [2])
                else:
                    ids(lost, out)
                print(
                    f"entrance/{kind}/L={length}: lost={lost['particleID'].astype(int).tolist()} "
                    f"expected=[1] (x²/a²+y²/b²=1.25>1); interior ID=2 survives"
                )
        for a, b, xp, yp, label in (
            (0.01, 0.01, 0.02, 0.02, "circle"),
            (A, B, 0.02, 0, "horizontal-axis"),
            (A, B, 0, 0.02, "vertical-axis"),
        ):
            expected = 1 / math.sqrt((xp / a) ** 2 + (yp / b) ** 2)
            lost, out, _ = track(
                aperture(0.8, a, b, 1e-13), [[0, xp, 0, yp]], p0=2000, global_coords=False
            )
            ids(lost, out, [1], [])
            close(lost["s"][0], expected, 1e-13)
            print(
                f"control/{label}: s={lost['s'][0]:.16g} expected={expected:.16g} m (implicit ellipse); tolerance=1e-13 m"
            )

        def taper_wall(s):
            return (
                ((0.001 + 0.012 * s) / (0.012 - 0.004 * s)) ** 2
                + ((0.0015 + 0.008 * s) / (0.007 - 0.002 * s)) ** 2
                - 1
            )

        def faulty_wall(s):
            x, y, a, b = 0.001 + 0.012 * s, 0.0015 + 0.008 * s, 0.012 - 0.004 * s, 0.007 - 0.002 * s
            # Compare r² with a²cos²(theta)+b²sin²(theta), theta=atan2(y,x).
            return x * x + y * y - (a * a * x * x + b * b * y * y) / (x * x + y * y)

        expected = brentq(taper_wall, 0, 0.8, xtol=5e-16, rtol=1e-15)
        faulty = brentq(faulty_wall, 0, 0.8, xtol=5e-16, rtol=1e-15)
        lost, out, _ = track(
            "A:TAPERAPE,L=.8,ASTART=.012,AEND=.0088,BSTART=.007,BEND=.0054,RESOLUTION=1e-13\nBL:LINE=(A)\n",
            [[0.001, 0.012, 0.0015, 0.008]],
            p0=2000,
            global_coords=False,
        )
        ids(lost, out, [1], [])
        defect |= fault(lost["s"][0], expected, faulty, 1e-13)
        print(
            f"shrinking-taper: s={lost['s'][0]:.16g} expected={expected:.16g} m (implicit wall + Brent root); "
            f"late={1000 * (lost['s'][0] - expected):.12g} mm; ellipse value={taper_wall(lost['s'][0]) + 1:.16g} expected=1"
        )
        for sticky in (0, 1):
            lattice = aperture(0.3, resolution=1e-13, extra=f",STICKY={sticky}").replace(
                "BL:LINE=(A)", "D:DRIF,L=.1\nBL:LINE=(A,D)"
            )
            lost, out, _ = track(lattice, [[0, V, 0, V], [0, 0, 0, 0]], p0=2000, global_coords=False)
            if sticky:
                ids(lost, out)
                defect |= fault(lost["s"][0], expected_s, 0.3, 1e-13)
            else:
                mismatch = acceptance(lost, out)
                ids(lost, out, [] if mismatch else [1], [1, 2] if mismatch else [2])
                defect |= mismatch
                if not mismatch:
                    close(lost["s"], [expected_s], 1e-13)
            print(
                f"sticky={sticky}: lost={lost['particleID'].astype(int).tolist()} s={lost['s'].tolist()} "
                f"expected lost=[1] s={expected_s:.16g} m (first implicit-wall intersection)"
            )
        tilt, dx, dy = 0.31, 0.002, -0.001
        x = 0.006 * math.cos(tilt) - 0.006 * math.sin(tilt) + dx
        y = 0.006 * math.sin(tilt) + 0.006 * math.cos(tilt) + dy
        lost, out, _ = track(
            aperture(0.4, extra=f",DX={dx},DY={dy},TILT={tilt}"),
            [[x, 0, y, 0], [dx, 0, dy, 0]],
            p0=2000,
            global_coords=False,
        )
        mismatch = acceptance(lost, out)
        ids(lost, out, [] if mismatch else [1], [1, 2] if mismatch else [2])
        defect |= mismatch
        print(
            f"common offset/roll: lost={lost['particleID'].astype(int).tolist()} expected=[1] (same implicit ellipse value 1.25)"
        )
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

