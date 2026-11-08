---
title: "IO-13: Placed apertures project loss coordinates before restoring the reference frame"
hide:
  - toc
---

# IO-13 — Placed apertures project loss coordinates before restoring the reference frame

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/limit_amplitudes.c:1737–1740](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1737-L1740)<br>[src/limit_amplitudes.c:1783–1789](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1783-L1789)<br>[src/limit_amplitudes.c:1801–1804](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1801-L1804)<br>[src/limit_amplitudes.c:1819–1822](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1819-L1822)<br>[src/limit_amplitudes.c:1905–1911](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1905-L1911)<br>[src/limit_amplitudes.c:1923–1926](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1923-L1926)<br>[src/limit_amplitudes.c:1495–1499](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1495-L1499)<br>[src/limit_amplitudes.c:1530–1536](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1530-L1536)<br>[src/limit_amplitudes.c:1582–1588](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1582-L1588)<br>[src/limit_amplitudes.c:1600–1603](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1600-L1603)<br>[src/floor.c:1221–1227](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/floor.c#L1221-L1227)<br>[src/track_data.c:3929–3931](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3929-L3931)<br>[src/track_data.c:3946–3948](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3946-L3948)<br>[src/track_data.c:3892–3910](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3892-L3910)<br>[doc/elegant.tex:7206–7207](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7206-L7207)<br>[doc/elegant.tex:12288–12290](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L12288-L12290)<br>[src/sdds_support.c:222–234](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L222-L234)<br>[src/floor.c:1050–1051](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/floor.c#L1050-L1051) |

With global loss coordinates enabled, displaced or rolled TAPERAPR, TAPERAPE and APCONTOUR apertures project aperture-frame particle coordinates as reference-frame coordinates. A 10.00 mm half-width shifted horizontally by 1.000 mm reports a wall at 10.00 mm from the reference line instead of 11.00 mm. Roll can also change the reported horizontal direction of an unchanged straight ray. Independent ray–wall intersections and Cartesian floor projection establish the discrepancy. Ordinary loss coordinates, longitudinal interception and survival pass; a pure vertical shift without roll does not corrupt X, Z or thetaX.

Run `python IO-13.py` with [_loss_geometry.py](../helpers/_loss_geometry.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: IO-13.py"

    [Download IO-13.py](IO-13.py)

    ```python
    """IO-13: Placed apertures project loss coordinates before restoring the reference frame

    With global loss coordinates enabled, displaced or rolled TAPERAPR, TAPERAPE and APCONTOUR apertures project aperture-frame particle coordinates as reference-frame coordinates. A 10.00 mm half-width shifted horizontally by 1.000 mm reports a wall at 10.00 mm from the reference line instead of 11.00 mm. Roll can also change the reported horizontal direction of an unchanged straight ray. Independent ray–wall intersections and Cartesian floor projection establish the discrepancy. Ordinary loss coordinates, longitudinal interception and survival pass; a pure vertical shift without roll does not corrupt X, Z or thetaX.

    Track fixed Cartesian rays through displaced and rolled tapered apertures and single/logical contours, solving each wall crossing independently. Compare the global columns with a rigid floor projection and check ordinary loss coordinates, survivor IDs, aligned and vertical-shift controls, and output enabled/disabled.
    """

    import math

    from _loss_geometry import close, fault, ids, same_physics, shown, track, world

    # isort: split
    # The helper sets numerical thread limits before NumPy/SciPy initialize.
    from _elegant import verdict
    from scipy.optimize import brentq

    CONTOUR = (
        "SDDS1\n&column name=x,type=double,units=m &end\n"
        "&column name=y,type=double,units=m &end\n&data mode=ascii &end\n"
        "5\n-.012 -.009\n.012 -.009\n.012 .009\n-.012 .009\n-.012 -.009\n"
    )
    PARAMS = {
        "TAPERAPR": "XSTART=.012,XEND=.008,YSTART=.009,YEND=.008",
        "TAPERAPE": "ASTART=.012,AEND=.008,BSTART=.012,BEND=.008,RESOLUTION=1e-14",
        "APCONTOUR": 'FILENAME="contour.sdds",XCOLUMN="x",YCOLUMN="y",RESOLUTION=1e-14',
    }


    def main():
        defect = False
        ray, heading = [0.002, 0.017, -0.001, 0.004], -0.19
        for kind, params in PARAMS.items():
            for label, dx, dy, tilt in (
                ("aligned", 0, 0, 0),
                ("dx", 0.001, 0, 0),
                ("dy", 0, -0.0015, 0),
                ("roll", 0, 0, 0.37),
                ("combined", -0.0013, 0.0007, -0.29),
            ):
                c, t = math.cos(tilt), math.sin(tilt)

                def body(s, *, dx=dx, dy=dy, c=c, t=t):
                    x, y = ray[0] + ray[1] * s - dx, ray[2] + ray[3] * s - dy
                    return c * x + t * y, -t * x + c * y

                def boundary(s, *, kind=kind, body=body):
                    x, y = body(s)
                    a = 0.012 if kind == "APCONTOUR" else 0.012 - 0.004 * s / 0.8
                    b = (
                        a
                        if kind == "TAPERAPE"
                        else (0.009 if kind == "APCONTOUR" else 0.009 - 0.001 * s / 0.8)
                    )
                    # Circular TAPERAPE avoids the independent unequal-axis shape fault.
                    return (
                        (x / a) ** 2 + (y / b) ** 2 - 1
                        if kind == "TAPERAPE"
                        else max(abs(x) / a, abs(y) / b) - 1
                    )

                crossing = brentq(boundary, 0, 0.8, xtol=5e-16, rtol=1e-15)
                lattice = f"A:{kind},L=.8,{params},DX={dx},DY={dy},TILT={tilt}\nBL:LINE=(A)\n"
                rays = [ray, [dx, 0, dy, 0]]
                lost, out, _ = track(lattice, rays, heading=heading, files={"contour.sdds": CONTOUR})
                ids(lost, out)
                s = float(lost["s"][0])
                close(s, crossing, 1e-14)
                close(
                    [lost[k][0] for k in ("x", "xp", "y", "yp")],
                    [ray[0] + ray[1] * s, ray[1], ray[2] + ray[3] * s, ray[3]],
                    2e-15,
                )
                expected = [*world(ray[0] + ray[1] * s, s, heading), heading + math.atan(ray[1])]
                faulty = [*world(body(s)[0], s, heading), heading + math.atan(c * ray[1] + t * ray[3])]
                actual = [lost[k][0] for k in ("X", "Z", "thetaX")]
                if label in ("aligned", "dy"):
                    close(actual, expected)
                else:
                    defect |= fault(actual, expected, faulty)
                if label == "combined":
                    off = track(
                        lattice,
                        rays,
                        heading=heading,
                        files={"contour.sdds": CONTOUR},
                        global_coords=False,
                    )
                    same_physics((lost, out), off[:2])
                print(
                    f"{kind}/{label}: X,Z,thetaX={shown(actual)} expected={shown(expected)} "
                    f"(Cartesian ray + floor rotation); s={s:.12g} expected={crossing:.12g}; ordinary/IDs=pass"
                )

        for kind, params in (
            ("TAPERAPR", "XSTART=.01,XEND=.01,YSTART=.008,YEND=.008"),
            ("TAPERAPE", "ASTART=.01,AEND=.01,BSTART=.008,BEND=.008,RESOLUTION=1e-14"),
        ):
            lost, out, _ = track(
                f"A:{kind},L=.8,{params},DX=.001\nBL:LINE=(A)\n",
                [[0.001, 0.02, 0, 0], [0.001, 0, 0, 0]],
            )
            ids(lost, out)
            close(lost["s"][0], 0.5, 1e-14)
            close(lost["x"][0], 0.011, 2e-15)
            defect |= fault(lost["X"][0], 0.421, 0.420, 2e-15)
            print(
                f"{kind}/11mm-wall: X={lost['X'][0]:.12g} expected=.421 m "
                f"(.41 + .001 + .010); ordinary x={lost['x'][0]:.12g} m, s={lost['s'][0]:.12g} m"
            )

        logical = (
            "SDDS1\n&parameter name=Logic,type=string &end\n"
            "&column name=x,type=double,units=m &end\n&column name=y,type=double,units=m &end\n"
            '&data mode=ascii &end\n""\n5\n-.01 -.01\n.01 -.01\n.01 .01\n-.01 .01\n-.01 -.01\n'
            '"&&"\n5\n-.02 -.02\n.02 -.02\n.02 .02\n-.02 .02\n-.02 -.02\n'
        )
        for kind in ("TAPERAPR", "TAPERAPE", "APCONTOUR", "logical-APCONTOUR", "TAPERAPC"):
            base = "APCONTOUR" if kind.startswith("logical") else kind
            params = PARAMS.get(base, "RSTART=.01,REND=.009")
            if base == "TAPERAPR":
                params = "XSTART=.01,XEND=.01,YSTART=.008,YEND=.008"
            elif base == "TAPERAPE":
                params = "ASTART=.01,AEND=.01,BSTART=.008,BEND=.008"
            length = 0.8 if base == "TAPERAPC" else 0
            lost, out, _ = track(
                f"A:{base},L={length},{params},DX=.001\nBL:LINE=(A)\n",
                [[0.015, 0.002, 0, 0], [0.001, 0, 0, 0]],
                files={"contour.sdds": logical if kind.startswith("logical") else CONTOUR},
            )
            ids(lost, out)
            close(lost["s"][0], 0, 0)
            expected = [0.425, -0.26, math.atan(0.002)]
            actual = [lost[k][0] for k in ("X", "Z", "thetaX")]
            if base == "TAPERAPC":
                close(actual, expected)
            else:
                defect |= fault(actual, expected, [0.424, -0.26, math.atan(0.002)])
            print(
                f"{kind}/entrance: X,Z,thetaX={shown(actual)} expected={shown(expected)} "
                "(reference ray at s=0); s=0, ordinary/IDs=pass"
            )
        print(
            "global-output on/off: ordinary loss and survivor columns bit-identical for all three placed apertures"
        )
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

