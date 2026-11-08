---
title: "IO-14: Sector-bend loss directions discard slopes and use an incorrect global angle"
hide:
  - toc
---

# IO-14 — Sector-bend loss directions discard slopes and use an incorrect global angle

| | |
|---|---|
| Code | [Elegant](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/csbend.c:1866–1876](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1866-L1876)<br>[src/csbend.c:2011–2021](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2011-L2021)<br>[src/csbend.c:1504–1533](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1504-L1533)<br>[src/floor.c:1150–1197](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/floor.c#L1150-L1197)<br>[src/floor.c:1066–1092](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/floor.c#L1066-L1092)<br>[src/floor.c:1221–1227](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/floor.c#L1221-L1227)<br>[src/csbend.c:1880–1881](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1880-L1881)<br>[src/obstructionData.c:299–299](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/obstructionData.c#L299-L299)<br>[src/obstructionData.c:311–361](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/obstructionData.c#L311-L361)<br>[doc/elegant.tex:7206–7207](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7206-L7207)<br>[doc/elegant.tex:12288–12290](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L12288-L12290)<br>[src/sdds_support.c:222–234](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/sdds_support.c#L222-L234) |

With EXPAND_HAMILTONIAN=0 and global loss coordinates enabled, CSBEND internal-aperture losses report a horizontal direction that omits the traversed bend and particle slope. With a horizontal entrance, a particle lost 0.3375 m into a 0.9000 m, 0.2400 rad bend is reported at zero instead of −0.08109 rad. The shared sector conversion also reverses SBEN exit-loss angles and corrupts CSBEND global-obstruction loss angles. Exact uniform-field circle intersections establish the expected directions, and a downstream RCOL returns the correct convention. Correcting the converter alone cannot recover slopes discarded by the CSBEND aperture writer.

Run `python IO-14.py` with [_loss_geometry.py](../helpers/_loss_geometry.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: IO-14.py"

    [Download IO-14.py](IO-14.py)

    ```python
    """IO-14: Sector-bend loss directions discard slopes and use an incorrect global angle

    With EXPAND_HAMILTONIAN=0 and global loss coordinates enabled, CSBEND internal-aperture losses report a horizontal direction that omits the traversed bend and particle slope. With a horizontal entrance, a particle lost 0.3375 m into a 0.9000 m, 0.2400 rad bend is reported at zero instead of −0.08109 rad. The shared sector conversion also reverses SBEN exit-loss angles and corrupts CSBEND global-obstruction loss angles. Exact uniform-field circle intersections establish the expected directions, and a downstream RCOL returns the correct convention. Correcting the converter alone cannot recover slopes discarded by the CSBEND aperture writer.

    Intersect an exact uniform-field orbit circle with each rejected sector sample and compare its tangent with the stored direction. Check final-sample CSBEND losses, persistent-MAXAMP SBEN exit losses, downstream RCOL controls and global obstructions.
    """

    import math

    from _loss_geometry import circle, close, fault, ids, obstruction, same_physics, track, world

    # isort: split
    from _elegant import verdict


    def bend(slices=64):
        return (
            f"B:CSBEND,L=.9,ANGLE=.24,N_SLICES={slices},INTEGRATION_ORDER=6,"
            "EXPAND_HAMILTONIAN=0,EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,SYNCH_RAD=0,ISR=0\n"
        )


    def main():
        defect = False
        for heading in (0, 0.17):
            for sign in (-1, 1):
                for slices in (32, 64):
                    x0, xp0 = sign * 0.002, sign * 0.009
                    lattice = "A:MAXAMP,X_MAX=.005,Y_MAX=.02\n" + bend(slices) + "BL:LINE=(A,B)\n"
                    rays = [[x0, xp0, 0, 0], [0, 0, 0, 0]]
                    lost, out, _ = track(lattice, rays, heading=heading)
                    ids(lost, out)
                    s = float(lost["s"][0])
                    x, z, alpha, local_x = circle(s, x0, xp0)
                    if not abs(circle(s - 0.9 / slices, x0, xp0)[3]) < 0.005 < abs(local_x):
                        raise RuntimeError("Circle does not bracket the first rejected aperture sample")
                    close([lost["X"][0], lost["Z"][0]], world(x, z, heading))
                    close(lost["x"][0], local_x)
                    close(lost["xp"][0], math.tan(alpha + s * 0.24 / 0.9))
                    actual, expected = float(lost["thetaX"][0]), heading + alpha
                    defect |= fault(actual, expected, -heading)
                    off = track(lattice, rays, heading=heading, global_coords=False)
                    same_physics((lost, out), off[:2])
                    print(
                        f"CSBEND/h={heading:g}/sign={sign}/n={slices}: thetaX={actual:.16g} "
                        f"expected={expected:.16g} rad (exact circle tangent); s={s:.12g} m, position/local/IDs=pass"
                    )
        bound = (circle(0.9 * 31 / 32, 0.002, 0.009)[3] + circle(0.9, 0.002, 0.009)[3]) / 2
        lost, out, _ = track(
            f"A:MAXAMP,X_MAX={bound:.17g},Y_MAX=.02\n" + bend(32) + "BL:LINE=(A,B)\n",
            [[0.002, 0.009, 0, 0], [0, 0, 0, 0]],
            heading=0.17,
        )
        ids(lost, out)
        close(lost["s"][0], 0.9, 2e-15)
        expected = 0.17 + circle(0.9, 0.002, 0.009)[2]
        defect |= fault(lost["thetaX"][0], expected, -0.17)
        print(
            f"CSBEND/final sample: thetaX={lost['thetaX'][0]:.16g} expected={expected:.16g} rad (circle at s=.9 m)"
        )
        for kind in ("CSBEND", "SBEN"):
            for endpoint in (False, True):
                magnet = bend() if kind == "CSBEND" else "B:SBEN,L=.9,ANGLE=.24,E1=0,E2=0\n"
                aperture = (
                    "A:RCOL,L=0,X_MAX=.005,Y_MAX=.02\n"
                    if endpoint
                    else "A:MAXAMP,X_MAX=.005,Y_MAX=.02\n"
                )
                lost, out, _ = track(
                    aperture + magnet + f"BL:LINE=({'B,A' if endpoint else 'A,B'})\n",
                    [[0.002, 0.009, 0, 0], [0, 0, 0, 0]],
                    heading=0.17,
                )
                ids(lost, out)
                # SBEN uses a polynomial map, so preserve its reported geometric slope.
                expected = 0.17 - 0.24 * lost["s"][0] / 0.9 + math.atan(lost["xp"][0])
                actual = lost["thetaX"][0]
                if endpoint:
                    close(actual, expected, 2e-15)
                else:
                    defect |= fault(actual, expected, -expected if kind == "SBEN" else -0.17)
                print(
                    f"{kind}/{'RCOL control' if endpoint else 'MAXAMP'}: thetaX={actual:.16g} "
                    f"expected={expected:.16g} rad (h - angle*s/L + atan(xp)); IDs=pass"
                )
        lost, out, _ = track(
            bend() + "BL:LINE=(B)\n",
            [[0.002, 0.009, 0, 0], [0, 0, 0, 0]],
            files={"obstruction.sdds": obstruction()},
            commands="&obstruction_data input=obstruction.sdds &end\n",
        )
        ids(lost, out, (1, 2), ())
        for i, (x0, xp0) in enumerate(((0.002, 0.009), (0, 0))):
            s = float(lost["s"][i])
            x, z, expected, _ = circle(s, x0, xp0)
            close([lost["X"][i], lost["Z"][i]], world(x, z))
            # This caller supplies canonical px=sin(alpha+phi), not geometric xp.
            faulty = -math.atan(math.sin(expected + s * 0.24 / 0.9))
            defect |= fault(lost["thetaX"][i], expected, faulty)
            print(
                f"CSBEND/obstruction/ID={i + 1}: thetaX={lost['thetaX'][i]:.16g} "
                f"expected={expected:.16g} rad (exact circle tangent); s={s:.12g} m, position=pass"
            )
        print(
            "global-output on/off: ordinary loss and survivor columns bit-identical in all eight body cases"
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

