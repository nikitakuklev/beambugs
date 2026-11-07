---
title: "TRK-58: Negative-angle CSBEND passes internally rotated coordinates to global obstruction tests and loss projection"
hide:
  - toc
---

# TRK-58 — Negative-angle CSBEND passes internally rotated coordinates to global obstruction tests and loss projection

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/csbend.c:1088–1107](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1088-L1107)<br>[src/track_data.c:2018–2018](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2018-L2018)<br>[src/csbend.c:1866–1881](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1866-L1881)<br>[src/csbend.c:2011–2026](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2011-L2026)<br>[src/csbend.c:2155–2157](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2155-L2157)<br>[src/csbend.c:2247–2250](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L2247-L2250)<br>[src/floor.c:1150–1195](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/floor.c#L1150-L1195)<br>[src/obstructionData.c:299–361](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/obstructionData.c#L299-L361)<br>[src/csbend.c:1504–1533](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L1504-L1533)<br>[src/multipole.c:1816–1834](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1816-L1834)<br>[src/multipole.c:1885–1893](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/multipole.c#L1885-L1893)<br>[doc/csbend.tex:55–62](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L55-L62)<br>[doc/elegant.tex:5895–5901](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5895-L5901)<br>[doc/elegant.tex:5931–5934](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5931-L5934)<br>[doc/elegant.tex:7206–7207](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L7206-L7207)<br>[src/floor.c:1049–1050](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/floor.c#L1049-L1050) |

CSBEND tracks a negative-angle bend with the default MALIGN_METHOD=0 as a positive bend rolled by π, and passes these internal coordinates, with x and y reversed, to global obstruction tests and loss-position conversion. A global obstruction can both miss a real hit and remove a different particle that should survive, even with global loss output disabled. In a 1.200 m, −0.1800 rad bend with a 6.000 mm horizontal half-aperture, the global loss output places the lost particle 12.01 mm from its correct horizontal position X. Independent Lorentz integration and the expanded-Hamiltonian analytic orbit establish the expected positions and intercepted particle. Both the exact and expanded Hamiltonians have the obstruction error; aligned negative bends with MALIGN_METHOD=1 or 2 give correct results.

Run `python TRK-58.py` with [_loss_geometry.py](../helpers/_loss_geometry.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-58.py"

    [Download TRK-58.py](TRK-58.py)

    ```python
    """TRK-58: Negative-angle CSBEND passes internally rotated coordinates to global obstruction tests and loss projection

    CSBEND tracks a negative-angle bend with the default MALIGN_METHOD=0 as a positive bend rolled by π, and passes these internal coordinates, with x and y reversed, to global obstruction tests and loss-position conversion. A global obstruction can both miss a real hit and remove a different particle that should survive, even with global loss output disabled. In a 1.200 m, −0.1800 rad bend with a 6.000 mm horizontal half-aperture, the global loss output places the lost particle 12.01 mm from its correct horizontal position X. Independent Lorentz integration and the expanded-Hamiltonian analytic orbit establish the expected positions and intercepted particle. Both the exact and expanded Hamiltonians have the obstruction error; aligned negative bends with MALIGN_METHOD=1 or 2 give correct results.

    Integrate laboratory Lorentz motion and intersect it with the reference radial planes; use the expanded-Hamiltonian analytic orbit for its obstruction expectation. Compare obstruction identities with both global-output settings, local-aperture positions, positive bends, alignment methods 1/2 and free transport.
    """

    import math

    from _loss_geometry import close, fault, ids, obstruction, same_physics, shown, track, world

    # isort: split
    import numpy as np
    from _elegant import verdict
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    LENGTH, ANGLE = 1.2, -0.18
    ORIGIN = (0.32, -0.17)
    RAYS = [[0.003, 0.006, 0, 0], [-0.003, -0.006, 0, 0], [0, 0, 0, 0]]


    def lorentz_oracle(angle, x0, xp0):
        # u is particle path length, s labels the reference radial plane, and alpha
        # is laboratory heading. This ODE does not use the native local trajectory.
        k = angle / LENGTH
        trajectory = solve_ivp(
            lambda u, q: [math.sin(q[2]), math.cos(q[2]), -k],
            (0, 1.1 * LENGTH),
            [x0, 0, math.atan(xp0)],
            method="DOP853",
            rtol=2.3e-14,
            atol=2e-16,
            max_step=LENGTH / 32,
            dense_output=True,
        )
        if not trajectory.success or trajectory.sol is None:
            raise RuntimeError("Lorentz integration failed")
        solution = trajectory.sol

        def sample(s):
            phi = k * s
            reference = np.array([(math.cos(phi) - 1) / k, math.sin(phi) / k])
            tangent = np.array([-math.sin(phi), math.cos(phi)])
            u = (
                0
                if s == 0
                else brentq(
                    lambda u: float(np.dot(solution(u)[:2] - reference, tangent)),
                    0,
                    1.1 * LENGTH,
                    xtol=5e-16,
                    rtol=1e-15,
                )
            )
            x, z, alpha = solution(u)
            local_x = float(np.dot(np.array([x, z]) - reference, [math.cos(phi), math.sin(phi)]))
            return [x, z], local_x, math.tan(alpha + phi)

        return sample


    def bend(angle=ANGLE, method=0, expanded=0, slices=64):
        method_text = "" if method == 0 else f",MALIGN_METHOD={method}"
        return (
            f"B:CSBEND,L={LENGTH},ANGLE={angle},N_SLICES={slices},INTEGRATION_ORDER=6,"
            f"EXPAND_HAMILTONIAN={expanded},EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,"
            f"SYNCH_RAD=0,ISR=0{method_text}\n"
        )


    def expanded_position(s, x0, xp0):
        # The leading-order uniform-field Hamiltonian gives x'' = -k²x.
        k, phi = ANGLE / LENGTH, ANGLE * s / LENGTH
        x = x0 * math.cos(phi) + xp0 * math.sin(phi) / k
        return [(math.cos(phi) - 1) / k + x * math.cos(phi), (1 / k + x) * math.sin(phi)]


    def hit_ids(oracles, bounds, expanded=False, slices=64):
        hits = []
        for pid, (ray, oracle) in enumerate(zip(RAYS, oracles, strict=True), 1):
            for s in np.linspace(0, LENGTH, slices + 1):
                point = expanded_position(s, ray[0], ray[1]) if expanded else oracle(s)[0]
                x, z = world(*point, origin=ORIGIN)
                if bounds[0] < x < bounds[1] and bounds[2] < z < bounds[3]:
                    hits.append(pid)
                    break
        return hits


    def obstruction_case(oracles, bounds, *, expanded=0, method=0, slices=64):
        expected = hit_ids(oracles, bounds, bool(expanded), slices)
        close(expected, [1], 0)
        lattice = bend(method=method, expanded=expanded, slices=slices) + "BL:LINE=(B)\n"
        pair = []
        defects = []
        for glob in (False, True):
            result = track(
                lattice,
                RAYS,
                p0=2000,
                origin=ORIGIN,
                global_coords=glob,
                files={"obstruction.sdds": obstruction(*bounds)},
                commands="&obstruction_data input=obstruction.sdds &end\n",
            )
            lost, out = result[:2]
            actual = lost["particleID"].astype(int).tolist()
            if method == 0:
                mismatch = actual != expected
                if mismatch:
                    close(actual, [2], 0)
            else:
                close(actual, expected, 0)
                mismatch = False
            ids(lost, out, actual, [pid for pid in (1, 2, 3) if pid not in actual])
            oracle_name = (
                "expanded analytic orbit" if expanded else "Lorentz ODE + radial-plane intersection"
            )
            print(
                f"obstruction/expanded={expanded}/method={method}/n={slices}/global={int(glob)}: "
                f"lost={actual} survivors={out['particleID'].astype(int).tolist()} expected lost={expected} "
                f"({oracle_name} + rectangle); s={lost['s'].tolist()} m"
            )
            pair.append(result[:2])
            defects.append(mismatch)
        same_physics(*pair)
        if defects[0] != defects[1]:
            raise RuntimeError("Global output changed obstruction verdict")
        return any(defects)


    def main():
        defect = False
        oracles = [lorentz_oracle(ANGLE, ray[0], ray[1]) for ray in RAYS]
        center = world(*oracles[0](0.6)[0], origin=ORIGIN)
        bounds = [center[0] - 0.0015, center[0] + 0.0015, center[1] - 0.022, center[1] + 0.022]
        for expanded in (0, 1):
            for method in (0, 1, 2):
                defect |= obstruction_case(oracles, bounds, expanded=expanded, method=method)
        defect |= obstruction_case(oracles, bounds, slices=128)
        for expanded in (0, 1):
            # Reflect the whole physical configuration about the incoming axis.
            reflected = [2 * ORIGIN[0] - bounds[1], 2 * ORIGIN[0] - bounds[0], *bounds[2:]]
            lost, out, _ = track(
                bend(angle=-ANGLE, expanded=expanded) + "BL:LINE=(B)\n",
                [[-r[0], -r[1], 0, 0] for r in RAYS],
                p0=2000,
                origin=ORIGIN,
                files={"obstruction.sdds": obstruction(*reflected)},
                commands="&obstruction_data input=obstruction.sdds &end\n",
            )
            ids(lost, out, [1], [2, 3])
            print(
                f"positive-obstruction/expanded={expanded}: lost=[1] expected=[1] (reflected physical configuration)"
            )
        for angle in (ANGLE, -ANGLE):
            oracle = lorentz_oracle(angle, *RAYS[0][:2])
            for method in (0, 1, 2):
                lost, out, _ = track(
                    "A:MAXAMP,X_MAX=.006,Y_MAX=.02\n"
                    + bend(angle=angle, method=method)
                    + "BL:LINE=(A,B)\n",
                    RAYS[:1],
                    p0=2000,
                    origin=ORIGIN,
                )
                ids(lost, out, [1], [])
                s = float(lost["s"][0])
                point, x, xp = oracle(s)
                close([lost["x"][0], lost["xp"][0]], [x, xp], 5e-13)
                if not abs(oracle(s - LENGTH / 64)[1]) < 0.006 < abs(x):
                    raise RuntimeError("ODE does not bracket the first rejected aperture sample")
                actual = np.array([lost["X"][0], lost["Z"][0]])
                expected = np.array(world(*point, origin=ORIGIN))
                if angle < 0 and method == 0:
                    phi = angle * s / LENGTH
                    faulty = expected - 2 * x * np.array([math.cos(phi), math.sin(phi)])
                    defect |= fault(actual, expected, faulty, 5e-13)
                else:
                    close(actual, expected, 5e-13)
                print(
                    f"MAXAMP/angle={angle}/method={method}: s={s:.16g} m; X,Z={shown(actual)} "
                    f"expected={shown(expected)} m (Lorentz ODE); dX={1000 * (actual[0] - expected[0]):.12g} mm; local/IDs=pass"
                )
        for method in (0, 1, 2):
            lost, out, _ = track(
                bend(method=method) + "BL:LINE=(B)\n", RAYS, p0=2000, origin=ORIGIN, global_coords=False
            )
            ids(lost, out, [], [1, 2, 3])
            errors = []
            for j, oracle in enumerate(oracles):
                _, x, xp = oracle(LENGTH)
                actual = np.array([out["x"][j], out["xp"][j]])
                close(actual, [x, xp], 5e-13)
                errors.append(float(np.max(np.abs(actual - [x, xp]))))
            print(
                f"free/method={method}: max local residual={max(errors):.4g} expected<=5e-13 (Lorentz ODE); survivors=[1,2,3]"
            )
        print(
            "global-output on/off: ordinary loss and survivor records bit-identical in all seven obstruction pairs"
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
| Confirmed | `claude-opus-5-5` | high | 2026-11-07 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial, one thread |

