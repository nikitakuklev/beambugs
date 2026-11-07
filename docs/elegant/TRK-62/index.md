---
title: "TRK-62: TAPERAPE integer division corrupts superellipse boundaries for exponents above 2"
hide:
  - toc
---

# TRK-62 — TAPERAPE integer division corrupts superellipse boundaries for exponents above 2

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [src/track.h:4034–4039](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track.h#L4034-L4039)<br>[src/limit_amplitudes.c:1699–1719](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1699-L1719)<br>[doc/elegant.tex:12784–12785](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L12784-L12785)<br>[src/track_data.c:3923–3935](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3923-L3935)<br>[src/limit_amplitudes.c:1728–1740](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1728-L1740)<br>[src/limit_amplitudes.c:1752–1777](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L1752-L1777)<br>[src/limit_amplitudes.c:372–389](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L372-L389)<br>[416–419](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L416-L419)<br>[src/limit_amplitudes.c:531–555](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L531-L555)<br>[584–592](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/limit_amplitudes.c#L584-L592) |

TAPERAPE with either superellipse exponent above 2 uses integer division in its boundary construction, accepting particles outside the specified opening. For equal 14.00 mm semi-axes and slopes x′=y′=0.02500, exponent 4 records interception at 0.5600 m instead of the analytic 0.4709 m, an 89.10 mm delay. The ray survives a 0.5200 m element that should intercept it. Independent implicit-boundary calculations and ECOL/MAXAMP controls establish the error for separate horizontal and vertical exponents, zero and positive lengths, and shrinking tapers. STICKY can restore loss counts while postponing missed interceptions to the exit.

Run `python TRK-62.py` with [_loss_geometry.py](../helpers/_loss_geometry.py) and [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-62.py"

    [Download TRK-62.py](TRK-62.py)

    ```python
    """TRK-62: TAPERAPE integer division corrupts superellipse boundaries for exponents above 2

    TAPERAPE with either superellipse exponent above 2 uses integer division in its boundary construction, accepting particles outside the specified opening. For equal 14.00 mm semi-axes and slopes x′=y′=0.02500, exponent 4 records interception at 0.5600 m instead of the analytic 0.4709 m, an 89.10 mm delay. The ray survives a 0.5200 m element that should intercept it. Independent implicit-boundary calculations and ECOL/MAXAMP controls establish the error for separate horizontal and vertical exponents, zero and positive lengths, and shrinking tapers. STICKY can restore loss counts while postponing missed interceptions to the exit.

    Compare native acceptance and first interception with the implicit superellipse, using closed forms and 90-digit roots for diagonal rays. Check separate exponents, zero and positive lengths, shrinking tapers, resolution, placement, exact axes and ECOL/MAXAMP controls. --mpi adds a three-rank Pelegant body case.
    """

    import itertools
    import math

    from _loss_geometry import close, fault

    # isort: split
    import _elegant as native
    import mpmath as mp
    import numpy as np

    mp.mp.dps = 90
    A, Q = 0.014, 0.025
    PAIRS = list(itertools.product((2, 4, 6), repeat=2))
    LEVELS = (0.0, -0.35, 0.35, -0.75, 0.75, -0.9, 0.9, -1.1, 1.1, -1.3, 1.3)


    def track(lattice, rays, mode="serial"):
        beam = {k: np.asarray(rays, dtype=float)[:, j] for j, k in enumerate(("x", "xp", "y", "yp"))}
        beam.update(p=np.full(len(rays), 1500.0), particleID=np.arange(1, len(rays) + 1))
        result = native.run(
            {
                "line.lte": lattice,
                "run.ele": (
                    "&run_setup lattice=line.lte,use_beamline=BL,p_central=1500,"
                    "output=out.sdds,losses=lost.sdds,random_number_seed=92743 &end\n"
                    "&run_control n_steps=1,n_passes=1 &end\n"
                    "&sdds_beam input=beam.sdds &end\n&track soft_failure=0 &end\n"
                ),
            },
            mode=mode,
            beams={"beam.sdds": beam},
            outputs=("lost.sdds", "out.sdds"),
        )
        lost, out = (native.particles(result[name]) for name in ("lost.sdds", "out.sdds"))
        close(sorted([*lost["particleID"], *out["particleID"]]), list(range(1, len(rays) + 1)), 0)
        return lost, out


    def taper(length, nx, ny, a0=A, a1=A, b0=A, b1=A, sticky=0, resolution=1e-13, placement=""):
        return (
            f"A:TAPERAPE,L={length:.17g},ASTART={a0:.17g},AEND={a1:.17g},"
            f"BSTART={b0:.17g},BEND={b1:.17g},XEXPONENT={nx},YEXPONENT={ny},"
            f"STICKY={sticky},RESOLUTION={resolution:.17g}{placement}\n"
        )


    def lost_ids(lost):
        return lost["particleID"].astype(int).tolist()


    def expected_ids(rays, a, b, nx, ny):
        values = [abs(mp.mpf(x) / a) ** nx + abs(mp.mpf(y) / b) ** ny for x, _, y, _ in rays]
        if min(abs(v - 1) for v in values) <= mp.mpf(".001"):
            raise RuntimeError("Classification fixture too close to the wall")
        return [i for i, value in enumerate(values, 1) if value > 1]


    def classification(lost, expected):
        actual = lost_ids(lost)
        if not set(actual).issubset(expected):
            raise RuntimeError(f"Unexpected false losses: {actual}; physical losses {expected}")
        return actual != expected


    def diagonal_root(a0, a1, length, nx, ny):
        # u=x/A(z)=y/A(z): u^nx+u^ny=1. Equal powers and (2,4) have closed forms.
        # Other pairs use a high-precision implicit root, without polar geometry.
        if nx == ny:
            u = mp.power(2, -mp.mpf(1) / nx)
        elif sorted((nx, ny)) == [2, 4]:
            u = mp.sqrt((mp.sqrt(5) - 1) / 2)
        else:
            u = mp.findroot(lambda t: t**nx + t**ny - 1, (0.6, 0.99))
        if abs(u**nx + u**ny - 1) >= mp.mpf("1e-85"):
            raise RuntimeError("Implicit root residual exceeds 1e-85")
        alpha = (mp.mpf(a1) - mp.mpf(a0)) / length
        # A(z)=A0+alpha*z and Q*z=u*A(z) imply z=A0*u/(Q-alpha*u).
        expected = mp.mpf(a0) * u / (mp.mpf(Q) - alpha * u)
        # Separate explanation of the faulty result: at theta=pi/4 the integer
        # quotients give u_code=1/sqrt(2), sqrt(3)/2 or 1 as zero, one or two powers vanish.
        u_code = (1 / mp.sqrt(2), mp.sqrt(3) / 2, mp.mpf(1))[(nx > 2) + (ny > 2)]
        faulty = mp.mpf(a0) * u_code / (mp.mpf(Q) - alpha * u_code)
        return float(expected), float(faulty)


    def body(
        label, nx, ny, length=0.7, a0=A, a1=A, sticky=0, resolution=1e-13, placement="", mode="serial"
    ):
        if placement:
            angle, dx, dy = 0.41, 0.0017, -0.0023
            c, s = math.cos(angle), math.sin(angle)
            rays = [[dx, (c - s) * Q, dy, (s + c) * Q], [dx, 0, dy, 0]]
            placement = f",DX={dx},DY={dy},TILT={angle}"
        else:
            rays = [[0, Q, 0, Q], [0, 0, 0, 0]]
        # MPI tracking requires at least as many initial particles as ranks.
        if mode == "mpi":
            rays += [[0, 0, 0, 0], [0, 0, 0, 0]]
        lost, _ = track(
            taper(length, nx, ny, a0, a1, a0, a1, sticky, resolution, placement) + "BL:LINE=(A)\n",
            rays,
            mode,
        )
        expected, faulty = diagonal_root(a0, a1, length, nx, ny)
        actual_ids = lost_ids(lost)
        if actual_ids == [1]:
            observed = float(lost["s"][0])
            defect = fault(observed, expected, min(length, faulty), max(resolution, 3e-15))
            shown = f"{observed:.16g}"
        elif not actual_ids and faulty > length and expected < length and not sticky:
            defect, shown = True, "survives"
        else:
            raise RuntimeError(f"{label}: unexpected loss IDs {actual_ids}")
        print(
            f"{label}: lost={actual_ids} s={shown}; expected lost=[1] s={expected:.16g} m "
            "(u^n+u^m=1, z=A0*u/(Q-alpha*u))"
        )
        return defect


    def main():
        modes = native.modes()
        defect = False
        grid = [[x * A, 0, y * A, 0] for x, y in itertools.product(LEVELS, repeat=2)]
        for nx, ny in PAIRS:
            expected = expected_ids(grid, A, A, nx, ny)
            for length in (0.0, 0.12):
                lost, _ = track(taper(length, nx, ny) + "BL:LINE=(A)\n", grid)
                defect |= classification(lost, expected)
                close(lost["s"], np.zeros(len(lost["s"])), 0)
                print(
                    f"entrance/{nx},{ny}/L={length:g}: losses={len(lost_ids(lost))}/121; "
                    f"expected={len(expected)} (|x/A|^n+|y/A|^m>1); "
                    f"false survivors={len(set(expected) - set(lost_ids(lost)))}"
                )
            for kind in ("ECOL", "MAXAMP"):
                params = f"X_MAX={A},Y_MAX={A},EXPONENT={nx},Y_EXPONENT={ny}"
                if kind == "MAXAMP":
                    params += ",ELLIPTICAL=1"
                lost, _ = track(f"A:{kind},{params}\nBL:LINE=(A)\n", grid)
                close(lost_ids(lost), expected, 0)
                print(
                    f"control/{kind}/{nx},{ny}: losses={len(expected)}/121; implicit loss IDs match exactly"
                )
        for nx, ny in [*PAIRS, (8, 8)]:
            defect |= body(f"body/{nx},{ny}", nx, ny)
        for n in (4, 6, 8):
            for sticky in (0, 1):
                defect |= body(f"short/{n}/sticky={sticky}", n, n, length=0.52, sticky=sticky)
        for nx, ny in ((2, 2), (2, 4), (4, 2), (4, 4)):
            defect |= body(f"shrinking/{nx},{ny}", nx, ny, length=0.8, a0=0.016, a1=0.010)
        for resolution in (1e-6, 1e-14):
            defect |= body(
                f"shrinking/4,4/resolution={resolution:g}",
                4,
                4,
                length=0.8,
                a0=0.016,
                a1=0.010,
                resolution=resolution,
            )
        defect |= body("placed/4,4", 4, 4, placement="common")
        for nx, ny in ((4, 2), (2, 6), (4, 6)):
            for length in (0.0, 0.1):
                total = length + 0.7
                rays = [
                    [0, x * A / total, 0, y * 0.011 / total]
                    for x, y in itertools.product(LEVELS, repeat=2)
                ]
                endpoint = [[xp * total, 0, yp * total, 0] for _, xp, _, yp in rays]
                expected = expected_ids(endpoint, A, 0.011, nx, ny)
                lost, _ = track(
                    taper(length, nx, ny, 0.018, A, 0.016, 0.011, sticky=1)
                    + "D:DRIF,L=.7\nBL:LINE=(A,D)\n",
                    rays,
                )
                close(lost_ids(lost), expected, 0)
                close(lost["s"], np.full(len(expected), total), 2e-15)
                print(
                    f"downstream/{nx},{ny}/L={length:g}: losses={len(expected)}/121; "
                    f"end-axis implicit IDs match, s={total:.1f} m (MAXAMP endpoint)"
                )
        axes = [
            [0.0168, 0, 0, 0],
            [-0.0168, 0, 0, 0],
            [0.0168, 0, 1e-15, 0],
            [0.0168, 0, -1e-15, 0],
            [0, 0, 0.0168, 0],
            [0, 0, -0.0168, 0],
            [0.007, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        expected = expected_ids(axes, A, A, 4, 4)
        lost, _ = track(taper(0, 4, 4) + "BL:LINE=(A)\n", axes)
        defect |= classification(lost, expected)
        print(
            f"axis/4,4: lost={lost_ids(lost)}; expected={expected} (implicit fourth powers); interior IDs 7,8 survive"
        )
        if "mpi" in modes:
            defect |= body("mpi/body/4,6", 4, 6, mode="mpi")
        native.verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source inspection; source-built CPU double serial with SDDS a3f730a14c49, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, double, serial and three-rank MPI, one thread |

