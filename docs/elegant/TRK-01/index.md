---
title: "TRK-01: Finite EHKICK bends sloped particles on the wrong radius and leaves yp unchanged"
hide:
  - toc
---

# TRK-01 — Finite EHKICK bends sloped particles on the wrong radius and leaves yp unchanged

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`trackThroughExactCorrector`, exactCorrector.c:244–257](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/exactCorrector.c#L244-L257)<br>[element description, track_data.c:333](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L333) |

A finite-length EHKICK with a nonzero kick mistracks particles with nonzero vertical slope y′: it uses total momentum to set the projected bend radius and changes horizontal slope x′ while keeping y′ fixed. A uniform vertical field conserves total and vertical momentum, requiring a helix whose radius depends on momentum perpendicular to the field, as independent Lorentz-force integrations confirm. For a 0.7 m magnet with a 0.09 rad kick, a particle 20% above reference momentum entering with x′ = 0.08 and y′ = 0.06 leaves with y′ = 0.06000 instead of 0.06054, losing 0.8842% of its vertical momentum. At 1 mrad kick and slopes, y′ is 1.500 nrad too small; particles without vertical slope are unaffected.

Run `python TRK-01.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-01.py"

    [Download TRK-01.py](TRK-01.py)

    ```python
    """TRK-01: Finite EHKICK bends sloped particles on the wrong radius and leaves yp unchanged

    A finite-length EHKICK with a nonzero kick mistracks particles with nonzero vertical slope y′: it
    uses total momentum to set the projected bend radius and changes horizontal slope x′ while keeping
    y′ fixed. A uniform vertical field conserves total and vertical momentum, requiring a helix whose
    radius depends on momentum perpendicular to the field, as independent Lorentz-force integrations
    confirm. For a 0.7 m magnet with a 0.09 rad kick, a particle 20% above reference momentum entering
    with x′ = 0.08 and y′ = 0.06 leaves with y′ = 0.06000 instead of 0.06054, losing 0.8842% of its
    vertical momentum. At 1 mrad kick and slopes, y′ is 1.500 nrad too small; particles without vertical
    slope are unaffected.

    The script tracks particles through a 0.7 m EHKICK with KICK = 0.09 rad, and through a 0.3 m
    EHKICK with KICK = 1 mrad, and compares each exit position, slope and path length with the
    closed-form helix of a uniform vertical field. It also reconstructs the vertical momentum before
    and after the element, which the field must conserve. A particle with zero vertical slope is the
    control: it stays in the horizontal plane, where all its momentum is perpendicular to the field,
    and must agree with the helix.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    C = 299792458.0  # m/s, c_mks of SDDS
    RTOL, ATOL = 3e-14, 2e-15
    COPIES = 4  # identical particles per row, so that each MPI worker tracks some


    def helix(p0, length, kick, w, xp, yp):
        """Exit (x, xp, y, yp, path) of the exact hard-edge uniform-field map, from x = y = 0.

        A vertical field B conserves p_y and |p|, so the particle moves on a helix. Its projection
        on the horizontal plane is a circle whose radius is set by the momentum perpendicular to B,
        p_perp = p/sqrt(1 + r^2), with r = p_y/p_perp = yp/sqrt(1 + xp^2). KICK is the bend angle of
        the reference particle (momentum p0, no slopes) over L, so its radius is rho0 = L/sin(KICK)
        and a particle of momentum w*p0 has projected radius R = rho0*w/sqrt(1 + r^2). Entering at
        angle a = atan(xp), it reaches the exit plane s = L at angle a + theta with
        sin(a + theta) = L/R + sin(a).
        """
        a = np.arctan(xp)
        r = yp / np.sqrt(1 + xp * xp)
        radius = length / np.sin(kick) * w / np.sqrt(1 + r * r)
        theta = np.arcsin(length / radius + np.sin(a)) - a
        return np.array(
            [
                radius * (np.cos(a) - np.cos(a + theta)),
                np.tan(a + theta),
                r * radius * theta,  # y advances by r per unit of projected arc length
                r / np.cos(a + theta),  # yp = p_y/p_s, with p_s = p_perp*cos(a + theta)
                radius * theta * np.sqrt(1 + r * r),  # path = projected arc length * |p|/p_perp
            ]
        )


    def vertical_momentum(p, xp, yp):
        # p_y = p*yp/sqrt(1 + xp^2 + yp^2), in units of m*c
        return p * yp / np.sqrt(1 + xp * xp + yp * yp)


    def track(mode, p0, length, kick, rows):
        """Exit (x, xp, y, yp, path) for each (p, xp, yp) row; particles start at x = y = t = 0."""
        lattice = f"E: EHKICK,L={length!r},KICK={kick!r}\nBL: LINE=(E)\n"
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={p0!r}, "
            "output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        rows = np.repeat(np.asarray(rows, dtype=float), COPIES, axis=0)
        beam = {"p": rows[:, 0], "xp": rows[:, 1], "yp": rows[:, 2]}
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})
        out = particles(result["out.sdds"])
        if len(out["p"]) != len(rows):
            sys.exit(f"{mode}: {len(rows) - len(out['p'])} particles lost")
        # The helix oracle and vertical_momentum assume the element leaves the total momentum unchanged.
        if not np.allclose(out["p"], rows[:, 0], rtol=RTOL, atol=0):
            sys.exit(f"{mode}: the element changed the total momentum: comparison not valid.")
        beta = out["p"] / np.sqrt(1 + out["p"] ** 2)
        columns = (out["x"], out["xp"], out["y"], out["yp"], out["t"] * beta * C)
        observed = np.column_stack(columns)
        # The copies of a row must agree; keep one.
        if not np.array_equal(observed, np.repeat(observed[::COPIES], COPIES, axis=0)):
            sys.exit(f"{mode}: identical particles left the element differently")
        return observed[::COPIES]


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected disagree."""
        bad = not np.allclose(observed, expected, rtol=RTOL, atol=ATOL)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    defect = False
    print("columns: x (m), xp, y (m), yp, path (m); expected from the uniform-field helix")
    for mode in modes():
        # Reference momentum 10 m_e c, particle 20% above it (w = 1.2).
        p0, length, kick, w, xp = 10.0, 0.7, 0.09, 1.2, 0.08
        sloped, planar = track(mode, p0, length, kick, [[p0 * w, xp, 0.06], [p0 * w, xp, 0.0]])
        if compare(f"{mode} control, yp=0", planar, helix(p0, length, kick, w, xp, 0.0)):
            sys.exit("The planar control disagrees with the helix: the comparison is not valid.")
        defect |= compare(
            f"{mode} KICK=0.09, xp=0.08, yp=0.06", sloped, helix(p0, length, kick, w, xp, 0.06)
        )
        # A vertical field exerts no vertical force: p_y at the exit equals p_y at the entrance.
        before = vertical_momentum(p0 * w, xp, 0.06)
        after = vertical_momentum(p0 * w, sloped[1], sloped[3])
        defect |= compare(f"{mode} p_y at exit (m c), conserved", after, before)
        # Milliradian corrector setting: reference particle with 1 mrad slopes in both planes.
        realistic = track(mode, p0, 0.3, 1e-3, [[p0, 1e-3, 1e-3]])[0]
        expected = helix(p0, 0.3, 1e-3, 1.0, 1e-3, 1e-3)
        defect |= compare(f"{mode} KICK=1e-3, xp=yp=1e-3", realistic, expected)
        print(f"{mode} KICK=1e-3, xp=yp=1e-3: observed-expected={fmt(realistic - expected)}")
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

