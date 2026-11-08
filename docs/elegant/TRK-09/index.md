---
title: "TRK-09: CCBEND plane changes add the axial projection of the flight, not its length, to the path"
hide:
  - toc
---

# TRK-09 — CCBEND plane changes add the axial projection of the flight, not its length, to the path

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Source | [`switchRbendPlane`, ccbend.c:1414–1418](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L1414-L1418)<br>[entrance call, ccbend.c:523](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L523)<br>[exit call, ccbend.c:643](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L643)<br>[`USE_NEW_SWITCH_CODE 0`, ccbend.c:1379](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L1379)<br>[disabled branch with the factor, ccbend.c:1398](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/ccbend.c#L1398)<br>[manual, elegant.tex:374](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L374)<br>[internals, elegantInternals.tex:37](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegantInternals.tex#L37)<br>[CCBEND model, ccbend.tex:1–15](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/ccbend.tex#L1-L15)<br>[parameter L, track_data.c:3786](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3786) |

CCBEND's entrance and exit plane changes add each straight flight's axial projection to the path instead of its signed length. For positive-angle uniform dipoles without radiation, an independent helix construction shows a timing error that survives increased slicing and reference correction. With L = 1.300 m, bend angle 0.2400 rad and reference momentum 10 mₑc, a particle at x = 12.00 mm, horizontal slope x′ = −0.02500 and 12% higher momentum has a path 2.635 μm too long, or 8.826 fs late. At x = 0.1000 mm and x′ = 0.1000 mrad on momentum, the error is −0.1980 μm. The on-axis, on-momentum reference particle is unaffected.

Run `python TRK-09.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: TRK-09.py"

    [Download TRK-09.py](TRK-09.py)

    ```python
    """TRK-09: CCBEND plane changes add the axial projection of the flight, not its length, to the path

    CCBEND's entrance and exit plane changes add each straight flight's axial projection to the path
    instead of its signed length. For positive-angle uniform dipoles without radiation, an independent
    helix construction shows a timing error that survives increased slicing and reference correction.
    With L = 1.300 m, bend angle 0.2400 rad and reference momentum 10 mₑc, a particle at x = 12.00 mm,
    horizontal slope x′ = −0.02500 and 12% higher momentum has a path 2.635 μm too long, or 8.826 fs
    late. At x = 0.1000 mm and x′ = 0.1000 mrad on momentum, the error is −0.1980 μm. The on-axis,
    on-momentum reference particle is unaffected.

    The script tracks particles with transverse offsets and slopes through a CCBEND, each time with an
    on-axis reference particle: a large horizontal offset and slope, a beam-sized one, and one with
    vertical offset and slope as well. It compares each exit position, slope and path length with an
    independent construction: the exact helix of a uniform vertical field, intersected with the two
    parallel end faces and the nominal exit plane. The control is the on-axis, on-momentum particle,
    whose plane changes add no distance error, together with the transverse coordinates, which the same
    construction reproduces.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    C = 299792458.0  # m/s, c_mks of SDDS
    ATOL = 1e-12  # m and rad; the observed agreement is about 1e-14 at 256 slices
    COPIES = 6  # identical particles per state, so that each MPI worker tracks some
    P0, LENGTH, ANGLE = 10.0, 1.3, 0.24  # reference momentum (m_e c), arc length (m), bend angle (rad)
    STATE = (0.012, -0.025, 0.0, 0.0, 1.12)  # x (m), xp, y (m), yp, p/p0
    STATE_3D = (0.012, -0.025, 0.008, 0.04, 1.12)
    SMALL = (1e-4, 1e-4, 0.0, 0.0, 1.0)  # a beam-sized amplitude, on momentum
    REFERENCE = (0.0, 0.0, 0.0, 0.0, 1.0)
    ISOLATED = "FRINGEMODEL=-1,OPTIMIZE_FSE=0,OPTIMIZE_DX=0,REFERENCE_CORRECTION=0"


    def solve(f, lo, hi):
        """Root of f in [lo, hi] by bisection (f changes sign there)."""
        flo = f(lo)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if (f(mid) > 0) == (flo > 0):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


    def construct(x, xp, y, yp, w):
        """Exit (x, xp, y, yp), physical path and the distance the plane changes add wrongly.

        Laboratory axes: the reference orbit starts at the origin along +z and bends toward -x about
        the vertical axis, so its tangent after path l is (-sin(l/rho), 0, cos(l/rho)) with
        rho = L/ANGLE. The magnet's end faces are parallel, with normal n at ANGLE/2 from z. A
        uniform vertical field conserves |p| and the vertical direction cosine, so a particle of
        momentum w*p0 with unit velocity v moves on a helix with velocity rotation rate per unit physical path k = 1/(rho*w):
        dv_x/dl = -k v_z, dv_z/dl = k v_x, dv_y/dl = 0. The particle starts in the plane z = 0, flies
        straight to the entrance face, follows the helix to the exit face, and flies straight to the
        plane through the reference exit point normal to the reference tangent. The physical path is
        the sum of the three signed lengths, the quantity the tracking coordinate s = beta*c*t counts.
        """
        rho = LENGTH / ANGLE
        k = 1 / (rho * w)
        v = np.array([xp, yp, 1.0]) / np.sqrt(1 + xp * xp + yp * yp)
        r = np.array([x, y, 0.0])
        n = np.array([-np.sin(ANGLE / 2), 0.0, np.cos(ANGLE / 2)])
        reference_exit = rho * np.array([np.cos(ANGLE) - 1, 0.0, np.sin(ANGLE)])
        ez = np.array([-np.sin(ANGLE), 0.0, np.cos(ANGLE)])
        ex = np.array([np.cos(ANGLE), 0.0, np.sin(ANGLE)])

        d_in = -(n @ r) / (n @ v)  # signed straight flight to the entrance face
        entry = r + d_in * v

        def orbit(path):
            s, c = np.sin(k * path), np.cos(k * path)
            return entry + np.array(
                [(v[0] * s + v[2] * (c - 1)) / k, v[1] * path, (v[2] * s + v[0] * (1 - c)) / k]
            )

        arc = solve(lambda path: n @ (orbit(path) - reference_exit), 0.5 * LENGTH, 1.5 * LENGTH)
        v_exit = np.array(
            [
                v[0] * np.cos(k * arc) - v[2] * np.sin(k * arc),
                v[1],
                v[2] * np.cos(k * arc) + v[0] * np.sin(k * arc),
            ]
        )
        d_out = -(ez @ (orbit(arc) - reference_exit)) / (ez @ v_exit)
        rel = orbit(arc) + d_out * v_exit - reference_exit
        exit_state = np.array(
            [ex @ rel, (ex @ v_exit) / (ez @ v_exit), rel[1], v_exit[1] / (ez @ v_exit)]
        )
        # Elegant adds, at each face, the displacement along the local axis, d*(unit velocity)_axis,
        # where the signed flight distance d is needed: the missing factor is (axis component - 1).
        omitted = d_in * (v[2] - 1) + d_out * (n @ v_exit - 1)
        return exit_state, d_in + arc + d_out, omitted


    def track(mode, element, states):
        """Exit (x, xp, y, yp) and path (m) of each (x, xp, y, yp, p/p0) state, one row per state."""
        lattice = f"E: {element}\nBL: LINE=(E)\n"
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
            "output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        rows = np.repeat(np.asarray(states, dtype=float), COPIES, axis=0)
        beam = {
            "x": rows[:, 0],
            "xp": rows[:, 1],
            "y": rows[:, 2],
            "yp": rows[:, 3],
            "p": P0 * rows[:, 4],
        }
        result = run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})
        out = particles(result["out.sdds"])
        if len(out["p"]) != len(rows):
            sys.exit(f"{mode}: {len(rows) - len(out['p'])} particles lost")
        # The helix oracle assumes constant momentum (radiation off): the element must not change p.
        if not np.allclose(out["p"], P0 * rows[:, 4], rtol=1e-14, atol=0):
            sys.exit(f"{mode}: the element changed the momentum: comparison not valid.")
        beta = out["p"] / np.sqrt(1 + out["p"] ** 2)
        observed = np.column_stack((out["x"], out["xp"], out["y"], out["yp"], out["t"] * beta * C))
        # Rows are ordered by particle ID, which follows the input order; the copies must agree.
        groups = observed.reshape(len(states), COPIES, 5)
        if np.ptp(groups, axis=1).max() > 1e-15:
            sys.exit(f"{mode}: identical particles left the element differently")
        return groups[:, 0]


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.atleast_1d(values)) + "]"


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected disagree."""
        bad = not np.allclose(observed, expected, rtol=0, atol=ATOL)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    DEFAULT = "CCBEND,L=1.3,ANGLE=0.24,N_SLICES={}"
    CASES = (
        # name, element, slice counts, particle states
        ("default settings", DEFAULT, (4, 32, 256), (STATE, SMALL, REFERENCE)),
        (
            "fringes, optimization, reference correction off",
            f"{DEFAULT},{ISOLATED}",
            (256,),
            (STATE_3D, REFERENCE),
        ),
    )

    defect = False
    print("path in m; expected is the physical flight distance of the construction")
    for mode in modes():
        for name, element, slice_counts, states in CASES:
            for n in slice_counts:
                observed = track(mode, element.format(n), states)
                for state, row in zip(states, observed):
                    label = f"{mode} {name}, N_SLICES={n}, x={state[0]}, xp={state[1]}, y={state[2]}, yp={state[3]}"
                    exit_state, physical, omitted = construct(*state)
                    excess = row[4] - physical
                    if n < 256 and state != states[0]:
                        continue
                    if n < 256:
                        # Coarse slicing leaves a body-integration error of order 3e-7 m in the
                        # transverse exit state and the path, so only the persistent excess is shown;
                        # the verdict comes from the 256-slice comparisons.
                        print(
                            f"{label}, path excess: observed={float(excess)!r} omitted-factor prediction={float(omitted)!r}"
                        )
                        continue
                    if compare(label + ", x xp y yp", row[:4], exit_state):
                        sys.exit(
                            "The transverse coordinates disagree with the construction: it is not valid."
                        )
                    bad = compare(label + ", path", row[4], physical)
                    if state == REFERENCE:
                        if bad:
                            sys.exit(
                                "The on-axis control path is not the arc length: comparison not valid."
                            )
                        continue
                    if bad:
                        # A measured path error must be the omitted plane-change factor.
                        if compare(label + ", path excess vs omitted factor", excess, omitted):
                            sys.exit("The path error is not the omitted plane-change factor.")
                        defect = True
    print(f"comparison: atol={ATOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-15</span> | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-15</span> | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-15</span> | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | high | <span class="bb-nowrap">2026-10-25</span> | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | high | <span class="bb-nowrap">2026-10-25</span> | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

