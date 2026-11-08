---
title: "COL-28: SCMULT keeps stale beam sizes across zero-length elements that change them"
hide:
  - toc
---

# COL-28 — SCMULT keeps stale beam sizes across zero-length elements that change them

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`0355dd6`](https://github.com/rtsoliday/elegant/commit/0355dd6aca12), committed 2026-11-03 |
| Tags | <span class="bb-tag">known-limitation</span> <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [call site, do_tracking.c:2964–2975](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2964-L2975)<br>[twiss0, insertSCeffects.cc:762](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L762)<br>[accumulation, insertSCeffects.cc:837–856](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L837-L856)<br>[EMATRIX L, track_data.c:3003](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3003)<br>[flags, track_data.c:4261](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L4261)<br>[flags, track_data.c:4200](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L4200)<br>[linear force, elegant.tex:4529–4538](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4529-L4538) |

SCMULT measures the beam sizes for its space-charge strength only at exits of elements with positive length, so after a zero-length element that changes the transverse sizes, such as an EMATRIX or ROTATE, the next segment pairs the old sizes with the new beta functions. Wrapping a segment in a zero-length telescope and its inverse, or in ±90° rotations, cannot change the physical kick, so the manual's linear kick for the plain ring is expected. For a 0.8000 nC bunch at momentum 10.00 mₑc with rms sizes 1.500 mm and 0.8000 mm, a 2× telescope makes the kick 2.500 times too large; the rotations scale the horizontal and vertical kicks by 1.438 and 0.7667. Slope-only elements do not trigger it.

Run `python COL-28.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: COL-28.py"

    [Download COL-28.py](COL-28.py)

    ```python
    """COL-28: SCMULT keeps stale beam sizes across zero-length elements that change them

    SCMULT measures the beam sizes for its space-charge strength only at exits of elements with positive
    length, so after a zero-length element that changes the transverse sizes, such as an EMATRIX or
    ROTATE, the next segment pairs the old sizes with the new beta functions. Wrapping a segment in a
    zero-length telescope and its inverse, or in ±90° rotations, cannot change the physical kick, so the
    manual's linear kick for the plain ring is expected. For a 0.8000 nC bunch at momentum 10.00 mₑc
    with rms sizes 1.500 mm and 0.8000 mm, a 2× telescope makes the kick 2.500 times too large; the
    rotations scale the horizontal and vertical kicks by 1.438 and 0.7667. Slope-only elements do not
    trigger it.

    The script tracks one turn of a 243-particle bunch with exact second moments (0.8 nC, p = 10 mₑc,
    rms sizes 1.5 mm and 0.8 mm) through a ring whose only lengthy element is a 1 m EMATRIX rotation
    followed by a linear SCMULT. It wraps that segment in a zero-length 2× EMATRIX telescope and its
    inverse (rescaling the segment's beta function to 4 m), or in ROTATE elements of +90° and −90°.
    Both wrappings leave the physical kick unchanged, so the expected kick is the manual's linear
    formula for the plain ring; the plain ring and a telescope of length 10⁻¹² m are the controls.
    """

    import itertools
    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    # SDDS constants.h, as Elegant uses them.
    C, QE, RE = 299792458.0, 1.60217733e-19, 2.81794092e-15
    P = 10.0  # electron momentum, units of m_e c
    GAMMA = np.hypot(1, P)
    BETA_C = C * P / GAMMA
    CHARGE = 0.8e-9  # C
    SX, SY, SZ = 1.5e-3, 0.8e-3, 0.025  # m
    LENGTH = 1.0  # m, integrating length of the segment
    M = 2.0  # telescope magnification
    RTOL, ATOL = 1e-11, 1e-18  # the 1e-12 m control adds 1e-12 of integrating length

    # Every combination of -sqrt(1.5), 0, +sqrt(1.5) in (x, xp, y, yp, z): exact unit variances, zero
    # means and no correlations, so the scaled bunch has rms sizes SX, SY, SZ and is matched to beta = 1 m.
    UNIT = np.array(list(itertools.product([-np.sqrt(1.5), 0.0, np.sqrt(1.5)], repeat=5)))
    N = len(UNIT)


    def ematrix(name, matrix, length=0.0):
        terms = [
            f"R{i + 1}{j + 1}={float(matrix[i, j])!r}"
            for i in range(6)
            for j in range(6)
            if matrix[i, j]
        ]
        return f"{name}: EMATRIX,L={length!r},ORDER=1," + ",".join(terms) + "\n"


    def rotation(beta):
        """Quarter-turn transverse rotation in both planes, matched at the given beta function."""
        r = np.eye(6)
        for k in (0, 2):
            r[k : k + 2, k : k + 2] = [[0.0, beta], [-1.0 / beta, 0.0]]
        return r


    def telescope(m):
        """Symplectic magnification: positions times m, slopes divided by m, in both planes."""
        return np.diag([m, 1 / m, m, 1 / m, 1.0, 1.0])


    # Lattices: the segment R is followed by the SCMULT (insert_sceffects after R). A telescope T maps
    # the matched beam at beta = 1 m to one matched at beta = m^2 with sizes times m; inside, the
    # Gaussian linear field scales as 1/m, and the inverse U multiplies the acquired slope by m, so
    # the final kick equals the plain ring's. ROTATE by +90 degrees exchanges the planes; R has the
    # same beta and phase in both planes, so it commutes with the rotation and the -90 degree ROTATE
    # returns the plain ring's kick.
    HEAD = f"Q: CHARGE,TOTAL={CHARGE!r}\n"
    CASES = (
        ("plain ring (control)", HEAD + ematrix("R", rotation(1.0), LENGTH) + "RING: LINE=(Q,R)\n"),
        (
            f"zero-length {M:g}x telescope",
            HEAD
            + ematrix("T", telescope(M))
            + ematrix("R", rotation(M * M), LENGTH)
            + ematrix("U", telescope(1 / M))
            + "RING: LINE=(Q,T,R,U)\n",
        ),
        (
            f"{M:g}x telescope of length 1e-12 m (control)",
            HEAD
            + ematrix("T", telescope(M), 1e-12)
            + ematrix("R", rotation(M * M), LENGTH)
            + ematrix("U", telescope(1 / M))
            + "RING: LINE=(Q,T,R,U)\n",
        ),
        (
            "ROTATE +90 and -90 degrees",
            HEAD
            + f"RA: ROTATE,TILT={np.pi / 2!r}\n"
            + ematrix("R", rotation(1.0), LENGTH)
            + f"RB: ROTATE,TILT={-np.pi / 2!r}\n"
            + "RING: LINE=(Q,RA,R,RB)\n",
        ),
    )
    COMMANDS = (
        "&global_settings echo_namelists=0 &end\n"
        '&insert_sceffects name="R",skip=1,horizontal=1,vertical=1 &end\n'
        f"&run_setup lattice=lattice.lte,use_beamline=RING,p_central={P!r},default_order=1,"
        "output=out.sdds &end\n"
        "&twiss_output matched=1 &end\n"
        "&run_control n_steps=1,n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track &end\n"
    )

    x, xp, y, yp, z = UNIT.T
    BEAM = {
        "x": SX * x,
        "xp": SX * xp,
        "y": SY * y,
        "yp": SY * yp,
        "t": SZ * z / BETA_C,
        "p": np.full(N, P),
        "particleID": np.arange(1, N + 1),
    }
    # Without space charge the ring is the plain quarter turn: (x, xp) -> (xp, -x) at beta = 1 m.
    IDEAL = np.column_stack([BEAM["xp"], -BEAM["x"], BEAM["yp"], -BEAM["y"]])
    # The manual's linear kick K L lambda(z) x_i/[sigma_i (sigma_x + sigma_y)] at the SCMULT of the
    # plain ring, K = 2 N r_e/(beta^2 gamma^3), lambda = exp(-z^2/(2 sigma_z^2))/(sqrt(2 pi) sigma_z),
    # with the rms sizes of the rotated bunch (the quarter turn exchanges positions and slopes, whose
    # spreads are equal at beta = 1 m).
    K = 2 * CHARGE / QE * RE / (P * P * GAMMA)
    sigma = np.array([IDEAL[:, 0].std(), IDEAL[:, 2].std()])
    s = SZ * z
    density = np.exp(-0.5 * ((s - s.mean()) / s.std()) ** 2) / (np.sqrt(2 * np.pi) * s.std())
    EXPECTED = K * LENGTH * density[:, None] * IDEAL[:, [0, 2]] / (sigma * sigma.sum())
    PROBE = int(np.argmax(EXPECTED[:, 0]))  # the reported particle: largest horizontal kick


    def fmt(values):
        return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values)) + "]"


    defect = False
    for mode in modes():
        for label, lattice in CASES:
            result = run({"lattice.lte": lattice, "run.ele": COMMANDS}, mode, {"beam.sdds": BEAM})
            out = particles(result["out.sdds"])
            if not np.array_equal(out["particleID"], np.arange(1, N + 1)):
                sys.exit(f"{mode} {label}: particles lost")
            if np.max(np.abs(np.column_stack([out["x"], out["y"]]) - IDEAL[:, [0, 2]])) > 1e-15:
                sys.exit(f"{mode} {label}: positions differ from the ideal quarter turn")
            observed = np.column_stack([out["xp"] - IDEAL[:, 1], out["yp"] - IDEAL[:, 3]])
            ratio = observed[PROBE] / EXPECTED[PROBE]
            print(
                f"{mode} {label}, particle {PROBE + 1}: observed={fmt(observed[PROBE])} "
                f"expected={fmt(EXPECTED[PROBE])} ratio={fmt(ratio)}"
            )
            matches = bool(np.allclose(observed, EXPECTED, rtol=RTOL, atol=ATOL))
            if "control" in label:
                if not matches:
                    sys.exit(f"{mode} {label}: the control disagrees with the linear formula")
            else:
                defect |= not matches
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g} against the plain ring's linear kick")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-04</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Found<br><small class="bb-scope">ROTATE case</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-04</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-04</span> | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed<br><small class="bb-scope">ROTATE case</small> | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-04</span> | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

