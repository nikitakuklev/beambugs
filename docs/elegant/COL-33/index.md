---
title: "COL-33: SCMULT discards the strength accumulated after a ring's last kick on every turn"
hide:
  - toc
---

# COL-33 — SCMULT discards the strength accumulated after a ring's last kick on every turn

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [pass start, do_tracking.c:804–812](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L804-L812)<br>[reset, insertSCeffects.cc:706–713](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L706-L713)<br>[call site, do_tracking.c:2964–2975](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2964-L2975)<br>[accumulation, insertSCeffects.cc:837–856](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L837-L856)<br>[kick reset, insertSCeffects.cc:491](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/insertSCeffects.cc#L491)<br>[insertSCMULT, get_beamline.c:2379–2403](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/get_beamline.c#L2379-L2403)<br>[end insertion, get_beamline.c:466–475](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/get_beamline.c#L466-L475)<br>[important notes, elegant.tex:4516–4519](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4516-L4519)<br>[linear force, elegant.tex:4529–4538](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L4529-L4538) |

When insert_sceffects places its last ring SCMULT before the end of the beamline, for example by name after an interior element, the strength accumulated along the later elements is erased at the start of every pass, so they never contribute. The same ring with the SCMULT last puts the whole circumference into each kick, as the manual's linear kick over the integrating length requires. In a 2.300 m ring whose only SCMULT follows the first 0.9000 m, each kick after the first turn integrates 0.9000 m instead of 2.300 m, 60.87% too weak. The default skip = 0, and a wildcard name with skip ≥ 1 and no exclusion, end the ring with an SCMULT and are unaffected.

Run `python COL-33.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: COL-33.py"

    [Download COL-33.py](COL-33.py)

    ```python
    """COL-33: SCMULT discards the strength accumulated after a ring's last kick on every turn

    When insert_sceffects places its last ring SCMULT before the end of the beamline, for example by
    name after an interior element, the strength accumulated along the later elements is erased at the
    start of every pass, so they never contribute. The same ring with the SCMULT last puts the whole
    circumference into each kick, as the manual's linear kick over the integrating length requires. In
    a 2.300 m ring whose only SCMULT follows the first 0.9000 m, each kick after the first turn
    integrates 0.9000 m instead of 2.300 m, 60.87% too weak. The default skip = 0, and a wildcard name
    with skip ≥ 1 and no exclusion, end the ring with an SCMULT and are unaffected.

    The script tracks four turns of a weak 243-particle bunch with exact second moments (p = 10 mₑc,
    rms sizes 1.5 mm and 0.8 mm) around a 2.3 m ring of two symplectic EMATRIX rotations matched at
    β = 1 m, A (0.9 m) and B (1.4 m), with one linear SCMULT inserted by name after a marker that
    follows A. WATCH elements around the SCMULT give each turn's kick, from which the script fits the
    integrating length against the manual's linear kick. The control is the same ring started at B,
    so that the SCMULT is the last element and every turn's kick must represent the whole ring.
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
    # A weak bunch: each kick changes the slopes by ~4e-8 of their spread, so the rms sizes stay
    # constant around the ring to that level and the kick measures the integrating length alone.
    CHARGE = 1e-16  # C
    SX, SY, SZ = 1.5e-3, 0.8e-3, 0.025  # m
    LA, LB = 0.9, 1.4  # m; the ring is 2.3 m
    PASSES = 4
    RTOL = 1e-6  # integrating length; the control deviates by less than 1e-7 (rounding, size drift)

    # Every combination of -sqrt(1.5), 0, +sqrt(1.5) in (x, xp, y, yp, z): exact unit variances, zero
    # means and no correlations, so the bunch has rms sizes SX, SY, SZ and is matched at beta = 1 m.
    _UNIT = np.array(list(itertools.product([-np.sqrt(1.5), 0.0, np.sqrt(1.5)], repeat=5)))
    N = len(_UNIT)
    BEAM = {
        "x": SX * _UNIT[:, 0],
        "xp": SX * _UNIT[:, 1],
        "y": SY * _UNIT[:, 2],
        "yp": SY * _UNIT[:, 3],
        "t": SZ * _UNIT[:, 4] / BETA_C,
        "p": np.full(N, P),
        "particleID": np.arange(1, N + 1),
    }


    def cell(name: str, length: float) -> str:
        """EMATRIX of the given length rotating (x, x') and (y, y') by length/beta, beta = 1 m.

        A phase-space rotation at beta = 1 m, alpha = 0 keeps the matched bunch's second moments, so
        the rms sizes and the beta function are the same at every element end.
        """
        c, s = float(np.cos(length)), float(np.sin(length))
        return (
            f"{name}: EMATRIX,L={length!r},ORDER=1,R11={c!r},R12={s!r},R21={-s!r},R22={c!r},"
            f"R33={c!r},R34={s!r},R43={-s!r},R44={c!r},R55=1,R66=1\n"
        )


    def kicks(mode: str, line: str) -> tuple[list[str], list[str], np.ndarray, list[dict]]:
        """Track the ring BL = (line) with one linear SCMULT inserted after the marker K.

        Returns the element names and types from the Twiss output, the beta functions there (shape
        (n, 2)) and, per pass, the particles before the SCMULT with their slope kicks.
        """
        lattice = (
            f"Q: CHARGE,TOTAL={CHARGE!r}\n"
            + cell("A", LA)
            + cell("B", LB)
            + 'W1: WATCH,FILENAME="before.sdds",MODE="coordinates"\n'
            + "K: MARK\n"
            + 'W2: WATCH,FILENAME="after.sdds",MODE="coordinates"\n'
            + f"BL: LINE=({line})\n"
        )
        commands = (
            "&global_settings echo_namelists=0 &end\n"
            '&insert_sceffects name="K",skip=1,horizontal=1,vertical=1 &end\n'
            f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},default_order=1,"
            "output=out.sdds &end\n"
            '&twiss_output matched=1,filename="twiss.sdds" &end\n'
            f"&run_control n_steps=1,n_passes={PASSES} &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        result = run(
            {"lattice.lte": lattice, "run.ele": commands},
            mode,
            {"beam.sdds": BEAM},
            outputs=("before.sdds", "after.sdds", "twiss.sdds"),
        )
        twiss = result["twiss.sdds"]
        names = [str(n) for n in twiss[0, "ElementName"]]
        types = [str(t) for t in twiss[0, "ElementType"]]
        beta = np.column_stack([np.asarray(twiss[0, "betax"]), np.asarray(twiss[0, "betay"])])
        passes = []
        for page in range(PASSES):
            before = particles(result["before.sdds"], page)
            after = particles(result["after.sdds"], page)
            if len(before["x"]) != N or len(after["x"]) != N:
                sys.exit(f"{mode} {line}: particles lost")
            kick = np.column_stack([after["xp"] - before["xp"], after["yp"] - before["yp"]])
            passes.append({"before": before, "kick": kick})
        return names, types, beta, passes


    def integrating_length(before: dict, kick: np.ndarray) -> np.ndarray:
        """Least-squares integrating length L of the manual's linear kick, per plane.

        The manual's kick is dx' = K L lambda(z) x/[sigma_x (sigma_x + sigma_y)], and dy' likewise, with
        K = 2 N r_e/(beta^2 gamma^3) and lambda = exp(-z^2/(2 sigma_z^2))/(sqrt(2 pi) sigma_z); the
        sizes are those at the SCMULT, z = beta c t, and coordinates are taken from the centroid.
        SCMULT weights the strength by beta_element/beta_kick, which is 1 here (checked from the Twiss
        output), so L is the plain integrated length.
        """
        k = 2 * CHARGE / QE * RE / (P * P * GAMMA)
        xy = np.column_stack([before["x"], before["y"]])
        xy = xy - xy.mean(axis=0)
        sigma = xy.std(axis=0)
        s = BETA_C * before["t"]
        s = s - s.mean()
        density = np.exp(-0.5 * (s / s.std()) ** 2) / (np.sqrt(2 * np.pi) * s.std())
        unit = k * density[:, None] * xy / (sigma * sigma.sum())  # kick per metre of integration
        return (unit * kick).sum(axis=0) / (unit * unit).sum(axis=0)


    def fmt(values) -> str:
        return "[" + ", ".join(f"{v:.10g}" for v in np.ravel(values)) + "]"


    CASES = (
        # SCMULT after A, followed by B: the last SCMULT is not at the end of the ring. On the first
        # pass the strength starts at the beginning of the beamline, so the kick integrates A only;
        # on every later pass the strength accumulated along B on the preceding turn, which no kick
        # has used, must be added, so the kick integrates B and A, the whole 2.3 m circumference.
        ("SCMULT after A, then B", "Q,A,W1,K,W2,B", False),
        # The same ring started at B: nothing with length follows the SCMULT, so every kick, the first
        # included, integrates B and A.
        ("ring started at B, SCMULT last (control)", "Q,B,A,W1,K,W2", True),
    )

    defect = False
    for mode in modes():
        for label, line, control in CASES:
            names, types, beta, passes = kicks(mode, line)
            scmult = [i for i, t in enumerate(types) if t == "SCMULT"]
            if len(scmult) != 1 or names[scmult[0] - 1] != "K":
                sys.exit(f"{mode} {label}: expected one SCMULT after K, Twiss lists {names}")
            if not np.allclose(beta, 1.0, rtol=0, atol=1e-12):
                sys.exit(f"{mode} {label}: beta functions differ from 1 m")
            for page, data in enumerate(passes):
                observed = integrating_length(data["before"], data["kick"])
                expected = LA if page == 0 and not control else LA + LB
                print(
                    f"{mode} {label}, pass {page}: integrating length (x, y) m "
                    f"observed={fmt(observed)} expected={expected:.10g}"
                )
                ok = bool(np.allclose(observed, expected, rtol=RTOL, atol=0))
                if control and not ok:
                    sys.exit(f"{mode} {label}: the control fails")
                defect |= not ok
            if not control:
                last = integrating_length(passes[-1]["before"], passes[-1]["kick"])
                print(
                    f"{mode} {label}: steady-turn kick short by {fmt(1 - last / (LA + LB))} "
                    f"(the 1.4 m after the SCMULT, {LB / (LA + LB):.4%} of the ring)"
                )
    print(f"comparison: integrating length rtol={RTOL:g} against the manual's linear kick")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-04</span> | master (`0355dd6aca12`), source reading, not executed for this claim |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-04</span> | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

