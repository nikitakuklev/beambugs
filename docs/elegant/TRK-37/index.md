---
title: "TRK-37: CSBEND edge model 5 adds a constant path offset and drops the y² path term when K3 is small"
hide:
  - toc
---

# TRK-37 — CSBEND edge model 5 adds a constant path offset and drops the y² path term when K3 is small

| | |
|---|---|
| Code | [Elegant](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`) |
| Source | [`curvedDipoleFringe`, csbend.c:6830–6839](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L6830-L6839)<br>[6833](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L6833)<br>[6896](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L6896)<br>[6967](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L6967)<br>[7008](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L7008)<br>[csbend.c:7026](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/csbend.c#L7026)<br>[manual, csbend.tex:96–101](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/csbend.tex#L96-L101)<br>[`curvedDipoleFringeCalc`, curvedDipoleFringeCalc.c:461–505](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/elegantTools/curvedDipoleFringeCalc.c#L461-L505) |

A CSBEND edge using model 5 misplaces a parenthesis when |K₃/(1+δ)| < 10⁻⁵, where K₃ is a fringe integral and δ is the fractional momentum offset. Its K₂ fringe-integral contribution adds the constant −K₂/(2(1+δ)²) to the path instead of −K₂y²/(2(1+δ)²), where y is vertical position. The correct expression follows from the lens's momentum-dependent vertical kick and the continuous larger-K₃ formula. A pure-dipole tanh fringe with width parameter 2.000 cm and curvature 0.1000 m⁻¹ has K₃ = 0 and K₂ = 1.000×10⁻⁴ m⁻¹, producing a false constant path offset of −50.00 μm per edge on momentum. All-zero fringe integrals produce no error.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)

??? example "Reproducer: TRK-37.py"

    [Download TRK-37.py](TRK-37.py)

    ```python
    """TRK-37: CSBEND edge model 5 adds a constant path offset and drops the y² path term when K3 is small

    A CSBEND edge using model 5 misplaces a parenthesis when |K₃/(1+δ)| < 10⁻⁵, where K₃ is a fringe
    integral and δ is the fractional momentum offset. Its K₂ fringe-integral contribution adds the
    constant −K₂/(2(1+δ)²) to the path instead of −K₂y²/(2(1+δ)²), where y is vertical position. The
    correct expression follows from the lens's momentum-dependent vertical kick and the continuous
    larger-K₃ formula. A pure-dipole tanh fringe with width parameter 2.000 cm and curvature 0.1000
    m⁻¹ has K₃ = 0 and K₂ = 1.000×10⁻⁴ m⁻¹, producing a false constant path offset of −50.00 μm per
    edge on momentum. All-zero fringe integrals produce no error.

    The script tracks electrons on axis and at y = ±1.5 mm through a CSBEND with edge model 5 on one
    edge, for both EDGE_ORDER settings, with the fringe integral K3 just below and just above the
    10⁻⁵ switch between the small-argument and exact formulas. The expected path is the K3 → 0 limit of
    the exact formula: no on-axis offset and a y² term −K₂y²/(2(1+δ)²). The cases above the switch are
    the control; a weak fringe with K₂ = 10⁻⁴ shows the realistic size.
    """

    import sys

    import numpy as np
    from _elegant import modes, particles, run, verdict

    C = 299792458.0  # speed of light (m/s)
    P0 = 6000.0  # reference momentum in m_e c
    LENGTH, ANGLE, EDGE_ANGLE, Y0 = 0.7, 0.07, 0.035, 1.5e-3
    # Branch continuity at the 1e-5 switch is about 4e-14 m in the y^2 term; path rounding is ~1e-16 m.
    TOL = 1e-12


    def paths(mode, edge, order, k2, k3, edge_angle):
        """Path minus L of electrons at y = 0, +Y0, -Y0, with model 5 on one edge only."""
        lattice = (
            f"B: CSBEND,L={LENGTH!r},ANGLE={ANGLE!r},E{edge}={edge_angle!r},N_SLICES=64,INTEGRATION_ORDER=6,"
            f"EDGE1_EFFECTS=0,EDGE2_EFFECTS=0,EDGE{edge}_EFFECTS=5,EDGE_ORDER={order},"
            f"FRINGE{edge}K2={k2!r},FRINGE{edge}K3={k3!r},SYNCH_RAD=0,ISR=0\n"
            "BL: LINE=(B)\n"
        )
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, output=out.sdds &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n"
            "&track &end\n"
        )
        y = np.tile([0.0, Y0, -Y0], 4)  # four copies so that every MPI worker tracks some
        beam = {"y": y, "p": np.full(len(y), P0), "particleID": np.arange(1, len(y) + 1)}
        out = particles(
            run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
        )
        if len(out["p"]) != len(y):
            sys.exit(f"{mode}: particles were lost")
        p = out["p"]
        path = out["t"] * C * p / np.sqrt(1 + p * p) - LENGTH
        return path[0], (path[1] + path[2]) / 2 - path[0]


    defect = []
    print(
        f"L={LENGTH} m, ANGLE={ANGLE} rad, p0={P0:g} m_e c, delta=0; y2 term is the path change at y=+-{Y0} m"
    )
    for mode in modes():
        cases = [(2, o, 0.006, k3, EDGE_ANGLE) for o in (1, 2) for k3 in (0.0, 9.9e-6, 1.01e-5, 2e-5)]
        cases += [(1, o, 0.006, k3, EDGE_ANGLE) for o in (1, 2) for k3 in (0.0, 1.01e-5)]
        # A pure dipole whose field falls off as (1 + tanh(u/w))/2 has, by the fringe-integral definitions
        # of curvedDipoleFringeCalc, K2 = h^2 w/2 and K3 = 0; h = 0.1 /m and w = 2 cm give K2 = 1e-4.
        cases += [(2, 1, 1e-4, k3, 0.0) for k3 in (0.0, 1.01e-5)]
        for edge, order, k2, k3, edge_angle in cases:
            on_axis, y2 = paths(mode, edge, order, k2, k3, edge_angle)
            # Exact branch: dtau gets 0.5*(focY + 0.5*focX*((exp(2m) - 1)/m - 2))*y^2/(1+delta) with
            # m = K3/(1+delta) and focY = K2/(1+delta); as m -> 0 the bracket tends to focY, and the
            # path is -dtau, so the path changes by -K2*y^2/(2(1+delta)^2) and has no constant term.
            # The body of a pure dipole adds no path change for a vertical offset.
            expected_y2 = -0.5 * k2 * Y0 * Y0
            label = (
                f"{mode} {'entrance' if edge == 1 else 'exit'} EDGE_ORDER={order} K2={k2:g} K3={k3:g}"
            )
            y2_label = "expected" if edge == 2 else "edge-only limit (not compared)"
            print(
                f"{label}: on-axis path - L observed={on_axis:.6g} expected=0;"
                f" y2 term observed={y2:.6g} {y2_label}={expected_y2:.6g}"
            )
            bad = abs(on_axis) > TOL or (edge == 2 and abs(y2 - expected_y2) > TOL)
            if k3 > 1e-5:
                if bad:
                    sys.exit(f"{label}: the exact-branch control disagrees: invalid")
            else:
                if abs(on_axis + 0.5 * k2) > TOL:
                    sys.exit(f"{label}: the offset is not -K2/2: the cause differs")
                defect.append(bad)
    print(f"comparison: abs tolerance {TOL:g} m")
    verdict(bool(defect) and all(defect))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

