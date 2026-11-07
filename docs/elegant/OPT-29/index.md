---
title: "OPT-29: Resonance driving terms ignore the roll of sextupoles, octupoles, KQUSE and bends"
hide:
  - toc
---

# OPT-29 — Resonance driving terms ignore the roll of sextupoles, octupoles, KQUSE and bends

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Elegant master (`0355dd6`, committed 2026-11-03) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [SEXT and KSEXT cases, src/twiss.cc:5229–5234](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5229-L5234)<br>[KQUSE, SBEN, RBEN, CSBEND, CCBEND and CSRCSBEND cases, src/twiss.cc:5235–5255](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5235-L5255)<br>[OCTU and KOCT cases, src/twiss.cc:5266–5271](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5266-L5271)<br>[QUAD and KQUAD cases, src/twiss.cc:5256–5264](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5256-L5264)<br>[linear coupling terms h10010 and h10100, src/twiss.cc:5311–5315](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5311-L5315)<br>[first-order geometric sextupole terms, src/twiss.cc:5330–5343](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5330-L5343)<br>[octupole tune shifts and driving terms, src/twiss.cc:5345–5363](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5345-L5363)<br>[tilt of SEXT, KSEXT, KQUSE, bends, QUAD and KQUAD, src/twiss.cc:5014–5066](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5014-L5066)<br>[applies cos(3 tilt) to the sextupole strength, src/twiss.cc:5076–5078](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L5076-L5078)<br>[`compute_driving_terms` sums the driving terms and tune shifts with amplitude over dipole, quadrupole, sextupole and octupole elements, with only the gradient and sextupole terms of dipoles, doc/elegant.tex:8176–8183](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8176-L8183)<br>[rotation about the longitudinal axis of SEXT, src/track_data.c:481](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L481)<br>[KQUSE, src/track_data.c:3416](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3416)<br>[rotation about the incoming longitudinal axis, src/track_data.c:411](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L411)<br>[potential with the factor (x + iy)ⁿ e^{−inΔφ}, doc/elegant.tex:10876–10886](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L10876-L10886) |

With `compute_driving_terms`, `twiss_output` computes resonance driving terms and tune shifts with amplitude of SEXT, KSEXT, OCTU, KOCT, KQUSE, SBEN, CSBEND, CSRCSBEND and CCBEND ignoring their TILT. A roll θ scales normal sextupole and octupole fields by cos 3θ and cos 4θ and adds a skew gradient K1 sin 2θ; tracking confirms this. A sextupole of K2 = 13 m⁻³ rolled by π/3 keeps the driving term h30000 = 0.5572 + 0.1962i instead of reversing it, and an octupole of K3 = 640 m⁻⁴ rolled by π/4 keeps the detuning dνx/dJx = 66.35 m⁻¹ instead of −66.35 m⁻¹. A KQUSE of K1 = 0.08 m⁻² rolled by 3 mrad, like each bend, reports no linear coupling instead of |h10010| = 1.103×10⁻⁴.

Run `python OPT-29.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: OPT-29.py"

    [Download OPT-29.py](OPT-29.py)

    ```python
    """OPT-29: Resonance driving terms ignore the roll of sextupoles, octupoles, KQUSE and bends

    With `compute_driving_terms`, `twiss_output` computes resonance driving terms and tune shifts with
    amplitude of SEXT, KSEXT, OCTU, KOCT, KQUSE, SBEN, CSBEND, CSRCSBEND and CCBEND ignoring their TILT.
    A roll θ scales normal sextupole and octupole fields by cos 3θ and cos 4θ and adds a skew gradient
    K1 sin 2θ; tracking confirms this. A sextupole of K2 = 13 m⁻³ rolled by π/3 keeps the driving term
    h30000 = 0.5572 + 0.1962i instead of reversing it, and an octupole of K3 = 640 m⁻⁴ rolled by π/4
    keeps the detuning dνx/dJx = 66.35 m⁻¹ instead of −66.35 m⁻¹. A KQUSE of K1 = 0.08 m⁻² rolled by 3
    mrad, like each bend, reports no linear coupling instead of |h10010| = 1.103×10⁻⁴.

    The script runs serial Elegant `twiss_output` with `compute_driving_terms=1` and
    `leading_order_driving_terms_only=1` on a periodic FODO cell holding one test element, and tracks
    three particles through it. SEXT, KSEXT, OCTU and KOCT are run unrolled, with negative strength and
    rolled by π/n and π/(2n) (n = 3 or 4), checked against analytic coefficients, and expected to scale
    by cos(n θ). KQUSE, SBEN, CSBEND, CSRCSBEND and CCBEND rolled by π are compared with the unrolled
    element of opposite K2 and ANGLE, whose tracking and linear optics are identical. Rolled by 3 mrad
    with a gradient, the same elements are compared with the skew-gradient coupling term that a KQUAD
    control reproduces.
    """

    import math
    import sys

    import numpy as np
    from _elegant import particles, run, verdict

    LENGTH = 0.16  # length of the nonlinear magnet (m)
    STRENGTH = {3: 13.0, 4: 640.0}  # K2 (m^-3) for sextupoles, K3 (m^-4) for octupoles
    # Every driving term Elegant accumulates linearly in the sextupole strength (first-order
    # geometric terms) or the octupole strength (leading-order terms). In the straight cell the
    # chromatic sextupole terms vanish (no dispersion); the bend cases below test them.
    TERMS = {
        3: ["h21000", "h30000", "h10110", "h10020", "h10200"],
        4: [
            "dnux/dJx", "dnux/dJy", "dnuy/dJy", "h22000", "h11110", "h00220", "h31000", "h40000",
            "h20110", "h11200", "h20020", "h20200", "h00310", "h00400",
        ],
    }  # fmt: skip
    P_CENTRAL = 1000.0  # beta*gamma of the reference electron
    CHROMATIC = ["h11001", "h00111", "h20001", "h00201", "h10002"]
    ALL_TERMS = ["dnux/dJx", "dnux/dJy", "dnuy/dJy", "h10100", "h10010", *TERMS[3], *CHROMATIC]
    ALL_TERMS += [t for t in TERMS[4] if t not in ALL_TERMS]
    TRACKED = {"x": [1.3e-3, -0.8e-3, 0.5e-3], "y": [-0.7e-3, 1.1e-3, 0.9e-3]}
    ANGLE = 0.006  # bend angle (rad) of the bend cases
    # Extra parameters of the elements with a quadrupole and sextupole field (KQUSE, bends); the
    # ANGLE placeholder takes the bend angle. RBEN shares the SBEN case of the source but is not run.
    QUSE_FAMILIES = {
        "KQUSE": ",N_SLICES=12",
        "SBEN": ",ANGLE={a!r}",
        "CSBEND": ",ANGLE={a!r},N_SLICES=12",
        "CSRCSBEND": ",ANGLE={a!r},N_SLICES=12,BINS=32,CSR=0,ISR=0,SYNCH_RAD=0",
        "CCBEND": ",ANGLE={a!r},N_SLICES=12,OPTIMIZE_FSE=0,OPTIMIZE_DX=0",
    }


    def drift(s):
        return np.array([[1.0, s], [0.0, 1.0]])


    def quad(s, k):
        t = math.sqrt(abs(k))
        if k > 0:
            c, v = math.cos(t * s), math.sin(t * s) / t
        else:
            c, v = math.cosh(t * s), math.sinh(t * s) / t
        return np.array([[c, v], [-k * v, c]])


    def optics_at_magnet(k1):
        """Periodic beta and phase at the magnet's ends from exact 2x2 matrices, averaged over the
        two ends as Elegant's driving-term sum does. The magnet has no linear field on axis, so it
        is a drift for the linear optics. k1 is the focusing strength of QF in this plane."""
        qf, qd = quad(0.22, k1), quad(0.22, -k1)
        turn = drift(1.02) @ qd @ drift(0.43) @ drift(LENGTH) @ drift(0.43) @ qf
        mu = math.acos(np.trace(turn) / 2)
        beta0 = turn[0, 1] / math.sin(mu)
        alpha0 = (turn[0, 0] - turn[1, 1]) / (2 * math.sin(mu))
        sigma0 = np.array([[beta0, -alpha0], [-alpha0, (1 + alpha0**2) / beta0]])
        ends = [drift(0.43) @ qf, drift(LENGTH) @ drift(0.43) @ qf]
        beta = sum((m @ sigma0 @ m.T)[0, 0] for m in ends) / 2
        # Betatron phase advance from the start: tan(phi) = M12 / (beta0 M11 - alpha0 M12).
        phase = sum(math.atan2(m[0, 1], beta0 * m[0, 0] - alpha0 * m[0, 1]) for m in ends) / 2
        return beta, phase


    BX, PX = optics_at_magnet(1.7)
    BY, _ = optics_at_magnet(-1.7)


    def analytic(order, k):
        """Absolute leading-order coefficients of an unrolled magnet of strength k, for the control.

        Sextupole: H3 = K2 (x^3 - 3 x y^2)/6 with x = sqrt(2 Jx bx) cos(phi); the e^{3i phi_x} harmonic
        gives h30000 = K2 L bx^{3/2} e^{3i phi_x} / 48 in Elegant's normalization.
        Octupole: phase-averaging H4 = K3 (x^4 - 6 x^2 y^2 + y^4)/24 gives
        <H4> L = K3 L (bx^2 Jx^2 - 4 bx by Jx Jy + by^2 Jy^2)/16, and dnu/dJ = (1/2 pi) d2<H>/dJ2.
        """
        if order == 3:
            return {"h30000": k * LENGTH * BX**1.5 / 48 * np.exp(3j * PX)}
        return {
            "dnux/dJx": k * LENGTH * BX**2 / (16 * math.pi),
            "dnux/dJy": -k * LENGTH * BX * BY / (8 * math.pi),
            "dnuy/dJy": k * LENGTH * BY**2 / (16 * math.pi),
        }


    def case(magnet):
        """Run twiss_output with driving terms and one tracking pass of TRACKED with the magnet M.

        Returns the driving terms (complex; tune shifts real), the Twiss columns betax, betay, etax,
        etay of every row, the endpoint-averaged betax, betay at M, and the tracked x, xp, y, yp."""
        lattice = (
            "QF: QUAD,L=0.22,K1=1.7\nQD: QUAD,L=0.22,K1=-1.7\nD: DRIF,L=0.43\nE: DRIF,L=1.02\n"
            f"M: {magnet}\nBL: LINE=(QF,D,M,D,QD,E)\n"
        )
        commands = (
            "&run_setup lattice=lattice.lte, use_beamline=BL, output=out.sdds,\n"
            f"  p_central={P_CENTRAL!r}, default_order=3 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            "&twiss_output filename=twiss.sdds, matched=1, output_at_each_step=1,\n"
            "  compute_driving_terms=1, leading_order_driving_terms_only=1 &end\n"
            "&sdds_beam input=beam.sdds &end\n&track &end\n"
        )
        n = len(TRACKED["x"])
        beam = {
            "x": np.array(TRACKED["x"]),
            "y": np.array(TRACKED["y"]),
            "p": np.full(n, P_CENTRAL),
            "particleID": np.arange(1, n + 1),
        }
        files = {"lattice.lte": lattice, "run.ele": commands}
        result = run(files, beams={"beam.sdds": beam}, outputs=("twiss.sdds", "out.sdds"))
        twiss = result["twiss.sdds"]
        terms = {}
        for name in ALL_TERMS:
            if name.startswith("dnu"):
                terms[name] = complex(float(twiss[name][0]))
            else:
                terms[name] = complex(float(twiss["Re" + name][0]), float(twiss["Im" + name][0]))
        optics = np.stack([np.asarray(twiss[0, c], float) for c in ("betax", "betay", "etax", "etay")])
        i = list(twiss[0, "ElementName"]).index("M")
        # Average of the beta functions at M's entrance (row i-1) and exit (row i), as in the sum.
        beta_m = [float(np.mean(twiss[0, f"beta{p}"][i - 1 : i + 1])) for p in "xy"]
        out = particles(result["out.sdds"])
        tracked = np.stack([out[c] for c in ("x", "xp", "y", "yp")])
        return terms, optics, beta_m, tracked


    def fmt(z):
        return f"{z.real:.10g}" if z.imag == 0 else f"{z.real:.10g}{z.imag:+.10g}i"


    def multipole(family, order, strength, tilt):
        kicks = ",N_KICKS=32" if family.startswith("K") else ""
        return f"{family},L={LENGTH},K{order - 1}={strength!r},TILT={tilt!r}{kicks}"


    def same_field(label, a, b):
        """Control: two specifications of one physical field track identically and have the same
        linear optics; otherwise the comparison is invalid."""
        diff = float(np.max(np.abs(a[3] - b[3])))
        if diff > 1e-15 or not np.allclose(a[1], b[1], rtol=1e-12, atol=1e-15):
            sys.exit(f"{label}: tracking or linear optics differ from the equivalent field: invalid.")
        return diff


    defect = False
    # Part 1. SEXT, KSEXT, OCTU, KOCT in the straight cell.
    for family, order in (("SEXT", 3), ("KSEXT", 3), ("OCTU", 4), ("KOCT", 4)):
        k = STRENGTH[order]
        normal = case(multipole(family, order, k, 0.0))
        negative = case(multipole(family, order, -k, 0.0))
        scale = max(abs(normal[0][t]) for t in TERMS[order])
        # Control 1: the unrolled magnet matches the analytic leading-order coefficients.
        for name, value in analytic(order, k).items():
            if abs(normal[0][name] - value) > 1e-12 * abs(value):
                sys.exit(f"{family} unrolled {name} disagrees with theory: invalid comparison.")
        # Control 2: every tested term is linear in the strength (negative strength = -normal).
        if max(abs(negative[0][t] + normal[0][t]) for t in TERMS[order]) > 1e-12 * scale:
            sys.exit(f"{family} terms are not odd in the strength: invalid comparison.")
        ref = "h30000" if order == 3 else "dnux/dJx"
        print(
            f"{family} control: unrolled {ref}={fmt(normal[0][ref])} matches theory; "
            "negative strength gives -1 x every term"
        )
        # A roll TILT multiplies the normal part of a 2n-pole by cos(n TILT) (manual: potential
        # proportional to (x + i y)^n e^{-i n TILT}). TILT = pi/n is the unrolled magnet with negative
        # strength; TILT = pi/(2n) is a pure skew magnet, whose normal terms are zero.
        for label, tilt in (
            (f"TILT=pi/{order}", math.pi / order),
            (f"TILT=pi/{2 * order}", math.pi / (2 * order)),
        ):
            rolled = case(multipole(family, order, k, tilt))
            factor = math.cos(order * tilt)
            expected = {t: normal[0][t] * factor for t in TERMS[order]}
            worst = max(abs(rolled[0][t] - expected[t]) for t in TERMS[order]) / scale
            print(
                f"{family} {label} {ref}: observed={fmt(rolled[0][ref])} expected={fmt(expected[ref])}"
                f" (unrolled x cos({order} TILT) = {factor:.4g}); max deviation over all"
                f" {len(TERMS[order])} terms = {worst:.4g} x the largest unrolled term"
            )
            defect |= worst > 1e-12
            if label == f"TILT=pi/{order}":
                # Control 3: tracking applies TILT, so this roll and the negative strength agree.
                diff = same_field(f"{family} {label}", rolled, negative)
                print(
                    f"{family} control: tracking with {label} equals the negative-strength magnet"
                    f" (max coordinate difference {diff:.3g})"
                )

    # Part 2. KQUSE and bends with K2 rolled by pi. A roll by pi reverses the normal sextupole
    # (cos 3 pi = -1) and the bending direction, and keeps K1 (cos 2 pi = 1); the same field is the
    # unrolled element with -K2 and -ANGLE. Its linear optics and tracking agree exactly with the
    # rolled element, so every driving term, including the chromatic terms that the bend's
    # dispersion makes nonzero, must agree too.
    for family, extra in QUSE_FAMILIES.items():
        rolled = case(f"{family},L={LENGTH},K2=13.0,TILT={math.pi!r}" + extra.format(a=ANGLE))
        equal = case(f"{family},L={LENGTH},K2=-13.0" + extra.format(a=-ANGLE))
        diff = same_field(f"{family} TILT=pi", rolled, equal)
        scale = max(abs(v) for v in equal[0].values())
        worst = max(abs(rolled[0][t] - equal[0][t]) for t in ALL_TERMS) / scale
        chrom = max(abs(rolled[0][t] - equal[0][t]) for t in CHROMATIC) / scale
        print(
            f"{family} TILT=pi K2=13: control: tracking and linear optics equal those of K2=-13"
            f"{', ANGLE=-' + repr(ANGLE) if 'ANGLE' in extra else ''} (max coordinate difference"
            f" {diff:.3g})"
        )
        print(
            f"{family} TILT=pi K2=13 h21000: observed={fmt(rolled[0]['h21000'])}"
            f" expected={fmt(equal[0]['h21000'])} (equivalent field); max deviation over all"
            f" {len(ALL_TERMS)} terms = {worst:.4g}, over the chromatic terms = {chrom:.4g}"
            " x the largest term"
        )
        print(
            f"{family} TILT=pi K2=13 h11001: observed={fmt(rolled[0]['h11001'])}"
            f" expected={fmt(equal[0]['h11001'])} (equivalent field)"
        )
        defect |= worst > 1e-12

    # Part 3. A small roll of a quadrupole gradient adds the skew gradient K1 L sin(2 TILT), whose
    # linear coupling terms Elegant computes for QUAD and KQUAD as
    # |h10010| = |h10100| = |K1 L sin(2 TILT)| sqrt(betax betay)/4 with endpoint-averaged betas.
    # KQUAD is the control.
    K1_SMALL, TILT_SMALL = 0.08, 0.003
    for family, extra in {"KQUAD": ",N_SLICES=12", **QUSE_FAMILIES}.items():
        result = case(
            f"{family},L={LENGTH},K1={K1_SMALL!r},TILT={TILT_SMALL!r}" + extra.format(a=ANGLE)
        )
        bx, by = result[2]
        expected = abs(K1_SMALL * LENGTH * math.sin(2 * TILT_SMALL)) * math.sqrt(bx * by) / 4
        observed = [abs(result[0]["h10010"]), abs(result[0]["h10100"])]
        print(
            f"{family} K1={K1_SMALL} TILT={TILT_SMALL} |h10010|, |h10100|: observed="
            f"[{observed[0]:.10g}, {observed[1]:.10g}] expected=[{expected:.10g}, {expected:.10g}]"
            " (|K1 L sin(2 TILT)| sqrt(betax betay)/4)"
        )
        bad = any(abs(o - expected) > 1e-10 * expected for o in observed)
        if family == "KQUAD" and bad:
            sys.exit("The KQUAD control disagrees with the coupling formula: invalid comparison.")
        defect |= bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Found<br><small>KQUSE and bend roll</small> | `gpt-6-astra` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed<br><small>KQUSE and bend roll</small> | `gpt-6.1-sol` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

