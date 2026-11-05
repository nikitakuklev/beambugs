---
title: "OPT-03: CCBEND and LGBEND radiation totals omit repeated periods"
hide:
  - toc
---

# OPT-03 — CCBEND and LGBEND radiation totals omit repeated periods

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P1 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Source | [`incrementRadIntegrals`, CCBEND branch, src/twiss.cc:3091–3102](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3091-L3102)<br>[`incrementRadIntegrals`, LGBEND branch, src/twiss.cc:3103–3124](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3103-L3124)<br>[the generic bend branch, src/twiss.cc:3331–3335](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3331-L3335)<br>[the common per-element output, src/twiss.cc:3346–3351](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3346-L3351)<br>[`addLgbendRadiationIntegrals`, src/lgbend.c:814–912](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/lgbend.c#L814-L912)<br>[`completeRadiationIntegralComputation`, src/twiss.cc:3363–3369](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L3363-L3369)<br>[`radiation_integrals` option, manual, doc/elegant.tex:8092–8095](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8092-L8095) |

With `twiss_output`, `radiation_integrals=1` and `n_periods` greater than 1, Elegant adds, for CCBEND (canonical combined-function bend) and LGBEND (multi-segment straight-pole bend) elements, the integrals I₁ to I₅ of one period to the totals without multiplying by `n_periods`, although the per-element contributions are multiplied. For a cell of two 1 m bends with curvature 0.1 m⁻¹ at 6000 mₑc, I₂ must be 0.02000 m⁻¹ for one period and 0.04000 m⁻¹ for two, but Elegant writes 0.02000 m⁻¹ for both. The energy loss per turn then stays at 24.88 keV instead of doubling to 49.77 keV, and the damping times double instead of staying unchanged. The same cell with SBEN bends, or written out twice with one period, is correct.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)

??? example "Reproducer: OPT-03.py"

    [Download OPT-03.py](OPT-03.py)

    ```python
    """OPT-03: CCBEND and LGBEND radiation totals omit repeated periods

    With `twiss_output`, `radiation_integrals=1` and `n_periods` greater than 1, Elegant adds, for
    CCBEND (canonical combined-function bend) and LGBEND (multi-segment straight-pole bend) elements,
    the integrals I₁ to I₅ of one period to the totals without multiplying by `n_periods`, although the
    per-element contributions are multiplied. For a cell of two 1 m bends with curvature 0.1 m⁻¹ at
    6000 mₑc, I₂ must be 0.02000 m⁻¹ for one period and 0.04000 m⁻¹ for two, but Elegant writes 0.02000
    m⁻¹ for both. The energy loss per turn then stays at 24.88 keV instead of doubling to 49.77 keV,
    and the damping times double instead of staying unchanged. The same cell with SBEN bends, or
    written out twice with one period, is correct.

    The script runs the same three-meter cell (two quadrupoles and two 1 m bends, 6000 mₑc) with
    `n_periods` = 1 and 2, in serial Elegant (and with `--mpi` in 3-rank Pelegant), with the bends as
    CCBEND and as a two-segment uniform-field LGBEND whose configuration it writes inline. For one period
    it checks I₂ = Σ L/ρ², the energy loss and the damping times against their closed-form definitions;
    for two periods the totals and the energy loss must double and the damping times be unchanged. The
    same cell with SBEN bends, and each CCBEND and LGBEND cell written out twice with one period, are the
    controls; the sum of the written per-element I₂ contributions is printed.
    """

    import sys

    import numpy as np
    from _elegant import modes, run, verdict

    # SDDS constants.h, as Elegant uses them.
    C, RE, ELECTRON_MEV = 299792458.0, 2.81794092e-15, 0.51099906
    P_CENTRAL = 6000.0  # m_e c
    BEND_L, BEND_ANGLE = 1.0, 0.1  # two bends per cell; curvature h = angle/L = 0.1 1/m
    CELL_L = 0.5 + 1.0 + 0.5 + 1.0  # m: QF, bend, QD, bend
    RTOL = 1e-6  # Elegant's rounded constants and relativistic energy

    NAMES = ("I1", "I2", "I3", "I4", "I5", "U0", "taux", "tauy", "taudelta")


    def lgbend_configuration():
        """LGBEND configuration of a uniform-field 1 m arc of 0.1 rad in two equal segments.

        Built from circular geometry with zero gradients and no fringe terms: each segment spans half
        the chord (longitudinal length) and half the angle; the reference enters at x = 0 with slope
        angle/2 relative to the chord, reaches the sagitta rho (1 - cos(angle/2)) at the shared face,
        which is perpendicular to the reference, and leaves at x = 0. The vertex is where the entry and
        exit tangents meet. OPTIMIZE_FSE=0 keeps the segment strengths as given.
        """
        rho = BEND_L / BEND_ANGLE
        chord = 2 * rho * np.sin(BEND_ANGLE / 2)
        half = BEND_ANGLE / 2
        first = {
            "LONGIT_L": chord / 2, "ANGLE": half, "ENTRY_X": 0, "ENTRY_ANGLE": half,
            "EXIT_X": rho * (1 - np.cos(half)), "EXIT_ANGLE": 0, "K1": 0, "K2": 0, "PREDRIFT": 0,
            "ZEntry": 0, "XEntry": 0, "ZExit": chord, "XExit": 0, "ZVertex": chord / 2,
            "XVertex": chord / 2 * np.tan(half),
        }  # fmt: skip
        second = {
            "LONGIT_L": chord / 2, "ANGLE": half, "EXIT_X": 0, "EXIT_ANGLE": -half, "K1": 0, "K2": 0,
            "POSTDRIFT": 0,
        }  # fmt: skip
        pages = "".join(
            f"{len(page)}\n" + "".join(f"{k} {v:.17g}\n" for k, v in page.items())
            for page in (first, second)
        )
        return (
            "SDDS1\n&column name=ParameterName,type=string &end\n"
            "&column name=ParameterValue,type=double &end\n&data mode=ascii &end\n" + pages
        )


    def lattice(bend, copies):
        element = {
            "CCBEND": f"D: CCBEND,L={BEND_L!r},ANGLE={BEND_ANGLE!r},N_SLICES=16",
            "LGBEND": 'D: LGBEND,CONFIGURATION="bend.sdds",N_SLICES=32,OPTIMIZE_FSE=0',
            "SBEN": f"D: SBEN,L={BEND_L!r},ANGLE={BEND_ANGLE!r}",
        }[bend]
        return (
            "QF: QUAD,L=0.5,K1=0.6,TILT=0\nQD: QUAD,L=0.5,K1=-0.54,TILT=0\n"
            f"{element}\nCELL: LINE=(QF,D,QD,D)\nBL: LINE=({copies}*CELL)\n"
        )


    def totals(mode, bend, periods, copies=1):
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P_CENTRAL!r}, "
            "default_order=2 &end\n"
            "&run_control n_steps=1, n_passes=1 &end\n"
            f"&twiss_output filename=twiss.sdds, radiation_integrals=1, n_periods={periods} &end\n"
            "&bunched_beam n_particles_per_bunch=12 &end\n"
            "&track &end\n"
        )
        files = {"lattice.lte": lattice(bend, copies), "run.ele": commands}
        if bend == "LGBEND":
            files["bend.sdds"] = lgbend_configuration()
        twiss = run(files, mode, outputs=("twiss.sdds",))["twiss.sdds"]
        values = np.array([twiss[name][0] for name in NAMES])
        return values, np.sum(twiss[0, "dI2"])


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in np.atleast_1d(values)) + "]"


    def compare(label, observed, expected):
        """Print one case; return True when observed and expected disagree."""
        bad = not np.allclose(observed, expected, rtol=RTOL, atol=0)
        print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
        return bad


    def single_period_oracle(i4_over_i2):
        """Closed-form I2, U0 and damping times of one cell (E in MeV, times in s).

        I2 = sum of L/rho^2 = 2 L h^2 for two uniform bends. U0 = C_gamma E^4 I2/(2 pi) with
        C_gamma = 4 pi r_e/(3 (m c^2)^3). The damping times are tau_u = 2 E T0/(J_u U0) with
        T0 = cell length/c; Jx = 1 - I4/I2, Jy = 1, Jdelta = 2 + I4/I2.
        """
        i2 = 2 * BEND_L * (BEND_ANGLE / BEND_L) ** 2
        energy = np.sqrt(1 + P_CENTRAL**2) * ELECTRON_MEV
        c_gamma = 4 * np.pi / 3 * RE / ELECTRON_MEV**3  # m/MeV^3
        u0 = c_gamma * energy**4 * i2 / (2 * np.pi)
        t0 = CELL_L / C
        return i2, u0, [2 * energy * t0 / (j * u0) for j in (1 - i4_over_i2, 1, 2 + i4_over_i2)]


    defect = False
    print(f"columns: {', '.join(NAMES)}; energies in MeV, times in s")
    for mode in modes():
        for bend in ("SBEN", "CCBEND", "LGBEND"):
            one, sum_one = totals(mode, bend, 1)
            two, sum_two = totals(mode, bend, 2)
            i2, u0, taus = single_period_oracle(one[3] / one[1])
            # One period (the control for the oracle): I2, U0 and the damping times follow from the
            # curvature, the energy and the cell length.
            if compare(
                f"{mode} {bend} 1 period (I2, U0, taux, tauy, taudelta)",
                one[[1, 5, 6, 7, 8]],
                [i2, u0, *taus],
            ):
                sys.exit("The one-period totals disagree with the closed form: comparison not valid.")
            # Two identical periods: every integral and U0 add, T0 doubles, so each damping time
            # 2 E T0/(J U0) is unchanged.
            expected = np.concatenate([2 * one[:6], one[6:]])
            if bend != "SBEN":
                # Control: the cell written out twice with n_periods = 1 must give the same totals, so
                # the bend's own integrals add over repetition and only n_periods loses the factor.
                explicit, _ = totals(mode, bend, 1, copies=2)
                if compare(f"{mode} {bend} cell written twice, 1 period", explicit, expected):
                    sys.exit(f"The explicit two-cell {bend} line does not add: comparison not valid.")
            bad = compare(f"{mode} {bend} 2 periods", two, expected)
            print(
                f"{mode} {bend} sum of dI2 over elements: 1 period={sum_one:.10g} 2 periods={sum_two:.10g}"
            )
            if bend == "SBEN" and bad:
                sys.exit("The SBEN control does not scale with n_periods: comparison not valid.")
            defect |= bad
    print(f"comparison: rtol={RTOL:g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Found<br><small>LGBEND branch</small> | `gpt-6-astra` | xhigh | 2026-11-04 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-15 | 2026.4.0 (`d177ff5347c7`), source with SDDS 3fa650ee9a1c, CPU, serial |
| Confirmed | `gpt-6-astra` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | high | 2026-10-25 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed<br><small>LGBEND branch</small> | `gpt-6.1-sol` | xhigh | 2026-11-05 | master (`0355dd6aca12`), source with SDDS a3f730a14c49, CPU, serial |

