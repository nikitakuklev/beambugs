---
title: "OPT-20: Matrix-method tune shift with amplitude uses the wrong betatron slope phase"
hide:
  - toc
---

# OPT-20 — Matrix-method tune shift with amplitude uses the wrong betatron slope phase

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`7514db3`, head build) |
| Source | [`computeTuneShiftWithAmplitudeM`, src/twiss.cc:4556–4561](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/twiss.cc#L4556-L4561)<br>[outputs are derivatives of the tunes with respect to A, doc/elegant.tex:8412–8413](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8412-L8413)<br>[concatenated multi-turn third-order matrix method appears reliable for many cases, doc/elegant.tex:8403–8404](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8403-L8404)<br>[deprecates the command, doc/elegant.tex:8389–8392](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8389-L8392)<br>[the matrix method does not work well with all lattices, doc/elegant.tex:8442–8446](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L8442-L8446) |

Elegant's `tune_shift_with_amplitude` with `turns=0` averages the multi-turn third-order map over the betatron ellipse with the slope phase written as α sin θ + cos θ instead of sin θ + α cos θ, where α is the Twiss alpha at the start of the ring. The coefficients then depend on the starting point, and the cross terms dνx/dAy and dνy/dAx differ. For eight FODO cells with one thin octupole of K3L = 3 m⁻³, first-order perturbation theory gives dνx/dAx = 1.411, dνx/dAy = dνy/dAx = −7.538 and dνy/dAy = 10.07 m⁻¹; Elegant gives 3.658, −29.47, −19.54 and 39.35 m⁻¹ from one start and 1.416, −86.53, −7.561 and 115.5 m⁻¹ from another. The tracking method (`turns` > 0) agrees with theory.

Run `python OPT-20.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: OPT-20.py"

    [Download OPT-20.py](OPT-20.py)

    ```python
    """OPT-20: Matrix-method tune shift with amplitude uses the wrong betatron slope phase

    Elegant's `tune_shift_with_amplitude` with `turns=0` averages the multi-turn third-order map over
    the betatron ellipse with the slope phase written as α sin θ + cos θ instead of sin θ + α cos θ,
    where α is the Twiss alpha at the start of the ring. The coefficients then depend on the starting
    point, and the cross terms dνx/dAy and dνy/dAx differ. For eight FODO cells with one thin octupole
    of K3L = 3 m⁻³, first-order perturbation theory gives dνx/dAx = 1.411, dνx/dAy = dνy/dAx = −7.538
    and dνy/dAy = 10.07 m⁻¹; Elegant gives 3.658, −29.47, −19.54 and 39.35 m⁻¹ from one start and 1.416,
    −86.53, −7.561 and 115.5 m⁻¹ from another. The tracking method (`turns` > 0) agrees with theory.

    The script runs serial Elegant `tune_shift_with_amplitude turns=0` with `twiss_output
    concat_order=3` on eight FODO cells of first-order quadrupoles and one thin octupole, starting
    the same ring at the octupole and at a drift. The expected coefficients are the first-order
    thin-octupole values K3L/(32π)·[βx², −2βxβy, −2βxβy, βy²], with the beta functions at the
    octupole computed from analytic thick-quadrupole and drift matrices; Elegant's own tune and beta
    functions at the octupole are controls.
    """

    import sys

    import numpy as np
    from _elegant import run, verdict

    K3L = 3.0  # thin octupole strength, m^-3
    QF, QD, LQ = 1.1, -1.0, 0.25  # quadrupole K1 (m^-2) and length (m)
    LD, LE = 0.7, 0.9  # drift lengths, m
    LATTICE = f"""F: QUAD,L={LQ},K1={QF},ORDER=1
    G: QUAD,L={LQ},K1={QD},ORDER=1
    D: DRIF,L={LD}
    E: DRIF,L={LE}
    O: MULT,ORDER=3,KNL={K3L}
    C: LINE=(F,D,G,E)
    A: LINE=(O,8*C)
    B: LINE=(D,G,E,7*C,O,F)
    """
    NAMES = ["dnux/dAx", "dnux/dAy", "dnuy/dAx", "dnuy/dAy"]
    RTOL = 1e-3


    def quad(k, length):
        if k > 0:
            w = np.sqrt(k)
            return np.array(
                [
                    [np.cos(w * length), np.sin(w * length) / w],
                    [-w * np.sin(w * length), np.cos(w * length)],
                ]
            )
        w = np.sqrt(-k)
        return np.array(
            [
                [np.cosh(w * length), np.sinh(w * length) / w],
                [w * np.sinh(w * length), np.cosh(w * length)],
            ]
        )


    def drift(length):
        return np.array([[1.0, length], [0.0, 1.0]])


    def periodic_beta(m):
        """Periodic beta function and tune of a 2x2 one-cell matrix (8 cells per turn)."""
        cos_mu = (m[0, 0] + m[1, 1]) / 2
        sin_mu = np.sign(m[0, 1]) * np.sqrt(1 - cos_mu**2)
        mu = np.arctan2(sin_mu, cos_mu) % (2 * np.pi)
        return m[0, 1] / sin_mu, 8 * mu / (2 * np.pi)


    # The octupole sits at the cell start (entrance of F). The cell matrix in transport order is
    # E G D F (the rightmost acts first); y sees -K1.
    beta_x, nux = periodic_beta(drift(LE) @ quad(QD, LQ) @ drift(LD) @ quad(QF, LQ))
    beta_y, nuy = periodic_beta(drift(LE) @ quad(-QD, LQ) @ drift(LD) @ quad(-QF, LQ))
    # Thin octupole H = K3L (x^4 - 6 x^2 y^2 + y^4)/24 at s = 0, x = sqrt(2 Jx betax) cos(phi): the
    # phase average gives <H> = K3L (betax^2 Jx^2 - 4 betax betay Jx Jy + betay^2 Jy^2)/16, so
    # dnu/dJ = (1/2 pi) d2<H>/dJ dJ, and with A = 2J (Elegant's amplitude) the coefficients below.
    expected = (
        K3L
        / (32 * np.pi)
        * np.array([beta_x**2, -2 * beta_x * beta_y, -2 * beta_x * beta_y, beta_y**2])
    )


    def fmt(values):
        return "[" + ", ".join(f"{v:.10g}" for v in values) + "]"


    defect = False
    print(f"analytic: nux={nux:.10g} nuy={nuy:.10g} betax(O)={beta_x:.10g} m betay(O)={beta_y:.10g} m")
    for line in ("A", "B"):
        commands = (
            f"&run_setup lattice=lattice.lte, use_beamline={line}, p_central_mev=2400,"
            " default_order=3 &end\n"
            "&tune_shift_with_amplitude turns=0 &end\n"
            "&twiss_output filename=twiss.sdds, matched=1, output_at_each_step=1, concat_order=3 &end\n"
            "&run_control &end\n&bunched_beam &end\n&track &end\n"
        )
        table = run({"lattice.lte": LATTICE, "run.ele": commands}, outputs=("twiss.sdds",))[
            "twiss.sdds"
        ]
        names = list(table[0, "ElementName"])
        at = names.index("O")
        tunes = [float(table["nux"][0]), float(table["nuy"][0])]
        betas = [float(table[0, "betax"][at]), float(table[0, "betay"][at])]
        alphas = [float(table[0, "alphax"][0]), float(table[0, "alphay"][0])]
        print(
            f"start {line} control nux, nuy, betax(O), betay(O): observed={fmt(tunes + betas)}"
            f" expected={fmt([nux, nuy, beta_x, beta_y])}"
        )
        if not np.allclose(tunes + betas, [nux, nuy, beta_x, beta_y], rtol=1e-9):
            sys.exit("Elegant's linear optics disagree with the analytic matrices: invalid comparison.")
        observed = np.array([float(table[name][0]) for name in NAMES])
        if not np.all(observed):
            sys.exit("Elegant wrote zero tune-shift coefficients: the command did not run.")
        print(f"start {line} alphax, alphay at the start: {fmt(alphas)}")
        print(
            f"start {line} {', '.join(NAMES)} (1/m): observed={fmt(observed)} expected={fmt(expected)}"
        )
        print(f"start {line} observed/expected: {fmt(observed / expected)}")
        defect |= not np.allclose(observed, expected, rtol=RTOL)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-04 | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

