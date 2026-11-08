---
title: "DIA-01: Transverse Twiss and dispersion getters return NaN for a bunch without energy spread"
hide:
  - toc
---

# DIA-01 — Transverse Twiss and dispersion getters return NaN for a bunch without energy spread

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Diagnostics |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`BunchTwissAnalysis.cc#L397-L464`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTwissAnalysis.cc#L397-L464)<br>[`BunchTwissAnalysis.cc#L368-L386`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTwissAnalysis.cc#L368-L386)<br>[`BunchTwissAnalysis.cc#L340-L343`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTwissAnalysis.cc#L340-L343)<br>[`BunchTwissAnalysis.cc#L241-L248`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTwissAnalysis.cc#L241-L248)<br>[`#L271-L278`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTwissAnalysis.cc#L271-L278)<br>[`wrap_bunch_twiss_analysis.cc#L192-L203`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/wrap_bunch_twiss_analysis.cc#L192-L203)<br>[`diagnostics.py#L30-L41`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/diagnostics.py#L30-L41) |

`BunchTwissAnalysis.getAlpha`, `getBeta`, `getGamma`, `getDispersion` and `getDispersionDerivative` for x and y remove the energy correlation by dividing by the energy variance ⟨dE²⟩ without the zero check that `getEmittance` has. For a bunch with no energy spread all of them return NaN, `getTwiss` returns NaN for α, β and γ beside a finite emittance, and `computeBunchMoments` returns NaN moments; for 16 particles at x = ±1 mm, x′ = ±0.1 mrad the correct values are α = 0, β = 10.00 m, γ = 0.1000 m⁻¹ and zero dispersion. With the same nonzero dE for every particle, rounding leaves a variance of −7.425e-22 GeV² at dE = 1 MeV, and the bunch reports a dispersion of 0.04205 m instead of 0.

Run `python DIA-01.py`.

??? example "Reproducer: DIA-01.py"

    [Download DIA-01.py](DIA-01.py)

    ```python
    """DIA-01: Transverse Twiss and dispersion getters return NaN for a bunch without energy spread

    `BunchTwissAnalysis.getAlpha`, `getBeta`, `getGamma`, `getDispersion` and `getDispersionDerivative`
    for x and y remove the energy correlation by dividing by the energy variance ⟨dE²⟩ without the zero
    check that `getEmittance` has. For a bunch with no energy spread all of them return NaN, `getTwiss`
    returns NaN for α, β and γ beside a finite emittance, and `computeBunchMoments` returns NaN moments;
    for 16 particles at x = ±1 mm, x′ = ±0.1 mrad the correct values are α = 0, β = 10.00 m, γ = 0.1000
    m⁻¹ and zero dispersion. With the same nonzero dE for every particle, rounding leaves a variance of
    −7.425e-22 GeV² at dE = 1 MeV, and the bunch reports a dispersion of 0.04205 m instead of 0.

    The script builds 16 particles from all sign combinations of x = ±1 mm, x′ = ±0.1 mrad, y = ±2 mm,
    y′ = ±0.1 mrad, analyzes them with dE = 0 and with dE = 1 MeV for every particle, and compares
    each getter and the order-2 moment (2, 0) with the closed-form values. A control repeats the
    analysis with dE = ±1 MeV, which leaves the transverse moments unchanged and gives the
    closed-form values.
    """

    import itertools
    import math

    from orbit.core.bunch import Bunch, BunchTwissAnalysis  # ty: ignore[unresolved-import]

    # Sign-factorial design: every cross moment vanishes and <x> = <x'> = 0, so
    # <x^2> = 1e-6 m^2, <x'^2> = 1e-8, eps_x = sqrt(<x^2><x'^2>) = 1e-7 m,
    # beta_x = <x^2>/eps_x = 10 m, alpha_x = -<x x'>/eps_x = 0, gamma_x = <x'^2>/eps_x = 0.1 1/m.
    # x is uncorrelated with dE, so the dispersion D = <x dE>/<dE^2> * p*beta and D' vanish.
    EXPECTED = {"alpha": 0.0, "beta": 10.0, "gamma": 0.1, "eps": 1e-7, "D": 0.0, "D'": 0.0}


    def analyze(dE_values):
        b = Bunch()
        b.mass(0.93827208816)
        b.getSyncParticle().kinEnergy(1.0)
        for sx, sxp, sy, syp in itertools.product((1, -1), repeat=4):
            for de in dE_values:
                b.addParticle(sx * 1e-3, sxp * 1e-4, sy * 2e-3, syp * 1e-4, 0.0, de)
        t = BunchTwissAnalysis()
        t.analyzeBunch(b)
        alpha, beta, gamma, eps = t.getTwiss(0)
        vals = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "eps": eps,
            "D": t.getDispersion(0),
            "D'": t.getDispersionDerivative(0),
        }
        # emitnormflag = -1: (2,0) = <x^2>/beta_x = 1e-7 m.
        t.computeBunchMoments(b, 2, -1, -1)
        return vals, t.getBunchMoment(2, 0), t.getCorrelation(5, 5)


    def line(label, vals):
        return f"{label}: " + ", ".join(
            f"{k} {v:.6g} (expected {EXPECTED[k]:.6g})" for k, v in vals.items()
        )


    mono, m20_mono, _ = analyze([0.0])
    print(line("dE = 0", mono))
    print(f"dE = 0: moment (2,0) {m20_mono:.6g} (expected 1e-07 = <x^2>/beta_x)")

    const, _, var_const = analyze([1e-3])
    print(line("dE = 1 MeV for all", const))
    print(f"dE = 1 MeV for all: <dE^2> - <dE>^2 evaluated as {var_const:.4g} GeV^2 (exact value 0)")

    ctrl, m20_ctrl, _ = analyze([1e-3, -1e-3])
    print(line("dE = +-1 MeV (control)", ctrl))
    print(f"dE = +-1 MeV (control): moment (2,0) {m20_ctrl:.6g} (expected 1e-07)")

    twiss_nan = all(math.isnan(mono[k]) for k in ("alpha", "beta", "gamma", "D", "D'"))
    eps_ok = abs(mono["eps"] - 1e-7) < 1e-20
    const_wrong = abs(const["D"]) > 1e-3
    ctrl_ok = all(abs(ctrl[k] - EXPECTED[k]) < 1e-12 for k in EXPECTED) and abs(m20_ctrl - 1e-7) < 1e-20
    if twiss_nan and math.isnan(m20_mono) and eps_ok and const_wrong and ctrl_ok:
        print("DEFECT REPRODUCED")
    else:
        print("NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

