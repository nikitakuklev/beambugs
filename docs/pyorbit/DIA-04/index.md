---
title: "DIA-04: Dispersion-subtracted computeBunchMoments centers x on the uncorrected mean"
hide:
  - toc
---

# DIA-04 — Dispersion-subtracted `computeBunchMoments` centers x on the uncorrected mean

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Diagnostics |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Issue | [[gh:https://github.com/PyORBIT-Collaboration/PyORBIT3/issues/121]] |
| Pull request | [[gh:https://github.com/PyORBIT-Collaboration/PyORBIT3/pull/132]], fix |
| Source | [`BunchTwissAnalysis.cc#L153`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTwissAnalysis.cc#L153-L153)<br>[`#L167-L204`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTwissAnalysis.cc#L167-L204)<br>[`#L237-L239`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTwissAnalysis.cc#L237-L239)<br>[`#L250-L251`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTwissAnalysis.cc#L250-L251)<br>[`#L267-L269`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTwissAnalysis.cc#L267-L269)<br>[`#L280-L281`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTwissAnalysis.cc#L280-L281)<br>[`TeapotDiagnosticsNode.py#L78-L81`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/TeapotDiagnosticsNode.py#L78-L81)<br>[`diagnostics.py#L158-L165`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/diagnostics.py#L158-L165) |

`BunchTwissAnalysis.computeBunchMoments(bunch, order, 1, …)`, the setting `TeapotMomentsNode` uses by default, subtracts the dispersive offset D·δ from each particle's x, with D the dispersion and δ the momentum deviation, but centers the result on the plain mean ⟨x⟩: the branch that would compute ⟨x − D·δ⟩ is guarded by a variable that is always zero. Whenever ⟨δ⟩ ≠ 0, (1, 0) reports ⟨x⟩ instead of the betatron centroid and the higher x moments are taken about a centre offset by D⟨δ⟩. For D = 2 m, ⟨δ⟩ = 1.5e-3, βx = 10 m and a 1 mm RMS betatron size, (2, 0) is 1.000e-6 m instead of 1.000e-7 m. Moments of y alone are unaffected.

Run `python DIA-04.py`.

??? example "Reproducer: DIA-04.py"

    [Download DIA-04.py](DIA-04.py)

    ```python
    """DIA-04: Dispersion-subtracted `computeBunchMoments` centers x on the uncorrected mean

    `BunchTwissAnalysis.computeBunchMoments(bunch, order, 1, …)`, the setting `TeapotMomentsNode` uses
    by default, subtracts the dispersive offset D·δ from each particle's x, with D the dispersion and δ
    the momentum deviation, but centers the result on the plain mean ⟨x⟩: the branch that would compute
    ⟨x − D·δ⟩ is guarded by a variable that is always zero. Whenever ⟨δ⟩ ≠ 0, (1, 0) reports ⟨x⟩ instead
    of the betatron centroid and the higher x moments are taken about a centre offset by D⟨δ⟩. For D = 2
    m, ⟨δ⟩ = 1.5e-3, βx = 10 m and a 1 mm RMS betatron size, (2, 0) is 1.000e-6 m instead of 1.000e-7 m.
    Moments of y alone are unaffected.

    The script builds 32 particles from betatron offsets x_β = ±1 mm, x′ = ±0.1 mrad, y = ±2 mm,
    y′ = ±0.1 mrad and two momentum deviations δ = ⟨δ⟩ ± 1e-3, places them at x = x_β + D·δ with
    D = 2 m, computes order-3 moments with dispersion subtraction and compares (1, 0), (2, 0) and
    (0, 2) with the moments of the betatron offsets. The control has ⟨δ⟩ = 0, where both centerings
    coincide.
    """

    import itertools

    import numpy as np
    from orbit.core.bunch import Bunch, BunchTwissAnalysis  # ty: ignore[unresolved-import]

    D = 2.0  # m


    def run(mean_delta):
        b = Bunch()
        sp = b.getSyncParticle()
        sp.kinEnergy(1.0)
        # The code converts dE to delta = dE / (beta^2 E), with E the total energy.
        k = sp.beta() ** 2 * (sp.kinEnergy() + sp.mass())
        rows = []
        for sx, sxp, sy, syp, sd in itertools.product((1, -1), repeat=5):
            xb, d = sx * 1e-3, mean_delta + sd * 1e-3
            rows.append((xb, d))
            b.addParticle(xb + D * d, sxp * 1e-4, sy * 2e-3, syp * 1e-4, 0.0, d * k)
        t = BunchTwissAnalysis()
        # dispersionflag = 1, emitnormflag = -1 (divide by sqrt(beta)); order 3 so that (0,2) is not the
        # top-order pure-y moment, which the code never accumulates.
        t.computeBunchMoments(b, 3, 1, -1)
        xb = np.array([r[0] for r in rows])
        return t, xb


    ok = True
    for label, mean_delta in (("<delta> = 1.5e-3", 1.5e-3), ("<delta> = 0 (control)", 0.0)):
        t, xb = run(mean_delta)
        disp, beta_x, beta_y = t.getDispersion(0), t.getBeta(0), t.getBeta(1)
        # Correct dispersion-subtracted moments are those of the betatron offsets x_b (exact by
        # construction): (1,0) = <x_b> = 0, (2,0) = <(x_b - <x_b>)^2>/beta_x = 1e-6/10 = 1e-7.
        # y carries no dispersion: (0,2) = <y^2>/beta_y = 4e-6/20 = 2e-7.
        exp10 = xb.mean()
        exp20 = ((xb - xb.mean()) ** 2).mean() / beta_x
        m10, m20, m02 = t.getBunchMoment(1, 0), t.getBunchMoment(2, 0), t.getBunchMoment(0, 2)
        print(
            f"{label}: D observed {disp:.6g} (built 2), beta_x {beta_x:.6g} (built 10); "
            f"(1,0) observed {m10:.6g}, expected {exp10:.6g}; (2,0) observed {m20:.6g}, expected {exp20:.6g}; "
            f"(0,2) observed {m02:.6g}, expected {4e-6 / beta_y:.6g}"
        )
        if mean_delta:
            # Centered on <x> = D<delta> instead of <x - D delta> = 0:
            # (2,0) = (<x_b^2> + (D<delta>)^2)/beta_x = (1e-6 + 9e-6)/10 = 1e-6, (1,0) = D<delta> = 3e-3.
            print(
                f"{label}: centering on <x> predicts (1,0) {D * mean_delta:.6g}, "
                f"(2,0) {(1e-6 + (D * mean_delta) ** 2) / beta_x:.6g}"
            )
            ok &= abs(m20 - exp20) > 0.5 * exp20 and abs(m10 - D * mean_delta) < 1e-15
        else:
            ok &= abs(m20 - exp20) < 1e-12 * exp20 and abs(m10) < 1e-15
        ok &= abs(m02 - 4e-6 / beta_y) < 1e-12 * m02

    print("DEFECT REPRODUCED" if ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed<br><small class="bb-scope">second moment with a mean energy offset</small> | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

