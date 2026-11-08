---
title: "DIA-02: computeBunchMoments returns zero for the top-order pure-y moment and crashes at orders 0 and 1"
hide:
  - toc
---

# DIA-02 — `computeBunchMoments` returns zero for the top-order pure-y moment and crashes at orders 0 and 1

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Diagnostics |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Issue | [[gh:https://github.com/PyORBIT-Collaboration/PyORBIT3/issues/121]] |
| Pull request | [[gh:https://github.com/PyORBIT-Collaboration/PyORBIT3/pull/132]], fix |
| Source | [`BunchTwissAnalysis.cc#L256-L260`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTwissAnalysis.cc#L256-L260)<br>[`#L286-L289`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTwissAnalysis.cc#L286-L289)<br>[`#L209-L212`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTwissAnalysis.cc#L209-L212)<br>[`#L332-L334`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTwissAnalysis.cc#L332-L334)<br>[`diagnostics.py#L190-L194`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/diagnostics.py#L190-L194) |

`BunchTwissAnalysis.computeBunchMoments(bunch, order, …)` accumulates the moments ⟨(x−⟨x⟩)ⁱ(y−⟨y⟩)ʲ⟩ only for j < order, so the pure-y moment (0, order) is never summed and is returned as 0, and `Moments` and `TeapotMomentsNode` write that zero. For ⟨y²⟩ = 4 mm² and βy = 20 m, the order-2 moment (0, 2) is 0 instead of 2.000e-7 m while (2, 0) is correct, and the order-4 moment (0, 4) is 0 instead of 4.000e-14 m². Without a `macrosize` attribute an order-1 request crashes the interpreter with a segmentation fault, because a missing pair of braces adds a term to the table entry (2, 1) after the loops; an order-0 request writes its first-order entries outside its one-row table and crashes too.

Run `python DIA-02.py`.

??? example "Reproducer: DIA-02.py"

    [Download DIA-02.py](DIA-02.py)

    ```python
    """DIA-02: `computeBunchMoments` returns zero for the top-order pure-y moment and crashes at orders 0 and 1

    `BunchTwissAnalysis.computeBunchMoments(bunch, order, …)` accumulates the moments ⟨(x−⟨x⟩)ⁱ(y−⟨y⟩)ʲ⟩
    only for j < order, so the pure-y moment (0, order) is never summed and is returned as 0, and
    `Moments` and `TeapotMomentsNode` write that zero. For ⟨y²⟩ = 4 mm² and βy = 20 m, the order-2
    moment (0, 2) is 0 instead of 2.000e-7 m while (2, 0) is correct, and the order-4 moment (0, 4) is 0
    instead of 4.000e-14 m². Without a `macrosize` attribute an order-1 request crashes the interpreter
    with a segmentation fault, because a missing pair of braces adds a term to the table entry (2, 1)
    after the loops; an order-0 request writes its first-order entries outside its one-row table and
    crashes too.

    The script builds 32 particles from all sign combinations of x = ±1 mm, x′ = ±0.1 mrad,
    y = ±2 mm, y′ = ±0.1 mrad and dE = ±1 MeV, computes moments of order 2, 3 and 4 without
    dispersion subtraction or emittance normalization, and compares (2, 0), (0, 2) and (0, 4) with
    ⟨x²⟩/βx, ⟨y²⟩/βy and ⟨y⁴⟩/βy². The (0, 2) moment of an order-3 request is the control. Child
    interpreters run the call at orders 1 and 0 and report their exit status.
    """

    import itertools
    import signal
    import subprocess
    import sys

    from orbit.core.bunch import Bunch, BunchTwissAnalysis  # ty: ignore[unresolved-import]


    def bunch():
        b = Bunch()
        b.getSyncParticle().kinEnergy(1.0)
        for sx, sxp, sy, syp, se in itertools.product((1, -1), repeat=5):
            b.addParticle(sx * 1e-3, sxp * 1e-4, sy * 2e-3, syp * 1e-4, 0.0, se * 1e-3)
        return b


    if len(sys.argv) == 3 and sys.argv[1] == "child":
        # Child mode: one call at the given order, then print the (0,0) entry.
        t = BunchTwissAnalysis()
        t.computeBunchMoments(bunch(), int(sys.argv[2]), -1, -1)
        print(t.getBunchMoment(0, 0))
        sys.exit(0)

    b, t = bunch(), BunchTwissAnalysis()
    # Sign-factorial design: <x^2> = 1e-6, <x'^2> = 1e-8 -> eps_x = 1e-7, beta_x = 10 m;
    # <y^2> = 4e-6, <y'^2> = 1e-8 -> eps_y = 2e-7, beta_y = 20 m; <y^4> = (2e-3)^4 = 1.6e-11.
    # With emitnormflag < 0 the code divides x^i y^j by beta_x^(i/2) beta_y^(j/2):
    # (2,0) = 1e-6/10 = 1e-7, (0,2) = 4e-6/20 = 2e-7, (0,4) = 1.6e-11/20^2 = 4e-14.
    EXP = {(2, 0): 1e-7, (0, 2): 2e-7, (0, 4): 4e-14}
    obs = {}
    for order, ij in ((2, (2, 0)), (2, (0, 2)), (3, (0, 2)), (4, (0, 4))):
        t.computeBunchMoments(b, order, -1, -1)
        obs[order, ij] = t.getBunchMoment(*ij)
        role = " (control: not the top order)" if (order, ij) == (3, (0, 2)) else ""
        print(f"order {order}: moment {ij} observed {obs[order, ij]:.6g}; expected {EXP[ij]:.6g}{role}")


    def child(order):
        r = subprocess.run(
            [sys.executable, "-I", __file__, "child", str(order)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        status = (
            f"killed by {signal.Signals(-r.returncode).name}"
            if r.returncode < 0
            else f"exit status {r.returncode}"
        )
        # Expected: a normal return; the code sets (0,0) to 1 for every order.
        print(
            f"order {order}: child {status}, stdout {r.stdout.strip()!r}; "
            "expected exit status 0 and (0,0) = 1.0"
        )
        return r.returncode


    rc1, rc0 = child(1), child(0)

    top_zero = obs[2, (0, 2)] == 0.0 and obs[4, (0, 4)] == 0.0
    others_ok = abs(obs[2, (2, 0)] - 1e-7) < 1e-20 and abs(obs[3, (0, 2)] - 2e-7) < 1e-20
    if top_zero and others_ok and rc1 != 0 and rc0 != 0:
        print("DEFECT REPRODUCED")
    else:
        print("NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

