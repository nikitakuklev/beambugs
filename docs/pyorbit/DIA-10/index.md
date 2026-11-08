---
title: "DIA-10: TeapotTuneAnalysisNode.getData(bunch, index) rejects every particle index"
hide:
  - toc
---

# DIA-10 — `TeapotTuneAnalysisNode.getData(bunch, index)` rejects every particle index

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Diagnostics |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`TeapotDiagnosticsNode.py#L289-L297`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/TeapotDiagnosticsNode.py#L289-L297)<br>[`TeapotDiagnosticsNode.py#L300-L326`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/TeapotDiagnosticsNode.py#L300-L326) |

`TeapotTuneAnalysisNode.getData(bunch, index)` raises "particle index < 0" when `index < bunch.getSize()` holds and "particle index > bunch.getSize() - 1" for every other index, so every integer index raises `ValueError`. The per-particle forms of `getData`, `getTunes` and `getActions` therefore cannot be used; only the all-particle form, with the index omitted, returns data. For a three-particle bunch, indices 0, 1 and 2 raise "particle index < 0" instead of returning that particle's phases, tunes and actions, such as the action 5.000e-8 m of particle 0.

Run `python DIA-10.py`.

??? example "Reproducer: DIA-10.py"

    [Download DIA-10.py](DIA-10.py)

    ```python
    """DIA-10: `TeapotTuneAnalysisNode.getData(bunch, index)` rejects every particle index

    `TeapotTuneAnalysisNode.getData(bunch, index)` raises "particle index < 0" when `index <
    bunch.getSize()` holds and "particle index > bunch.getSize() - 1" for every other index, so every
    integer index raises `ValueError`. The per-particle forms of `getData`, `getTunes` and `getActions`
    therefore cannot be used; only the all-particle form, with the index omitted, returns data. For a
    three-particle bunch, indices 0, 1 and 2 raise "particle index < 0" instead of returning that
    particle's phases, tunes and actions, such as the action 5.000e-8 m of particle 0.

    The script analyzes a three-particle bunch twice with a Twiss normalization, requests
    `getData`, `getTunes` and `getActions` for each valid index and `getData` for the invalid indices
    −1 and 3, and compares each valid result with the corresponding entry of the all-particle
    `getData` result, which is the control.
    """

    import numpy as np
    from orbit.core.bunch import Bunch  # ty: ignore[unresolved-import]
    from orbit.diagnostics import TeapotTuneAnalysisNode

    node = TeapotTuneAnalysisNode()
    node.setNormMatrixFromTwiss(10.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    b = Bunch()
    b.getSyncParticle().kinEnergy(1.0)
    for k in (1, 2, 3):
        b.addParticle(1e-3 * k, 0.0, 0.0, 1e-4 * k, 0.0, 0.0)
    node.track({"bunch": b})
    node.track({"bunch": b})
    all_data = {k: np.asarray(v) for k, v in node.getData(b).items()}
    # With beta = 10 m, alpha = 0: action J = (x^2/beta + beta x'^2)/2, so particle k has
    # J_x = (1e-3 k)^2/20 = 5e-8 k^2 and J_y = 10 (1e-4 k)^2/2 = 5e-8 k^2.
    print(
        "all-particle getData (control): action_1 "
        + ", ".join(f"{v:.6g}" for v in all_data["action_1"])
        + "; expected 5e-08, 2e-07, 4.5e-07"
    )
    ok = all(abs(v - 5e-8 * k * k) < 1e-20 for k, v in zip((1, 2, 3), all_data["action_1"]))


    def call(f, *args):
        try:
            return f(*args)
        except ValueError as err:
            return f"ValueError: {err}"


    for i in range(3):
        exp_data = {k: float(v[i]) for k, v in all_data.items()}
        for name, f, exp in (
            ("getData", node.getData, exp_data),
            ("getTunes", node.getTunes, (exp_data["tune_1"], exp_data["tune_2"])),
            ("getActions", node.getActions, (exp_data["action_1"], exp_data["action_2"])),
        ):
            obs = call(f, b, i)
            print(f"{name}(bunch, {i}): observed {obs!r}; expected {exp!r}")
            ok &= isinstance(obs, str)
    for i in (-1, 3):
        print(
            f"getData(bunch, {i}): observed {call(node.getData, b, i)!r}; expected a ValueError (index out of range)"
        )

    print("DEFECT REPRODUCED" if ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

