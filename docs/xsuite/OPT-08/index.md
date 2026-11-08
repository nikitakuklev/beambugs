---
title: "OPT-08: Extracted Twiss init loses a downstream reference-momentum change"
hide:
  - toc
---

# OPT-08 — Extracted Twiss init loses a downstream reference-momentum change

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack release [v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4) (`3cd8bfc`) |
| Source | [`TwissTable.get_twiss_init`, twiss_table.py:359–369](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_table.py#L359-L369) |

`TwissTable.get_twiss_init()` loses a reference-momentum change that lies between the table start and the selected row, so an open Twiss restarted from the extracted initial conditions follows a particle with a different physical momentum and orbit. `ReferenceEnergyChange` only changes the normalization, so exact momentum identities and drift transport require the restarted and uninterrupted trajectories to agree. Starting at 1 GeV/c with x = 3 mm and normalized horizontal momentum pₓ = 0.001, rebasing to 2 GeV/c and restarting before a 1 m exact drift gives a final x of 6.069 mm instead of 4.000 mm; the extracted reference momentum is 1.000 GeV/c instead of 2.000 GeV/c. Without a reference change the restart is correct.

Run `python OPT-08.py` with [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: OPT-08.py"

    [Download OPT-08.py](OPT-08.py)

    ```python
    """OPT-08: Extracted Twiss init loses a downstream reference-momentum change

    `TwissTable.get_twiss_init()` loses a reference-momentum change that lies between the table start
    and the selected row, so an open Twiss restarted from the extracted initial conditions follows a
    particle with a different physical momentum and orbit. `ReferenceEnergyChange` only changes the
    normalization, so exact momentum identities and drift transport require the restarted and
    uninterrupted trajectories to agree. Starting at 1 GeV/c with x = 3 mm and normalized horizontal
    momentum pₓ = 0.001, rebasing to 2 GeV/c and restarting before a 1 m exact drift gives a final x of
    6.069 mm instead of 4.000 mm; the extracted reference momentum is 1.000 GeV/c instead of 2.000
    GeV/c. Without a reference change the restart is correct.

    The script tracks that proton through a `ReferenceEnergyChange`, a marker and an exact 1 m
    `DriftExact`, and compares a full open Twiss with one restarted from `get_twiss_init` at the marker.
    The expected final x is x₀ + pₓ₀/√(1 − pₓ₀²) from the exact drift, and the expected extracted state
    (2 GeV/c, δ = −0.5, pₓ = 0.0005) comes from the momentum identities, checked first by direct
    tracking. A rebase to the unchanged 1 GeV/c is the control.
    """

    import numpy as np
    import xtrack as xt

    RTOL, ATOL = 2e-13, 2e-15


    def evaluate():
        cases = {}
        x0, px0, old_p0c = 0.003, 0.001, 1e9
        expected_x = x0 + px0 / np.sqrt(1 - px0**2)
        for new_p0c in (1e9, 2e9):
            line = xt.Line(
                elements={
                    "rebase": xt.ReferenceEnergyChange(p0c=new_p0c),
                    "handoff": xt.Marker(),
                    "d": xt.DriftExact(length=1),
                }
            )
            reference = xt.Particles(p0c=old_p0c, x=x0, px=px0)
            line.particle_ref = reference.copy()
            initial = xt.TwissInit(
                particle_on_co=reference.copy(), W_matrix=np.eye(6), element_name="rebase"
            )
            whole = line.twiss(init=initial, chrom=False)
            np.testing.assert_allclose(whole.x[-1], expected_x, rtol=RTOL, atol=ATOL)
            expected_local = np.array([new_p0c, old_p0c / new_p0c - 1, px0 * old_p0c / new_p0c])
            direct = reference.copy()
            line.track(direct, ele_start="rebase", ele_stop="handoff")
            np.testing.assert_allclose(
                [direct.p0c[0], direct.delta[0], direct.px[0]], expected_local, rtol=RTOL, atol=ATOL
            )
            extracted = whole.get_twiss_init("handoff")
            part = extracted.particle_on_co
            observed = np.array([part.p0c[0], part.delta[0], part.px[0]])
            continued = line.twiss(start="handoff", end="d", init=extracted, chrom=False)
            if new_p0c == old_p0c:
                np.testing.assert_allclose(observed, expected_local, rtol=RTOL, atol=ATOL)
                np.testing.assert_allclose(continued.x[-1], expected_x, rtol=RTOL, atol=ATOL)
            cases[f"new p0c={new_p0c}, extracted p0c,delta,px"] = observed, expected_local
            cases[f"new p0c={new_p0c}, continued final x"] = continued.x[-1], expected_x
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print(
            "Expected: reference rebasing preserves physical momentum; extracted local p0c and a restarted Twiss trajectory must agree."
        )
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-26</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, prebuilt kernels, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-30</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

