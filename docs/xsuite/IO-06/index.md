---
title: "IO-06: Negative-charge reference rigidity fails a constructor round trip"
hide:
  - toc
---

# IO-06 — Negative-charge reference rigidity fails a constructor round trip

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack release [v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4) (`3cd8bfc`) |
| Source | [`Particles._update_refs`, particles.py:1704–1708](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/particles.py#L1704-L1708)<br>[`rigidity0` getter and linked setter](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/particles.py#L1342-L1360) |

Constructing `Particles` from the reference rigidity `rigidity0` computes the momentum from `rigidity0`·|q₀|·c, while the `rigidity0` getter and setter use the signed reference charge q₀, so a negative-charge reference's own reported rigidity reconstructs a negative momentum. The manual defines signed rigidity, and property-to-constructor round trips must preserve the reference state. For electron-mass references with q₀ = −1 or −2 at 1 GeV/c, the constructor returns the reference momentum `p0c` = −1.000×10⁹ eV and reference speed β₀ = v/c = −1.000, while the particle remains alive; the correct momentum and speed are positive. Positive-charge construction, the linked rigidity setter, and construction from `p0c` or `energy0` pass their controls.

Run `python IO-06.py` with [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-06.py"

    [Download IO-06.py](IO-06.py)

    ```python
    """IO-06: Negative-charge reference rigidity fails a constructor round trip

    Constructing `Particles` from the reference rigidity `rigidity0` computes the momentum from
    `rigidity0`·|q₀|·c, while the `rigidity0` getter and setter use the signed reference charge q₀, so a
    negative-charge reference's own reported rigidity reconstructs a negative momentum. The manual
    defines signed rigidity, and property-to-constructor round trips must preserve the reference state.
    For electron-mass references with q₀ = −1 or −2 at 1 GeV/c, the constructor returns the reference
    momentum `p0c` = −1.000×10⁹ eV and reference speed β₀ = v/c = −1.000, while the particle remains
    alive; the correct momentum and speed are positive. Positive-charge construction, the linked
    rigidity setter, and construction from `p0c` or `energy0` pass their controls.

    For electron-mass references with q₀ = ±1 and ±2 at 1 GeV/c, the script reads `rigidity0`, checks it
    against the signed value p₀c/(c·q₀), assigns it back through the linked setter and passes it to a
    new `Particles` with unchanged mass and charge. The rebuilt `p0c` and `beta0` must equal the
    originals. The setter and the positive charges are controls; the negative charges show the bug.
    """

    import numpy as np
    import xtrack as xt
    from scipy.constants import c

    RTOL, ATOL = 2e-15, 0


    def evaluate():
        cases = {}
        for charge in (1, -1, 2, -2):
            original = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=charge, p0c=1e9)
            rigidity = float(original.rigidity0[0])
            # The getter and linked setter use signed rigidity, p0c/(c*q0).
            np.testing.assert_allclose(rigidity, 1e9 / (c * charge), rtol=RTOL, atol=ATOL)
            assigned = original.copy()
            assigned.rigidity0 = rigidity
            np.testing.assert_allclose(assigned.p0c, original.p0c, rtol=RTOL, atol=ATOL)
            rebuilt = xt.Particles(mass0=original.mass0, q0=charge, rigidity0=rigidity)
            assert rebuilt.state[0] == 1
            if charge > 0:
                np.testing.assert_allclose(rebuilt.p0c, original.p0c, rtol=RTOL, atol=ATOL)
                np.testing.assert_allclose(rebuilt.beta0, original.beta0, rtol=RTOL, atol=ATOL)
            else:
                cases[f"q0={charge}, reconstructed p0c [eV]"] = rebuilt.p0c.copy(), original.p0c.copy()
                cases[f"q0={charge}, reconstructed beta0"] = rebuilt.beta0.copy(), original.beta0.copy()
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print(
            "Expected: constructing from the same object's rigidity0 preserves p0c and beta0, as assigning that property does."
        )
        print(
            "Signed Lorentz rigidity and positive-charge controls are checked first; no convention change can justify a failed round trip."
        )
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

