---
title: "OPT-21: Integral bunch length uses summed cavity M65 instead of one-turn M65"
hide:
  - toc
---

# OPT-21 — Integral bunch length uses summed cavity M65 instead of one-turn M65

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P4 |
| Failure | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">known-limitation</span> [?](../../methodology.md#tags) |
| Source | [cavity-kick accumulation, bmad/code/radiation_integrals.f90:436–438](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/radiation_integrals.f90#L436-L438)<br>[bunch-length formula, bmad/code/radiation_integrals.f90:648–654](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/radiation_integrals.f90#L648-L654)<br>[Bmad manual, “Synchrotron Radiation Integrals”, small-synchrotron-frequency approximation, bmad/doc/synchrotron-radiation.tex:405–410](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/synchrotron-radiation.tex#L405-L410) |

The radiation-integral bunch-length calculation sums individual RF-cavity M₆₅ values, while the manual defines its denominator as the composed one-turn M₆₅. These coincide to first order in RF strength and differ at second order, within the documented weak-synchrotron approximation. In a ring with two 10 kV cavities, the implemented denominator is 1.708 × 10⁻⁴ m⁻¹ and the one-turn value is 1.707 × 10⁻⁴ m⁻¹, giving 0.03239 m versus 0.03240 m in the stated formula. The denominator description should identify the cavity sum used by the approximation.

Run `python OPT-21.py` with [_bmad.py](../helpers/_bmad.py) and [_dispersion_and_moments.py](../helpers/_dispersion_and_moments.py) in the same directory.

??? example "Reproducer: OPT-21.py"

    [Download OPT-21.py](OPT-21.py)

    ```python
    """OPT-21: Integral bunch length uses summed cavity M65 instead of one-turn M65

    The radiation-integral bunch-length calculation sums individual RF-cavity M₆₅ values, while
    the manual defines its denominator as the composed one-turn M₆₅. These coincide to first
    order in RF strength and differ at second order, within the documented weak-synchrotron
    approximation. In a ring with two 10 kV cavities, the implemented denominator is 1.708 ×
    10⁻⁴ m⁻¹ and the one-turn value is 1.707 × 10⁻⁴ m⁻¹, giving 0.03239 m versus 0.03240 m in
    the stated formula. The denominator description should identify the cavity sum used by the
    approximation.

    The script executes radiation_integrals on a ring with two separated RF cavities. It validates the
    returned bunch length against the cavity sum, compares it with the composed one-turn formula, and
    reads the manual’s denominator definition through BMAD_SOURCE.
    """

    import numpy as np
    from _bmad import compare, source, verdict
    from _dispersion_and_moments import RADIATION, execute, radiation_ring, require

    rows = execute(RADIATION, {"lat.bmad": radiation_ring(rf=True)})
    cavity_sum, one_turn = rows["m65"]
    i1 = rows["integrals"][1]
    sig_z, sig_delta, _ = rows["spread"]
    if min(i1 / cavity_sum, i1 / one_turn) <= 0:
        raise RuntimeError("Expected stable longitudinal focusing")
    # The implemented weak-synchrotron approximation uses the sum of the two cavity kicks.
    require(sig_z, np.sqrt(i1 / cavity_sum) * sig_delta, atol=0, rtol=3e-14)
    manual = (source() / "bmad/doc/synchrotron-radiation.tex").read_text()
    if "element for the 1--turn transfer matrix" not in manual:
        raise RuntimeError("The documented denominator changed; inspect its definition")
    defect = compare(
        "bunch-length denominator M65 (1/m), documented one-turn value",
        cavity_sum,
        one_turn,
        atol=1e-15,
        rtol=3e-14,
    )
    defect |= compare(
        "sig_z (m), using the documented one-turn denominator",
        sig_z,
        np.sqrt(i1 / one_turn) * sig_delta,
        atol=0,
        rtol=3e-14,
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small class="bb-scope">source check</small> | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small class="bb-scope">source check</small> | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small class="bb-scope">weak-RF bunch-length denominator definition</small> | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

