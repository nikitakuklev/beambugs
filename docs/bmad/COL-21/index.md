---
title: "COL-21: Transverse slice space charge inherits the last shielding-image factor"
hide:
  - toc
---

# COL-21 — Transverse slice space charge inherits the last shielding-image factor

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [track1_bunch_csr shielding-image loop, bmad/space_charge/csr_and_space_charge_mod.f90:347–368](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L347-L368)<br>[transverse slice kick, bmad/space_charge/csr_and_space_charge_mod.f90:1605–1610](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1605-L1610)<br>[direct-source condition, csr_and_space_charge_mod.f90:828–833](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L828-L833) |

Combining transverse slice space charge with shielding images multiplies the kick by −2 for an odd image count and by +2 for an even nonzero count. Shielding images belong to the coherent synchrotron radiation calculation and should not change the slice-model force. In a 1 cm drift with 10 MeV/c electrons and 10 pC bunch charge, the space-charge change in the position–momentum covariance is 3.231e−11, −6.462e−11 and 6.462e−11 m for zero, one and two images. Longitudinal slice space charge is calculated at the direct-source iteration and avoids this multiplier.

Run `python COL-21.py` with [_bmad.py](../helpers/_bmad.py) and [_collective_fields.py](../helpers/_collective_fields.py) in the same directory.

??? example "Reproducer: COL-21.py"

    [Download COL-21.py](COL-21.py)

    ```python
    """COL-21: Transverse slice space charge inherits the last shielding-image factor

    Combining transverse slice space charge with shielding images multiplies the kick by −2 for an odd image count and by +2 for an even nonzero count. Shielding images belong to the coherent synchrotron radiation calculation and should not change the slice-model force. In a 1 cm drift with 10 MeV/c electrons and 10 pC bunch charge, the space-charge change in the position–momentum covariance is 3.231e−11, −6.462e−11 and 6.462e−11 m for zero, one and two images. Longitudinal slice space charge is calculated at the direct-source iteration and avoids this multiplier.

    The script runs fresh deterministic Tao bunches with space charge off and with slice space charge at
    zero, one and two shielding images. It subtracts the drift-only covariance and requires all 2000
    particles to survive. With CSR off, the tracked covariance must be independent of image count.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _collective_fields import tao_covariance

    off, rejected = tao_covariance("off")
    if rejected:
        raise RuntimeError("Space-charge-off control rejected")
    changes = []
    for images in (0, 1, 2):
        result, rejected = tao_covariance("slice", images)
        if rejected or not np.isfinite(result).all():
            raise RuntimeError("Slice tracking failed")
        changes.append(result[0] - off[0])
        print(
            f"images={images}: exit sigma_12, sigma_56 (m)={result.tolist()}; SC change sigma_12={changes[-1]:.16g}"
        )
    if changes[0] <= 1e-12:
        raise RuntimeError("Unshielded transverse-force control failed")
    # With CSR off, shielding images must not affect the slice-model force or the
    # evolved bunch. This end-to-end test executes the image loop itself.
    ratios = np.array(changes[1:]) / changes[0]
    defect = compare(
        "tracked slice covariance ratios for images 1,2; image-independent force",
        ratios,
        np.ones(2),
        atol=0,
        rtol=2e-10,
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-22 | fft-sc-endpoint-kicks (`f8173e0253f1`), source, production, CPU, double, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |

