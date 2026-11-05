---
title: "COL-38: Beam–beam slices use mid-quantiles instead of documented charge centroids"
hide:
  - toc
---

# COL-38 — Beam–beam slices use mid-quantiles instead of documented charge centroids

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [bbi_slice_calc, bmad/low_level/bbi_slice_calc.f90:30–39](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/bbi_slice_calc.f90#L30-L39)<br>[Bmad manual, BeamBeam (s:beambeam), bmad/doc/elements.tex:335–343](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L335-L343)<br>[BeamBeam Tracking (s:beambeam.std), bmad/doc/charged-tracking.tex:321–330](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/charged-tracking.tex#L321-L330) |

A beam–beam element places equal-charge Gaussian slices at probability midpoints instead of the charge centroids specified in the manual. For two slices, the positions are ±0.6745 times the bunch RMS length rather than the conditional means ±0.7979 times that length. The offsets are 15.46% smaller than the documented values. A single slice is correctly centered at zero.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)

??? example "Reproducer: COL-38.py"

    [Download COL-38.py](COL-38.py)

    ```python
    """COL-38: Beam–beam slices use mid-quantiles instead of documented charge centroids

    A beam–beam element places equal-charge Gaussian slices at probability midpoints instead of the charge centroids specified in the manual. For two slices, the positions are ±0.6745 times the bunch RMS length rather than the conditional means ±0.7979 times that length. The offsets are 15.46% smaller than the documented values. A single slice is correctly centered at zero.

    The script calls bbi_slice_calc from BMAD_DIST for one and two slices. It compares the two-slice positions with the analytic Gaussian half-bunch mean and checks the centered single-slice control.
    """

    import math

    import numpy as np
    from _bmad import compare, fortran, verdict

    PROGRAM = """program probe
    use bmad
    implicit none
    type(ele_struct) ele
    real(rp) z(2)
    ele%value(sig_z$)=1.0_rp
    call bbi_slice_calc(ele,1,z)
    write(*,'(2es25.16)') z
    call bbi_slice_calc(ele,2,z)
    write(*,'(2es25.16)') z
    end program
    """
    rows = np.array([[float(x) for x in line.split()] for line in fortran(PROGRAM).log.splitlines()])
    if rows.shape != (2, 2):
        raise RuntimeError("Expected one- and two-slice results")
    if compare("single-slice control and zero padding", rows[0], [0, 0], atol=0):
        raise RuntimeError("Single-slice control failed")
    # The conditional mean is integral_0^inf z*phi(z) dz / (1/2) = sqrt(2/pi).
    verdict(
        compare(
            "two-slice positions / sigma_z, Gaussian charge centroids",
            rows[1],
            [-math.sqrt(2 / math.pi), math.sqrt(2 / math.pi)],
            atol=1e-5,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>source check</small> | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>three- and five-slice Gaussian charge centroids</small> | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

