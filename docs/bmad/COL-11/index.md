---
title: "COL-11: Slice-width flooring discards all but the last numerator"
hide:
  - toc
---

# COL-11 — Slice-width flooring discards all but the last numerator

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [csr_bin_particles, bmad/space_charge/csr_and_space_charge_mod.f90:631–673](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L631-L673) |

With slice-width flooring enabled, the charge-weighted average width retains only the final eligible bin’s numerator while its denominator includes all eligible charge. Narrow slices consequently receive too small a minimum width, affecting the longitudinal and transverse space-charge forces. In a 400-particle example, the two central widths remain 0.2864 mm instead of reaching the required 0.5265 mm floor. Independent triangular-overlap integration reproduces the baseline charges and widths before applying the full weighted average.

Run `python COL-11.py` with [_bmad.py](../helpers/_bmad.py) and [_collective_fields.py](../helpers/_collective_fields.py) in the same directory.

??? example "Reproducer: COL-11.py"

    [Download COL-11.py](COL-11.py)

    ```python
    """COL-11: Slice-width flooring discards all but the last numerator

    With slice-width flooring enabled, the charge-weighted average width retains only the final
    eligible bin’s numerator while its denominator includes all eligible charge. Narrow slices
    consequently receive too small a minimum width, affecting the longitudinal and transverse
    space-charge forces. In a 400-particle example, the two central widths remain 0.2864 mm instead of
    reaching the required 0.5265 mm floor. Independent triangular-overlap integration reproduces the
    baseline charges and widths before applying the full weighted average.

    The script bins 400 deterministic particles with flooring off and on, then independently integrates
    each triangular particle shape over the bins. Baseline charges and widths must agree before the
    floored widths are compared with the full charge-weighted average.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _collective_fields import rows

    PROGRAM = r"""

    program binning_floor
      use bmad
      use csr_and_space_charge_mod
      implicit none
      type(ele_struct) ele
      type(csr_struct) csr
      type(coord_struct) p(400)
      integer i, j, k
      logical err
      real(rp) width, baseline(16), baseline_y(16), q(16)
      call init_ele(ele, drift$)
      ele%space_charge_method = slice$
      space_charge_com%n_bin = 16
      space_charge_com%particle_bin_span = 2
      space_charge_com%sc_min_in_bin = 1
      k = 0
      do i = 1, 100
        do j = 1, 4
          k = k+1
          p(k)%state = alive$
          p(k)%charge = 1.0e-12_rp/400
          p(k)%vec(5) = (i-50.5_rp)*0.001_rp
          width = 0.001_rp
          if (abs(p(k)%vec(5)) < 0.008_rp) width = 0.000001_rp
          p(k)%vec(1) = width * (-1)**j
          p(k)%vec(3) = width * (-1)**(j/2)
        enddo
      enddo
      space_charge_com%lsc_sigma_cutoff = 0
      call csr_bin_particles(ele, p, csr, err)
      if (err) error stop 'binning failed'
      baseline = csr%slice%sig_x
      baseline_y = csr%slice%sig_y
      q = csr%slice%charge
      space_charge_com%lsc_sigma_cutoff = 0.5_rp
      call csr_bin_particles(ele, p, csr, err)
      if (err) error stop 'binning with floor failed'
      write(*, '(a,16es25.16e3)') 'RESULT baseline ', baseline
      write(*, '(a,16es25.16e3)') 'RESULT charge ', q
      write(*, '(a,16es25.16e3)') 'RESULT floored ', csr%slice%sig_x
      write(*, '(a,16es25.16e3)') 'RESULT baseline_y ', baseline_y
      write(*, '(a,16es25.16e3)') 'RESULT floored_y ', csr%slice%sig_y
    end program
    """


    data = rows(fortran(PROGRAM).log)
    base, charge = data["baseline"], data["charge"]
    # Integrate the triangular macroparticle shape over every bin. Its padding
    # literal is a discretization convention, separate from the averaging defect.
    z = (np.arange(1, 101) - 50.5) * 0.001
    width = np.where(abs(z) < 0.008, 1e-6, 1e-3)
    dz = (z[-1] - z[0]) / (16 - 3 - 2) * float(np.float32(1.0000001))
    centers = (np.arange(16) + 0.5 - 16 / 2) * dz
    shape_width = 2 * dz


    def triangle_cdf(u):
        u = np.clip(u, -shape_width / 2, shape_width / 2)
        return np.where(
            u < 0,
            2 * (u + shape_width / 2) ** 2 / shape_width**2,
            1 - 2 * (shape_width / 2 - u) ** 2 / shape_width**2,
        )


    overlap = triangle_cdf(centers[:, None] + dz / 2 - z) - triangle_cdf(centers[:, None] - dz / 2 - z)
    expected_charge = overlap.sum(axis=1) * 1e-14
    populated = expected_charge > 0
    expected_width = np.sqrt(np.pi / 2) * (overlap[populated] @ width) / overlap[populated].sum(axis=1)
    if not np.allclose(charge, expected_charge, rtol=2e-13, atol=1e-28):
        raise RuntimeError("Charge-overlap control failed")
    if compare(
        "baseline populated-bin widths (m), triangular-overlap integral",
        base[populated],
        expected_width,
        atol=1e-18,
        rtol=2e-13,
    ):
        raise RuntimeError("Baseline-width control failed")
    floor = 0.5 * np.average(expected_width, weights=expected_charge[populated])
    if not np.allclose(data["baseline_y"][populated], expected_width, rtol=2e-13, atol=1e-18):
        raise RuntimeError("Vertical baseline-width control failed")
    if not np.array_equal(data["floored_y"], data["floored"]):
        raise RuntimeError("Symmetric horizontal and vertical floors differ")
    print(f"Required half-weighted-average floor (m): {floor:.16g}")
    actual = data["floored"][populated]
    defect = compare(
        "floored populated-bin widths (m), full charge-weighted average",
        actual,
        np.maximum(expected_width, floor),
        atol=1e-18,
        rtol=2e-13,
    )
    # Identify the overwrite separately from the independently derived intended floor.
    last_bin_floor = 0.5 * expected_width[-1] * expected_charge[populated][-1] / expected_charge.sum()
    if defect and not np.allclose(
        actual, np.maximum(expected_width, last_bin_floor), rtol=2e-13, atol=1e-18
    ):
        raise RuntimeError("Floored widths do not match the last-bin numerator signature")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

