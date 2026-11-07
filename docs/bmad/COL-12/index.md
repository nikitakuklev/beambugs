---
title: "COL-12: CSR3D boundary derivatives use integer division"
hide:
  - toc
---

# COL-12 — CSR3D boundary derivatives use integer division

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Tags | <span class="bb-tag">experimental</span> [?](../../methodology.md#tags) |
| Source | [calc_density_derivative_complex, bmad/space_charge/csr3d_mod.f90:182–204](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr3d_mod.f90#L182-L204)<br>[track1_bunch_csr3d status, bmad/space_charge/csr_and_space_charge_mod.f90:1854–1860](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1854-L1860) |

The experimental three-dimensional coherent synchrotron radiation solver uses integer division in its one-sided density-derivative stencils. The coefficients 3/2 and 1/2 become 1 and 0, so even a constant density acquires nonzero endpoint derivatives. For f(z) = 1 + 0.3z + 0.7z² on a grid of spacing 0.2, the endpoint derivatives are 5.880 and −5.880 instead of 0.3000 and 1.420. Interior derivatives and zero FFT padding are correct.

Run `python COL-12.py` with [_bmad.py](../helpers/_bmad.py) and [_collective_fields.py](../helpers/_collective_fields.py) in the same directory.

??? example "Reproducer: COL-12.py"

    [Download COL-12.py](COL-12.py)

    ```python
    """COL-12: CSR3D boundary derivatives use integer division

    The experimental three-dimensional coherent synchrotron radiation solver uses integer division in
    its one-sided density-derivative stencils. The coefficients 3/2 and 1/2 become 1 and 0, so even a
    constant density acquires nonzero endpoint derivatives. For f(z) = 1 + 0.3z + 0.7z² on a grid of
    spacing 0.2, the endpoint derivatives are 5.880 and −5.880 instead of 0.3000 and 1.420. Interior
    derivatives and zero FFT padding are correct.

    The script differentiates a quadratic density on five evenly spaced points and compares against its
    analytic derivative. It requires the interior stencil and doubled-grid padding to pass before
    checking both endpoints.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _collective_fields import rows

    PROGRAM = r"""

    program csr3d_derivative
      use bmad
      use csr3d_mod, only: calc_density_derivative_complex
      implicit none
      real(rp) density(1,1,5), z
      complex(rp) derivative(2,2,10)
      integer i
      do i = 1, 5
        z = (i-1)*0.2_rp
        density(1,1,i) = 1 + 0.3_rp*z + 0.7_rp*z*z
      enddo
      call calc_density_derivative_complex(density, derivative, 0.2_rp)
      write(*,'(a,5es25.16e3)') 'RESULT derivative ', real(derivative(1,1,1:5),rp)
      derivative(1,1,1:5) = 0
      write(*,'(a,es25.16e3)') 'RESULT padding ', maxval(abs(derivative))
    end program
    """


    data = rows(fortran(PROGRAM).log)
    actual = data["derivative"]
    expected = 0.3 + 1.4 * np.arange(5) * 0.2
    if not np.array_equal(data["padding"], [0]):
        raise RuntimeError("Zero-padding control failed")
    if compare(
        "interior density derivatives, analytic quadratic derivative",
        actual[1:-1],
        expected[1:-1],
        atol=2e-15,
        rtol=2e-15,
    ):
        raise RuntimeError("Interior-stencil control failed")
    defect = compare(
        "endpoint density derivatives, analytic quadratic derivative",
        actual[[0, -1]],
        expected[[0, -1]],
        atol=3e-15,
        rtol=3e-15,
    )
    z = np.arange(5) * 0.2
    density = 1 + 0.3 * z + 0.7 * z * z
    # Integer division removes the third sample and changes each 3/2 to 1.
    integer_ends = np.array([-density[0] + 2 * density[1], -2 * density[-2] + density[-1]]) / 0.2
    if defect and not np.allclose(actual[[0, -1]], integer_ends, rtol=3e-15, atol=3e-15):
        raise RuntimeError("Endpoint mismatch does not match the integer-division stencil")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

