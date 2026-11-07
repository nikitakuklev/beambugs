---
title: "COL-10: The FFT Coulomb prefactor loses default-real precision"
hide:
  - toc
---

# COL-10 — The FFT Coulomb prefactor loses default-real precision

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [osc_freespace_solver2, bmad/space_charge/open_spacecharge_mod.f90:742–759](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L742-L759)<br>[CSR3D prefactor, bmad/space_charge/csr3d_mod.f90:100–102](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr3d_mod.f90#L100-L102)<br>[space_charge_3d speed literal and B-field conversion, bmad/space_charge/open_spacecharge_mod.f90:639–663](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L639-L663) |

The FFT free-space solver squares the speed-of-light literal in default real precision before converting to double precision. Its Coulomb prefactor falls short of the stated formula by 290.0 V m/C, making the electric field 32.27 parts per billion too small. Independent uniform-cell quadrature confirms the common normalization error while field direction and relative component strengths agree.

Run `python COL-10.py` with [_bmad.py](../helpers/_bmad.py) and [_collective_fields.py](../helpers/_collective_fields.py) in the same directory.

??? example "Reproducer: COL-10.py"

    [Download COL-10.py](COL-10.py)

    ```python
    """COL-10: The FFT Coulomb prefactor loses default-real precision

    The FFT free-space solver squares the speed-of-light literal in default real precision before
    converting to double precision. Its Coulomb prefactor falls short of the stated formula by 290.0
    V m/C, making the electric field 32.27 parts per billion too small. Independent uniform-cell
    quadrature confirms the common normalization error while field direction and relative component
    strengths agree.

    The script solves a uniformly charged cell and compares three field vectors with converged
    Gauss–Legendre volume quadrature. Compiler-kind, prefactor, field-direction and relative-strength
    checks isolate the shared normalization error.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _collective_fields import rows

    PROGRAM = r"""

    program fft_field_normalization
      use bmad
      use open_spacecharge_mod
      implicit none
      real(rp) rho(3,3,3), field(3,3,3,3), delta(3)
      rho = 0; rho(2,2,2) = 1.0e-12_rp
      delta = 1
      call osc_freespace_solver2(rho, 1.0_rp, delta, efield=field)
      write(*, '(a,9es25.16e3)') 'RESULT field ', field(3,2,2,:), field(3,3,2,:), field(3,3,3,:)
      write(*, '(a,4i8)') 'RESULT precision ', kind(1.0), digits(1.0), kind(1.0_rp), digits(1.0_rp)
      write(*, '(a,2es25.16e3)') 'RESULT prefactor ', 299792458.0**2*1.00000000055d-7, &
        299792458.0_rp**2*1.00000000055d-7
    end program
    """


    def _uniform_cube_field(order):
        """Unit-charge cell-average of r/r³, independently integrated on [-1/2,1/2]³."""
        nodes, weights = np.polynomial.legendre.leggauss(order)
        xyz = np.stack(np.meshgrid(nodes / 2, nodes / 2, nodes / 2, indexing="ij"), axis=-1)
        product_weights = np.einsum("i,j,k->ijk", weights, weights, weights) / 8
        points = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        displacement = points[:, None, None, None, :] - xyz
        radius2 = np.sum(displacement**2, axis=-1)
        return np.sum(
            displacement / radius2[..., None] ** 1.5 * product_weights[None, ..., None],
            axis=(1, 2, 3),
        )


    data = rows(fortran(PROGRAM).log)
    if not np.array_equal(data["precision"], [4, 24, 8, 53]):
        raise RuntimeError("Compiler real-kind control failed")
    kernel = _uniform_cube_field(40)
    if not np.allclose(kernel, _uniform_cube_field(28), rtol=2e-13, atol=1e-15):
        raise RuntimeError("Cell quadrature failed to converge")
    intended = 299792458.0**2 * 1.00000000055e-7
    rounded = float(np.float32(299792458) * np.float32(299792458)) * 1.00000000055e-7
    if not np.allclose(data["prefactor"], [rounded, intended], rtol=2e-15, atol=0):
        raise RuntimeError("Compiler prefactor control failed")
    actual = data["field"].reshape(3, 3)
    expected = kernel * 1e-12 * intended
    nonzero = abs(expected) > 1e-12
    measured = np.mean(actual[nonzero] / (1e-12 * kernel[nonzero]))
    if compare(
        "field shape (V/m), quadrature at measured normalization",
        actual,
        kernel * 1e-12 * measured,
        atol=1e-16,
        rtol=2e-12,
    ):
        raise RuntimeError("Field direction and relative-strength controls failed")
    print(f"Fortran default-real/double prefactors (V m/C): {data['prefactor'].tolist()}")
    print(f"Relative normalization error (ppb): {(measured / intended - 1) * 1e9:.9f}")
    defect = compare(
        "absolute electric field (V/m), double-precision prefactor",
        actual,
        expected,
        atol=1e-16,
        rtol=2e-12,
    )
    # A different normalization error cannot identify default-real rounding.
    if defect and not np.allclose(actual, kernel * 1e-12 * rounded, rtol=2e-12, atol=1e-16):
        raise RuntimeError("Field mismatch does not match the default-real prefactor")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

