---
title: "COL-13: The high-energy SC matrix rotates the kick with the wrong sign"
hide:
  - toc
---

# COL-13 — The high-energy SC matrix rotates the kick with the wrong sign

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [make_mat6_high_energy_space_charge, bmad/space_charge/high_energy_space_charge_mod.f90:248–257](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/high_energy_space_charge_mod.f90#L248-L257)<br>[track1_high_energy_space_charge, bmad/space_charge/high_energy_space_charge_mod.f90:187–208](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/high_energy_space_charge_mod.f90#L187-L208) |

With linear high-energy space-charge tracking enabled for a tilted, anisotropic Gaussian bunch, the transfer matrix rotates the kick opposite to particle tracking. The matrix must equal the derivative of the particle map. At a tilt of 0.37 rad, both tracked transverse cross-plane responses are +2.118e−6 m⁻¹ while the matrix gives −2.118e−6 m⁻¹. An independently rotated Gaussian response agrees with tracking, and the matrix diagonals agree.

Run `python COL-13.py` with [_bmad.py](../helpers/_bmad.py) and [_collective_fields.py](../helpers/_collective_fields.py) in the same directory.

??? example "Reproducer: COL-13.py"

    [Download COL-13.py](COL-13.py)

    ```python
    """COL-13: The high-energy SC matrix rotates the kick with the wrong sign

    With linear high-energy space-charge tracking enabled for a tilted, anisotropic Gaussian bunch,
    the transfer matrix rotates the kick opposite to particle tracking. The matrix must equal the
    derivative of the particle map. At a tilt of 0.37 rad, both tracked transverse cross-plane
    responses are +2.118e−6 m⁻¹ while the matrix gives −2.118e−6 m⁻¹. An independently rotated
    Gaussian response agrees with tracking, and the matrix diagonals agree.

    The script compares the high-energy space-charge matrix with two tracked basis displacements and an
    independently rotated linear Gaussian kick. The diagonal entries control the kick strength and the
    cross-plane entries reveal the rotation sign.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _collective_fields import rows

    PROGRAM = r"""

    program high_energy_checks
      use bmad
      use high_energy_space_charge_mod
      implicit none
      type(ele_struct) ele
      type(lat_struct) lat
      type(normal_modes_struct) mode
      type(coord_struct) p
      type(lat_param_struct) param
      real(rp) response(2,2), phi
      integer j
      call init_ele(ele, drift$)
      allocate(ele%high_energy_space_charge)
      phi = 0.37_rp
      ele%high_energy_space_charge%phi = phi
      ele%high_energy_space_charge%cos_phi = cos(phi)
      ele%high_energy_space_charge%sin_phi = sin(phi)
      ele%high_energy_space_charge%sig_x = 0.001_rp
      ele%high_energy_space_charge%sig_y = 0.002_rp
      ele%high_energy_space_charge%sig_z = 0.01_rp
      ele%high_energy_space_charge%kick_const = 1.0e-9_rp
      ele%high_energy_space_charge%closed_orb%vec = 0
      bmad_com%high_energy_space_charge_linear = .true.
      do j = 1, 2
        p%vec = 0
        p%vec(2*j-1) = 1.0e-6_rp
        call track1_high_energy_space_charge(ele, param, p)
        response(:,j) = [p%vec(2),p%vec(4)]/1.0e-6_rp
      enddo
      call mat_make_unit(ele%mat6)
      call make_mat6_high_energy_space_charge(ele,param)
      write(*,'(a,4es25.16e3)') 'RESULT tracked ',response(1,:),response(2,:)
      write(*,'(a,4es25.16e3)') 'RESULT matrix ',ele%mat6(2,1),ele%mat6(2,3),ele%mat6(4,1),ele%mat6(4,3)
    end program
    """


    data = rows(fortran(PROGRAM).log)
    phi = 0.37
    rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    # Linear Gaussian kick in its principal axes, transformed back to laboratory axes.
    expected = rotation @ np.diag(4 * np.pi * 1e-9 / np.array([0.001, 0.002])) @ rotation.T
    if compare(
        "particle response (1/m), rotated Gaussian",
        data["tracked"].reshape(2, 2),
        expected,
        atol=1e-20,
        rtol=3e-15,
    ):
        raise RuntimeError("Particle-response control failed")
    matrix = data["matrix"].reshape(2, 2)
    if not np.allclose(matrix.diagonal(), expected.diagonal(), rtol=3e-15, atol=0):
        raise RuntimeError("Matrix-diagonal control failed")
    defect = compare(
        "matrix response (1/m), derivative of particle map",
        matrix,
        expected,
        atol=1e-20,
        rtol=3e-15,
    )
    reverse_rotation = rotation.T @ np.diag(4 * np.pi * 1e-9 / np.array([0.001, 0.002])) @ rotation
    if defect and not np.allclose(matrix, reverse_rotation, rtol=3e-15, atol=1e-20):
        raise RuntimeError("Matrix mismatch does not match the reversed-rotation signature")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

