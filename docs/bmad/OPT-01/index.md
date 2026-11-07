---
title: "OPT-01: Ohmi damping integrand omits forward transport"
hide:
  - toc
---

# OPT-01 — Ohmi damping integrand omits forward transport

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [calc_rad_at_pt, bmad/modules/rad_6d_mod.f90:624–662](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/rad_6d_mod.f90#L624-L662)<br>[rad1_damp_and_stoc_mats, bmad/modules/rad_6d_mod.f90:573–582](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/rad_6d_mod.f90#L573-L582)<br>[transport equation, bmad/doc/synchrotron-radiation.tex:116–141](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/synchrotron-radiation.tex#L116-L141) |

Computing a radiation damping map for a bend omits transport from the entrance to each radiation-emission point. The first-order damping correction must include transport on both sides of the local damping matrix, as the transport equation and an independent matrix-exponential calculation establish. For a 2 m sector bend of curvature 0.1 m⁻¹ at 3 GeV, the momentum response to entrance horizontal momentum is zero instead of −7.578 × 10⁻⁷. Splitting the bend into two halves gives −3.795 × 10⁻⁷, so the result depends on segmentation.

Run `python OPT-01.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_14_optics_matrix_radiation.py](../helpers/_s3_14_optics_matrix_radiation.py) in the same directory.

??? example "Reproducer: OPT-01.py"

    [Download OPT-01.py](OPT-01.py)

    ```python
    """OPT-01: Ohmi damping integrand omits forward transport

    Computing a radiation damping map for a bend omits transport from the entrance to each radiation-emission point. The first-order damping correction must include transport on both sides of the local damping matrix, as the transport equation and an independent matrix-exponential calculation establish. For a 2 m sector bend of curvature 0.1 m⁻¹ at 3 GeV, the momentum response to entrance horizontal momentum is zero instead of −7.578 × 10⁻⁷. Splitting the bend into two halves gives −3.795 × 10⁻⁷, so the result depends on segmentation.

    The script computes the damping correction for a full sector bend and for two half-bends. It compares the result with a Fréchet derivative of a matrix exponential, which integrates the constant-field transport equation independently. SciPy supplies the matrix exponential.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _s3_14_optics_matrix_radiation import control, rows
    from scipy.linalg import expm_frechet

    PROGRAM = """
    program probe
    use bmad
    use rad_6d_mod
    use radiation_mod
    implicit none
    type(lat_struct), target :: lat
    type(coord_struct), allocatable :: orb(:)
    type(coord_struct) o1, o2
    type(ele_struct), pointer :: ele
    type(rad_map_struct) rm, rm1, rm2
    real(rp) mat(6,6)
    logical err
    integer stat, i
    call bmad_parser('lat.bmad', lat, err_flag=err)
    if (err) error stop 'parse failed'
    call twiss_and_track(lat, orb, stat)
    if (stat /= ok$) error stop 'twiss failed'
      ele => lat%ele(2)
      err = .false.
      call rad1_damp_and_stoc_mats(ele, .true., orb(1), orb(2), rm, 1e-10_rp, 1e-10_rp, err)
      if (err) error stop 'radiation failed'
      call row('constants', [ele%value(e_tot$), mass_of(ele%ref_species), classical_radius(ele%ref_species)])
      do i=1,6
        write(*,'(a,i1,6es24.15)') 'RESULT D',i,rm%damp_dmat(i,:)
      enddo
      call init_coord(o1, [0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp], lat%ele(3), upstream_end$)
      call make_mat6(lat%ele(3), lat%param, o1, o2, err)
      if (err) error stop 'first half matrix failed'
      call rad1_damp_and_stoc_mats(lat%ele(3), .true., o1, o2, rm1, 1e-10_rp, 1e-10_rp, err)
      if (err) error stop 'first half radiation failed'
      call make_mat6(lat%ele(4), lat%param, o2, o1, err)
      if (err) error stop 'second half matrix failed'
      call rad1_damp_and_stoc_mats(lat%ele(4), .true., o2, o1, rm2, 1e-10_rp, 1e-10_rp, err)
      if (err) error stop 'split radiation failed'
      mat=matmul(lat%ele(4)%mat6,rm1%damp_dmat)+matmul(rm2%damp_dmat,lat%ele(3)%mat6)
      do i=1,6
        write(*,'(a,i1,6es24.15)') 'RESULT S',i,mat(i,:)
      enddo
    contains
    subroutine row(name,values)
    character(*) name
    real(rp) values(:)
    write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
    end subroutine
    end program
    """
    LATTICE = "no_digested\nparameter[geometry] = open\nparameter[e_tot] = 3e9\nparameter[no_end_marker] = T\nbeginning[beta_a] = 10\nbeginning[beta_b] = 12\nd: drift, l = 2\nb: sbend, l = 2, g = 0.1, fringe_type = none\nbh: sbend, l = 1, g = 0.1, fringe_type = none\nq: quadrupole, l = 1, k1 = 0.5\nln: line = (d, b, bh, bh, q)\nuse, ln\n"

    r = rows(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
    energy, mass, radius = r["constants"]
    length, g = 2.0, 0.1
    gamma = energy / mass
    kd = 2 * radius * gamma**3 / 3
    # Bmad's z sign gives A51=-g; the local damping generator includes the
    # curvature/path derivative d61=-kd*g^3. Z=integral exp(A*(L-s))*d*exp(A*s) ds.
    a = np.zeros((6, 6))
    a[0, 1] = a[2, 3] = 1
    a[1, 0], a[1, 5], a[4, 0], a[4, 5] = -g * g, g, -g, 1 / gamma**2
    d = np.zeros((6, 6))
    d[1, 1] = d[3, 3] = -kd * g * g
    d[5, 5], d[5, 0] = -2 * kd * g * g, -kd * g**3
    expected = expm_frechet(a * length, d * length, compute_expm=False)
    observed = np.array([r[f"D{i}"] for i in range(1, 7)])
    split = np.array([r[f"S{i}"] for i in range(1, 7)])
    control("vertical momentum damping control", observed[3, 3], expected[3, 3], atol=2e-19)
    defect = compare(
        "whole bend D62, two-sided transport integral", observed[5, 1], expected[5, 1], atol=2e-13
    )
    defect |= compare("two half-bends D62, same integral", split[5, 1], expected[5, 1], atol=2e-13)
    print(f"whole matrix maximum absolute error={np.max(np.abs(observed - expected)):.16g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

