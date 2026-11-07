---
title: "OPT-07: Curvature-vector rotation overwrites its second input"
hide:
  - toc
---

# OPT-07 — Curvature-vector rotation overwrites its second input

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [rad_g_integrals, bmad/modules/rad_6d_mod.f90:878–884](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/rad_6d_mod.f90#L878-L884)<br>[curvature-integral outputs, bmad/modules/rad_6d_mod.f90:738–744](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/rad_6d_mod.f90#L738-L744) |

Computing radiation curvature integrals for a tilted element rotates the second component using an already-rotated first component. An orthogonal rotation must use the original vector for both components and preserve its norm. For a short quadrupole tilted by 45°, the second integrated-curvature component is 2.500 × 10⁻⁷ instead of 3.536 × 10⁻⁷, an underestimate of 29.29%. The untilted case provides the rotation control.

Run `python OPT-07.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_14_optics_matrix_radiation.py](../helpers/_s3_14_optics_matrix_radiation.py) in the same directory.

??? example "Reproducer: OPT-07.py"

    [Download OPT-07.py](OPT-07.py)

    ```python
    """OPT-07: Curvature-vector rotation overwrites its second input

    Computing radiation curvature integrals for a tilted element rotates the second component using an already-rotated first component. An orthogonal rotation must use the original vector for both components and preserve its norm. For a short quadrupole tilted by 45°, the second integrated-curvature component is 2.500 × 10⁻⁷ instead of 3.536 × 10⁻⁷, an underestimate of 29.29%. The untilted case provides the rotation control.

    The script integrates curvature in a short quadrupole, then rotates both the quadrupole and its incoming particle by 45°. It compares the tilted vector with a two-component orthogonal rotation of the untilted result.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _s3_14_optics_matrix_radiation import control, rows

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
    real(rp) ig(2), ig0(2), ig2, ig3, v0(6)
    logical err
    integer stat
    call bmad_parser('lat.bmad', lat, err_flag=err)
    if (err) error stop 'parse failed'
    call twiss_and_track(lat, orb, stat)
    if (stat /= ok$) error stop 'twiss failed'
      ele => lat%ele(5)
      ele%value(l$) = 0.001_rp
      call set_flags_for_changed_attribute(ele)
      call lattice_bookkeeper(lat)
      v0 = [0.001_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp]
      call init_coord(o1, v0, ele, upstream_end$)
      call make_mat6(ele, lat%param, o1, o2, err)
      if (err) error stop 'matrix failed'
      call rad_g_integrals(ele, all$, o1, o2, ig0, ig2, ig3, 1e-10_rp, 1e-10_rp, 1e-10_rp)
      call row('untilted',ig0)
      ele%value(tilt$) = pi/4
      call set_flags_for_changed_attribute(ele, ele%value(tilt$))
      call lattice_bookkeeper(lat)
      v0(1) = 0.001_rp/sqrt_2
      v0(3) = 0.001_rp/sqrt_2
      call init_coord(o1, v0, ele, upstream_end$)
      call make_mat6(ele, lat%param, o1, o2, err)
      if (err) error stop 'matrix failed'
      call rad_g_integrals(ele, all$, o1, o2, ig, ig2, ig3, 1e-10_rp, 1e-10_rp, 1e-10_rp)
      call row('tilted',ig)
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
    control(
        "untilted quadrupole integral (rad), analytic k*x orbit",
        r["untilted"],
        [0.001 * np.sqrt(0.5) * np.sin(np.sqrt(0.5) * 0.001), 0],
        atol=3e-20,
    )
    rotation = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
    expected = rotation @ r["untilted"]
    defect = compare(
        "tilted integrated curvature (rad), orthogonal rotation", r["tilted"], expected, atol=3e-14
    )
    defect |= compare(
        "tilted vector norm (rad), rotation preserves norm",
        np.linalg.norm(r["tilted"]),
        np.linalg.norm(r["untilted"]),
        atol=3e-14,
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

