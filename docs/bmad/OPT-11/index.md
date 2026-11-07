---
title: "OPT-11: Tracking-derived quaternion derivatives have reversed signs"
hide:
  - toc
---

# OPT-11 — Tracking-derived quaternion derivatives have reversed signs

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [make_mat6_tracking, bmad/low_level/make_mat6_tracking.f90:69–93](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/make_mat6_tracking.f90#L69-L93)<br>[custom spin method, bmad/doc/methods.tex:436–452](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L436-L452) |

Computing a spin map by tracking finite orbital perturbations returns quaternion derivatives with reversed signs. They must use the forward central difference, as the orbital matrix does. For a custom spin rotation by 2κx, where x is horizontal position and κ = 1 m⁻¹, the quaternion is q(x) = (cos κx, 0, 0, sin κx). At x = 0.03 m with a 0.0001 m step, its derivative is (0.03000, 0, 0, −0.9996) m⁻¹ instead of (−0.03000, 0, 0, 0.9996) m⁻¹. The central quaternion and orbital matrix are correct.

Run `python OPT-11.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_14_optics_matrix_radiation.py](../helpers/_s3_14_optics_matrix_radiation.py) in the same directory.

??? example "Reproducer: OPT-11.py"

    [Download OPT-11.py](OPT-11.py)

    ```python
    """OPT-11: Tracking-derived quaternion derivatives have reversed signs

    Computing a spin map by tracking finite orbital perturbations returns quaternion derivatives with reversed signs. They must use the forward central difference, as the orbital matrix does. For a custom spin rotation by 2κx, where x is horizontal position and κ = 1 m⁻¹, the quaternion is q(x) = (cos κx, 0, 0, sin κx). At x = 0.03 m with a 0.0001 m step, its derivative is (0.03000, 0, 0, −0.9996) m⁻¹ instead of (−0.03000, 0, 0, 0.9996) m⁻¹. The central quaternion and orbital matrix are correct.

    The script uses a custom spin rotation with an exactly known quaternion and a marker with identity orbital transport. It checks the central quaternion, orbital matrix and the five zero derivatives before comparing the x derivative with the exact centered difference.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _s3_14_optics_matrix_radiation import control, rows

    PROGRAM = """
    module architecture_contract_hooks
    use bmad
    implicit none
    contains
    subroutine spin(start, ele, param, ending, err, make_quaternion)
    type(coord_struct) start, ending
    type(ele_struct) ele
    type(lat_param_struct) param
    logical err
    logical, optional :: make_quaternion
    real(rp) c, s
    real(rp), parameter :: kappa=1.0_rp ! Inverse metres; rotation angle is 2*kappa*x.
    c=cos(2*kappa*start%vec(1)); s=sin(2*kappa*start%vec(1))
    ending%spin=[c*start%spin(1)-s*start%spin(2),s*start%spin(1)+c*start%spin(2),start%spin(3)]
    err=.false.
    end subroutine

    end module

    program architecture_contracts
    use architecture_contract_hooks
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: q
    type(coord_struct) start, ending
    real(rp) v(6)
    integer i
    logical err
    character(40) label
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) stop 1
    q=>lat%ele(1)
    q%spin_tracking_method=custom$
    track1_spin_custom_ptr=>spin
    bmad_com%spin_tracking_on=.true.
    bmad_com%d_orb(1)=1e-4_rp
    v=0; v(1)=0.03_rp
    call init_coord(start,v,q,upstream_end$)
    call make_mat6_tracking(q,lat%param,start,ending,err)
    if (err) stop 6
    call row('q0',q%spin_q(:,0))
    call row('qx',q%spin_q(:,1))
    call row('other_spin', [maxval(abs(q%spin_q(:,2:6)))])
    do i=1,6
      write(label,'(a,i0)') 'matrix',i
      call row(trim(label),q%mat6(i,:))
    enddo
    call deallocate_lat_pointers(lat)
    contains
    subroutine row(name,values)
    character(*) name
    real(rp) values(:)
    write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
    end subroutine
    end program
    """
    LATTICE = "\nno_digested\nparameter[particle] = electron\nparameter[p0c] = 1e9\nparameter[geometry] = open\nbeginning[beta_a] = 10\nbeginning[beta_b] = 10\nq: marker\nline1: line = (q)\n\nuse, line1\n"

    r = rows(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
    x, h, kappa = 0.03, 1e-4, 1.0
    control(
        "central quaternion",
        r["q0"],
        [np.cos(kappa * x), 0, 0, np.sin(kappa * x)],
        atol=2e-16,
        rtol=2e-14,
    )
    control("other five spin derivatives", r["other_spin"], [0], atol=0, rtol=0)
    control(
        "marker orbital matrix",
        [r[f"matrix{i}"] for i in range(1, 7)],
        np.eye(6),
        atol=2e-14,
        rtol=2e-11,
    )
    # q(x+h)-q(x-h) over 2h has units 1/m; keep its exact finite-step factor.
    expected = np.array([-np.sin(kappa * x), 0, 0, np.cos(kappa * x)]) * np.sin(kappa * h) / h
    verdict(
        compare(
            "quaternion x derivative (1/m), forward central difference",
            r["qx"],
            expected,
            atol=2e-12,
            rtol=2e-11,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

