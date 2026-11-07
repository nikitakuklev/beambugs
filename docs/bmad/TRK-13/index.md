---
title: "TRK-13: PTC electric multipoles remain active after being set to zero"
hide:
  - toc
---

# TRK-13 — PTC electric multipoles remain active after being set to zero

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [update_fibre_from_ele contract, bmad/ptc/update_fibre_from_ele.f90:1–13](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/update_fibre_from_ele.f90#L1-L13)<br>[electric coefficient updates, bmad/ptc/update_fibre_from_ele.f90:105–124](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/update_fibre_from_ele.f90#L105-L124) |

Editing an electric multipole to exactly zero leaves its old field active in a persistent PTC layout. A 0.4 m electric quadrupole with a 1.3 GeV/c proton reference retains normalized horizontal momentum 5.161e-7 after its gradient is changed from 800 kV/m² to zero. A separately constructed zero-field element gives zero kick, and setting a tiny nonzero gradient clears the macroscopic kick. The update loop skips zero electric coefficients instead of overwriting their stored values.

Run `python TRK-13.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_19_fields_photons_ptc.py](../helpers/_s3_19_fields_photons_ptc.py) in the same directory.

??? example "Reproducer: TRK-13.py"

    [Download TRK-13.py](TRK-13.py)

    ```python
    """TRK-13: PTC electric multipoles remain active after being set to zero

    Editing an electric multipole to exactly zero leaves its old field active in a persistent PTC
    layout. A 0.4 m electric quadrupole with a 1.3 GeV/c proton reference retains normalized horizontal
    momentum 5.161e-7 after its gradient is changed from 800 kV/m² to zero. A separately constructed
    zero-field element gives zero kick, and setting a tiny nonzero gradient clears the macroscopic kick.
    The update loop skips zero electric coefficients instead of overwriting their stored values.

    The script creates a persistent PTC layout, edits B1_ELEC through the public setter and bookkeeping
    sequence, and tracks between the element’s integration nodes. It compares exact-zero updates with a
    separately constructed zero-field element and a 1e-70 nonzero coefficient control;
    """

    import os

    # Set numerical pool sizes before NumPy/SciPy are imported.
    for _thread_variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_thread_variable] = "1"


    import numpy as np
    from _bmad import compare, verdict
    from _s3_19_fields_photons_ptc import control, execute, rows

    PROGRAM = r"""program ptc_update
    use bmad
    use ptc_layout_mod, only: lat_to_ptc_layout
    use ptc_spin, only: track_probe_x
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: e
    real(rp) v(6),px
    integer i
    logical err
    character(100) setting
    call bmad_parser('ptc.bmad',lat,err_flag=err)
    if(err) error stop 'parse failed'
      call lat_to_ptc_layout(lat)
      e=>lat%ele(1)
      do i=1,5
        select case(i)
        case(1); setting='B1_ELEC = 2e6'
        case(2); setting='B1_ELEC = 8e5'
        case(3); setting='B1_ELEC = 0'
        case(4); setting='B1_ELEC = 1e-70'
        case(5); setting='B1_ELEC = 0'
        end select
        call set_ele_attribute(e,trim(setting),err)
        if(err) error stop 'PTC setter failed'
        call lattice_bookkeeper(lat)
        call ptc_bookkeeper(lat)
        v=[0.0017_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp]
        call track_probe_x(v,ptc_private%base_state,node1=e%ptc_fibre%t1,node2=e%ptc_fibre%t2%next)
        print '(a,i3,4es25.16)', 'PTC_UPDATE ',i,e%b_pole_elec(1),v(2),v(1),v(6)
        px=v(2)
      enddo
      e=>lat%ele(2)
      v=[0.0017_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp]
      call track_probe_x(v,ptc_private%base_state,node1=e%ptc_fibre%t1,node2=e%ptc_fibre%t2%next)
      print '(a,3es25.16)', 'PTC_FRESH_ZERO ',v(2),v(1),v(6)
      if(v(2)/=0) error stop 'fresh zero control failed'
    print '(a)', 'DONE'
    end program
    """
    LATTICE = """parameter[geometry]=open
    beginning[beta_a]=10
    beginning[beta_b]=10
    parameter[particle]=proton
    parameter[p0c]=1.3e9
    q1: quadrupole, l=0.4, k1=0, b1_elec=2e6, tracking_method=symp_lie_ptc, num_steps=8
    q2: q1, b1_elec=0
    channel: line=(q1,q2)
    use,channel
    """
    output = execute(PROGRAM, [], {"ptc.bmad": LATTICE})
    updates = np.array([list(map(float, row)) for row in rows(output, "PTC_UPDATE")])
    fresh = np.array(list(map(float, rows(output, "PTC_FRESH_ZERO")[0])))
    assert updates.shape == (5, 5) and np.isfinite(updates).all()
    control(updates[:, 0], [1, 2, 3, 4, 5], 0)
    control(updates[:, 1], [2e6, 8e5, 0, 1e-70, 0], 0)
    control(fresh, [0, 0.0017, 0], 0)
    assert abs(updates[0, 2]) > 1e-7 and abs(updates[1, 2]) > 1e-7
    assert abs(updates[0, 2] - updates[1, 2]) > 1e-7
    control(updates[3, 2:], [0, 0.0017, 0], 1e-17)
    print("Nonzero edits change the kick; fresh zero and 1e-70 coefficient controls give the drift.")
    defect = False
    for row in updates:
        print(
            f"step={int(row[0])} b1_elec={row[1]:.16g} px={row[2]:.16g} x={row[3]:.16g} delta={row[4]:.16g}"
        )
        if row[1] == 0:
            defect |= compare(
                f"step={int(row[0])}, (px,x,delta), zero-field drift", row[2:], fresh, atol=1e-17
            )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

