---
title: "TRK-24: A zero-frequency lcavity gives NaN with absolute-time tracking"
hide:
  - toc
---

# TRK-24 — A zero-frequency lcavity gives NaN with absolute-time tracking

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [zero-frequency input guard, bmad/low_level/track_a_lcavity.f90:57–61](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L57-L61)<br>[this_rf_phase absolute-time reduction, bmad/low_level/track_a_lcavity.f90:499–510](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L499-L510) |

A zero-voltage, zero-frequency linear cavity produces NaN coordinates and transfer matrices when absolute-time tracking is enabled. The phase calculation divides by the zero frequency before reducing the time modulo an infinite period. The same 1 m cavity is a finite drift in relative-time tracking, as required when the field is zero. The NaN occurs with the cavity both on and off.

Run `python TRK-24.py` with [_bmad.py](../helpers/_bmad.py) and [_rf_and_spin.py](../helpers/_rf_and_spin.py) in the same directory.

??? example "Reproducer: TRK-24.py"

    [Download TRK-24.py](TRK-24.py)

    ```python
    """TRK-24: A zero-frequency lcavity gives NaN with absolute-time tracking

    A zero-voltage, zero-frequency linear cavity produces NaN coordinates and transfer matrices when absolute-time tracking is enabled. The phase calculation divides by the zero frequency before reducing the time modulo an infinite period. The same 1 m cavity is a finite drift in relative-time tracking, as required when the field is zero. The NaN occurs with the cavity both on and off.

    The script tracks the same zero-field cavity with relative and absolute time, using BMAD_DIST. It compares relative-time coordinates and matrices with the analytic drift, then tests coordinates and matrices for NaNs with absolute time enabled.
    """

    import numpy as np
    from _bmad import fortran, verdict
    from _rf_and_spin import control, rows

    PROGRAM = """program zero_frequency
    use bmad
    implicit none
    type(lat_struct),target::lat
    type(ele_struct),pointer::cav
    type(coord_struct)::a,b
    real(rp)::v(6),mat(6,6)
    integer k
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) error stop 'parse'
    cav=>lat%ele(1)
    do k=1,2
     cav%is_on=(k==1)
     call set_flags_for_changed_attribute(cav,cav%is_on)
     call lattice_bookkeeper(lat,err)
     if(err) error stop 'bookkeeping'
     v=0;v(1)=1e-3_rp;v(5)=0.003_rp
     call init_coord(a,v,cav,upstream_end$,electron$)
     call track1(a,cav,lat%param,b,err_flag=err)
     if(err) error stop 'track'
     write(*,'(a,i0,6es25.16)') 'RESULT orbit',k,b%vec
     b=a
     call mat_make_unit(mat)
     call track1_bmad(b,cav,lat%param,mat6=mat,make_matrix=.true.)
     write(*,'(a,i0,36es25.16)') 'RESULT matrix',k,mat
    enddo
    end program
    """
    LATTICE = """no_digested
    beginning[p0c]=5e6
    parameter[particle]=electron
    parameter[geometry]=open
    c: lcavity, l=1, voltage=0, rf_frequency=0
    ln: line=(c)
    use, ln
    """
    defect = False
    for absolute in (False, True):
        lattice = LATTICE + f"\nbmad_com[absolute_time_tracking]={'T' if absolute else 'F'}\n"
        d = rows(fortran(PROGRAM, files={"lat.bmad": lattice}).log)
        for k in (1, 2):
            orb = d[f"orbit{k}"]
            mat = d[f"matrix{k}"]
            state = "on" if k == 1 else "off"
            if not absolute:
                control(
                    f"relative-time {state} drift orbit", orb, [0.001, 0, 0, 0, 0.003, 0], atol=2e-15
                )
                if not np.isfinite(mat).all():
                    raise AssertionError("Relative-time control matrix is nonfinite")
                # At px=py=delta=0, the exact drift has R12=R34=L and
                # R56=L/gamma^2 in Bmad's z=beta*c*(t_ref-t) convention.
                expected_mat = np.eye(6)
                expected_mat[0, 1] = expected_mat[2, 3] = 1
                expected_mat[4, 5] = 510998.95069**2 / (5e6**2 + 510998.95069**2)
                control(
                    f"relative-time {state} analytic drift matrix",
                    mat.reshape(6, 6, order="F"),
                    expected_mat,
                    atol=3e-15,
                )
            else:
                if np.isinf(orb).any() or np.isinf(mat).any():
                    raise AssertionError("Unexpected infinite output")
                orbit_bad = bool(np.isnan(orb).any())
                matrix_bad = bool(np.isnan(mat).any())
                print(
                    f"absolute-time {state}: orbit NaNs={np.isnan(orb).sum()}/6 matrix NaNs={np.isnan(mat).sum()}/36; expected=0 NaNs"
                )
                defect |= orbit_bad or matrix_bad
                if not orbit_bad:
                    control(
                        f"absolute-time {state} drift orbit",
                        orb,
                        [0.001, 0, 0, 0, 0.003, 0],
                        atol=2e-15,
                    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-22</span> | lcavity-is-on (`1df84a7745db`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-10-22</span> | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |

