---
title: "TRK-20: The LCavity tracking manual doubles the implemented fringe coefficient"
hide:
  - toc
---

# TRK-20 — The LCavity tracking manual doubles the implemented fringe coefficient

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P4 |
| Failure | Documentation or example mismatch |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [track_a_lcavity::fringe_kick, bmad/low_level/track_a_lcavity.f90:269–276](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L269-L276)<br>[transverse fringe application, bmad/low_level/track_a_lcavity.f90:309–312](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L309-L312)<br>[Bmad Manual, LCavity Tracking / Edge kick, bmad/doc/charged-tracking.tex:1203–1235](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/charged-tracking.tex#L1203-L1235)<br>[Bmad Manual, RF fringe fields, equation `pqe2c`, bmad/doc/fringe-fields.tex:455–465](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/fringe-fields.tex#L455-L465) |

The displayed LCavity tracking equations predict twice the transverse entrance focusing applied by the native tracker. For a 0.8 MV cavity with reference momentum 2 MeV/c and active length 0.5996 m, the measured horizontal focusing coefficient is −0.3336 m⁻¹, while those equations give −0.6671 m⁻¹. Removing the body transport from finite-difference tracking matrices isolates the edge. The manual’s separate RF-fringe derivation agrees with the implemented half-gradient coefficient.

Run `python TRK-20.py` with [_bmad.py](../helpers/_bmad.py) and [_rf_and_spin.py](../helpers/_rf_and_spin.py) in the same directory.

??? example "Reproducer: TRK-20.py"

    [Download TRK-20.py](TRK-20.py)

    ```python
    """TRK-20: The LCavity tracking manual doubles the implemented fringe coefficient

    The displayed LCavity tracking equations predict twice the transverse entrance focusing applied by the native tracker. For a 0.8 MV cavity with reference momentum 2 MeV/c and active length 0.5996 m, the measured horizontal focusing coefficient is −0.3336 m⁻¹, while those equations give −0.6671 m⁻¹. Removing the body transport from finite-difference tracking matrices isolates the edge. The manual’s separate RF-fringe derivation agrees with the implemented half-gradient coefficient.

    The script uses BMAD_DIST to obtain entrance-edge matrices from finite differences and remove the body transport. It uses BMAD_SOURCE to verify the displayed manual coefficient, then compares it with the isolated edge and the separate RF-fringe derivation.
    """

    import numpy as np
    from _bmad import compare, fortran, source, verdict
    from _rf_and_spin import control, rows

    PROGRAM = """program edge
    use bmad
    implicit none
    type(lat_struct) lat
    type(coord_struct) a,b
    real(rp) v(6)
    integer i,j
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) error stop 'parse'
    do i=1,1
     do j=1,4
      v=0
      if(j<=2) then
       v(1)=merge(1e-7_rp,-1e-7_rp,j==1)
      else
       v(2)=merge(1e-7_rp,-1e-7_rp,j==3)
      endif
      call init_coord(a,v,lat%ele(i),upstream_end$)
      call track1(a,lat%ele(i),lat%param,b,err_flag=err)
      if(err .or. b%state/=alive$) error stop 'track'
      write(*,'(a,i0,i0,2es25.16)') 'RESULT case',i,j,b%vec(1:2)
     enddo
    enddo
    end program
    """
    manual = (source() / "bmad/doc/charged-tracking.tex").read_text()
    formula = r"= \mp \frac{q}{P_0 \, c} G_t \, \cos(\phi_t + \phi_\REF) \, x"
    if formula not in manual:
        raise RuntimeError("LCavity displayed edge equation changed; resolve its coefficient")
    matrices = []
    for fringe in ("no_end", "entrance_end"):
        lattice = f"""no_digested
    parameter[particle]=electron
    parameter[p0c]=2e6
    parameter[geometry]=open
    e: lcavity, l=0.6, voltage=8e5, rf_frequency=5e8, phi0=0, n_rf_steps=8, cavity_type=traveling_wave, fringe_at={fringe}
    ln: line=(e)
    use,ln
    """
        d = rows(fortran(PROGRAM, files={"lat.bmad": lattice}).log)
        matrices.append(
            np.column_stack([(d["case11"] - d["case12"]) / 2e-7, (d["case13"] - d["case14"]) / 2e-7])
        )
    # M_on = M_body * D_margin^-1 * E_edge * D_margin.
    active_length = 299792458.0 / 5e8
    margin = (0.6 - active_length) / 2
    drift = np.array([[1, margin], [0, 1]])
    edge = drift @ np.linalg.solve(matrices[0], matrices[1]) @ np.linalg.inv(drift)
    gradient_over_p0 = 8e5 / active_length / 2e6
    control("edge diagonal and R12", [edge[0, 0], edge[0, 1], edge[1, 1]], [1, 0, 1], atol=2e-13)
    control(
        "RF-fringe derivation R21 (1/m), -G/(2 p0c)",
        edge[1, 0],
        -gradient_over_p0 / 2,
        atol=0,
        rtol=2e-12,
    )
    verdict(
        compare(
            "displayed LCavity R21 (1/m), -G/p0c", edge[1, 0], -gradient_over_p0, atol=0, rtol=2e-12
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

