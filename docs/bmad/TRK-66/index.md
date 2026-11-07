---
title: "TRK-66: Full bend fringes omit the tracked particle charge factor"
hide:
  - toc
---

# TRK-66 — Full bend fringes omit the tracked particle charge factor

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad 20261031-0 (`c4c629f`, committed 2026-10-31) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [full-fringe dispatcher, bmad/modules/fringe_mod.f90:72–78](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/fringe_mod.f90#L72-L78)<br>[magnetic strength, bmad/modules/fringe_mod.f90:1536–1541](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/fringe_mod.f90#L1536-L1541)<br>[both exact edge compositions, bmad/modules/fringe_mod.f90:1563–1579](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/fringe_mod.f90#L1563-L1579)<br>[linear fringes, bmad/modules/fringe_mod.f90:174–197](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/fringe_mod.f90#L174-L197)<br>[basic fringes, bmad/modules/fringe_mod.f90:266–299](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/fringe_mod.f90#L266-L299)<br>[production edge caller, bmad/low_level/apply_element_edge_kick.f90:188–189](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/apply_element_edge_kick.f90#L188-L189)<br>[field/potential edges, bmad/low_level/apply_element_edge_kick.f90:130–148](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/apply_element_edge_kick.f90#L130-L148)<br>[PTC-coordinate reference helper, bmad/modules/fringe_mod.f90:2021–2025](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/fringe_mod.f90#L2021-L2025)<br>[bend-field contract, bmad/doc/elements.tex:715–730](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L715-L730)<br>[exact fringe equations, bmad/doc/fringe-fields.tex:244–335](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/fringe-fields.tex#L244-L335)<br>[fringe option contract, bmad/doc/attributes.tex:3431–3454](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L3431-L3454) |

With `fringe_type=full`, native bend edges use the reference particle's magnetic coupling when the tracked particle has another charge. Both edge kicks and their orbital matrices miss the charge factor. At 0.8000 mm vertical displacement, a particle with twice the reference charge receives 0.2774 µrad of finite-gap focusing instead of 1.110 µrad, as obtained directly from the exact fringe equations. The omission also halves the nonlinear zero-gap displacement and reverses finite-face-angle kicks for opposite charge. Forward matched-charge full fringes and the tested linear and default basic fringe paths obey their charge-scaling equations.

Run `python TRK-66.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: TRK-66.py"

    [Download TRK-66.py](TRK-66.py)

    ```python
    """TRK-66: Full bend fringes omit the tracked particle charge factor

    With `fringe_type=full`, native bend edges use the reference particle's magnetic coupling when the tracked particle has another charge. Both edge kicks and their orbital matrices miss the charge factor. At 0.8000 mm vertical displacement, a particle with twice the reference charge receives 0.2774 µrad of finite-gap focusing instead of 1.110 µrad, as obtained directly from the exact fringe equations. The omission also halves the nonlinear zero-gap displacement and reverses finite-face-angle kicks for opposite charge. Forward matched-charge full fringes and the tested linear and default basic fringe paths obey their charge-scaling equations.

    The script uses BMAD_DIST and the shared runner to call the production bend-edge dispatcher at both ends for equal-mass particles with relative charges −1, 1 and 2. Direct reductions of the exact fringe equations supply finite-gap kicks and matrices, zero-gap displacements and finite-face-angle kicks. Matched-charge full fringes and the linear and default basic fringe models provide controls with their own equations.
    """

    from __future__ import annotations

    import math

    from _bmad import fortran, verdict

    PROGRAM = r"""
    program confirmation
    use bmad
    use fringe_mod
    implicit none
    type(lat_struct) :: lat
    type(ele_struct) :: e
    type(coord_struct) :: a
    real(rp) :: v(6), mat(6,6)
    integer :: j,sid,method,orientation,edge,q
    logical :: err
     call bmad_parser('fringe.bmad',lat,err_flag=err)
     if(err) error stop 'fringe parse'
     do method=1,3
     do orientation=1,1
     do q=-2,4,2
     if(q==0) cycle
     do edge=1,2
     do j=0,2
      e=lat%ele(1)
      e%orientation=orientation
      select case(method)
      case(1); e%value(fringe_type$)=full$
      case(2); e%value(fringe_type$)=linear_edge$
      case(3); e%value(fringe_type$)=basic_bend$
      end select
      if(j==1) then
       e%value(hgap$)=0; e%value(hgapx$)=0
      endif
      if(j==2) then
       e%value(hgap$)=0; e%value(hgapx$)=0
       e%value(e1$)=0.13_rp; e%value(e2$)=0.13_rp
      endif
      if(q<0) then
       sid=species_id('@M16'//trim(int_str(q)))
      else
       sid=species_id('@M16+'//trim(int_str(q)))
      endif
      v=0; v(3)=0.0008_rp
      call init_coord(a,v,e,upstream_end$,particle=sid)
      call convert_pc_to(a%p0c,sid,beta=a%beta)
      call mat_make_unit(mat)
      if(edge==1) then
       call bend_edge_kick(e,lat%param,first_track_edge$,a,mat,.true.)
      else
       call bend_edge_kick(e,lat%param,second_track_edge$,a,mat,.true.)
      endif
      if(a%state/=alive$) error stop 'edge tracking'
      write(*,'(a,5i4,8es26.17)') 'EDGE ',method,orientation,q,edge,j, &
         rel_tracking_charge_to_mass(a,lat%param%particle),a%vec,mat(4,3)
     enddo
     enddo
     enddo
     enddo
     enddo
    end program
    """

    LATTICE = """no_digested
    parameter[particle]=@M16+2
    parameter[p0c]=800e6
    parameter[geometry]=open
    b: sbend,l=0.3,g=0,dg=0.17,fint=0.4,fintx=0.4,hgap=0.015,hgapx=0.015,fringe_type=full
    ln: line=(b)
    use,ln
    """

    def records(log: str, tag: str, count: int) -> list[list[float]]:
        if any(marker in log for marker in ("[ERROR", "[FATAL", "[ABORT")):
            raise RuntimeError(log)
        result = [[float(x) for x in line.split()[1:]] for line in log.splitlines()
                  if line.startswith(tag + " ")]
        if not result or any(len(row) != count or not all(map(math.isfinite, row)) for row in result):
            raise RuntimeError(f"Missing or invalid {tag} records:\n{log}")
        return result


    def close(actual: list[float], expected: list[float], tolerance: float, label: str) -> bool:
        if len(actual) != len(expected) or not actual:
            raise RuntimeError(f"{label}: shape mismatch")
        return all(math.isclose(a, e, abs_tol=tolerance, rel_tol=0)
                   for a, e in zip(actual, expected, strict=True))


    def require(actual: list[float], expected: list[float], tolerance: float, label: str) -> None:
        if not close(actual, expected, tolerance, label):
            raise RuntimeError(f"{label}: observed={actual}, expected={expected}, atol={tolerance}")


    def main() -> None:
        rows = records(fortran(PROGRAM, files={"fringe.bmad": LATTICE}).log, "EDGE", 13)
        if len(rows) != 54:
            raise RuntimeError(f"Expected 54 edge cases, received {len(rows)}")
        wrong = False
        controls = {2: 0, 3: 0}
        for row in rows:
            method, orientation, charge, edge, case = map(int, row[:5])
            chi, x, _, _, py, _, _, m43 = row[5:]
            require([chi, float(orientation)], [charge/2, 1.0], 0, "Charge/geometry inputs")
            k, y = 0.17*chi, 0.0008
            gap = 0.4*0.015 if case == 0 else 0.0
            face = 0.13 if case == 2 else 0.0
            # At zero incoming momenta and x=delta=0, the exact manual map reduces
            # to py=k*tan(2*K*k)*y, M43=k*tan(2*K*k) at zero face angle.
            # At zero gap the nonlinear x shift is +/-k*y^2/2. A finite face
            # instead gives py=-k*tan(e)*y; the wedge maps preserve py.
            expected = -k*math.tan(face-2*gap*k*(1+math.sin(face)**2)/math.cos(face))
            if method == 3 and case == 0:
                expected = 2*gap*k*k  # Basic fringes retain the second-order expression.
            if method != 1:
                require([py, m43*y], [expected*y, expected*y], 2e-19, "Linear/basic equation")
                controls[method] += 1
                continue
            observed, target, tolerance = ([py, m43*y], [expected*y, expected*y], 2e-19)
            if case == 1:
                observed = [x, py]
                target = [(1 if edge == 1 else -1)*k*y*y/2, 0.0]
                tolerance = 2e-20
            if charge == 2:
                require(observed, target, tolerance, "Matched-charge exact-map control")
            wrong |= not close(observed, target, tolerance, "Exact fringe equation")
            kind = ("finite_gap", "zero_gap", "finite_face")[case]
            print(f"full chi={chi:g} edge={edge} {kind}: "
                  f"py={py:.17g} expected_exact={expected*y:.17g}; "
                  f"M43={m43:.17g} expected_exact={expected:.17g} m^-1"
                  + (f"; x={x:.17g} expected_exact={target[0]:.17g} m" if case == 1 else ""))
        print(f"Matched full-fringe controls pass; linear={controls[2]} and "
              f"default basic={controls[3]} kick/matrix equation controls pass")
        verdict(wrong)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

