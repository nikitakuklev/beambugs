---
title: "OPT-10: Linear tracking stores incompatible map-reference records"
hide:
  - toc
---

# OPT-10 — Linear tracking stores incompatible map-reference records

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [make_mat6 input reference, bmad/code/make_mat6.f90:105–117](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/make_mat6.f90#L105-L117)<br>[returned and stored output references, bmad/code/make_mat6.f90:174–196](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/make_mat6.f90#L174-L196)<br>[map-reference contract, bmad/doc/ele-struct.tex:540–550](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/ele-struct.tex#L540-L550) |

Constructing a linear tracking map at a nonzero incident orbit stores that orbit as the input reference but stores the zero-orbit expansion as the output reference. The two records should satisfy the documented affine relation between the matrix, constant vector and reference coordinates. In a 1.2 m quadrupole, the stored output is zero while its affine reconstruction begins at x = 0.8394 mm and normalized horizontal momentum −0.0005447. The returned particle orbit is correct at zero momentum offset.

Run `python OPT-10.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_14_optics_matrix_radiation.py](../helpers/_s3_14_optics_matrix_radiation.py) in the same directory.

??? example "Reproducer: OPT-10.py"

    [Download OPT-10.py](OPT-10.py)

    ```python
    """OPT-10: Linear tracking stores incompatible map-reference records

    Constructing a linear tracking map at a nonzero incident orbit stores that orbit as the input reference but stores the zero-orbit expansion as the output reference. The two records should satisfy the documented affine relation between the matrix, constant vector and reference coordinates. In a 1.2 m quadrupole, the stored output is zero while its affine reconstruction begins at x = 0.8394 mm and normalized horizontal momentum −0.0005447. The returned particle orbit is correct at zero momentum offset.

    The script constructs a linear quadrupole map for a nonzero transverse input at zero momentum offset. It verifies the returned particle against a thick-lens solution, then tests the documented affine identity for the stored reference records.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _s3_14_optics_matrix_radiation import control, quad, rows

    PROGRAM = """
    program architecture_contracts
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: q
    type(coord_struct) start, ending
    real(rp) v(6)
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) stop 1
    q=>lat%ele(1)
    q%tracking_method=linear$
    v=0; v(1)=0.001_rp; v(2)=0.0003_rp
    call init_coord(start,v,q,upstream_end$)
    call make_mat6(q,lat%param,start,ending,err)
    if (err) stop 5
    call row('map_in',q%map_ref_orb_in%vec)
    call row('map_out',q%map_ref_orb_out%vec)
    call row('returned',ending%vec)
    call row('affine',matmul(q%mat6,q%map_ref_orb_in%vec)+q%vec0)
    call deallocate_lat_pointers(lat)
    contains
    subroutine row(name,values)
    character(*) name
    real(rp) values(:)
    write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
    end subroutine
    end program
    """
    LATTICE = "\nno_digested\nparameter[particle] = electron\nparameter[p0c] = 1e9\nparameter[geometry] = open\nbeginning[beta_a] = 10\nbeginning[beta_b] = 10\nq: quadrupole, l = 1.2, k1 = 0.7, fringe_type = none\nline1: line = (q)\n\nuse, line1\n"

    r = rows(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
    expected = np.zeros(6)
    expected[:2] = quad(0.7, 1.2) @ [0.001, 0.0003]
    control("returned orbit, analytic linear quadrupole", r["returned"], expected, atol=2e-16)
    verdict(compare("stored map output, M*map_input+vec0", r["map_out"], r["affine"], atol=2e-16))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

