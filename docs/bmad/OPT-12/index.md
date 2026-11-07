---
title: "OPT-12: Single-element matrix cache hits report an error"
hide:
  - toc
---

# OPT-12 — Single-element matrix cache hits report an error

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [lat_make_mat6, bmad/code/lat_make_mat6.f90:240–250](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/lat_make_mat6.f90#L240-L250)<br>[initialization, bmad/code/lat_make_mat6.f90:59–64](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/lat_make_mat6.f90#L59-L64)<br>[output contract, bmad/code/lat_make_mat6.f90:21–26](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/lat_make_mat6.f90#L21-L26) |

Requesting a single-element transfer matrix whose cached reference orbit already matches returns an error flag despite a valid matrix. This happens both with and without an explicit reference-orbit array. In a 1.2 m quadrupole of strength 0.7 m⁻², the cached horizontal position coefficient remains the correct 0.5369, but the call reports failure. A successful cache hit should report success just like recomputation.

Run `python OPT-12.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_14_optics_matrix_radiation.py](../helpers/_s3_14_optics_matrix_radiation.py) in the same directory.

??? example "Reproducer: OPT-12.py"

    [Download OPT-12.py](OPT-12.py)

    ```python
    """OPT-12: Single-element matrix cache hits report an error

    Requesting a single-element transfer matrix whose cached reference orbit already matches returns an error flag despite a valid matrix. This happens both with and without an explicit reference-orbit array. In a 1.2 m quadrupole of strength 0.7 m⁻², the cached horizontal position coefficient remains the correct 0.5369, but the call reports failure. A successful cache hit should report success just like recomputation.

    The script exercises both single-element cache branches in separate processes, with and without an explicit orbit array. It verifies the cached quadrupole coefficient against the analytic thick-lens cosine before inspecting the error flag.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _s3_14_optics_matrix_radiation import control, rows

    PROGRAM = """
    program architecture_contracts
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: q
    type(coord_struct), allocatable :: orbit(:)
    real(rp) v(6)
    logical err
    character(40) mode
    call get_command_argument(1,mode)
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) stop 1
    q=>lat%ele(1)
    call row('cache_setup', [real(merge(1,0,q%bookkeeping_state%mat6==ok$),rp), q%mat6(1,1)])
    if (mode == 'cache_ref') then
      call reallocate_coord(orbit, lat%n_ele_max)
      v=0
      call init_coord(orbit(0),v,lat%ele(0),downstream_end$)
      call track_all(lat,orbit,err_flag=err)
      if (err) stop 7
      call lat_make_mat6(lat,ix_ele=1,ref_orb=orbit,err_flag=err)
    else
      call lat_make_mat6(lat,ix_ele=1,err_flag=err)
    endif
    call row('cache_error', [real(merge(1,0,err),rp)])
    call row('cache_matrix', [q%mat6(1,1)])
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

    defect = False
    expected = np.cos(np.sqrt(0.7) * 1.2)
    for mode in ("cache", "cache_ref"):
        r = rows(fortran(PROGRAM, [mode], files={"lat.bmad": LATTICE}).log)
        control(f"{mode}: [cache valid, M11]", r["cache_setup"], [1, expected], atol=2e-16, rtol=2e-14)
        control(
            f"{mode}: cached M11, thick-lens cosine",
            r["cache_matrix"],
            [expected],
            atol=2e-16,
            rtol=2e-14,
        )
        defect |= compare(f"{mode}: error flag, successful cache hit", r["cache_error"], [0], atol=0)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

