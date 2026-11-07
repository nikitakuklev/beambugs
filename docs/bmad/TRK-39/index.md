---
title: "TRK-39: track_all accepts an orbit array one record too short"
hide:
  - toc
---

# TRK-39 — track_all accepts an orbit array one record too short

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [`track_all` allocation guard and output access, bmad/code/track_all.f90:57–77](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/track_all.f90#L57-L77)<br>[`reallocate_coord_n`, bmad/low_level/reallocate_coord.f90:26–52](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/reallocate_coord.f90#L26-L52) |

Calling track_all with an orbit array one record shorter than the lattice requires can write beyond the allocation. The routine grows smaller arrays but compares their size with the last required index, forgetting that the orbit starts at index zero. A two-element lattice supplied with indices 0:1 bypasses reallocation and attempts to write index 2; bounds checking reports the invalid access. A correctly sized array tracks successfully.

Run `python TRK-39.py` with [_bmad.py](../helpers/_bmad.py) and [_orbital_matrices_radiation.py](../helpers/_orbital_matrices_radiation.py) in the same directory.

??? example "Reproducer: TRK-39.py"

    [Download TRK-39.py](TRK-39.py)

    ```python
    """TRK-39: track_all accepts an orbit array one record too short

    Calling track_all with an orbit array one record shorter than the lattice requires can write beyond the allocation. The routine grows smaller arrays but compares their size with the last required index, forgetting that the orbit starts at index zero. A two-element lattice supplied with indices 0:1 bypasses reallocation and attempts to write index 2; bounds checking reports the invalid access. A correctly sized array tracks successfully.

    The script first checks a correctly sized array against the production library and a bounds-checked copy of the pin’s unmodified track_all source. It then runs the one-short array against that checked copy and requires the specific index-2 bounds failure.
    """

    from _bmad import RunFailure, compare, fortran, source, verdict
    from _orbital_matrices_radiation import checked_fortran, records

    PROGRAM = """
    program probe
    use bmad

    implicit none
    type(lat_struct), target :: lat
    type(coord_struct) a,b
    real(rp) v(6)
    logical err
    type(coord_struct),allocatable::orb(:)
    integer n,short,st
    character(10) arg
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) error stop 'parse failed'
    call get_command_argument(1,arg)
    read(arg,*) short
    n=lat%n_ele_max
    allocate(orb(0:n-short))
    v=0
    call init_coord(orb(0),v,lat%ele(0),downstream_end$)
    write(*,'(a,3i5)') 'RESULT before ',lat%n_ele_track,n,ubound(orb,1)
    flush(6)
    call track_all(lat,orb,track_state=st,err_flag=err)
    if(err) error stop 'track_all returned error'
    write(*,'(a,3i5)') 'RESULT after ',ubound(orb,1),n,st
    if(ubound(orb,1)<n) error stop 'orbit allocation remains short'
    write(*,'(a,es26.17)') 'RESULT final_x ',orb(lat%n_ele_track)%vec(1)
    end program
    """
    LATTICE = """
    no_digested
    parameter[geometry]=open
    parameter[particle]=electron
    beginning[p0c]=1e9
    e: sbend,l=2,g=.1,k1=1,fringe_type=none
    lat: line=(e)
    use,lat
    """
    # Compile the pin's unmodified routine with bounds checks, linked ahead of its archive.
    routine = (source() / "bmad/code/track_all.f90").read_text()
    control = records(fortran(PROGRAM, ("0",), files={"lat.bmad": LATTICE}).log)
    checked = records(checked_fortran(routine + "\n" + PROGRAM, LATTICE, ("0",)).log)
    for label, rows in (("production", control), ("bounds checked", checked)):
        if compare(f"{label} full-size array upper bound", rows["after"][0], rows["after"][1], atol=0):
            raise RuntimeError("Full-size control failed")
        if compare(f"{label} on-axis final x (m)", rows["final_x"], [0], atol=0):
            raise RuntimeError("Full-size orbit control failed")
    try:
        result = checked_fortran(routine + "\n" + PROGRAM, LATTICE, ("1",))
    except RunFailure as failure:
        rows = records(failure.log)
        if list(rows["before"]) != [2, 2, 1]:
            raise RuntimeError("Unexpected short-array setup") from failure
        expected = "Index '2' of dimension 1 of array 'orbit' above upper bound of 1"
        if failure.status != 2 or expected not in failure.log:
            raise
        print(
            f"one-short array: observed bounds failure: {expected}; expected automatic growth to upper bound 2"
        )
        verdict(True)
    else:
        rows = records(result.log)
        if rows["after"][0] < rows["after"][1]:
            raise RuntimeError("One-short array was not enlarged")
        print("one-short array: observed successful growth; expected upper bound 2")
        verdict(False)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

