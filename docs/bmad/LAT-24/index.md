---
title: "LAT-24: Multipass bookkeeping deletes multipole arrays, losing kicks or crashing"
hide:
  - toc
---

# LAT-24 — Multipass bookkeeping deletes multipole arrays, losing kicks or crashing

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`makeup_multipass_slave`, bmad/modules/bookkeeper_mod.f90:394–460](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bookkeeper_mod.f90#L394-L460)<br>[`multipole_ele_to_kt`, bmad/code/multipole_ele_to_kt.f90:49–80](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/multipole_ele_to_kt.f90#L49-L80)<br>[Bmad manual, “Multipass Fundamentals” (`c:multipass.fund`), bmad/doc/multipass.tex:13–59](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/multipass.tex#L13-L59) |

A multipass line containing magnetic multipole elements loses the multipole arrays of its tracked passes. For a particle at horizontal position x = 1 mm, two thin quadrupole passes, each with integrated strength 0.25 m⁻¹, produce zero normalized horizontal momentum kick instead of −0.0005000, and their matrix focusing terms vanish. The alternative multipole representation crashes when its deleted arrays are read. Ordinary repeated elements retain their kicks and matrices.

Run `python LAT-24.py` with [_bmad.py](../helpers/_bmad.py) and [_multipass_slicing.py](../helpers/_multipass_slicing.py) in the same directory.

??? example "Reproducer: LAT-24.py"

    [Download LAT-24.py](LAT-24.py)

    ```python
    """LAT-24: Multipass bookkeeping deletes multipole arrays, losing kicks or crashing

    A multipass line containing magnetic multipole elements loses the multipole arrays of its tracked passes. For a particle at horizontal position x = 1 mm, two thin quadrupole passes, each with integrated strength 0.25 m⁻¹, produce zero normalized horizontal momentum kick instead of −0.0005000, and their matrix focusing terms vanish. The alternative multipole representation crashes when its deleted arrays are read. Ordinary repeated elements retain their kicks and matrices.

    The script compares ordinary and multipass lines at the same reference momentum and checks the thin-quadrupole formula. It also checks the alternative multipole representation against a successful ordinary-element control and identifies the missing-array crash.
    """

    import resource
    import signal

    from _bmad import RunFailure, compare, fortran, verdict
    from _multipass_slicing import HEADER, control, rows

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    PROGRAM = r"""
    program architecture_multipass
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: lord
    type(coord_struct), allocatable :: orbit(:)
    real(rp) v(6), knl(0:n_pole_maxx), tilt(0:n_pole_maxx)
    integer n
    logical err
    character(8) mode
    call get_command_argument(1, mode)
    call bmad_parser('lat.bmad', lat, make_mats6=.false., err_flag=err)
    if (err) stop 1
    lord => pointer_to_multipass_lord(lat%ele(1))
    if (associated(lord)) then
      write(*,'(a,3i4)') 'RESULT topology', lat%n_ele_track, lord%n_slave, &
        merge(1,0,lat%ele(1)%slave_status == multipass_slave$)
    else
      lord => lat%ele(1)
      write(*,'(a,3i4)') 'RESULT topology', lat%n_ele_track, 0, 0
    endif
    if (mode == 'kt') then
      if (lord%key /= multipole$ .or. .not. associated(lord%a_pole)) stop 2
      write(*,'(a,es25.16)') 'RESULT lord_strength', lord%a_pole(1)
      write(*,'(a,3i4)') 'RESULT pointers', merge(1,0,associated(lat%ele(1)%a_pole)), &
        merge(1,0,associated(lat%ele(1)%b_pole)), merge(1,0,associated(lat%ele(1)%a_pole_elec))
      print *, 'CHECKPOINT multipole_ele_to_kt'
      flush(6)
      call multipole_ele_to_kt(lat%ele(1), .false., n, knl, tilt)
      write(*,'(a,3es25.16)') 'RESULT multipole', real(n,rp), knl(1), tilt(1)
    else
      if (lord%key /= ab_multipole$ .or. .not. associated(lord%b_pole)) stop 3
      write(*,'(a,es25.16)') 'RESULT lord_strength', lord%b_pole(1)
      call reallocate_coord(orbit, lat%n_ele_max)
      v=0
      v(1)=0.001_rp
      call init_coord(orbit(0), v, lat%ele(0), downstream_end$)
      call track_all(lat, orbit, err_flag=err)
      if (err) stop 4
      call lat_make_mat6(lat, ix_branch=-1, err_flag=err)
      if (err) stop 5
      write(*,'(a,6es25.16)') 'RESULT orbit', orbit(lat%n_ele_track)%vec
      write(*,'(a,2es25.16)') 'RESULT matrices', lat%ele(1)%mat6(2,1), lat%ele(2)%mat6(2,1)
    endif
    call deallocate_lat_pointers(lat)
    end program
    """


    def lattice(definition, multipass):
        return (
            HEADER
            + f"m: {definition}\nsection: line{'[multipass]' if multipass else ''} = (m)\nbl: line=(section,section)\nuse,bl\n"
        )


    def run(mode, definition, multipass):
        return rows(fortran(PROGRAM, [mode], files={"lat.bmad": lattice(definition, multipass)}).log)


    normal = run("ab", "ab_multipole, b1=0.25", False)
    subject = run("ab", "ab_multipole, b1=0.25", True)
    for name, data, topology in (("ordinary", normal, [3, 0, 0]), ("multipass", subject, [3, 2, 1])):
        control(f"{name} topology", data["topology"], topology)
        control(f"{name} lord strength (m^-1)", data["lord_strength"], [0.25])
    # A thin normal quadrupole gives delta px = -b1*x; two zero-length passes add.
    expected_orbit = [0.001, -2 * 0.25 * 0.001, 0, 0, 0, 0]
    control("ordinary orbit: two thin-quadrupole kicks", normal["orbit"], expected_orbit, atol=2e-16)
    control("ordinary dpx/dx (m^-1)", normal["matrices"], [-0.25, -0.25], atol=2e-16)
    defect = compare(
        "multipass orbit: two thin-quadrupole kicks", subject["orbit"], expected_orbit, atol=2e-16
    )
    defect |= compare("multipass dpx/dx (m^-1)", subject["matrices"], [-0.25, -0.25], atol=2e-16)
    normal_kt = run("kt", "multipole, k1l=0.3", False)
    control("ordinary multipole [order, k1l, tilt]", normal_kt["multipole"], [1, 0.3, 0], atol=2e-16)
    try:
        subject_kt = run("kt", "multipole, k1l=0.3", True)
    except RunFailure as exc:
        data = rows(exc.log)
        control("multipole crash topology", data["topology"], [3, 2, 1])
        control("multipole crash lord strength", data["lord_strength"], [0.3])
        if (
            exc.status != -signal.SIGSEGV
            or data.get("pointers") != [0, 0, 0]
            or "CHECKPOINT multipole_ele_to_kt" not in exc.log
            or "Program received signal SIGSEGV" not in exc.log
            or "multipole" in data
        ):
            raise
        print(
            "multipass multipole: observed=SIGSEGV after missing-array checkpoint expected=successful [1, 0.3, 0] dispatch"
        )
        defect = True
    else:
        defect |= compare(
            "multipass multipole [order, k1l, tilt]", subject_kt["multipole"], [1, 0.3, 0], atol=2e-16
        )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

