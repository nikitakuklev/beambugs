---
title: "LAT-25: Overlay bookkeeping overruns its 20-attribute buffers"
hide:
  - toc
---

# LAT-25 — Overlay bookkeeping overruns its 20-attribute buffers

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P3 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`makeup_control_slave` buffers, bmad/modules/bookkeeper_mod.f90:1564–1567](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bookkeeper_mod.f90#L1564-L1567)<br>[`overlay_change_this` append, bmad/modules/bookkeeper_mod.f90:1815–1829](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bookkeeper_mod.f90#L1815-L1829) |

An overlay assigning more than twenty distinct real attributes of one element overruns fixed bookkeeping buffers. Assigning twenty normal multipole coefficients of a quadrupole succeeds, while adding the valid twenty-first coefficient crashes during lattice parsing. Bounds checking identifies an access to element 21 of a twenty-element pointer array. All twenty-one attributes are valid and should receive the requested value 0.001 without memory corruption.

Run `python LAT-25.py` with [_bmad.py](../helpers/_bmad.py) and [_multipass_slicing.py](../helpers/_multipass_slicing.py) in the same directory.

??? example "Reproducer: LAT-25.py"

    [Download LAT-25.py](LAT-25.py)

    ```python
    """LAT-25: Overlay bookkeeping overruns its 20-attribute buffers

    An overlay assigning more than twenty distinct real attributes of one element overruns fixed bookkeeping buffers. Assigning twenty normal multipole coefficients of a quadrupole succeeds, while adding the valid twenty-first coefficient crashes during lattice parsing. Bounds checking identifies an access to element 21 of a twenty-element pointer array. All twenty-one attributes are valid and should receive the requested value 0.001 without memory corruption.

    The script parses overlays controlling twenty and twenty-one coefficients, checking every value in the successful control. It identifies the production crash only after the checkpoint before parsing the twenty-one-target case; the companion bounds script locates the overflowing write.
    """

    import resource
    import signal

    from _bmad import RunFailure, compare, fortran, verdict
    from _multipass_slicing import HEADER, control, rows

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    PROGRAM = r"""program probe
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: ele, lord, slave
    type(ele_struct) :: s1,s2
    type(coord_struct) :: start, orb, ending, whole
    type(coord_struct), allocatable :: orbit(:)
    integer i, j, n
    real(rp) v(6)
    logical err
    character(40) label, arg
    call get_command_argument(1,arg)
    read(arg,*) n
    write(*,'(a,i0)') 'CHECKPOINT overlay attribute count ', n
    flush(6)
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) stop 1
    if (.not. associated(lat%ele(1)%b_pole)) stop 2
    call row('assigned',lat%ele(1)%b_pole(0:n-1))
    call deallocate_lat_pointers(lat)
    contains
    subroutine row(name,values)
    character(*) name
    real(rp) values(:)
    write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
    end subroutine
    end program
    """


    def lattice(count):
        targets = ", ".join(f"q[b{i}]:v" for i in range(count))
        return (
            HEADER
            + f"q:quadrupole,l=1,k1=0\no:overlay={{{targets}}},var={{v}},v=0.001\nbl:line=(q)\nuse,bl\n"
        )


    normal = rows(fortran(PROGRAM, ["20"], files={"lat.bmad": lattice(20)}).log)
    control(
        "twenty controlled coefficients, each v=0.001", normal["assigned"], [0.001] * 20, atol=1e-17
    )
    try:
        subject = rows(fortran(PROGRAM, ["21"], files={"lat.bmad": lattice(21)}).log)
    except RunFailure as exc:
        if (
            exc.status != -signal.SIGSEGV
            or "CHECKPOINT overlay attribute count 21" not in exc.log
            or "Parsing lattice file" not in exc.log
            or "Program received signal SIGSEGV" not in exc.log
            or "RESULT assigned" in exc.log
        ):
            raise
        print(
            "twenty-one controlled coefficients: observed=SIGSEGV during parsing expected=21 assignments of 0.001"
        )
        defect = True
    else:
        defect = compare(
            "twenty-one controlled coefficients", subject["assigned"], [0.001] * 21, atol=1e-17
        )
    verdict(defect)

    ```

??? example "Reproducer: LAT-25-bounds.py"

    [Download LAT-25-bounds.py](LAT-25-bounds.py)

    ```python
    """LAT-25: Identify the twenty-first overlay target with bounds checking

    The script recompiles the unchanged bookkeeper module from BMAD_SOURCE with bounds checking, links it ahead of the BMAD_DIST production library, and parses overlays with twenty and twenty-one valid targets. It checks all twenty control values before identifying the out-of-bounds pointer-buffer access.
    """

    from pathlib import Path

    from _bmad import RunFailure, _execute, _recipe, _scratch, source, verdict
    from _multipass_slicing import HEADER, control, rows

    PROGRAM = r"""program probe
    use bmad
    implicit none
    type(lat_struct) lat
    logical err
    integer n
    character(20) arg
    call get_command_argument(1,arg)
    read(arg,*) n
    write(*,'(a,i0)') 'CHECKPOINT overlay attribute count ',n
    flush(6)
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) stop 1
    if (.not. associated(lat%ele(1)%b_pole)) stop 2
    write(*,'(a,30es25.16)') 'RESULT assigned ',lat%ele(1)%b_pole(0:n-1)
    call deallocate_lat_pointers(lat)
    end program
    """


    def lattice(count):
        targets = ", ".join(f"q[b{i}]:v" for i in range(count))
        return (
            HEADER
            + f"q:quadrupole,l=1,k1=0\no:overlay={{{targets}}},var={{v}},v=0.001\nbl:line=(q)\nuse,bl\n"
        )


    compile_argv, link_argv = _recipe(False)
    with _scratch() as tmp:
        work = Path(tmp)
        module = source() / "bmad/modules/bookkeeper_mod.f90"
        # Override optimization and enable bounds checking only for this diagnostic object.
        module_compile = [
            str(module) if a == "probe.f90" else "bookkeeper_bounds.o" if a == "probe.o" else a
            for a in compile_argv
        ]
        _execute([*module_compile, "-g", "-O0", "-fcheck=bounds"], work, "", 290)
        (work / "probe.f90").write_text(PROGRAM)
        _execute(compile_argv, work, "", 290)
        link_argv.insert(link_argv.index("probe.o") + 1, "bookkeeper_bounds.o")
        _execute(link_argv, work, "", 290)
        for count in (20, 21):
            run_dir = work / f"case{count}"
            run_dir.mkdir()
            (run_dir / "lat.bmad").write_text(lattice(count))
            try:
                log = _execute([str(work / "probe"), str(count)], run_dir, "", 290)
            except RunFailure as exc:
                diagnostic = "Index '21' of dimension 1 of array 'ptr_attrib' above upper bound of 20"
                if (
                    count != 21
                    or exc.status != 2
                    or diagnostic not in exc.log
                    or "CHECKPOINT overlay attribute count 21" not in exc.log
                ):
                    raise
                print(
                    f"bounds-checked twenty-first target: observed={diagnostic} expected=successful assignment"
                )
                defect = True
            else:
                control(
                    f"bounds-checked {count} assigned coefficients",
                    rows(log)["assigned"],
                    [0.001] * count,
                    atol=1e-17,
                )
                if count == 21:
                    defect = False
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

