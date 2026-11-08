---
title: "LAT-34: An explicit END marker does not suppress the automatic marker"
hide:
  - toc
---

# LAT-34 — An explicit END marker does not suppress the automatic marker

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [`parser_expand_line` automatic END insertion, bmad/parsing/bmad_parser_mod.f90:5951–5958](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/bmad_parser_mod.f90#L5951-L5958)<br>[Bmad manual, “Branch Construction Overview” (`s:branch.construct`), bmad/doc/lines-and-lists.tex:25–28](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/lines-and-lists.tex#L25-L28) |

A beam line ending with an explicit END marker gains a second END marker by default. A drift followed by END expands to D, END, END, with three tracked elements and two identical terminal names, instead of the documented D, END sequence. The manual explicitly promises to suppress automatic insertion when an ending END marker is already present. Setting no_end_marker to true produces the expected two-element sequence.

Run `python LAT-34.py` with [_bmad.py](../helpers/_bmad.py) and [_multipass_slicing.py](../helpers/_multipass_slicing.py) in the same directory.

??? example "Reproducer: LAT-34.py"

    [Download LAT-34.py](LAT-34.py)

    ```python
    """LAT-34: An explicit END marker does not suppress the automatic marker

    A beam line ending with an explicit END marker gains a second END marker by default. A drift followed by END expands to D, END, END, with three tracked elements and two identical terminal names, instead of the documented D, END sequence. The manual explicitly promises to suppress automatic insertion when an ending END marker is already present. Setting no_end_marker to true produces the expected two-element sequence.

    The script parses the same drift and explicit END marker with automatic insertion enabled and disabled. It checks branch length and the suppression control, then compares the default element names and counts with the manual's contract.
    """

    from _bmad import compare, fortran, verdict
    from _multipass_slicing import HEADER, control, rows

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
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) stop 1
    n=0
    do i=1,lat%n_ele_track
      write(*,'(a,a)') 'ELEMENT ',trim(lat%ele(i)%name)
      if (lat%ele(i)%name=='END') n=n+1
    enddo
    call row('counts',real([lat%n_ele_track,n],rp))
    call row('length',[lat%ele(lat%n_ele_track)%s])
    call deallocate_lat_pointers(lat)
    contains
    subroutine row(name,values)
    character(*) name
    real(rp) values(:)
    write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
    end subroutine
    end program
    """
    defect = False
    for suppressed in (True, False):
        lattice = (
            HEADER
            + f"parameter[no_end_marker]={'T' if suppressed else 'F'}\nd:drift,l=1\nend:marker\nbl:line=(d,end)\nuse,bl\n"
        )
        result = fortran(PROGRAM, files={"lat.bmad": lattice})
        data = rows(result.log)
        names = [
            line.removeprefix("ELEMENT ")
            for line in result.log.splitlines()
            if line.startswith("ELEMENT ")
        ]
        control(f"no_end_marker={suppressed}: branch length (m)", data["length"], [1], atol=0)
        if suppressed:
            control("suppression control [tracked, END count]", data["counts"], [2, 1])
            if names != ["D", "END"]:
                raise RuntimeError(f"Suppression control names: {names}")
        else:
            print(
                f"default names: observed={names} expected=['D', 'END'] from the terminal-marker contract"
            )
            defect = compare("default [tracked, END count]", data["counts"], [2, 1], atol=0)
            defect |= names != ["D", "END"]
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

