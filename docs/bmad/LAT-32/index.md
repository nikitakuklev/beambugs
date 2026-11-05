---
title: "LAT-32: Indexed attribute access omits the spin method and cannot reach electric multipoles"
hide:
  - toc
---

# LAT-32 — Indexed attribute access omits the spin method and cannot reach electric multipoles

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [indexed-access contract and deprecation, bmad/low_level/pointer_to_indexed_attribute.f90:1–29](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/pointer_to_indexed_attribute.f90#L1-L29)<br>[multipole and component dispatch, bmad/low_level/pointer_to_indexed_attribute.f90:104–176](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/pointer_to_indexed_attribute.f90#L104-L176) |

The deprecated indexed-attribute routine rejects the spin tracking method and electric multipoles that named attribute lookup can access. On a quadrupole, named access returns the spin-method pointer and an electric B₀ coefficient set to 2, while both indexed calls report an error and return no pointer. The routine's indexed-access contract and its unreachable electric dispatch branch establish the discrepancy; named lookup provides a working access path.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _multipass_slicing.py](../helpers/_multipass_slicing.py) · [View source](../helpers/index.md#_multipass_slicing)

??? example "Reproducer: LAT-32.py"

    [Download LAT-32.py](LAT-32.py)

    ```python
    """LAT-32: Indexed attribute access omits the spin method and cannot reach electric multipoles

    The deprecated indexed-attribute routine rejects the spin tracking method and electric multipoles that named attribute lookup can access. On a quadrupole, named access returns the spin-method pointer and an electric B₀ coefficient set to 2, while both indexed calls report an error and return no pointer. The routine's indexed-access contract and its unreachable electric dispatch branch establish the discrepancy; named lookup provides a working access path.

    The script requests the same attributes through named and indexed lookup on one quadrupole. Successful named pointers, indexed TRACKING_METHOD access and the assigned electric coefficient establish the controls. Indexed results must identify the exact attribute target. BMAD_DIST selects the build.
    """

    from _bmad import compare, fortran, verdict
    from _multipass_slicing import HEADER, control, rows

    PROGRAM = r"""program probe
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: q
    type(ele_struct), target :: sliced, src, dst
    type(coord_struct) start, ending
    type(all_pointer_struct) ptr
    real(rp) v(6)
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) stop 1
    q=>lat%ele(1)
    call pointer_to_attribute(q,'SPIN_TRACKING_METHOD',.true.,ptr,err)
    if (err .or. .not. associated(ptr%i)) stop 8
    call row('named_spin', [real(merge(1,0,associated(ptr%i,q%spin_tracking_method)),rp)])
    call pointer_to_indexed_attribute(q,tracking_method$,.true.,ptr,err,err_print_flag=.false.)
    call row('indexed_control', [real(merge(1,0,err),rp), &
      real(merge(1,0,associated(ptr%i,q%tracking_method)),rp)])
    call pointer_to_indexed_attribute(q,spin_tracking_method$,.true.,ptr,err,err_print_flag=.false.)
    call row('indexed_spin', [real(merge(1,0,err),rp),real(merge(1,0,associated(ptr%i)),rp), &
      real(merge(1,0,associated(ptr%i,q%spin_tracking_method)),rp)])
    call pointer_to_attribute(q,'B0_ELEC',.true.,ptr,err)
    if (err .or. .not. associated(ptr%r)) stop 9
    ptr%r=2
    call row('named_electric', [q%b_pole_elec(0)])
    call pointer_to_indexed_attribute(q,b0_elec$,.true.,ptr,err,err_print_flag=.false.)
    call row('indexed_electric', [real(merge(1,0,err),rp),real(merge(1,0,associated(ptr%r)),rp), &
      real(merge(1,0,associated(ptr%r,q%b_pole_elec(0))),rp)])
    call deallocate_lat_pointers(lat)
    contains
    subroutine row(name,values)
    character(*) name
    real(rp) values(:)
    write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
    end subroutine
    end program
    """
    data = rows(
        fortran(
            PROGRAM,
            files={
                "lat.bmad": HEADER + "q:quadrupole,l=1.2,k1=0.7,fringe_type=none\nbl:line=(q)\nuse,bl\n"
            },
        ).log
    )
    control("named spin pointer associated", data["named_spin"], [1])
    control("indexed tracking method [error, correct target]", data["indexed_control"], [0, 1])
    control("named electric B0 value", data["named_electric"], [2])
    defect = compare(
        "indexed spin [error, associated, correct target], indexed access contract",
        data["indexed_spin"],
        [0, 1, 1],
        atol=0,
    )
    defect |= compare(
        "indexed electric B0 [error, associated, correct target], indexed access contract",
        data["indexed_electric"],
        [0, 1, 1],
        atol=0,
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

