---
title: "LAT-33: The field-map assignment manual overstates shared storage"
hide:
  - toc
---

# LAT-33 — The field-map assignment manual overstates shared storage

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [Bmad manual, “Initialization and Pointers”, assignment contract, bmad/doc/ele-struct.tex:170–178](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/ele-struct.tex#L170-L178)<br>[Cartesian payload sharing, bmad/low_level/transfer_fieldmap.f90:55–60](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/transfer_fieldmap.f90#L55-L60)<br>[cylindrical sharing and generalized-gradient copy, bmad/low_level/transfer_fieldmap.f90:94–130](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/transfer_fieldmap.f90#L94-L130)<br>[grid sharing, bmad/low_level/transfer_fieldmap.f90:163–168](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/transfer_fieldmap.f90#L163-L168)<br>[explicit Cartesian detachment, bmad/code/pointer_to_attribute.f90:279–283](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/pointer_to_attribute.f90#L279-L283) |

The element-assignment manual says that all four field-map pointer arrays share storage after an element copy. Assignment instead creates separate outer descriptors, shares the inner Cartesian, cylindrical and grid payloads, and independently copies generalized-gradient data. Editing a copied Cartesian coefficient changes the source from 1 to 2, while descriptor and generalized-gradient edits leave the source values at 1 and 5. Explicit Cartesian unlinking separates the coefficients, so the documentation needs to distinguish these ownership rules.

Run `python LAT-33.py` with [_bmad.py](../helpers/_bmad.py) and [_multipass_slicing.py](../helpers/_multipass_slicing.py) in the same directory.

??? example "Reproducer: LAT-33.py"

    [Download LAT-33.py](LAT-33.py)

    ```python
    """LAT-33: The field-map assignment manual overstates shared storage

    The element-assignment manual says that all four field-map pointer arrays share storage after an element copy. Assignment instead creates separate outer descriptors, shares the inner Cartesian, cylindrical and grid payloads, and independently copies generalized-gradient data. Editing a copied Cartesian coefficient changes the source from 1 to 2, while descriptor and generalized-gradient edits leave the source values at 1 and 5. Explicit Cartesian unlinking separates the coefficients, so the documentation needs to distinguish these ownership rules.

    The script copies four elements, each containing one map family. It checks pointer associations and edits immediately after assignment, then repeated-transfer reference counts and explicit unlinking. It reads the assignment contract from BMAD_SOURCE and compares its shared-storage statement with the observed copy behavior.
    """

    from _bmad import compare, fortran, source, verdict
    from _multipass_slicing import control, rows

    PROGRAM = r"""program probe
    use bmad
    implicit none
    type(ele_struct), target :: src(4), dst(4)
    type(all_pointer_struct) ptr
    logical err
    integer i
    ! A generalized-gradient map is exclusive of other map families in an element.
    do i=1,4
      call init_ele(src(i),quadrupole$)
    enddo
    allocate(src(1)%cartesian_map(1),src(2)%cylindrical_map(1),src(3)%grid_field(1),src(4)%gen_gradients(1))
    allocate(src(1)%cartesian_map(1)%ptr,src(2)%cylindrical_map(1)%ptr,src(3)%grid_field(1)%ptr,src(4)%gen_gradients(1)%curve(1))
    allocate(src(1)%cartesian_map(1)%ptr%term(1),src(2)%cylindrical_map(1)%ptr%term(1), &
             src(3)%grid_field(1)%ptr%pt(1,1,1),src(4)%gen_gradients(1)%curve(1)%deriv(0:0,0:0))
    src(4)%gen_gradients(1)%curve(1)%deriv=5
    src(1)%cartesian_map(1)%ptr%term(1)%coef=1
    do i=1,4
      dst(i)=src(i)
    enddo
    call row('shared', [real(merge(1,0,associated(src(1)%cartesian_map(1)%ptr,dst(1)%cartesian_map(1)%ptr)),rp), &
      real(merge(1,0,associated(src(2)%cylindrical_map(1)%ptr,dst(2)%cylindrical_map(1)%ptr)),rp), &
      real(merge(1,0,associated(src(3)%grid_field(1)%ptr,dst(3)%grid_field(1)%ptr)),rp)])
    call row('links', real([src(1)%cartesian_map(1)%ptr%n_link,src(2)%cylindrical_map(1)%ptr%n_link, &
                          src(3)%grid_field(1)%ptr%n_link],rp))
    call row('outer_shared', [real(merge(1,0,associated(src(1)%cartesian_map,dst(1)%cartesian_map)),rp), &
                            real(merge(1,0,associated(src(4)%gen_gradients,dst(4)%gen_gradients)),rp)])
    dst(1)%cartesian_map(1)%ptr%term(1)%coef=2
    dst(1)%cartesian_map(1)%field_scale=7
    dst(4)%gen_gradients(1)%curve(1)%deriv=7
    call row('source_after_edit', [src(1)%cartesian_map(1)%ptr%term(1)%coef,src(1)%cartesian_map(1)%field_scale, &
                                 src(4)%gen_gradients(1)%curve(1)%deriv(0,0)])
    ! Test assignment before any second transfer can change descriptor or gradient ownership.
    do i=1,4
      call transfer_fieldmap(src(i),dst(i),all$)
    enddo
    call row('repeat_links', real([src(1)%cartesian_map(1)%ptr%n_link,src(2)%cylindrical_map(1)%ptr%n_link, &
                                src(3)%grid_field(1)%ptr%n_link],rp))
    call pointer_to_attribute(dst(1),'CARTESIAN_MAP(1)%TERM(1)%A',.false.,ptr,err,do_unlink=.true.)
    if (err .or. .not. associated(ptr%r)) stop 10
    ptr%r=3
    call row('detached_values',[src(1)%cartesian_map(1)%ptr%term(1)%coef,dst(1)%cartesian_map(1)%ptr%term(1)%coef])
    call row('detached_links',real([src(1)%cartesian_map(1)%ptr%n_link,dst(1)%cartesian_map(1)%ptr%n_link],rp))
    do i=1,4
      call deallocate_ele_pointers(dst(i))
    enddo
    call row('remaining_links',real([src(1)%cartesian_map(1)%ptr%n_link,src(2)%cylindrical_map(1)%ptr%n_link, &
                                   src(3)%grid_field(1)%ptr%n_link],rp))
    do i=1,4
      call deallocate_ele_pointers(src(i))
    enddo
    contains
    subroutine row(name,values)
    character(*) name
    real(rp) values(:)
    write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
    end subroutine
    end program
    """
    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in (source() / "bmad/doc/ele-struct.tex").read_text().split("\n\n")
    ]
    claim = r"field component pointers of \vn{ele1} and \vn{ele2} after the set \vn{ele1 = ele2} will point at the same memory locations."
    # The sharing sentence must still refer to all four outer arrays in its own
    # paragraph. A statement scoped to the three shared inner payloads is consistent.
    families = ("cartesian_map", "cylindrical_map", "grid_field", "gen_gradients")
    shared_claim = any(
        claim in paragraph
        and "pointer arrays" in paragraph
        and all(r"\vn{ele%" + family + "}" in paragraph for family in families)
        for paragraph in paragraphs
    )
    print(
        f"manual unqualified field-array sharing claim: observed={shared_claim} expected=False for the measured ownership rules"
    )
    data = rows(fortran(PROGRAM).log)
    control("shared inner Cartesian/cylindrical/grid payloads", data["shared"], [1, 1, 1])
    control("payload reference counts after assignment", data["links"], [2, 2, 2])
    outer_differs = compare(
        "outer Cartesian/generalized-gradient sharing, manual expectation",
        data["outer_shared"],
        [1, 1],
        atol=0,
    )
    edit_differs = compare(
        "source [Cartesian coef, field scale, generalized derivative], manual sharing expectation",
        data["source_after_edit"],
        [2, 7, 7],
        atol=0,
    )
    control("payload reference counts after repeated transfer", data["repeat_links"], [2, 2, 2])
    control("explicit Cartesian detach coefficients", data["detached_values"], [2, 3])
    control("detached reference counts", data["detached_links"], [1, 1])
    control("reference counts after destination cleanup", data["remaining_links"], [1, 1, 1])
    verdict(shared_claim and (outer_differs or edit_differs))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gemini-4-argon-high`</span> |  | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

