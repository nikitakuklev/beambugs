---
title: "COL-15: Cathode image cutoff compares against the total field instead of the bunch field"
hide:
  - toc
---

# COL-15 — Cathode image cutoff compares against the total field instead of the bunch field

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [sc_field_calc cutoff, bmad/space_charge/space_charge_mod.f90:126–138](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/space_charge_mod.f90#L126-L138)<br>[direct-plus-image assembly, bmad/space_charge/open_spacecharge_mod.f90:665–691](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L665-L691)<br>[Bmad manual, Space_Charge_Common_Struct, s:sc.com, bmad/doc/param-structs.tex:503–508](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/param-structs.tex#L503-L508) |

When cathode image fields are enabled, the cutoff compares the image field with the sum of direct and image fields instead of the direct bunch field. Cancellation changes whether images remain enabled. In a 27-particle example, the largest image/direct component ratio is 0.6762, while the image/total ratio is 0.7083. A cutoff of 0.6922 therefore retains images even though the documented bunch-field criterion would disable them.

Run `python COL-15.py` with [_bmad.py](../helpers/_bmad.py) and [_collective_contracts.py](../helpers/_collective_contracts.py) in the same directory.

??? example "Reproducer: COL-15.py"

    [Download COL-15.py](COL-15.py)

    ```python
    """COL-15: Cathode image cutoff compares against the total field instead of the bunch field

    When cathode image fields are enabled, the cutoff compares the image field with the sum of direct and image fields instead of the direct bunch field. Cancellation changes whether images remain enabled. In a 27-particle example, the largest image/direct component ratio is 0.6762, while the image/total ratio is 0.7083. A cutoff of 0.6922 therefore retains images even though the documented bunch-field criterion would disable them.

    The script evaluates the direct and total fields for the same 27 stationary particles on a 16³ mesh.
    It derives the image field by subtraction, tests a cutoff between the two component ratios, and
    checks zero and high cutoffs as controls.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _collective_contracts import rows

    PROGRAM = """
    program cathode_cutoff
      use bmad
      use space_charge_mod
      implicit none
      type(lat_struct) lat
      type(bunch_struct) bunch
      type(em_field_struct) direct(27), total(27)
      integer i, j, k, n
      logical include_image
      character(40) arg
      call get_command_argument(1, arg)
      read(arg, *) space_charge_com%cathode_strength_cutoff
      call bmad_parser('lat.bmad', lat)
      space_charge_com%space_charge_mesh_size = [16,16,16]
      call reallocate_bunch(bunch, 27)
      n = 0
      do i = -1, 1
        do j = -1, 1
          do k = 1, 3
            n = n+1
            bunch%particle(n)%state = alive$
            bunch%particle(n)%species = electron$
            bunch%particle(n)%ix_ele = 1
            bunch%particle(n)%vec(1) = i*0.01_rp
            bunch%particle(n)%vec(3) = j*0.02_rp
            bunch%particle(n)%s = k*0.01_rp
            bunch%particle(n)%beta = 0
            bunch%particle(n)%charge = 1.0e-12_rp/27
          enddo
        enddo
      enddo
      include_image = .false.
      call sc_field_calc(bunch, lat%branch(0), include_image, 1.0_rp, direct)
      include_image = .true.
      call sc_field_calc(bunch, lat%branch(0), include_image, 1.0_rp, total)
      ! Particle one is the first minimum-s particle selected by minloc.
      write(*, '(a,3es25.16e3)') 'RESULT direct ', direct(1)%E
      write(*, '(a,3es25.16e3)') 'RESULT total ', total(1)%E
      write(*, '(a,i4)') 'RESULT retained ', merge(1, 0, include_image)
    end program
    """
    FILES = {
        "lat.bmad": """parameter[particle]=electron
    parameter[p0c]=1e6
    d: drift, l=1
    l: line=(d)
    use, l
    """
    }
    initial = rows(fortran(PROGRAM, ["0"], files=FILES).log)
    direct, total = initial["direct"], initial["total"]
    if not np.isfinite([direct, total]).all() or np.any(direct == 0) or np.any(total == 0):
        raise RuntimeError("The field-ratio fixture requires finite nonzero components")
    image = total - direct
    rd = float(np.max(abs(image / direct)))
    rt = float(np.max(abs(image / total)))
    print(f"image/direct={rd:.16g}; image/total={rt:.16g}")
    if initial["retained"].tolist() != [1]:
        raise RuntimeError("Zero-cutoff control failed")
    defect = False
    for cutoff in [(rd + rt) / 2, 1.01 * max(rd, rt)]:
        result = rows(fortran(PROGRAM, [f"{cutoff:.17g}"], files=FILES).log)
        for key in ["direct", "total"]:
            if compare(f"field control {key} (V/m)", result[key], initial[key], atol=0, rtol=2e-14):
                raise RuntimeError("Changing the cutoff changed the evaluated field")
        # The manual compares image field with bunch field; the direct run isolates the latter.
        mismatch = compare(
            f"cutoff={cutoff:.16g}, retain image iff max|image/direct|>cutoff",
            result["retained"],
            [int(rd > cutoff)],
            atol=0,
        )
        if cutoff > max(rd, rt) and mismatch:
            raise RuntimeError("High-cutoff control failed")
        defect |= mismatch
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

