---
title: "LAT-21: Bunch-index documentation disagrees with initialization"
hide:
  - toc
---

# LAT-21 — Bunch-index documentation disagrees with initialization

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [bunch_struct index comment, bmad/modules/bmad_struct.f90:1130–1142](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bmad_struct.f90#L1130-L1142)<br>[init_beam_distribution, bmad/multiparticle/beam_utils.f90:253–275](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L253-L275)<br>[bunch_init_end_calc, bmad/multiparticle/beam_utils.f90:1985–1989](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L1985-L1989)<br>[init_bunch_distribution argument contract, bmad/multiparticle/beam_utils.f90:318–322](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L318-L322) |

Initializing a beam assigns stored bunch indices starting at zero, while the bunch structure comment says the head bunch has index one. The initializer uses the same zero-based value to set the bunch time offset. Three bunches separated by 4 ns have stored indices 0, 1 and 2 and time offsets 0, 4 and 8 ns. Documentation should distinguish the one-based array position from the stored index and timing convention.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _s3_12_particle_contracts.py](../helpers/_s3_12_particle_contracts.py) · [View source](../helpers/index.md#_s3_12_particle_contracts)

??? example "Reproducer: LAT-21.py"

    [Download LAT-21.py](LAT-21.py)

    ```python
    """LAT-21: Bunch-index documentation disagrees with initialization

    Initializing a beam assigns stored bunch indices starting at zero, while the bunch structure comment says the head bunch has index one. The initializer uses the same zero-based value to set the bunch time offset. Three bunches separated by 4 ns have stored indices 0, 1 and 2 and time offsets 0, 4 and 8 ns. Documentation should distinguish the one-based array position from the stored index and timing convention.

    The script initializes three bunches and prints their array positions, stored indices and time offsets. It checks the spacing against the zero-based timing rule and compares the indices with the structure comment supplied through BMAD_SOURCE.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _s3_12_particle_contracts import document_claim, rows

    program = """
    program indices
    use bmad
    use beam_utils
    implicit none
    type(ele_struct) ele
    type(lat_param_struct) param
    type(beam_init_struct) bi
    type(beam_struct) beam
    integer i
    logical err
    call init_ele(ele, drift$)
    ele%value(p0c$)=1e6_rp
    ele%value(p0c_start$)=1e6_rp
    ele%value(e_tot$)=hypot(1e6_rp,m_electron)
    ele%a%beta=1; ele%b%beta=1
    param%particle=electron$
    bi%n_particle=1; bi%n_bunch=3; bi%dt_bunch=4e-9_rp
    call init_beam_distribution(ele,param,bi,beam,err_flag=err)
    if (err) error stop 'initialization failed'
    write(*,'(a,3i8)') 'RESULT indices ', beam%bunch%ix_bunch
    write(*,'(a,3es26.17e3)') 'RESULT times ', beam%bunch%t_center
    end program
    """
    result = rows(fortran(program).log)
    np.testing.assert_array_equal(result["indices"], [0, 1, 2])
    if compare(
        "bunch t_center (s), zero-based spacing", result["times"], [0, 4e-9, 8e-9], atol=0, rtol=2e-15
    ):
        raise RuntimeError("Bunch spacing control failed")
    print(
        f"array positions=[1, 2, 3]; stored indices={result['indices'].tolist()}; comment expects=[1, 2, 3]"
    )
    verdict(document_claim("bmad/modules/bmad_struct.f90", "Bunch index. Head bunch = 1, etc."))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found (source/contract discrepancy) | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed (source/contract discrepancy) | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

