---
title: "COL-36: Touschek emittance comments reverse the open and closed conventions"
hide:
  - toc
---

# COL-36 — Touschek emittance comments reverse the open and closed conventions

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P5 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [touschek_rate1 input comments, bmad/multiparticle/touschek_mod.f90:400–404](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/touschek_mod.f90#L400-L404)<br>[geometry-dependent emittance assignment, bmad/multiparticle/touschek_mod.f90:474–495](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/touschek_mod.f90#L474-L495)<br>[beam-size variances, bmad/multiparticle/touschek_mod.f90:513–516](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/touschek_mod.f90#L513-L516) |

When callers follow the Touschek rate helper's comments, they provide normalized emittances for a ring and geometric emittances for an open line. The calculation does the reverse: it uses ring inputs directly in beam-size variances and divides open-line inputs by the Lorentz factor γ. For identical local optics at γ = 1e5, an open-line input scaled by γ exactly matches the closed-ring loss rate of 0.5845 particles/s; the same unscaled input gives 6.160 particles/s. The comments should describe geometric ring inputs and the open branch's ultrarelativistic normalized convention.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _collective_contracts.py](../helpers/_collective_contracts.py) · [View source](../helpers/index.md#_collective_contracts)

??? example "Reproducer: COL-36.py"

    [Download COL-36.py](COL-36.py)

    ```python
    """COL-36: Touschek emittance comments reverse the open and closed conventions

    When callers follow the Touschek rate helper's comments, they provide normalized emittances for a ring and geometric emittances for an open line. The calculation does the reverse: it uses ring inputs directly in beam-size variances and divides open-line inputs by the Lorentz factor γ. For identical local optics at γ = 1e5, an open-line input scaled by γ exactly matches the closed-ring loss rate of 0.5845 particles/s; the same unscaled input gives 6.160 particles/s. The comments should describe geometric ring inputs and the open branch's ultrarelativistic normalized convention.

    The script calls the production Touschek routine with identical local optics and longitudinal
    beam sizes in closed and open geometries. It checks equality after multiplying the open
    emittances by γ and contrasts the same unscaled input. BMAD_DIST selects the library and
    BMAD_SOURCE supplies the disputed comments.
    """

    from _bmad import compare, fortran, verdict
    from _collective_contracts import rows, source_text

    r = rows(
        fortran("""program emittance
    use touschek_mod
    implicit none
    type(lat_struct) lat
    type(normal_modes_struct) mode
    real(rp) rate
    integer i,j
    call init_lat(lat,2)
    lat%n_ele_track=1
    lat%param%particle=electron$
    lat%param%n_part=1e10_rp
    do i=0,2
     lat%ele(i)%value(e_tot$)=1e5_rp*m_electron
     lat%ele(i)%a%beta=10
     lat%ele(i)%b%beta=20
     lat%ele(i)%z%sigma_p=0.001_rp
     lat%ele(i)%z%sigma=0.01_rp
    enddo
    mode%sigE_E=0.001_rp
    mode%sig_z=0.01_rp
    mode%pz_aperture=0.03_rp
    do j=1,3
     lat%param%geometry=closed$
     mode%a%emittance=1e-8_rp
     mode%b%emittance=1e-8_rp
     if (j>=2) lat%param%geometry=open$
     if (j==3) then
      mode%a%emittance=1e5_rp*mode%a%emittance
      mode%b%emittance=1e5_rp*mode%b%emittance
     endif
     call touschek_rate1(mode,rate,lat,ix=1)
     write(*,'(a,i0,es25.16)') 'RESULT case',j,rate
    enddo
    end program
    """).log
    )
    if compare(
        "closed geometric vs open gamma*geometric emittances, loss rate (particles/s)",
        r["case1"],
        r["case3"],
        atol=0,
        rtol=1e-12,
    ):
        raise RuntimeError("Equivalent geometric beam-size control failed")
    mismatch = compare(
        "same supplied emittances, open vs closed loss rate (particles/s)",
        r["case2"],
        r["case1"],
        atol=0,
        rtol=1e-12,
    )
    text = source_text("bmad/multiparticle/touschek_mod.f90")
    wrong = "contain the normalized emittences" in text
    print(
        f"header says closed normalized / open unnormalized={wrong}; execution requires closed geometric / open gamma*geometric"
    )
    verdict(wrong and mismatch)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found (source/contract discrepancy) | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed (source/contract discrepancy) | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

