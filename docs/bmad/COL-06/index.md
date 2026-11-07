---
title: "COL-06: The 1D CSR fast kick leaves beta and z inconsistent"
hide:
  - toc
---

# COL-06 — The 1D CSR fast kick leaves beta and z inconsistent

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [csr_and_sc_apply_kicks fast branch, bmad/space_charge/csr_and_space_charge_mod.f90:1532–1549](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1532-L1549)<br>[coordinate update, bmad/space_charge/csr_and_space_charge_mod.f90:1599–1603](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1599-L1603)<br>[subsequent FFT dispatch, bmad/space_charge/csr_and_space_charge_mod.f90:1646–1650](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1646-L1650)<br>[Charged Particle Phase Space Coordinates, s:phase.space / zbctt, phase-space.tex:80–101](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/phase-space.tex#L80-L101) |

The one-dimensional coherent synchrotron radiation fast path changes particle momentum without refreshing speed β or longitudinal coordinate z. Bmad defines z = −cβΔt, where Δt is the arrival-time offset, so an instantaneous kick must update β and rescale z to preserve arrival time. For an electron at reference momentum 2 MeV/c receiving a normalized momentum kick of −1e−5, β stays too large by 5.937e−7 and z stays too large by 4.596e−7 m. The general branch with slice space charge performs the required refresh.

Run `python COL-06.py` with [_bmad.py](../helpers/_bmad.py) and [_collective_fields.py](../helpers/_collective_fields.py) in the same directory.

??? example "Reproducer: COL-06.py"

    [Download COL-06.py](COL-06.py)

    ```python
    """COL-06: The 1D CSR fast kick leaves beta and z inconsistent

    The one-dimensional coherent synchrotron radiation fast path changes particle momentum without
    refreshing speed β or longitudinal coordinate z. Bmad defines z = −cβΔt, where Δt is the
    arrival-time offset, so an instantaneous kick must update β and rescale z to preserve arrival time.
    For an electron at reference momentum 2 MeV/c receiving a normalized momentum kick of −1e−5, β
    stays too large by 5.937e−7 and z stays too large by 4.596e−7 m. The general branch with slice
    space charge performs the required refresh.

    The script applies a zero kick, a small negative kick and a larger diagnostic positive kick to
    electrons. It derives β from the mass shell and z from fixed arrival time, independently of Bmad’s
    conversion routine.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _collective_fields import rows

    PROGRAM = r"""
    program csr_clock
    use csr_and_space_charge_mod
    implicit none
    type(ele_struct) e
    type(csr_struct) c
    type(coord_struct) p(1)
    integer j
    real(rp) pc, z, kick
    character(16) label
    allocate(c%slice(3))
    space_charge_com%n_bin=3
    c%dz_slice=1
    c%slice(1)%z_center=0
    c%slice(2)%z_center=1
    c%slice(3)%z_center=2
    e%csr_method=one_dim$
    e%space_charge_method=off$
    do j=0,2
     pc=2e6_rp; z=.75_rp; kick=-1e-5_rp
     if(j==0) kick=0
     if(j==2) then
      pc=1e6_rp; z=1; kick=.2_rp
     endif
     c%slice%kick_csr=kick
     p(1)=coord_struct()
     p(1)%p0c=pc; p(1)%species=electron$; p(1)%state=alive$
     p(1)%vec(5)=z
     p(1)%beta=pc/sqrt(pc**2+m_electron**2)
     call csr_and_sc_apply_kicks(e,c,p)
     write(label,'(a,i0)') 'case',j
     print '(3a,8es24.15)', 'RESULT ',trim(label),' ',pc,z,kick,m_electron,p(1)%vec(6),p(1)%beta,p(1)%vec(5)
    enddo
    end program
    """


    data = rows(fortran(PROGRAM).log)
    defect = False
    for j in range(3):
        pc, z, kick, mass, delta, beta, actual_z = data[f"case{j}"]
        if not np.isclose(delta, kick, rtol=2e-14, atol=1e-17):
            raise RuntimeError("Momentum-kick control failed")
        # Mass shell fixes beta; z/beta = -c*delta_t must survive an instantaneous kick.
        before = pc / np.hypot(pc, mass)
        after = pc * (1 + kick) / np.hypot(pc * (1 + kick), mass)
        bad = compare(
            f"case {j} beta and z (m), mass shell and fixed arrival time",
            [beta, actual_z],
            [after, z * after / before],
            atol=2e-15,
            rtol=2e-14,
        )
        if j == 0 and bad:
            raise RuntimeError("Zero-kick control failed")
        if j:
            defect |= bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

