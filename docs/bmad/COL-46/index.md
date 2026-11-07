---
title: "COL-46: multi_coulomb_log uses the root branch species for secondary branches"
hide:
  - toc
---

# COL-46 — multi_coulomb_log uses the root branch species for secondary branches

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [multi_coulomb_log species lookup](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/ibs_mod.f90#L737-L759)<br>[lat%param alias](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/allocate_branch_array.f90#L76-L80)<br>[bjmt1 species lookup](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/ibs_rates_mod.f90#L76-L85) |

Calling multi_coulomb_log for an ion element in a secondary branch uses the root branch’s
species to compute its Lorentz factor and classical radius. The same ion element must give the
same Coulomb logarithm wherever its branch is stored. For carbon ions with charge 6 at Lorentz
factor 120, placing the element beneath an electron root branch changes the classic logarithm
from 28.23 to 41.81. The rate routine at a fixed logarithm agrees between placements;
lattice-level IBS drivers use the root branch and avoid this mismatch.

Run `python COL-46.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-46.py"

    [Download COL-46.py](COL-46.py)

    ```python
    """COL-46: multi_coulomb_log uses the root branch species for secondary branches

    Calling multi_coulomb_log for an ion element in a secondary branch uses the root branch’s
    species to compute its Lorentz factor and classical radius. The same ion element must give the
    same Coulomb logarithm wherever its branch is stored. For carbon ions with charge 6 at Lorentz
    factor 120, placing the element beneath an electron root branch changes the classic logarithm
    from 28.23 to 41.81. The rate routine at a fixed logarithm agrees between placements;
    lattice-level IBS drivers use the root branch and avoid this mismatch.

    The script places the same carbon drift in a secondary branch beneath an electron root and in a
    single-branch lattice. It compares all four Coulomb-log options and requires the fixed-log rate
    routine to agree. The classic-log mass and radius scaling explains the placement-dependent
    difference.
    """

    from _bmad import compare, fortran, verdict

    PROGRAM = """! multi_coulomb_log and bjmt1 for the same 12C6+ drift, once in branch 1 under an electron
    ! root branch and once as the root branch. The species, gamma, emittances and density are
    ! identical, so every Coulomb log and rate must agree between the two placements.
    program probe
    use bmad
    use ibs_mod
    use ibs_rates_mod
    implicit none
    type (lat_struct), target :: lat2, lat1
    type (ele_struct), pointer :: e2, e1
    type (ibs_sim_param_struct) :: cfg
    type (ibs_struct) :: r2, r1
    real(rp) :: clog2, clog1, gam
    real(rp), parameter :: lambda = 2e10_rp / 0.05_rp
    integer :: i

    call bmad_parser('two_branch.bmad', lat2)
    call bmad_parser('one_branch.bmad', lat1)
    e2 => lat2%branch(1)%ele(1)
    e1 => lat1%branch(0)%ele(1)
    if(e2%branch%param%particle/=species_id('#12C+6')) error stop 'secondary species mismatch'
    if(e1%branch%param%particle/=e2%branch%param%particle) error stop 'control species mismatch'
    if(e2%branch%lat%param%particle/=electron$) error stop 'root species mismatch'
    if(abs(e2%value(e_tot$)/e1%value(e_tot$)-1)>1e-14_rp) error stop 'branch energies differ'
    call convert_total_energy_to(e2%value(e_tot$), e2%branch%param%particle, gam)
    if(abs(gam-120.0_rp)>1e-12_rp) error stop 'ion gamma mismatch'
    call set_beam(e2)
    call set_beam(e1)
    write(*, '(a, a, a, a)') 'branch-1 element species ', trim(species_name(e2%branch%param%particle)), &
                             ';  ele%branch%lat%param species ', trim(species_name(e2%branch%lat%param%particle))
    call convert_total_energy_to(e2%value(e_tot$), e2%branch%param%particle, gam)
    write(*, '(a, f0.9, a, es14.7)') 'element gamma (own species) ', gam, '   E_tot (eV) ', e2%value(e_tot$)
    call convert_total_energy_to(e2%value(e_tot$), e2%branch%lat%param%particle, gam)
    write(*, '(a, es14.7)') 'gamma from the root-branch species ', gam
    ! The classic-log shift from replacing carbon by electrons is ln(m_C/m_e)+ln(Q^2).
    write(*, '(a,es25.16)') 'EXPECTED_SHIFT ', log(mass_of(e2%branch%param%particle) / m_electron) + log(36.0_rp)
    cfg%tau_a = 1e-2_rp
    do i = 1, 4
      cfg%clog_to_use = i
      call multi_coulomb_log(cfg, e2, clog2, lambda)
      call multi_coulomb_log(cfg, e1, clog1, lambda)
      write(*, '(a,i1,2es25.16)') 'RESULT log', i, clog2, clog1
    enddo
    ! Rate routine at a fixed log of 20: it reads ele%branch%param, so both placements agree.
    call bjmt1(e2, 20.0_rp, r2, lambda)
    call bjmt1(e1, 20.0_rp, r1, lambda)
    write(*, '(a,2es25.16)') 'RESULT rate ', r2%inv_Tp, r1%inv_Tp

    contains
    subroutine set_beam(ele)
    type (ele_struct) :: ele
    ele%a%beta = 7;  ele%a%alpha = 0;  ele%a%gamma = 1 / ele%a%beta;  ele%a%eta = 0;  ele%a%etap = 0
    ele%b%beta = 4;  ele%b%alpha = 0;  ele%b%gamma = 1 / ele%b%beta;  ele%b%eta = 0;  ele%b%etap = 0
    ele%a%emit = 2.3e-6_rp
    ele%b%emit = 8e-7_rp
    ele%z%sigma_p = 1.7e-3_rp
    ele%z%sigma = 0.05_rp
    end subroutine
    end program
    """
    log = fortran(
        PROGRAM,
        files={
            "one_branch.bmad": """! Control: the same 12C6+ line as the only (root) branch.
    parameter[particle] = #12C+6
    parameter[geometry] = open
    beginning[e_tot] = 120 * mass_of(#12C+6)
    d: drift, l = 1
    lc: line = (d)
    use, lc
    """,
            "two_branch.bmad": """! Branch 0: electrons. Branch 1: 12C6+ at gamma = 120 The probe sets identical Twiss and beam sizes on both placements.
    parameter[particle] = electron
    parameter[geometry] = open
    beginning[e_tot] = 1e9
    d: drift, l = 1
    le: line = (d)
    lc: line = (d)
    lc[particle] = #12C+6
    lc[geometry] = open
    lc[e_tot] = 120 * mass_of(#12C+6)
    use, le, lc
    """,
        },
    ).log
    rows = {
        parts[1]: list(map(float, parts[2:]))
        for line in log.splitlines()
        if (parts := line.split()) and parts[0] == "RESULT"
    }
    if set(rows) != {"log1", "log2", "log3", "log4", "rate"}:
        raise RuntimeError(log)
    if compare(
        "fixed-log bjmt1 rate, s^-1; identical ion element",
        rows["rate"][0],
        rows["rate"][1],
        atol=0,
        rtol=2e-14,
    ):
        raise RuntimeError("Fixed-log branch control failed")
    defect = False
    for key in ["log1", "log2", "log3", "log4"]:
        defect |= compare(
            key + "; secondary branch vs identical root placement",
            rows[key][0],
            rows[key][1],
            atol=2e-12,
        )
    # The classic logarithm shift is ln(m_C/m_e) + ln(Q^2) on the pin's Q^2/m radius.
    shift_line = next(line for line in log.splitlines() if line.startswith("EXPECTED_SHIFT "))
    expected_shift = float(shift_line.split()[1])
    if compare(
        "classic log shift; ln(m_C/m_e)+ln(36) from root-species substitution",
        rows["log1"][0] - rows["log1"][1],
        expected_shift,
        atol=2e-12,
    ):
        raise RuntimeError("Branch difference does not match the species-substitution law")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-24 | ibs-ion-classical-radius (`886607535af8`), source, production, CPU, double, serial |
| Confirmed | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |

