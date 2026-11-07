---
title: "COL-48: Classic and Raubenheimer Coulomb logs omit the documented upper impact-parameter cap"
hide:
  - toc
---

# COL-48 — Classic and Raubenheimer Coulomb logs omit the documented upper impact-parameter cap

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [multi_coulomb_log upper impact-parameter choices](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/ibs_mod.f90#L759-L804)<br>[“Coulomb Logarithm”](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_ring/doc/ibs_ring.tex#L300-L345) |

Selecting the classic or Raubenheimer Coulomb logarithm uses the vertical beam size as the
largest impact parameter even when the mean interparticle distance is smaller. The manual
specifies the smaller distance, and the other two logarithm options apply that cap. The classic
logarithm is 24.43 instead of 22.12 for the LHC-like proton example, overstating the scattering
rate by 10.45%; the ATF-like electron example is 5.660% high. Low-density round-beam cases
agree because the cap is inactive.

Run `python COL-48.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-48.py"

    [Download COL-48.py](COL-48.py)

    ```python
    """COL-48: Classic and Raubenheimer Coulomb logs omit the documented upper impact-parameter cap

    Selecting the classic or Raubenheimer Coulomb logarithm uses the vertical beam size as the
    largest impact parameter even when the mean interparticle distance is smaller. The manual
    specifies the smaller distance, and the other two logarithm options apply that cap. The classic
    logarithm is 24.43 instead of 22.12 for the LHC-like proton example, overstating the scattering
    rate by 10.45%; the ATF-like electron example is 5.660% high. Low-density round-beam cases
    agree because the cap is inactive.

    The script scans density and evaluates LHC-like and ATF-like beams. It derives the documented
    upper impact parameter from the beam dimensions and rest-frame density, while checking the two
    capped options and low-density cases as controls. The option-2 extension is established by its
    source expression for the minimum momentum transfer.
    """

    import math

    from _bmad import compare, fortran, verdict

    PROGRAM = """! Prints multi_coulomb_log options 1, 3 and 4 for elements without dispersion:
    !   scan      -- the COL-47 proton element (gamma 100, beta_a = beta_b = 10 m, emit 1e-8 m,
    !                sigma_p 1e-3) over line densities 1e4 to 1e14 /m, tau_a = 1 s;
    !   lhc       -- LHC-like 7 TeV protons (beta 100 m, normalized emittance 3.75 um,
    !                sigma_p 1.13e-4, N = 1.15e11, sigma_z = 7.55 cm);
    !   atf       -- ATF-like 1.28 GeV electrons (beta 5 m, emit 1.2 nm / 12 pm, sigma_p 6e-4,
    !                N = 1e10, sigma_z = 6 mm).
    ! Columns: label, gamma, beta_a, beta_b, emit_a, emit_b, sigma_p, lambda (/m), tau_a (s),
    ! clog1, clog3, clog4. The Python oracle recomputes the logs from these inputs.
    program probe
    use bmad
    use ibs_mod
    implicit none
    type (lat_struct), target :: lat_p, lat_e
    type (ibs_sim_param_struct) :: cfg
    integer :: k
    real(rp) :: gam_lhc

    call bmad_parser('proton.bmad', lat_p)
    call bmad_parser('electron.bmad', lat_e)
    write(*, '(a,2es25.16)') 'RADII ', classical_radius(proton$), classical_radius(electron$)
    cfg%tau_a = 1
    do k = 4, 14, 2
      call one('scan', lat_p%ele(1), 100.0_rp, 10.0_rp, 10.0_rp, 1e-8_rp, 1e-8_rp, 1e-3_rp, 10.0_rp**k)
    enddo
    gam_lhc = 7e12_rp / mass_of(proton$)
    call one('lhc', lat_p%ele(1), gam_lhc, 100.0_rp, 100.0_rp, 3.75e-6_rp / sqrt(gam_lhc**2 - 1), &
             3.75e-6_rp / sqrt(gam_lhc**2 - 1), 1.13e-4_rp, 1.15e11_rp / 0.0755_rp)
    call one('atf', lat_e%ele(1), 1.28e9_rp / mass_of(electron$), 5.0_rp, 5.0_rp, 1.2e-9_rp, 1.2e-11_rp, &
             6e-4_rp, 1e10_rp / 6e-3_rp)

    contains

    subroutine one(label, ele, gamma, beta_a, beta_b, emit_a, emit_b, sigma_p, lambda)
    character(*) :: label
    type (ele_struct) :: ele
    real(rp) :: gamma, beta_a, beta_b, emit_a, emit_b, sigma_p, lambda, clog(4)
    integer :: i
    ele%value(e_tot$) = gamma * mass_of(ele%branch%param%particle)
    ele%a%beta = beta_a;  ele%a%alpha = 0;  ele%a%gamma = 1 / beta_a;  ele%a%eta = 0;  ele%a%etap = 0
    ele%b%beta = beta_b;  ele%b%alpha = 0;  ele%b%gamma = 1 / beta_b;  ele%b%eta = 0;  ele%b%etap = 0
    ele%a%emit = emit_a;  ele%b%emit = emit_b;  ele%z%sigma_p = sigma_p
    do i = 1, 4
      if (i == 2) cycle
      cfg%clog_to_use = i
      call multi_coulomb_log(cfg, ele, clog(i), lambda)
    enddo
    write(*, '(a, 1x, 8es24.16, 3f22.15)') label, gamma, beta_a, beta_b, emit_a, emit_b, sigma_p, lambda, &
                                         cfg%tau_a, clog(1), clog(3), clog(4)
    end subroutine

    end program
    """
    log = fortran(
        PROGRAM,
        files={
            "electron.bmad": """! One electron drift; the probe sets the energy, Twiss and beam sizes.
    parameter[particle] = electron
    parameter[geometry] = open
    beginning[e_tot] = 100 * mass_of(electron)
    d: drift, l = 1
    l1: line = (d)
    use, l1
    """,
            "proton.bmad": """! One proton drift; the probe sets the energy, Twiss and beam sizes.
    parameter[particle] = proton
    parameter[geometry] = open
    beginning[e_tot] = 100 * mass_of(proton)
    d: drift, l = 1
    l1: line = (d)
    use, l1
    """,
        },
    ).log
    radii = [
        list(map(float, line.split()[1:])) for line in log.splitlines() if line.startswith("RADII ")
    ]
    rows = [line.split() for line in log.splitlines() if line.startswith(("scan ", "lhc ", "atf "))]
    if len(radii) != 1 or len(rows) != 8:
        raise RuntimeError(log)
    defect = False
    for row in rows:
        label = row[0]
        gamma, ba, bb, ea, eb, sp, density, tau, observed, log3, log4 = map(float, row[1:])
        r = radii[0][int(label == "atf")]
        sa, sb = math.sqrt(ba * ea), math.sqrt(bb * eb)
        volume = (4 * math.pi) ** 1.5 * sa * sb * gamma
        distance = (volume / density) ** (1 / 3)
        bmax = min(sa, sb, distance)
        bmin = r / (gamma**2 * ea / ba)
        expected = math.log(bmax / bmin)
        for option, angle, got in [
            (3, math.sqrt(ea / ba), log3),
            (4, math.sqrt(ea / ba + eb / bb + (sp / gamma) ** 2), log4),
        ]:
            tail = math.sqrt(volume / density / math.pi / (tau / gamma)) / math.sqrt(
                299792458 * gamma * angle
            )
            if compare(
                f"{label} density={density:g} option {option} capped control",
                got,
                math.log(bmax / tail),
                atol=2e-13,
            ):
                raise RuntimeError("Capped-option control failed")
        bad = compare(
            f"{label} density={density:g} classic log; ln(min(sigma_a,sigma_b,d)/bmin)",
            observed,
            expected,
            atol=2e-13,
        )
        if label == "scan" and density <= 1e6 and bad:
            raise RuntimeError("Low-density control failed")
        print(
            f"{label} sigma_b={sb:.12g} m d={distance:.12g} m rate excess={observed / expected - 1:.10%}"
        )
        defect |= bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>maximum-impact-parameter cap (source check)</small> | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`) |
| Confirmed | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed<br><small>classic-log density cap; option 2 by source only</small> | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

