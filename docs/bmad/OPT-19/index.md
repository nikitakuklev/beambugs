---
title: "OPT-19: Chromaticity documentation labels a momentum derivative as energy"
hide:
  - toc
---

# OPT-19 — Chromaticity documentation labels a momentum derivative as energy

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P5 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`chrom_calc` argument descriptions, bmad/code/chrom_calc.f90:12–29](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/chrom_calc.f90#L12-L29)<br>[sixth-coordinate perturbation, bmad/code/chrom_calc.f90:204–217](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/chrom_calc.f90#L204-L217)<br>[Bmad manual, “Charged Particle Phase Space Coordinates”, `s:phase.space`, `ppp`, bmad/doc/phase-space.tex:62–78](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/phase-space.tex#L62-L78) |

The chromaticity API describes its perturbation arguments as energy changes, although the routine differentiates tune Q with respect to fractional momentum δ. At reference speed β₀c, fractional energy and momentum changes obey dE/E = β₀² dδ, so the energy derivative requires division by β₀². In a 12-cell electron FODO ring at 1 MeV/c, the horizontal result is −0.3572 per fractional momentum instead of the labelled energy derivative −0.4505.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _dispersion_and_moments.py](../helpers/_dispersion_and_moments.py) · [View source](../helpers/index.md#_dispersion_and_moments)

??? example "Reproducer: OPT-19.py"

    [Download OPT-19.py](OPT-19.py)

    ```python
    """OPT-19: Chromaticity documentation labels a momentum derivative as energy

    The chromaticity API describes its perturbation arguments as energy changes, although the routine differentiates tune Q with respect to fractional momentum δ. At reference speed β₀c, fractional energy and momentum changes obey dE/E = β₀² dδ, so the energy derivative requires division by β₀². In a 12-cell electron FODO ring at 1 MeV/c, the horizontal result is −0.3572 per fractional momentum instead of the labelled energy derivative −0.4505.

    The script executes chrom_calc on a finite-beta FODO ring and compares its result with an
    analytic momentum secant. It converts that secant to an energy derivative and reads the chrom_calc argument descriptions
    using BMAD_SOURCE. BMAD_DIST selects the build.
    """

    import numpy as np
    from _bmad import compare, source, verdict
    from _dispersion_and_moments import execute, fodo_matrix, lattice, require

    PROGRAM = r"""
    program optics_oracle
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(coord_struct), allocatable :: orb(:)
    type(normal_modes_struct) modes
    real(rp) m(6,6), v(4,4), h, ca, cb
    integer stat, i, n
    logical err
    character(40) option
    character(8) tag
    call get_command_argument(1, option)
    call bmad_parser('lat.bmad', lat)
    call twiss_and_track(lat, orb, stat, use_particle_start=.true.)
    if (stat /= ok$) error stop 'twiss_and_track failed'
    n = lat%n_ele_track
    call transfer_matrix_calc(lat, m)
    do i = 1, 6
      write(tag, '(a,i0)') 'm', i
      call row(trim(tag), m(i,:))
    enddo
    call make_v_mats(lat%ele(0), v_mat=v)
    do i = 1, 4
      write(tag, '(a,i0)') 'v', i
      call row(trim(tag), v(i,:))
    enddo
    call row('twiss', [lat%ele(0)%a%beta, lat%ele(0)%a%alpha, lat%ele(0)%b%beta, lat%ele(0)%b%alpha])
    call row('phase', [lat%ele(n)%a%phi, lat%ele(n)%b%phi])
    call row('dispersion', [lat%ele(0)%x%eta, lat%ele(0)%x%etap, lat%ele(0)%y%eta, lat%ele(0)%y%etap])
    call row('orbit0', orb(0)%vec)
    call row('orbit1', orb(n)%vec)
    if (lat%param%geometry == closed$) call row('alpha_c', [momentum_compaction(lat%branch(0))])
    call row('exit_dispersion', [lat%ele(n)%x%etap, lat%ele(n)%y%etap, &
                                lat%ele(n)%x%deta_ds, lat%ele(n)%y%deta_ds])
    if (option == 'chrom') then
      h = 1d-4
      call chrom_calc(lat, h, ca, cb, err)
      if (err) error stop 'chrom_calc failed'
      call row('chrom', [ca, cb, h])
    endif
    if (option == 'radiation') then
      call radiation_integrals(lat, orb, modes)
      call row('i0123', modes%synch_int)
      call row('i456', [modes%a%synch_int(4:5), modes%b%synch_int(4:6)])
      call row('emit', [modes%a%emittance, modes%b%emittance])
      call row('damping', [modes%a%j_damp, modes%b%j_damp, modes%z%j_damp, &
                          modes%a%alpha_damp, modes%b%alpha_damp, modes%z%alpha_damp])
      call row('energy', [modes%e_loss, modes%sigE_E])
    endif
    contains
    subroutine row(name, values)
    character(*) name
    real(rp) values(:)
    write(*, '(a,1x,a,100(1x,es25.17))') 'RESULT', name, values
    end subroutine
    end program
    """


    def tune(delta):
        mat = fodo_matrix(delta)
        return np.array(
            [12 * np.arccos(np.trace(mat[j : j + 2, j : j + 2]) / 2) / (2 * np.pi) for j in (0, 2)]
        )


    lat = lattice(
        "qf: quadrupole, l=0.2, k1=1, fringe_type=none\nqd: quadrupole, l=0.2, k1=-0.96, fringe_type=none\nd: drift, l=0.8",
        p0c=1e6,
        line=",".join(["qf,d,qd,d"] * 12),
        closed=True,
    )
    values = execute(PROGRAM, {"lat.bmad": lat}, args=["chrom"])["chrom"]
    h = values[2]
    secant = (tune(h) - tune(-h)) / (2 * h)
    require(values[:2], secant, atol=2e-10, rtol=3e-9)
    # Differentiate E^2=(pc)^2+(mc^2)^2 at p=p0: dE/E=beta0^2 d(delta).
    # The mass is Bmad's compiled electron mass (its CODATA value is a separate ledger entry).
    mass = 510998.95069
    beta2 = 1e12 / (1e12 + mass**2)
    expected = secant / beta2
    label = (source() / "bmad/code/chrom_calc.f90").read_text()
    if "+/- Delta energy used for the calculation" not in label:
        raise RuntimeError("Chromaticity argument description changed; inspect its definition")
    print(f"momentum secant control={secant.tolist()}; beta0^2={beta2:.17g}")
    verdict(
        compare(
            "API-described dQ/(dE/E), expected momentum derivative/beta0^2",
            values[:2],
            expected,
            atol=2e-10,
            rtol=3e-9,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found (source/contract discrepancy) | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed (source/contract discrepancy) | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`) |
| Confirmed (finite-beta energy label versus independently integrated tune derivative) | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

