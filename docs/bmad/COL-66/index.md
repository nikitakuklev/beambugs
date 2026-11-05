---
title: "COL-66: Wake kicks omit the witness charge state and source species sign"
hide:
  - toc
---

# COL-66 — Wake kicks omit the witness charge state and source species sign

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [long-range deposits and kicks](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L197-L259)<br>[short-range longitudinal modal path](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L324-L381)<br>[short-range transverse modal path](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L420-L458)<br>[z_long table path](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L538-L608)<br>[“Short-Range Wakes”, delvp1–delvp3 and pelqxw1–pelqxw2](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/wakefields.tex#L28-L70)<br>[“Long-Range Wakes” strength and charge definitions](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/wakefields.tex#L112-L147) |

Short-range modal, tabulated and long-range wakes omit the witness charge state and the source species’ sign. At fixed source-charge magnitude in coulombs and reference momentum, the ultrarelativistic wake kick must scale with Qw sign(Qs), where Qw and Qs are witness and source charges in elementary-charge units. Carbon ions with charge 6 receive kicks six times too small, while positrons following electrons receive kicks with the wrong sign. Source self-kicks also miss the factor six; a short-range longitudinal example gives a fractional momentum change −2.219 × 10⁻⁷ for a carbon witness instead of −1.331 × 10⁻⁶.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)

??? example "Reproducer: COL-66.py"

    [Download COL-66.py](COL-66.py)

    ```python
    """COL-66: Wake kicks omit the witness charge state and source species sign

    Short-range modal, tabulated and long-range wakes omit the witness charge state and the source species’ sign. At fixed source-charge magnitude in coulombs and reference momentum, the ultrarelativistic wake kick must scale with Qw sign(Qs), where Qw and Qs are witness and source charges in elementary-charge units. Carbon ions with charge 6 receive kicks six times too small, while positrons following electrons receive kicks with the wrong sign. Source self-kicks also miss the factor six; a short-range longitudinal example gives a fractional momentum change −2.219 × 10⁻⁷ for a carbon witness instead of −1.331 × 10⁻⁶.

    The script embeds the wake definitions and tracks electron, positron and carbon witnesses
    through all three wake paths. It compares kicks and self-kicks with the analytic modal, table
    and resonator responses including charge factors. Electron tracking is the unit-charge control;
    mixed-species direct calls isolate witness scaling.
    """

    import math

    import numpy as np
    from _bmad import compare, fortran, verdict

    PROGRAM = """! Witness-charge dependence of every Bmad wake kick path.
    !
    ! Usage: ./probe <lattice> prod   <witness species> ...
    !        ./probe <lattice> direct <witness species> ...
    !        ./probe <lattice> direct_negq <witness species> ...
    !
    ! The source is one macroparticle of the lattice reference species with weight
    ! q = 1 nC, offset x = 1 mm.  Witnesses have weight 0, so they feel the source
    ! wake but add none of their own.
    !   prod:   one witness per run, tracked with track1_bunch through each element
    !           (the path used by track_beam, Tao, long_term_tracking and bbu).
    !   direct: all witnesses in one bunch, wake routines called directly with
    !           exactly placed coordinates.
    !   direct_negq: as direct, but the source weight is -1 nC.
    ! SR witnesses trail by dz = -2e-5 m; LR witnesses arrive dt = 0.125 ns later
    ! in a separate bunch.

    program probe
    use bmad
    use beam_mod
    use wake_mod
    implicit none

    type (lat_struct), target :: lat
    type (bunch_struct) :: b, src
    real(rp) :: q = 1e-9_rp
    real(rp), parameter :: dz = -2e-5_rp, xs = 1e-3_rp, dt = 0.125e-9_rp
    integer, parameter :: ix_srm = 1, ix_srz = 2, ix_lrw = 3
    integer :: i, j, nw, ref, wsp(10)
    logical :: err
    character(200) :: lat_file, mode, arg

    call get_command_argument(1, lat_file)
    call get_command_argument(2, mode)
    nw = command_argument_count() - 2
    do i = 1, nw
      call get_command_argument(i+2, arg)
      wsp(i) = species_id(trim(arg))
    enddo

    call bmad_parser(lat_file, lat)
    bmad_com%sr_wakes_on = .true.
    bmad_com%lr_wakes_on = .true.
    ref = lat%param%particle
    if (mode == 'direct_negq') q = -q

    print '(a,1x,a,1x,a,1x,i0,2(1x,es24.16))', 'REF', trim(mode), trim(species_name(ref)), charge_of(ref), lat%ele(ix_srm)%value(p0c$), q

    select case (mode)
    case ('prod')
      do j = 1, nw
        ! SR pseudo-modes and SR z_long table: source and witness in one bunch.
        do i = ix_srm, ix_srz
          call reallocate_bunch(b, 2)
          call init_coord(b%particle(1), [xs, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp], lat%ele(0), downstream_end$)
          call init_coord(b%particle(2), [0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, dz, 0.0_rp], lat%ele(0), downstream_end$, particle = wsp(j))
          b%particle(1)%charge = q
          b%particle(2)%charge = 0
          b%charge_tot = q
          call track1_bunch(b, lat%ele(i), err)
          if(err) error stop 'bunch tracking failed'
          call out(trim(lat%ele(i)%name), b%particle(1), b%particle(2))
        enddo

        ! LR: source bunch, then witness bunch dt later.
        call zero_lr_wakes_in_lat(lat)
        call reallocate_bunch(src, 1)
        call init_coord(src%particle(1), [xs, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp], lat%ele(0), downstream_end$)
        src%particle(1)%charge = q
        src%charge_tot = q
        call track1_bunch(src, lat%ele(ix_lrw), err)
          if(err) error stop 'bunch tracking failed'
        call reallocate_bunch(b, 1)
        call init_coord(b%particle(1), [0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp], lat%ele(0), downstream_end$, particle = wsp(j))
        b%particle(1)%t = b%particle(1)%t + dt
        b%particle(1)%vec(5) = b%particle(1)%vec(5) - b%particle(1)%beta * c_light * dt
        b%particle(1)%charge = 0
        call track1_bunch(b, lat%ele(ix_lrw), err)
          if(err) error stop 'bunch tracking failed'
        call out('LRW', src%particle(1), b%particle(1))
      enddo

    case ('direct', 'direct_negq')
      ! SR: source at head, all witnesses at the same z behind it.
      do i = ix_srm, ix_srz
        call reallocate_bunch(b, nw+1)
        call init_coord(b%particle(1), [xs, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp], lat%ele(0), downstream_end$)
        b%particle(1)%charge = q
        do j = 1, nw
          call init_coord(b%particle(j+1), [0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, dz, 0.0_rp], lat%ele(0), downstream_end$, particle = wsp(j))
          b%particle(j+1)%charge = 0
        enddo
        call order_particles_in_z(b)
        call track1_sr_wake(b, lat%ele(i))
        do j = 1, nw
          call out(trim(lat%ele(i)%name), b%particle(1), b%particle(j+1))
        enddo
      enddo

      ! LR: source bunch, then one witness bunch holding all witnesses.
      call zero_lr_wakes_in_lat(lat)
      call reallocate_bunch(src, 1)
      call init_coord(src%particle(1), [xs, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp], lat%ele(0), downstream_end$)
      src%particle(1)%charge = q
      src%particle(1)%t = 0
      call order_particles_in_z(src)
      call track1_lr_wake(src, lat%ele(ix_lrw))
      call reallocate_bunch(b, nw)
      do j = 1, nw
        call init_coord(b%particle(j), [0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp], lat%ele(0), downstream_end$, particle = wsp(j))
        b%particle(j)%t = dt
        b%particle(j)%charge = 0
      enddo
      call order_particles_in_z(b)
      call track1_lr_wake(b, lat%ele(ix_lrw))
      do j = 1, nw
        call out('LRW', src%particle(1), b%particle(j))
      enddo
    end select

    contains

    ! One line per (path, witness): witness species, its charge number, witness
    ! dpz and dpx, source self dpz, and the source-witness separation used.
    subroutine out(path, s, w)
    character(*) path
    type (coord_struct) s, w
    if(s%state/=alive$ .or. w%state/=alive$) error stop 'particle lost'
    print '(a,1x,a,1x,a,1x,i0,5(1x,es24.16))', trim(mode), path, trim(species_name(w%species)), charge_of(w%species), &
                         w%vec(6), w%vec(2), s%vec(6), w%vec(5) - s%vec(5), w%t - s%t
    end subroutine

    end program
    """
    LATTICE = """! Electron reference. c6.bmad is identical except for the species.
    no_digested
    parameter[particle] = electron
    parameter[p0c] = 1e12
    parameter[geometry] = open
    beginning[beta_a] = 10
    beginning[beta_b] = 10

    ! SR pseudo-modes: monopole longitudinal, dipole transverse (leading).
    srm: pipe, l = 1, sr_wake = {amp_scale = 1, z_scale = 1, scale_with_length = F, z_max = 1,
      longitudinal = {1e15, 1e3, 2e4, 0.1, none},
      transverse = {1e17, 1e3, 3e4, 0.15, none, leading}}

    ! SR z_long table, no smoothing; nodes every 1e-5 m.
    srz: pipe, l = 1, sr_wake = {amp_scale = 1, z_scale = 1, scale_with_length = F, z_max = 1,
      z_long = {smoothing_sigma = 0, w = {
        -5e-5 3e14, -4e-5 4e14, -3e-5 5e14, -2e-5 7e14, -1e-5 9e14, 0 1e15,}}}

    ! LR modes: m = 0 at 1 GHz, m = 1 at 1.5 GHz; no damping, zero phase.
    lrw: pipe, l = 1, lr_wake = {mode = {1e9, 100, 0, 0, 0, unpolarized},
      mode = {1.5e9, 2e4, 0, 0, 1, unpolarized}}

    ln: line = (srm, srz, lrw)
    use, ln
    """
    C = 2.99792458e8
    TWOPI = 2 * math.pi
    DZ, XS, DT = -2e-5, 1e-3, 0.125e-9
    # Lattice wake parameters (e.bmad / c6.bmad).
    AL, DL, KL, PL = 1e15, 1e3, 2e4, 0.1  # SR longitudinal mode
    AT, DT_, KT, PT = 1e17, 1e3, 3e4, 0.15  # SR transverse mode, leading
    WTAB = {-5e-5: 3e14, -4e-5: 4e14, -3e-5: 5e14, -2e-5: 7e14, -1e-5: 9e14, 0.0: 1e15}
    F0, R0 = 1e9, 100.0  # LR m = 0
    F1, R1 = 1.5e9, 2e4  # LR m = 1
    TOL = 1e-12


    def unit_witness(path, q, p0c):
        """(witness dpz, witness dpx, source self dpz) with witness charge +e."""
        aq = abs(q)
        if path == "SRM":
            w = AL * math.exp(DL * DZ) * math.sin(KL * DZ + TWOPI * PL)
            wt = AT * math.exp(DT_ * DZ) * math.sin(KT * DZ + TWOPI * PT)
            return -aq * w / p0c, -aq * wt * XS / p0c, -aq * AL * math.sin(TWOPI * PL) / 2 / p0c
        if path == "SRZ":  # implemented with the signed weight
            return -q * WTAB[DZ] / p0c, 0.0, -q * WTAB[0.0] / 2 / p0c
        if path == "LRW":
            w0, w1 = TWOPI * F0, TWOPI * F1
            return (
                -aq * R0 * w0 * math.cos(w0 * DT) / p0c,
                aq * R1 * C * XS * math.sin(w1 * DT) / p0c,
                -aq / (2 * p0c) * (R0 * w0 + R1 * w1 * XS**2),
            )
        raise ValueError(path)


    defect = False
    for source_species, mode, witnesses in [
        ("electron", "prod", ["electron", "positron"]),
        ("#12C+6", "prod", ["#12C+6"]),
        ("#12C+6", "direct", ["#12C+6", "#12C+5", "electron", "positron"]),
        ("electron", "direct", ["#12C+6"]),
    ]:
        lattice = LATTICE.replace(
            "parameter[particle] = electron", f"parameter[particle] = {source_species}"
        )
        log = fortran(PROGRAM, ["lat.bmad", mode, *witnesses], files={"lat.bmad": lattice}).log
        refs = [p for line in log.splitlines() if (p := line.split()) and p[0] == "REF"]
        rows = [p for line in log.splitlines() if (p := line.split()) and p[0] == mode]
        if len(refs) != 1 or len(rows) != 3 * len(witnesses):
            raise RuntimeError(log)
        qs, p0c, weight = int(refs[0][3]), float(refs[0][4]), float(refs[0][5])
        for row in rows:
            _, path, witness, qw, *values = row
            qw = int(qw)
            observed = np.array(list(map(float, values[:3])))
            unit = np.array(unit_witness(path, weight, p0c))
            # Coulombs of source weight already include |Qs|. The witness Lorentz force
            # adds Qw and the field sign adds sign(Qs); self-wakes carry |Qs|.
            expected = unit * np.array([qw * math.copysign(1, qs)] * 2 + [abs(qs)])
            label = f"{mode} {source_species} source / {witness} witness {path} [dpz,dpx,self dpz]"
            bad = compare(label, observed, expected, atol=1e-25, rtol=2e-13)
            if source_species == "electron" and witness == "Electron" and bad:
                raise RuntimeError("Electron wake normalization control failed")
            defect |= bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>wake charge-state scaling (source check)</small> | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`) |
| Confirmed | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed<br><small>longitudinal wake charge/sign scaling; transverse by source only</small> | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

