---
title: "COL-49: Polarized long-range wake modes project onto the wrong geometric axis"
hide:
  - toc
---

# COL-49 — Polarized long-range wake modes project onto the wrong geometric axis

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [track1_lr_wake polarization axis](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L185-L187)<br>[self-wake projection](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L227-L229)<br>[source projection](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L263-L276)<br>[“Long-Range Wakes” syntax, s:lr.wake.syntax](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L3243-L3249)<br>[“Long-Range Wakes”, s:lr.wake.eq](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/wakefields.tex#L220-L244) |

Polarized long-range wake modes use a reflected dipole axis and the wrong angular dependence
for higher multipoles. A source at transverse coordinates x = y = 1 mm on a 45° dipole axis
produces essentially zero witness kick instead of 2.998 × 10⁻¹¹ in each transverse momentum
normalized to the reference momentum. An x-axis source on a 45° quadrupole node excites two
nonzero stored amplitudes, although the geometric projection requires zero. Modes polarized
along the x-axis give the expected dipole and quadrupole responses.

Run `python COL-49.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-49.py"

    [Download COL-49.py](COL-49.py)

    ```python
    """COL-49: Polarized long-range wake modes project onto the wrong geometric axis

    Polarized long-range wake modes use a reflected dipole axis and the wrong angular dependence
    for higher multipoles. A source at transverse coordinates x = y = 1 mm on a 45° dipole axis
    produces essentially zero witness kick instead of 2.998 × 10⁻¹¹ in each transverse momentum
    normalized to the reference momentum. An x-axis source on a 45° quadrupole node excites two
    nonzero stored amplitudes, although the geometric projection requires zero. Modes polarized
    along the x-axis give the expected dipole and quadrupole responses.

    The script excites 45° dipole and quadrupole modes and compares their stored amplitudes with
    the geometric multipole projection. It also tracks a delayed dipole witness and checks zero-
    angle controls. The amplitude basis is derived explicitly from the normal and skew multipole
    coefficients.
    """

    import math

    import numpy as np
    from _bmad import compare, fortran, verdict

    PROGRAM = """program probe
    use bmad
    use wake_mod
    implicit none
    type(ele_struct) ele
    type(bunch_struct) bunch
    integer m, ia
    real(rp) angle
    call init_ele(ele,lcavity$)
    ele%value(p0c$)=1e9_rp
    bmad_com%absolute_time_tracking=.true.
    do m=1,2
      do ia=0,1
        call init_wake(ele%wake,0,0,0,1)
        ele%wake%lr=wake_lr_struct(mode=ele%wake%lr%mode)
        ele%wake%lr%mode(1)=wake_lr_mode_struct()
        ele%wake%lr%self_wake_on=.false.
        ele%wake%lr%mode(1)%freq=1e9_rp
        ele%wake%lr%mode(1)%freq_in=1e9_rp
        ele%wake%lr%mode(1)%r_over_q=100
        ele%wake%lr%mode(1)%m=m
        ele%wake%lr%mode(1)%polarized=.true.
        angle=0.125_rp*ia
        ele%wake%lr%mode(1)%angle=angle
        call reallocate_bunch(bunch,1)
        call init_coord(bunch%particle(1),[0.001_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp],ele,upstream_end$,electron$)
        if(m==1) bunch%particle(1)%vec(3)=0.001_rp
        bunch%particle(1)%charge=1e-9_rp
        bunch%particle(1)%t=0
        call order_particles_in_z(bunch)
        call track1_lr_wake(bunch,ele)
        write(*,'(a,2i3,2es25.16)') 'AMPLITUDE ',m,ia,ele%wake%lr%mode(1)%b_sin,ele%wake%lr%mode(1)%a_sin
        bunch%particle(1)%charge=0
        bunch%particle(1)%t=0.25e-9_rp
        call track1_lr_wake(bunch,ele)
        write(*,'(a,2i3,2es25.16)') 'KICK ',m,ia,bunch%particle(1)%vec(2),bunch%particle(1)%vec(4)
      enddo
    enddo
    end program
    """
    log = fortran(PROGRAM).log
    rows = {
        (p[0], int(p[1]), int(p[2])): list(map(float, p[3:]))
        for line in log.splitlines()
        if (p := line.split()) and p[0] in ("AMPLITUDE", "KICK")
    }
    if len(rows) != 8:
        raise RuntimeError(log)
    defect = False
    for m in (1, 2):
        for ia in (0, 1):
            theta = ia * math.pi / 4
            # The coefficient basis is (-Re(z^m), Im(z^m)); a geometric axis
            # therefore has direction (cos(m theta), -sin(m theta)).
            axis = np.array([math.cos(m * theta), -math.sin(m * theta)])
            z = complex(1e-3, 1e-3 if m == 1 else 0)
            moment = np.array([-(z**m).real, (z**m).imag])
            amplitude = -1e-9 * 299792458 * 100 * axis * (axis @ moment)
            bad = compare(
                f"m={m} angle={ia * 45} deg amplitudes; geometric multipole projection",
                rows["AMPLITUDE", m, ia],
                amplitude,
                atol=1e-18,
                rtol=2e-14,
            )
            if ia == 0 and bad:
                raise RuntimeError("Zero-angle polarization control failed")
            defect |= bad
            if m == 1:
                # At a quarter period, delta px=b_sin/p0c and delta py=-a_sin/p0c.
                expected = amplitude * np.array([1, -1]) / 1e9
                bad = compare(
                    f"dipole angle={ia * 45} deg witness normalized momenta",
                    rows["KICK", m, ia],
                    expected,
                    atol=1e-25,
                    rtol=2e-14,
                )
                if ia == 0 and bad:
                    raise RuntimeError("Zero-angle dipole kick control failed")
                defect |= bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed<br><small>geometric polarization and delayed physical kicks for orders 1 and 2</small> | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

