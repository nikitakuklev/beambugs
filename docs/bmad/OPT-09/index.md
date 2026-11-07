---
title: "OPT-09: Linear-map construction retains off-momentum beta"
hide:
  - toc
---

# OPT-09 — Linear-map construction retains off-momentum beta

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [make_mat6, bmad/code/make_mat6.f90:105–117](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/make_mat6.f90#L105-L117)<br>[track_a_drift, bmad/low_level/track_a_drift.f90:94–109](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_drift.f90#L94-L109) |

Linear drift tracking constructs its reference map with zero momentum offset but retains the incident particle’s off-momentum speed. The longitudinal coefficient must use the reference speed and equal Lm²/(p₀² + m²), where L is drift length, m is particle mass and p₀ is reference momentum in units with c = 1. For a 1.7 m electron drift at p₀c = 1 MeV with fractional momentum offset 0.2 and initial longitudinal coordinate 0.03 m, the exit coordinate is 0.1076 m instead of 0.1004 m. The transverse first-order map remains correct.

Run `python OPT-09.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_14_optics_matrix_radiation.py](../helpers/_s3_14_optics_matrix_radiation.py) in the same directory.

??? example "Reproducer: OPT-09.py"

    [Download OPT-09.py](OPT-09.py)

    ```python
    """OPT-09: Linear-map construction retains off-momentum beta

    Linear drift tracking constructs its reference map with zero momentum offset but retains the incident particle’s off-momentum speed. The longitudinal coefficient must use the reference speed and equal Lm²/(p₀² + m²), where L is drift length, m is particle mass and p₀ is reference momentum in units with c = 1. For a 1.7 m electron drift at p₀c = 1 MeV with fractional momentum offset 0.2 and initial longitudinal coordinate 0.03 m, the exit coordinate is 0.1076 m instead of 0.1004 m. The transverse first-order map remains correct.

    The script constructs and applies the linear map for a 1.7 m drift at reference and off-reference momentum. It checks the transverse map and reference-momentum control, then compares the longitudinal coefficient with the derivative of the exact drift at the reference speed.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _s3_14_optics_matrix_radiation import control, rows

    PROGRAM = """
    program drift
    use bmad
    implicit none
    type(lat_struct) lat
    type(coord_struct) start, ending
    real(rp) v(6)
    logical err
    integer i
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) error stop 'parse failed'
    do i=0,1
     v=[0.001_rp,0.12_rp,-0.002_rp,-0.06_rp,0.03_rp,0.2_rp*i]
     call init_coord(start,v,lat%ele(1),upstream_end$)
     call make_mat6(lat%ele(1),lat%param,start,ending,err)
     if (err .or. ending%state /= alive$) error stop 'matrix or tracking failed'
     if (i==0) then
       call row('control',ending%vec)
       call row('control_r56',[lat%ele(1)%mat6(5,6)])
     else
       call row('off_momentum',ending%vec)
       call row('off_momentum_r56',[lat%ele(1)%mat6(5,6)])
     endif
    enddo
    call row('mass',[mass_of(electron$)])
    contains
    subroutine row(name,values)
    character(*) name
    real(rp) values(:)
    write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
    end subroutine
    end program
    """
    LATTICE = "no_digested\nparameter[particle]=electron\nparameter[p0c]=1e6\nparameter[geometry]=open\nbeginning[beta_a]=3\nbeginning[beta_b]=4\ne: drift, l=1.7, tracking_method=linear\nline1: line=(e)\nuse, line1\n"

    r = rows(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
    mass = r["mass"][0]
    length, pc = 1.7, 1e6
    # Derivative at delta=0: R56=L/gamma0^2, with p0c and mc^2 in eV.
    r56 = length * mass**2 / (pc**2 + mass**2)
    initial = np.array([0.001, 0.12, -0.002, -0.06, 0.03, 0.2])
    expected = initial.copy()
    expected[[0, 2]] += length * initial[[1, 3]]
    expected[4] += r56 * initial[5]
    control_expected = expected.copy()
    control_expected[4], control_expected[5] = initial[4], 0
    control("reference momentum output", r["control"], control_expected, atol=2e-16, rtol=3e-15)
    control("reference momentum R56 (m), L/gamma0^2", r["control_r56"], [r56], atol=2e-16, rtol=3e-15)
    control(
        "off-momentum transverse and delta components",
        r["off_momentum"][[0, 1, 2, 3, 5]],
        expected[[0, 1, 2, 3, 5]],
        atol=2e-16,
        rtol=3e-15,
    )
    defect = compare(
        "off-momentum R56 (m), reference-speed derivative",
        r["off_momentum_r56"],
        [r56],
        atol=2e-16,
        rtol=3e-15,
    )
    defect |= compare(
        "exit z (m), first-order drift", r["off_momentum"][4], expected[4], atol=2e-16, rtol=3e-15
    )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

