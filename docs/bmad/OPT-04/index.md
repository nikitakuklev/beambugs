---
title: "OPT-04: Chromaticity optional orbit outputs select the wrong branch"
hide:
  - toc
---

# OPT-04 — Chromaticity optional orbit outputs select the wrong branch

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [chrom_calc, bmad/code/chrom_calc.f90:208–225](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/chrom_calc.f90#L208-L225)<br>[open-line selection, bmad/code/chrom_calc.f90:232–241](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/chrom_calc.f90#L232-L241)<br>[argument contract, bmad/code/chrom_calc.f90:9–29](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/chrom_calc.f90#L9-L29) |

Requesting only the high-momentum orbit from the chromaticity routine returns success with the wrong chromaticity and an untracked orbit endpoint. Requesting only the low-momentum orbit crashes. Each optional orbit output should work independently and leave the chromaticity unchanged. In a four-cell FODO ring, the high-only call gives −0.07250 instead of −0.1450, while supplying both outputs or neither agrees with independent thick-lens differentiation.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _s3_14_optics_matrix_radiation.py](../helpers/_s3_14_optics_matrix_radiation.py) · [View source](../helpers/index.md#_s3_14_optics_matrix_radiation)

??? example "Reproducer: OPT-04.py"

    [Download OPT-04.py](OPT-04.py)

    ```python
    """OPT-04: Chromaticity optional orbit outputs select the wrong branch

    Requesting only the high-momentum orbit from the chromaticity routine returns success with the wrong chromaticity and an untracked orbit endpoint. Requesting only the low-momentum orbit crashes. Each optional orbit output should work independently and leave the chromaticity unchanged. In a four-cell FODO ring, the high-only call gives −0.07250 instead of −0.1450, while supplying both outputs or neither agrees with independent thick-lens differentiation.

    The script runs the four combinations of optional orbit outputs in separate processes. A thick-lens FODO calculation supplies the tune derivative, and the both-output and neither-output cases are controls. It recognizes only a segmentation fault after successful lattice setup as the low-only crash; BMAD_DIST supplies the build.
    """

    import resource
    import signal

    import numpy as np
    from _bmad import RunFailure, compare, fortran, verdict
    from _s3_14_optics_matrix_radiation import control, quad, rows

    PROGRAM = """
    program chrom
    use bmad
    implicit none
    type(lat_struct) lat
    type(coord_struct),allocatable :: orb(:),low(:),high(:)
    real(rp) h,ca,cb
    integer stat
    logical err
    character(12) mode
    call get_command_argument(1,mode)
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) error stop 'parse failed'
    call twiss_and_track(lat,orb,stat)
    if (stat /= ok$) error stop 'twiss failed'
    print *, 'PREREQUISITES PASSED'
    flush(6)
    h=1e-4_rp
    select case(mode)
    case('none')
     call chrom_calc(lat,h,ca,cb,err)
    case('both')
     call chrom_calc(lat,h,ca,cb,err,low_E_orb=low,high_E_orb=high)
    case('low')
     call chrom_calc(lat,h,ca,cb,err,low_E_orb=low)
    case('high')
     call chrom_calc(lat,h,ca,cb,err,high_E_orb=high)
    case default
     error stop 'bad mode'
    end select
    if (err) error stop 'chromaticity failed'
    call row('chrom', [ca,cb])
    if (allocated(high)) call row('high', [high(lat%n_ele_track)%vec(6), &
     real(merge(1,0,high(lat%n_ele_track)%state==alive$),rp)])
    contains
    subroutine row(name,values)
    character(*) name
    real(rp) values(:)
    write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
    end subroutine
    end program
    """
    LATTICE = "no_digested\nparameter[geometry] = closed\nparameter[e_tot] = 3e9\nparameter[no_end_marker] = T\nd: drift, l = 1\nqf: quadrupole, l = 0.2, k1 = 1\nqd: quadrupole, l = 0.2, k1 = -1\ncell: line = (qf, d, qd, d)\nring: line = (4*cell)\nuse, ring\n"

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


    def tune(delta):
        drift = np.array([[1.0, 1.0], [0.0, 1.0]])
        cell = drift @ quad(-1 / (1 + delta), 0.2) @ drift @ quad(1 / (1 + delta), 0.2)
        return 4 * np.arccos(np.trace(cell) / 2) / (2 * np.pi)


    expected = (tune(1e-4) - tune(-1e-4)) / 2e-4
    defect = False
    for mode in ("none", "both", "high", "low"):
        try:
            result = fortran(PROGRAM, [mode], files={"lat.bmad": LATTICE})
        except RunFailure as exc:
            if (
                mode != "low"
                or exc.status != -signal.SIGSEGV
                or "PREREQUISITES PASSED" not in exc.log
                or "Program received signal SIGSEGV" not in exc.log
            ):
                raise
            print(
                "low only: observed=SIGSEGV in chrom_calc expected=successful chromaticity and low orbit"
            )
            defect = True
            continue
        r = rows(result.log)
        if mode in ("none", "both"):
            control(
                f"{mode} outputs, chromaticity from FODO differentiation",
                r["chrom"],
                [expected, expected],
                atol=5e-11,
            )
            if mode == "both":
                control("both outputs, high endpoint [delta, alive]", r["high"], [1e-4, 1], atol=2e-14)
        else:
            defect |= compare(
                f"{mode} only chromaticity, FODO differentiation",
                r["chrom"],
                [expected, expected],
                atol=5e-11,
            )
            if mode == "high":
                defect |= compare("high endpoint [delta, alive]", r["high"], [1e-4, 1], atol=2e-14)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

