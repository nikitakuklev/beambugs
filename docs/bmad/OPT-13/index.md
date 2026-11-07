---
title: "OPT-13: Omitting the transfer endpoint returns identity instead of one turn"
hide:
  - toc
---

# OPT-13 — Omitting the transfer endpoint returns identity instead of one turn

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [transfer_matrix_calc, bmad/code/transfer_matrix_calc.f90:84–88](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/transfer_matrix_calc.f90#L84-L88)<br>[endpoint defaults, bmad/code/transfer_matrix_calc.f90:21–36](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/transfer_matrix_calc.f90#L21-L36) |

Requesting a closed-lattice transfer matrix with a start element and no end element returns identity. The documented omitted-endpoint contract promises a full turn back to the start element. In a 2 m FODO cell, the horizontal angle-to-position coefficient is zero instead of 2.150 m, as an independent thick-lens map establishes. Explicitly requesting one turn avoids the defect.

Run `python OPT-13.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_14_optics_matrix_radiation.py](../helpers/_s3_14_optics_matrix_radiation.py) in the same directory.

??? example "Reproducer: OPT-13.py"

    [Download OPT-13.py](OPT-13.py)

    ```python
    """OPT-13: Omitting the transfer endpoint returns identity instead of one turn

    Requesting a closed-lattice transfer matrix with a start element and no end element returns identity. The documented omitted-endpoint contract promises a full turn back to the start element. In a 2 m FODO cell, the horizontal angle-to-position coefficient is zero instead of 2.150 m, as an independent thick-lens map establishes. Explicitly requesting one turn avoids the defect.

    The script compares omitted and explicit one-turn endpoints in a FODO cell. It constructs the expected map by multiplying independent thick-lens quadrupole and drift matrices, then conjugates the cell map to the exit of the first quadrupole.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _s3_14_optics_matrix_radiation import control, quad, rows

    PROGRAM = """
    program transfer_oracle
    use bmad
    implicit none
    type(lat_struct) lat
    real(rp) m(6,6), explicit(6,6)
    logical err
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if (err) error stop 'parse failed'
    call transfer_matrix_calc(lat,m,ix1=1)
    call transfer_matrix_calc(lat,explicit,ix1=1,ix2=1,one_turn=.true.)
    call row('implicit',reshape(m,[36]))
    call row('explicit',reshape(explicit,[36]))
    contains
    subroutine row(name,values)
    character(*) name
    real(rp) values(:)
    write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
    end subroutine
    end program
    """
    LATTICE = "no_digested\nparameter[particle]=electron\nparameter[p0c]=1e9\nparameter[geometry]=closed\nqf: quadrupole, l=0.2, k1=1, fringe_type=none\nqd: quadrupole, l=0.2, k1=-0.96, fringe_type=none\nd: drift, l=0.8\nline1: line=(qf,d,qd,d)\nuse, line1\n"

    r = rows(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
    drift = np.array([[1, 0.8], [0, 1]])
    expected = np.eye(6)
    for block, sign in ((slice(0, 2), 1), (slice(2, 4), -1)):
        q = quad(sign, 0.2)
        cell = drift @ quad(-0.96 * sign, 0.2) @ drift @ q
        # Move the start of the cell from before the first quad to after it.
        expected[block, block] = q @ cell @ np.linalg.inv(q)
    expected[4, 5] = 2 * 510998.95069**2 / (1e18 + 510998.95069**2)
    explicit = r["explicit"].reshape((6, 6), order="F")
    implicit = r["implicit"].reshape((6, 6), order="F")
    control(
        "explicit one turn, independent FODO conjugation", explicit, expected, rtol=2e-12, atol=2e-14
    )
    verdict(
        compare("omitted endpoint, documented one turn", implicit, expected, rtol=2e-12, atol=2e-14)
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

