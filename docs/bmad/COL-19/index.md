---
title: "COL-19: The CSR manual describes a second staggered deposition absent from the calculation"
hide:
  - toc
---

# COL-19 — The CSR manual describes a second staggered deposition absent from the calculation

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [csr_bin_particles deposition and derivatives, bmad/space_charge/csr_and_space_charge_mod.f90:513–578](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L513-L578)<br>[csr_bin_kicks derivative input, bmad/space_charge/csr_and_space_charge_mod.f90:776–788](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L776-L788)<br>[Bmad manual, 1_Dim CSR Calculation, s:csr.1d, bmad/doc/multiparticle.tex:398–411](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/multiparticle.tex#L398-L411) |

When describing one-dimensional CSR binning, the manual says that particle density is deposited again into bins shifted by half a bin width. The implementation deposits triangular particle charge once and obtains the edge density derivative from adjacent bin-charge differences. A three-particle calculation agrees with that single deposition and finite-difference rule. The manual should describe the single charge array and the derivative used for the CSR kick.

Run `python COL-19.py` with [_bmad.py](../helpers/_bmad.py) and [_collective_contracts.py](../helpers/_collective_contracts.py) in the same directory.

??? example "Reproducer: COL-19.py"

    [Download COL-19.py](COL-19.py)

    ```python
    """COL-19: The CSR manual describes a second staggered deposition absent from the calculation

    When describing one-dimensional CSR binning, the manual says that particle density is deposited again into bins shifted by half a bin width. The implementation deposits triangular particle charge once and obtains the edge density derivative from adjacent bin-charge differences. A three-particle calculation agrees with that single deposition and finite-difference rule. The manual should describe the single charge array and the derivative used for the CSR kick.

    The script calls the production CSR binning routine for three fixed charged particles. It checks
    the bin charges against analytic integrals of triangular particle shapes and the edge
    derivatives against adjacent charge differences. It reads the disputed algorithm description
    through BMAD_SOURCE and uses BMAD_DIST for the library.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _collective_contracts import rows, section, source_text

    result = rows(
        fortran("""program bins
    use csr_and_space_charge_mod
    implicit none
    type(csr_struct) c
    type(ele_struct) e
    type(coord_struct) p(3)
    logical err
    integer i
    e%csr_method=one_dim$
    space_charge_com%n_bin=12
    space_charge_com%particle_bin_span=2
    p%state=alive$
    p%charge=1e-12_rp
    p(1)%vec(5)=-0.001_rp
    p(2)%vec(5)=0.00013_rp
    p(3)%vec(5)=0.001_rp
    call csr_bin_particles(e,p,c,err)
    if (err) error stop 'Binning failed'
    write(*,'(a,es25.16)') 'RESULT dz ',c%dz_slice
    write(*,'(a,12es25.16)') 'RESULT centers ',c%slice%z_center
    write(*,'(a,12es25.16)') 'RESULT charge ',c%slice%charge
    write(*,'(a,12es25.16)') 'RESULT derivative ',c%slice%edge_dcharge_density_dz
    end program
    """).log
    )
    q, dz = result["charge"], result["dz"][0]
    # For a triangular macroparticle with half-width h, the normalized cumulative
    # charge is (x+h)^2/(2h^2) on [-h,0], and 1-(h-x)^2/(2h^2) on [0,h].
    h = dz
    z = np.array([-0.001, 0.00013, 0.001])


    def cumulative(x):
        x = np.clip(x, -h, h)
        return np.where(x < 0, (x + h) ** 2 / (2 * h * h), 1 - (h - x) ** 2 / (2 * h * h))


    def deposit(centers):
        a = centers[:, None] - dz / 2 - z
        b = centers[:, None] + dz / 2 - z
        return 1e-12 * np.sum(cumulative(b) - cumulative(a), axis=1)


    expected_q = deposit(result["centers"])
    if compare("single triangular deposition Q (C)", q, expected_q, atol=3e-27):
        raise RuntimeError("Triangular deposition control failed")
    expected_d = np.r_[0, np.diff(q) / dz**2]
    if compare(
        "edge derivative (C/m^2), adjacent Q differences divided by dz^2",
        result["derivative"],
        expected_d,
        atol=2e-20,
    ):
        raise RuntimeError("Finite-difference implementation changed")
    manual = section(
        source_text("bmad/doc/multiparticle.tex"),
        r"\subsection{1\_Dim CSR Calculation}",
        r"\subsection{Slice Space Charge Calculation}",
    )
    claimed = "second set of staggered bins" in manual
    print(
        f"manual promises a second staggered deposition={claimed}; execution matches single-deposition finite differences"
    )
    verdict(claimed)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`) |
| Confirmed<br><small>source check</small> | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`) |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

