---
title: "COL-43: ibs_linac converts normalized emittance with gamma instead of beta gamma"
hide:
  - toc
---

# COL-43 — ibs_linac converts normalized emittance with gamma instead of beta gamma

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [ibs_linac::set_ele](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_linac/ibs_linac.f90#L221-L232)<br>[calc_emittances_and_twiss_from_sigma_matrix](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L1491-L1532) |

When ibs_linac derives geometric emittance from a bunch centered on the reference momentum, it divides normalized emittance by the Lorentz factor γ instead of βγ, where β is speed divided by the speed of light. The emittances passed to the intrabeam-scattering calculation are therefore too small by the factor β. At γ = 10.74, the underestimate is 0.4345%; it increases at lower energies. The bunch-moment routine defines normalized emittance using βγ, which fixes the required inverse conversion.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)

??? example "Reproducer: COL-43.py"

    [Download COL-43.py](COL-43.py)

    ```python
    """COL-43: ibs_linac converts normalized emittance with gamma instead of beta gamma

    When ibs_linac derives geometric emittance from a bunch centered on the reference momentum, it divides normalized emittance by the Lorentz factor γ instead of βγ, where β is speed divided by the speed of light. The emittances passed to the intrabeam-scattering calculation are therefore too small by the factor β. At γ = 10.74, the underestimate is 0.4345%; it increases at lower energies. The bunch-moment routine defines normalized emittance using βγ, which fixes the required inverse conversion.

    The script reads BMAD_SOURCE and compiles the actual internal set_ele routine without changing
    its body. Bmad’s bunch-moment routine supplies geometric and normalized emittances for a
    diagonal covariance matrix, providing the inverse-conversion control. Three energies expose the
    β factor and its ultrarelativistic limit.
    """

    import math
    import re

    from _bmad import compare, fortran, source, verdict

    text = (source() / "bsim/ibs_linac/ibs_linac.f90").read_text()
    match = re.search(
        r"^subroutine set_ele\(.*?^end subroutine\s*$", text, re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    if match is None:
        raise RuntimeError("Cannot locate the ibs_linac set_ele routine")
    program = (
        r"""program probe
    use bmad
    use beam_utils
    implicit none
    type(lat_struct),target :: lat
    type(bunch_params_struct) params
    type(ele_struct) ele
    real(rp) sigma(6,6),gam,beta,expected(2),before(2)
    integer i,j
    logical err
    call bmad_parser('lat.bmad',lat)
    do j=1,3
      if(j==1) gam=2
      if(j==2) gam=10.7383864_rp
      if(j==3) gam=1e6_rp
      beta=sqrt(1-1/gam**2)
      ele%value(e_tot$)=gam*mass_of(lat%param%particle)
      params%centroid%species=lat%param%particle
      params%centroid%p0c=beta*gam*mass_of(lat%param%particle)
      params%centroid%vec=0
      sigma=0
      sigma(1,1)=2e-6_rp; sigma(2,2)=2e-6_rp
      sigma(3,3)=1e-6_rp; sigma(4,4)=1e-6_rp
      sigma(5,5)=1e-6_rp; sigma(6,6)=1e-8_rp
      params%sigma=sigma
      call calc_emittances_and_twiss_from_sigma_matrix(sigma,params,err)
      if(err) error stop 'bunch eigensystem failed'
      expected=[params%a%emit,params%b%emit]
      before=[params%a%norm_emit,params%b%norm_emit]
      call set_ele(ele,params,1000.0_rp)
      write(*,'(a,7es25.16)') 'RESULT ',gam,ele%a%emit,ele%b%emit,expected,before
    enddo
    contains
    """
        + match.group(0)
        + """
    end program
    """
    )
    lattice = """parameter[particle]=#12C+6
    parameter[e_tot]=1e12
    parameter[geometry]=open
    d: drift,l=1
    l: line=(d)
    use,l
    """
    log = fortran(program, files={"lat.bmad": lattice}).log
    rows = [
        list(map(float, line.split()[1:])) for line in log.splitlines() if line.startswith("RESULT ")
    ]
    if len(rows) != 3:
        raise RuntimeError(log)
    defect = False
    for gamma, a, b, ea, eb, na, nb in rows:
        bg = math.sqrt(gamma**2 - 1)
        if compare(
            f"gamma={gamma:g} normalized-emittance control; epsilon_n=beta gamma epsilon",
            [na, nb],
            [bg * ea, bg * eb],
            atol=0,
            rtol=2e-14,
        ):
            raise RuntimeError("Normalized-emittance contract control failed")
        defect |= compare(
            f"gamma={gamma:g} set_ele geometric emittances, m", [a, b], [ea, eb], atol=0, rtol=2e-14
        )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>source check</small> | `gpt-6.1-sol` | xhigh | 2026-10-22 | ibs-ion-classical-radius (`1bf1982abf8e`) |
| Confirmed<br><small>source check</small> | `claude-opus-5-5` | high | 2026-10-22 | 20261021-0 (`d00a8f17e55a`) |
| Confirmed<br><small>on-reference-momentum emittance conversion</small> | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

