---
title: "COL-03: Off-axis slice LSC coefficients omit the kick normalization"
hide:
  - toc
---

# COL-03 — Off-axis slice LSC coefficients omit the kick normalization

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [lsc_kick_params_calc, bmad/space_charge/csr_and_space_charge_mod.f90:1288–1320](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1288-L1320)<br>[Slice Space Charge Calculation, s:sc.slice / kelsz, bmad/doc/multiparticle.tex:438–474](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/multiparticle.tex#L438-L474) |

With transverse dependence enabled for slice longitudinal space charge, changing the interaction length changes how the longitudinal kick depends on transverse position in a frozen bunch. The kick per metre at a 1 mm horizontal displacement is −5.174e−4, −5.757e−4 and −5.816e−4 m⁻¹ for steps of 0.001, 0.01 and 0.1 m. It should be independent of step length, as the on-axis control is. Four derivative primitives omit the normalization carried by the on-axis kick, breaking the scaling of the rational approximation.

Run `python COL-03.py` with [_bmad.py](../helpers/_bmad.py) and [_collective_fields.py](../helpers/_collective_fields.py) in the same directory.

??? example "Reproducer: COL-03.py"

    [Download COL-03.py](COL-03.py)

    ```python
    """COL-03: Off-axis slice LSC coefficients omit the kick normalization

    With transverse dependence enabled for slice longitudinal space charge, changing the interaction
    length changes how the longitudinal kick depends on transverse position in a frozen bunch. The
    kick per metre at a 1 mm horizontal displacement is −5.174e−4, −5.757e−4 and −5.816e−4 m⁻¹ for
    steps of 0.001, 0.01 and 0.1 m. It should be independent of step length, as the on-axis control
    is. Four derivative primitives omit the normalization carried by the on-axis kick, breaking the
    scaling of the rational approximation.

    The script holds three electron slices fixed and evaluates the on-axis and off-axis kicks at three
    interaction lengths. It requires the on-axis scaling control to pass and tests the same linearity
    off axis; a separate quadrature evaluates the manual’s slice integral.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _collective_fields import rows
    from scipy.integrate import quad

    PROGRAM = r"""
    program lsc_probe
    use csr_and_space_charge_mod
    use da2_mod
    implicit none
    type(csr_struct) c
    type(ele_struct) e
    real(rp) amp,k0,koff,cv
    integer i,j
    character(16) label
    e%space_charge_method=slice$
    c%gamma=10; c%gamma2=100; c%rel_mass=1; c%species=electron$
    c%kick_factor=1; c%actual_track_step=0.01_rp; c%dz_slice=.0001_rp
    space_charge_com%n_bin=3
    space_charge_com%lsc_kick_transverse_dependence=.true.
    allocate(c%slice(3))
    do j=0,2
     amp=10.0_rp**(j-1)
     c%actual_track_step=amp*0.01_rp
     c%slice=csr_bunch_slice_struct()
     do i=1,3
      c%slice(i)%z_center=(i-2)*c%dz_slice
      c%slice(i)%sig_x=.001_rp
      c%slice(i)%sig_y=.002_rp
      c%slice(i)%charge=1e-12_rp
     enddo
     call lsc_kick_params_calc(e,c)
     k0=0; koff=0
     do i=1,2
      if(i==1)then
       cv=c%slice(1)%coef_lsc_plus(0,0)
       if(cv==0) cycle
       k0=k0+1/da2_evaluate(c%slice(1)%coef_lsc_plus,0.0_rp,0.0_rp)
       koff=koff+1/da2_evaluate(c%slice(1)%coef_lsc_plus,1e-6_rp,0.0_rp)
      else
       cv=c%slice(1)%coef_lsc_minus(0,0)
       if(cv==0) cycle
       k0=k0+1/da2_evaluate(c%slice(1)%coef_lsc_minus,0.0_rp,0.0_rp)
       koff=koff+1/da2_evaluate(c%slice(1)%coef_lsc_minus,1e-6_rp,0.0_rp)
      endif
     enddo
     write(label,'(a,i0)') 'step',j
     print '(3a,4es24.15)', 'RESULT ',trim(label),' ',c%actual_track_step,c%slice(1)%kick_lsc,k0,koff
    enddo
    end program
    """


    data = rows(fortran(PROGRAM).log)
    table = np.array([data[f"step{i}"] for i in range(3)])
    steps, scalar, on_axis, off_axis = table.T
    if not np.allclose(steps, [0.001, 0.01, 0.1], rtol=2e-15, atol=0):
        raise RuntimeError("Step setup failed")
    if compare(
        "on-axis normalized kick per metre",
        on_axis / steps,
        np.full(3, on_axis[0] / steps[0]),
        atol=1e-17,
        rtol=2e-13,
    ):
        raise RuntimeError("On-axis scaling control failed")
    if not np.allclose(scalar, on_axis, rtol=2e-13, atol=1e-20):
        raise RuntimeError("On-axis scalar and rational evaluations disagree")
    # Manual equation kelsz integrated over the two other slices; self-field is odd.
    # This is an exact reference for the model integral, not for its Padé truncation.
    sx, sy, gamma, x = 0.001, 0.002, 10, 0.001
    a = sx * sy * np.exp(x * x / (2 * sx * sx))
    b = gamma * (sx * sx + sy * sy) / (sx + sy)
    factor_per_m = 2.8179403262e-15 / (1.602176634e-19 * gamma)
    integral = -sum(
        quad(
            lambda z: factor_per_m * 1e-8 / (a + b * z + gamma**2 * z * z),
            z - 0.00005,
            z + 0.00005,
            epsabs=1e-17,
        )[0]
        for z in [0.0001, 0.0002]
    )
    print(f"Independent slice integral: normalized kick per metre={integral:.16g}")
    verdict(
        compare(
            "off-axis normalized kick per metre, fixed-distribution linearity",
            off_axis / steps,
            np.full(3, off_axis[0] / steps[0]),
            atol=1e-17,
            rtol=2e-13,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

