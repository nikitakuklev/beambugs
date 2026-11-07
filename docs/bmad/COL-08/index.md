---
title: "COL-08: CSR and both slice-SC coefficients divide by mass twice"
hide:
  - toc
---

# COL-08 — CSR and both slice-SC coefficients divide by mass twice

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Issue | [[gh:https://github.com/bmad-sim/bmad-ecosystem/issues/2214]] |
| Pull request | [[gh:https://github.com/bmad-sim/bmad-ecosystem/pull/2215]], fix |
| Source | [csr_bin_kicks, bmad/space_charge/csr_and_space_charge_mod.f90:769–770](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L769-L770)<br>[lsc_kick_params_calc, bmad/space_charge/csr_and_space_charge_mod.f90:1141–1142](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1141-L1142)<br>[csr_and_sc_apply_kicks, bmad/space_charge/csr_and_space_charge_mod.f90:1609–1610](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1609-L1610)<br>[classical_radius, bmad/code/classical_radius.f90:14–23](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/classical_radius.f90#L14-L23) |

One-dimensional coherent synchrotron radiation and both slice space-charge kicks divide by species mass twice. At fixed Lorentz factor, geometry, total charge and unit charge state, normalized kicks must scale inversely with mass because the force is unchanged and the reference momentum scales with mass. The proton-to-electron ratio is 2.966e−7 in all three kicks instead of 5.446e−4, suppressing the proton result by another factor of about 1836. The classical-radius routine already supplies the required inverse mass before the extra divisor.

Run `python COL-08.py` with [_bmad.py](../helpers/_bmad.py) and [_collective_fields.py](../helpers/_collective_fields.py) in the same directory.

??? example "Reproducer: COL-08.py"

    [Download COL-08.py](COL-08.py)

    ```python
    """COL-08: CSR and both slice-SC coefficients divide by mass twice

    One-dimensional coherent synchrotron radiation and both slice space-charge kicks divide by species
    mass twice. At fixed Lorentz factor, geometry, total charge and unit charge state, normalized kicks
    must scale inversely with mass because the force is unchanged and the reference momentum scales
    with mass. The proton-to-electron ratio is 2.966e−7 in all three kicks instead of 5.446e−4,
    suppressing the proton result by another factor of about 1836. The classical-radius routine already
    supplies the required inverse mass before the extra divisor.

    The script compares electron and proton CSR, longitudinal slice and transverse slice kicks at fixed
    Lorentz factor and source charge. It checks that the classical-radius ratio already equals the
    inverse mass ratio before comparing all three normalized kicks.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _collective_fields import rows

    SLICE = r"""

    program species_slice
      use bmad
      use csr_and_space_charge_mod
      implicit none
      type(ele_struct) ele
      type(csr_struct) csr
      type(coord_struct) p(1)
      integer i, j, species
      real(rp) lsc, z
      character(12) name
      call init_ele(ele, drift$)
      ele%space_charge_method = slice$
      space_charge_com%n_bin = 8
      space_charge_com%lsc_kick_transverse_dependence = .false.
      allocate(csr%slice(8))
      csr%gamma = 20; csr%gamma2 = 400
      csr%kick_factor = 1; csr%actual_track_step = 0.2_rp; csr%dz_slice = 0.02_rp
      do j = 1, 2
        species = electron$
        if (j == 2) species = proton$
        csr%species = species; csr%rel_mass = mass_of(species)/m_electron
        ele%ref_species = species
        ele%value(p0c$) = mass_of(species)*sqrt(399.0_rp)
        ele%value(p0c_start$) = ele%value(p0c$)
        do i = 1, 8
          csr%slice(i) = csr_bunch_slice_struct()
          z = (i-4.5_rp)*csr%dz_slice
          csr%slice(i)%z_center = z
          csr%slice(i)%charge = 2.0e-9_rp*csr%dz_slice*exp(-0.5_rp*(z/0.03_rp)**2)
          csr%slice(i)%dcharge_density_dz = -z/0.03_rp**2*csr%slice(i)%charge/csr%dz_slice
          csr%slice(i)%sig_x = 0.001_rp; csr%slice(i)%sig_y = 0.001_rp
        enddo
        call lsc_kick_params_calc(ele, csr)
        lsc = csr%slice(6)%kick_lsc
        csr%slice%kick_lsc = 0
        call init_coord(p(1), [0.0007_rp,0.0_rp,-0.0004_rp,0.0_rp,0.0_rp,0.1_rp],ele,upstream_end$,species)
        call csr_and_sc_apply_kicks(ele, csr, p)
        write(name,'(a,i0)') 'species',j
        write(*,'(3a,4es25.16e3)') 'RESULT ',trim(name),' ',lsc,p(1)%vec(2),mass_of(species),classical_radius(species)
      enddo
    end program
    """

    CSR = r"""

    program csr_gaussian
      use bmad
      use csr_and_space_charge_mod
      implicit none
      type(ele_struct), target :: ele
      type(csr_struct) csr
      integer i, n
      real(rp) z, sigma, rho, q
      logical err
      character(32) arg
      call get_command_argument(1, arg); read(arg, *) n
      call init_ele(ele, sbend$)
      ele%csr_method = one_dim$
      ele%ix_ele = 1
      sigma = 1.0e-7_rp; rho = 100.0_rp; q = 1.0e-12_rp
      space_charge_com%n_bin = n
      csr%species = electron$; csr%rel_mass = 1
      csr%gamma = 1.0e6_rp; csr%gamma2 = csr%gamma**2
      csr%beta = sqrt(1-1/csr%gamma2)
      csr%kick_factor = 1; csr%actual_track_step = 0.1_rp
      csr%dz_slice = 16*sigma/n
      csr%ix_ele_kick = 1; csr%kick_ele => ele
      csr%y_source = 0; csr%s_chord_kick = 5
      allocate(csr%slice(n), csr%kick1(-n:n), csr%eleinfo(0:1))
      csr%eleinfo(1)%ele => ele
      csr%eleinfo(1)%theta_chord = 0; csr%eleinfo(1)%L_chord = 10
      csr%eleinfo(1)%floor0%r = 0
      csr%eleinfo(1)%spline%x0 = 0
      csr%eleinfo(1)%spline%coef = [0.0_rp, 5/rho, -0.5_rp/rho, 0.0_rp]
      csr%floor_k%r = [12.5_rp/rho, 0.0_rp, 5.0_rp]
      csr%floor_k%theta = 0
      do i = 1, n
        z = (i-0.5_rp)*csr%dz_slice-8*sigma
        csr%slice(i)%z_center = z
        csr%slice(i)%charge = q * csr%dz_slice * exp(-0.5_rp*(z/sigma)**2)/(sqrt(twopi)*sigma)
        if (i > 1) csr%slice(i)%edge_dcharge_density_dz = &
          (csr%slice(i)%charge-csr%slice(i-1)%charge)/csr%dz_slice**2
      enddo
      call csr_bin_kicks(ele, 5.0_rp, csr, err)
      if (err) error stop 'CSR failed'
      write(*, '(a,4es25.16e3)') 'RESULT csr ', sum(csr%slice%charge*csr%slice%kick_csr)/sum(csr%slice%charge), &
        r_e, e_charge, m_electron
    end program
    """


    data = rows(fortran(SLICE).log)
    electron, proton = data["species1"], data["species2"]
    ratio = electron[2] / proton[2]
    if not (1 / 1840 < ratio < 1 / 1830) or not np.all(np.isfinite([electron, proton])):
        raise RuntimeError("Species setup failed")
    if np.any(electron[:2] == 0) or np.any(proton[:2] == 0):
        raise RuntimeError("Nonzero slice-kick control failed")
    if compare(
        "classical-radius proton/electron ratio, inverse mass",
        proton[3] / electron[3],
        ratio,
        atol=0,
        rtol=2e-15,
    ):
        raise RuntimeError("Classical-radius control failed")
    species_setup = "csr%species = electron$; csr%rel_mass = 1"
    if CSR.count(species_setup) != 1:
        raise RuntimeError("CSR species setup must have exactly one substitution site")
    proton_csr = CSR.replace(species_setup, "csr%species = proton$; csr%rel_mass = m_proton/m_electron")
    e_loss = rows(fortran(CSR, args=["512"]).log)["csr"][0]
    p_loss = rows(fortran(proton_csr, args=["512"]).log)["csr"][0]
    if not np.isfinite([e_loss, p_loss]).all() or not (e_loss < 0 and p_loss < 0):
        raise RuntimeError("Nonzero CSR loss control failed")
    print(f"Electron/proton normalized CSR kicks: {e_loss:.16g}, {p_loss:.16g}")
    print(f"Electron/proton normalized LSC and transverse kicks: {electron[:2]}, {proton[:2]}")
    verdict(
        compare(
            "CSR, LSC, transverse SC proton/electron ratios, inverse mass",
            np.r_[p_loss / e_loss, proton[:2] / electron[:2]],
            np.full(3, ratio),
            atol=0,
            rtol=3e-12,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gemini-4-argon-high` |  | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

