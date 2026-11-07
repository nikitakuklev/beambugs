---
title: "COL-73: Distant cathode images lose integrated-cell field accuracy and produce NaNs"
hide:
  - toc
---

# COL-73 — Distant cathode images lose integrated-cell field accuracy and produce NaNs

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [Integrated Green-function deposition, FFT convolution and gathering, bmad/doc/multiparticle.tex:488–505](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/multiparticle.tex#L488-L505)<br>[the cathode-method contract, bmad/doc/methods.tex:641–650](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L641-L650)<br>[The public offset contract, bmad/space_charge/open_spacecharge_mod.f90:604–611](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L604-L611)<br>[Cell-charge deposition, bmad/space_charge/open_spacecharge_mod.f90:373–389](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L373-L389)<br>[Rest-frame scaling and observation offsets, bmad/space_charge/open_spacecharge_mod.f90:903–923](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L903-L923)<br>[osc_get_cgrn_freespace, bmad/space_charge/open_spacecharge_mod.f90:929–960](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L929-L960)<br>[xlafun2, bmad/space_charge/open_spacecharge_mod.f90:967–980](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L967-L980)<br>[The reflected-charge solve and field assembly, bmad/space_charge/open_spacecharge_mod.f90:659–691](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L659-L691)<br>[Current live-particle source velocity, bmad/space_charge/space_charge_mod.f90:75–99](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/space_charge_mod.f90#L75-L99)<br>[sc_field_calc, bmad/space_charge/space_charge_mod.f90:115–138](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/space_charge_mod.f90#L115-L138)<br>[Per-element image initialization, bmad/space_charge/track1_bunch_space_charge.f90:43–48](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/track1_bunch_space_charge.f90#L43-L48)<br>[fixed and adaptive time-step dispatch, bmad/space_charge/track1_bunch_space_charge.f90:107–112](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/track1_bunch_space_charge.f90#L107-L112) |

Fine transverse meshes can give large erroneous cathode-image fields when a bunch is far downstream, and larger separations produce NaN horizontal fields. For a 1 pC electron bunch spanning 50 µm transversely and 0.98 mm longitudinally, with Lorentz factor γ=30 and a 32×32×64 mesh, a cathode 1 m away adds up to 1.231×10⁴ V/m although direct Coulomb integration gives image fields of only micovolts per metre. At 2 m, all 1,331 particles receive NaN horizontal fields while image-free controls remain finite. Cancellation corrupts valid integrated-cell kernels before the FFT. The image cutoff is evaluated after the current field calculation.

Run `python COL-73.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-73.py"

    [Download COL-73.py](COL-73.py)

    ```python
    """COL-73: Distant cathode images lose integrated-cell field accuracy and produce NaNs

    Fine transverse meshes can give large erroneous cathode-image fields when a bunch is far downstream, and larger separations produce NaN horizontal fields. For a 1 pC electron bunch spanning 50 µm transversely and 0.98 mm longitudinally, with Lorentz factor γ=30 and a 32×32×64 mesh, a cathode 1 m away adds up to 1.231×10⁴ V/m although direct Coulomb integration gives image fields of only micovolts per metre. At 2 m, all 1,331 particles receive NaN horizontal fields while image-free controls remain finite. Cancellation corrupts valid integrated-cell kernels before the FFT. The image cutoff is evaluated after the current field calculation.

    The script calls the production particle-depositing field callback with cathode images off and on in fresh native processes. It integrates the reflected deposited cell charges independently with converged positive-volume Coulomb quadrature and checks a valid kernel cell before the FFT plus the primitive at nonzero transverse coordinates. The shared runner compiles the inline client and removes its scratch files.
    """

    from __future__ import annotations

    import os

    for _variable in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NPROC", "PJRT_NPROC",
    ):
        os.environ[_variable] = "1"

    import numpy as np
    from _bmad import compare, fortran, verdict
    from numpy.polynomial.legendre import leggauss

    NATIVE = r"""
    ! Exercise the production time-based collective-field callback, including its
    ! ordinary particle deposition and post-solve cathode cutoff.
    program callback
    use bmad
    use space_charge_mod
    use, intrinsic :: ieee_arithmetic
    implicit none
    type(lat_struct) :: lat
    type(bunch_struct),target :: bunch
    type(em_field_struct), allocatable :: fields(:),control(:)
    type(coord_struct),pointer :: p
    type(mesh3d_struct) :: mesh
    integer :: i,j,k,l,ni,no,finite_image,finite_control,iworst
    real(rp) :: gam,d,beta,scale,error,gam_eff,xa(1331),ya(1331),za(1331),qa(1331),x,y,z
    logical :: include_image
    read(*,*) gam,d
    call bmad_parser('callback.bmad',lat)
    allocate(bunch%particle(11**3),fields(11**3),control(11**3))
    beta=sqrt(1-1/gam**2)
    l=0
    do k=1,11
      do j=1,11
        do i=1,11
          l=l+1
          p=>bunch%particle(l)
          p%vec=0
          p%vec(1)=(i-6)*5e-6_rp
          p%vec(3)=(j-6)*5e-6_rp
          p%s=d+(k-6)*98e-6_rp
          p%t=0
          p%beta=beta
          p%charge=1e-12_rp/(11**3)
          p%species=electron$
          p%p0c=mass_of(p%species)*sqrt(gam**2-1)
          p%vec(5)=p%s
          p%vec(6)=p%p0c
          p%state=alive$
          p%ix_ele=1
          xa(l)=p%vec(1);ya(l)=p%vec(3);za(l)=p%s;qa(l)=-p%charge
        enddo
      enddo
    enddo
    gam_eff=1/sqrt(1-(sum(bunch%particle%beta)/size(bunch%particle))**2)
    mesh%nhi=space_charge_com%space_charge_mesh_size
    mesh%gamma=gam_eff
    call deposit_particles(xa,ya,za,mesh,qa=qa,mesh_growth_factor=space_charge_com%mesh_growth_factor, &
      mesh_shrink_factor=space_charge_com%mesh_shrink_factor)
    write(*,'(a,1x,es25.16)')'PREF',299792458.0**2*1.00000000055d-7
    write(*,'(a,10(1x,es25.16))')'MESH',mesh%min,mesh%max,mesh%delta,gam_eff
    open(newunit=ni,file='callback-rho.bin',access='stream',form='unformatted',status='replace')
    write(ni)mesh%rho
    close(ni)
    x=mesh%delta(1)/2;y=mesh%delta(2)/2;z=2*gam_eff*d
    write(*,'(a,5(1x,es25.16))')'PRIMITIVE',x,y,z,sqrt(x*x+y*y+z*z)-z,xlafun2(x,y,z)
    include_image=.false.
    call sc_field_calc(bunch,lat%branch(0),include_image,0.0_rp,control)
    include_image=.true.
    call sc_field_calc(bunch,lat%branch(0),include_image,0.0_rp,fields)
    finite_image=0;finite_control=0;scale=0;error=0;iworst=0
    do i=1,size(fields)
      finite_image=finite_image+count(ieee_is_finite(fields(i)%E))
      finite_control=finite_control+count(ieee_is_finite(control(i)%E))
      scale=max(scale,sqrt(sum(control(i)%E**2)))
      if(all(ieee_is_finite(fields(i)%E))) then
        if(sqrt(sum((fields(i)%E-control(i)%E)**2))>error)then
          error=sqrt(sum((fields(i)%E-control(i)%E)**2));iworst=i
        endif
      endif
    enddo
    if(finite_image<3*size(fields))error=ieee_value(0.0_rp,ieee_quiet_nan)
    write(*,'(a,2(1x,es25.16))')'GEOMETRY',gam,d
    write(*,'(a,3(1x,i0),1x,l1)')'FINITE',finite_image,finite_control,3*size(fields),include_image
    do j=1,3
      write(*,'(a,3(1x,i0))')'AXIS',j,count([(ieee_is_finite(fields(i)%E(j)),i=1,size(fields))]), &
        count([(ieee_is_nan(fields(i)%E(j)),i=1,size(fields))])
    enddo
    write(*,'(a,2(1x,es25.16))')'SCALE',scale,error
    write(*,'(a,6(1x,es25.16))')'CENTER',fields(666)%E,control(666)%E
    if(iworst>0)then
      p=>bunch%particle(iworst)
      write(*,'(a,1x,i0,9(1x,es25.16))')'WORST',iworst,p%vec(1),p%vec(3),p%s,fields(iworst)%E,control(iworst)%E
    endif
    end program
    """

    KERNEL = r"""
    program valid_cell
    use bmad
    use open_spacecharge_mod, only: osc_get_cgrn_freespace
    implicit none
    complex(dp) :: cgrn(32,32,32)
    real(rp) :: h, prefactor
    h=2e-6_rp
    prefactor=299792458.0**2*1.00000000055d-7
    call osc_get_cgrn_freespace(cgrn,[h,h,h],1.0_rp,1,offset=[0.0_rp,0.0_rp,0.02_rp])
    ! The cell at indices (17,16,16) is centered at (h,0,0.02) in target-minus-source coordinates.
    write(*,'(a,2(1x,es25.16))') 'CELL',real(cgrn(17,16,16),rp)*1e-9_rp*prefactor,prefactor
    end program
    """

    LATTICE = """no_digested
    beginning[beta_a] = 1
    beginning[beta_b] = 1
    beginning[p0c] = 1.53214315854e7
    parameter[particle] = electron
    parameter[geometry] = open
    d: drift, l = 4, space_charge_method = cathode_fft_3d, &
      tracking_method = fixed_step_time_runge_kutta
    lat: line = (d)
    use, lat
    """


    def image_oracle(mesh, rho, target, order, prefactor):
        """Integrate uniform deposited cells after reflection and the rest-frame boost."""
        indices = np.array(np.nonzero(rho)).T
        # rho is charge per cell in C. Reflection reverses charge and longitudinal position.
        charge = -rho[tuple(indices.T)]
        h = mesh[6:9]
        centers = mesh[:3] + indices * h
        gamma = mesh[9]
        nodes, weights = leggauss(order)
        total = np.zeros(3, dtype=np.longdouble)
        for i, x in enumerate(nodes):
            for j, y in enumerate(nodes):
                for k, z in enumerate(nodes):
                    sample = centers + np.array([x, y, z]) * h / 2
                    sample[:, 2] *= -1
                    delta = target - sample
                    delta[:, 2] *= gamma
                    radius = np.linalg.norm(delta, axis=1)
                    # Tensor weights / 8 average the physical cell volume; charge already
                    # includes that volume. Accumulate positive-volume Coulomb integrals.
                    total += (
                        np.sum(charge[:, None] * delta / radius[:, None] ** 3,
                               axis=0, dtype=np.longdouble)
                        * weights[i] * weights[j] * weights[k] / 8
                    )
        total *= prefactor
        total[:2] *= gamma
        return np.array(total, dtype=float)


    def main():
        result = fortran(KERNEL)
        rows = [line.split()[1:] for line in result.log.splitlines() if line.startswith("CELL ")]
        if len(rows) != 1:
            raise RuntimeError("Expected one valid-cell diagnostic")
        observed, prefactor = map(float, rows[0])
        h, distance = 2e-6, 0.02
        cell_mesh = np.array([0.0] * 6 + [h, h, h, 1.0])
        cell_rho = np.array([[[-1e-9]]])
        target = np.array([h, 0.0, distance])
        expected4 = image_oracle(cell_mesh, cell_rho, target, 4, prefactor)
        expected6 = image_oracle(cell_mesh, cell_rho, target, 6, prefactor)
        np.testing.assert_allclose(expected4, expected6,
                                   atol=2e-14 * np.linalg.norm(expected6), rtol=0)
        # Allow accumulated binary64 mesh-coordinate subtraction error. This uses the
        # physical field scale, independent of the ill-conditioned primitive representation.
        budget = 1024 * np.finfo(float).eps * (distance / h) * np.linalg.norm(expected6)
        cell_defect = compare("valid cell before FFT; Q=1 nC h=2 um R=2 cm Ex (V/m); "
                              "direct Coulomb volume integral", observed, expected6[0], atol=budget)
        defects = [cell_defect]
        for gamma, distance in ((3, 0.01), (10, 1), (30, 1), (30, 2), (100, 1)):
            result = fortran(
                NATIVE, files={"callback.bmad": LATTICE},
                outputs=("callback-rho.bin",), stdin=f"{gamma} {distance}\n",
            )
            records = {}
            axes = []
            for line in result.log.splitlines():
                words = line.split()
                if not words:
                    continue
                if words[0] == "AXIS":
                    axes.append(tuple(map(int, words[1:])))
                elif words[0] in {"MESH", "PREF", "PRIMITIVE", "FINITE", "SCALE", "CENTER", "WORST"}:
                    if words[0] in records:
                        raise RuntimeError(f"Duplicate native record: {line}")
                    records[words[0]] = words[1:]
            mesh = np.array(records["MESH"], dtype=float)
            if mesh.shape != (10,) or len(axes) != 3:
                raise RuntimeError("Incomplete mesh or component diagnostics")
            rho = np.frombuffer(result.outputs["callback-rho.bin"], dtype=np.float64).reshape(
                (32, 32, 64), order="F"
            )
            np.testing.assert_allclose(np.sum(rho), -1e-12, atol=0, rtol=2e-14)
            prefactor = float(records["PREF"][0])
            center = np.array([0.0, 0.0, distance])
            expected3 = image_oracle(mesh, rho, center, 3, prefactor)
            expected5 = image_oracle(mesh, rho, center, 5, prefactor)
            convergence = np.linalg.norm(expected3 - expected5) / np.linalg.norm(expected5)
            if convergence > 2e-13:
                raise RuntimeError(f"Coulomb quadrature did not converge: {convergence}")
            finite, control, total = map(int, records["FINITE"][:3])
            if control != total or total != 3993:
                raise RuntimeError("Image-free control must have 3993 finite components")
            label = f"gamma={gamma} d={distance:g} m"
            print(f"{label} mesh delta (m)={mesh[6:9].tolist()}; charge={np.sum(rho):.16e} C")
            print(f"{label} image-free finite={control}/{total}; image-on finite={finite}/{total}; "
                  f"(axis, finite, NaN)={axes}; cutoff include_image={records['FINITE'][3]}")
            print(f"{label} direct reflected-cell center E (V/m)={expected5.tolist()}; "
                  f"3/5-point relative convergence={convergence:.6e}")
            primitive = np.array(records["PRIMITIVE"], dtype=float)
            print(f"{label} native primitive (x,y,z,r-z,value)={primitive.tolist()}; "
                  "expected finite integral at nonzero x,y")
            # Callback finite-field comparisons also contain unused-plane FFT roundoff.
            # Assign this entry's verdict only when an isolated valid-cell/primitive
            # failure accompanies them, so the rejected padding candidate cannot decide it.
            primitive_defect = not np.isfinite(primitive[-1])
            defect = (finite != total) and (cell_defect or primitive_defect)
            if finite == total:
                worst = np.array(records["WORST"][1:], dtype=float)
                target, actual = worst[:3], worst[3:6] - worst[6:9]
                expected = image_oracle(mesh, rho, target, 5, prefactor)
                expected_low = image_oracle(mesh, rho, target, 3, prefactor)
                np.testing.assert_allclose(expected_low, expected,
                                           atol=2e-13 * np.linalg.norm(expected), rtol=0)
                # Assess practical total-field accuracy, rather than divide by a nearly
                # vanishing remote transverse field. Near-cathode controls pass this bound.
                self_scale = float(records["SCALE"][0])
                discrepancy = compare(
                    f"{label} largest image increment (V/m); positive-volume Coulomb oracle",
                    actual, expected, atol=1e-8 * self_scale,
                )
                local_scale = np.linalg.norm(worst[6:9])
                error = np.linalg.norm(actual - expected)
                print(f"{label} worst position (m)={target.tolist()}; "
                      f"image-error norm={error:.9e} V/m; "
                      f"error/local self-field norm={error/local_scale:.9e}")
                if gamma == 3:
                    if discrepancy:
                        raise RuntimeError("Near-cathode practical-accuracy control failed")
                else:
                    defect |= discrepancy and cell_defect
            defects.append(defect)
        verdict(any(defects))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

