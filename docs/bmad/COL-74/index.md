---
title: "COL-74: Time-based FFT space charge reconstructs the boost from accumulated rounded speeds"
hide:
  - toc
---

# COL-74 — Time-based FFT space charge reconstructs the boost from accumulated rounded speeds

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad 20261031-0 (`c4c629f`, committed 2026-10-31) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [FFT deposition/Poisson/gather contract](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/multiparticle.tex#L488-L502)<br>[space-charge selectors and time-tracking restrictions](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L641-L650)<br>[live-particle positions, signed charges and accumulated mean β](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/space_charge_mod.f90#L75-L99)<br>[γ = 1/√(1−β²), deposition and solve](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/space_charge_mod.f90#L115-L120)<br>[The rest-frame Green-function contract](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L858-L883)<br>[longitudinal stretching and transverse boost](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L904-L910)<br>[convert_pc_to](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/convert_pc_to.f90#L39-L48)<br>[time-coordinate momenta](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/convert_particle_coordinates_s_to_t.f90#L6-L12)<br>[track1_bunch dispatch](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_mod.f90#L282-L310)<br>[fixed/adaptive stepping](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/track1_bunch_space_charge.f90#L105-L113)<br>[sc_step field evaluation](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/space_charge_mod.f90#L244-L252)<br>[adaptive full/half-step evaluation](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/space_charge_mod.f90#L326-L329)<br>[S-based FFT initialization](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L169-L172)<br>[CSR3D initialization](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1925-L1928) |

Time-based FFT space charge reconstructs the Lorentz factor γ from an accumulated mean of rounded particle speeds. At 20.44 GeV, a 100 pC electron bunch represented by one million macroparticles gives a longitudinal field of −10.86 V/m instead of −10.45 V/m, a 3.976% error. Independent Coulomb integration of the deposited charges establishes the expected field; retaining momentum-derived γ reproduces it. Representing the same charge source with 100,000 macroparticles changes the error to −0.6909%. Photoinjector-energy errors are small, while γ=10⁸ produces nonfinite fields. S-based FFT obtains γ from momentum and avoids this reconstruction.

Run `python COL-74.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-74.py"

    [Download COL-74.py](COL-74.py)

    ```python
    """COL-74: Time-based FFT space charge reconstructs the boost from accumulated rounded speeds

    Time-based FFT space charge reconstructs the Lorentz factor γ from an accumulated mean of rounded particle speeds. At 20.44 GeV, a 100 pC electron bunch represented by one million macroparticles gives a longitudinal field of −10.86 V/m instead of −10.45 V/m, a 3.976% error. Independent Coulomb integration of the deposited charges establishes the expected field; retaining momentum-derived γ reproduces it. Representing the same charge source with 100,000 macroparticles changes the error to −0.6909%. Photoinjector-energy errors are small, while γ=10⁸ produces nonfinite fields. S-based FFT obtains γ from momentum and avoids this reconstruction.

    The script samples the public time-based field callback for cold electron bunches and independently integrates the actual deposited cell charges with 45-digit arithmetic. It varies macroparticle count at fixed physical source, checks photoinjector and retained-momentum controls, and repeats the integral at 65 digits. Every native case runs in a fresh process.
    """

    import math
    from functools import lru_cache

    import mpmath as mp
    import numpy as np
    from _bmad import compare, fortran, verdict

    LATTICE = 'no_digested\nbeginning[beta_a]=1\nbeginning[beta_b]=1\nbeginning[p0c]=1e9\nparameter[particle]=electron\nparameter[geometry]=open\nd:drift,l=1,tracking_method=fixed_step_time_runge_kutta,space_charge_method=cathode_fft_3d\nlat:line=(d)\nuse,lat\n'

    CLIENT = r"""
    program check_gamma
    use bmad
    use space_charge_mod
    implicit none
    type(lat_struct) lat
    type(bunch_struct) bunch
    type(mesh3d_struct) mesh
    type(em_field_struct),allocatable::fields(:)
    real(rp),allocatable::x(:),y(:),z(:),q(:)
    real(rp) g,gr,b,p,av,radius,half_length,totalq,kernel_k,e(3),bb(3)
    integer count,m,i,j,k,n,a,ix,iy,iz,mode,probe_index
    logical images
    read(*,*)g,count,m,radius,half_length,totalq,mode
    call bmad_parser('lat.bmad',lat)
    space_charge_com%space_charge_mesh_size=[9,9,9]
    if(mode==1)then
     space_charge_com%mesh_growth_factor=0
     space_charge_com%mesh_shrink_factor=0
    endif
    probe_index=m**3
    if(mode==1)probe_index=(m**3+1)/2+(m-1)/2
    ! Use the tracking defaults, including nonzero padding and its small-span floor.
    allocate(bunch%particle(count),fields(count),x(count),y(count),z(count),q(count))
    p=mass_of(electron$)*sqrt((g-1)*(g+1))
    call convert_pc_to(p,electron$,beta=b,gamma=gr)
    ! Identical momenta isolate the mean-speed rounding from a choice of bunch frame.
    do n=1,count
     a=modulo(n-1,m**3)
     ix=modulo(a,m);iy=modulo(a/m,m);iz=a/(m*m)
     x(n)=radius*(2d0*ix/(m-1)-1d0)
     y(n)=radius*(2d0*iy/(m-1)-1d0)
     z(n)=half_length*(2d0*iz/(m-1)-1d0)
     q(n)=-totalq/count
     bunch%particle(n)%vec=[x(n),0d0,y(n),0d0,z(n),p]
     bunch%particle(n)%s=z(n);bunch%particle(n)%p0c=p
     bunch%particle(n)%beta=b;bunch%particle(n)%species=electron$
     bunch%particle(n)%charge=-q(n);bunch%particle(n)%state=alive$;bunch%particle(n)%ix_ele=1
    enddo
    av=0
    do n=1,count
     av=av+bunch%particle(n)%beta
    enddo
    av=av/count
    mesh%nhi=[9,9,9];mesh%gamma=gr
    call deposit_particles(x,y,z,mesh,qa=q,mesh_growth_factor=space_charge_com%mesh_growth_factor, &
     mesh_shrink_factor=space_charge_com%mesh_shrink_factor)
    call space_charge_3d(mesh,calc_bfield=.true.)
    call interpolate_field(x(probe_index),y(probe_index),z(probe_index),mesh,E=e,B=bb)
    images=.false.
    call sc_field_calc(bunch,lat%branch(0),images,0d0,fields)
    kernel_k=299792458.0**2*1.00000000055d-7
    write(*,'(a,8es26.17)')'FACTORS ',gr,b,av,1/sqrt(1-av**2),mass_of(electron$),kernel_k,sum(mesh%rho),sum(q)
    write(*,'(a,3es26.17)')'MIN ',mesh%min
    write(*,'(a,3es26.17)')'DELTA ',mesh%delta
    write(*,'(a,3es26.17)')'TARGET ',x(probe_index),y(probe_index),z(probe_index)
    write(*,'(a,9es26.17)')'FIELD ',fields(probe_index)%E,e,fields(probe_index)%B
    do k=1,9
    do j=1,9
    do i=1,9
     if(mesh%rho(i,j,k)/=0)write(*,'(a,3i4,es26.17)')'RHO ',i,j,k,mesh%rho(i,j,k)
    enddo
    enddo
    enddo
    end program
    """

    def direct(rows, gamma, dps=45):
        # Integrate Coulomb's law over each constant-density box by the divergence
        # theorem. Only two faces contribute to a chosen Cartesian component.
        # The rectangle potential is the double antiderivative of 1/r below.
        with mp.workdps(dps):
            to_mp = lambda v: mp.mpf(float(v))
            delta = list(map(to_mp, rows["DELTA"]))
            delta[2] *= to_mp(gamma)
            vol = mp.fprod(delta)
            rho = rows["RHO"]
            target = list(map(to_mp, rows["TARGET"]))
            minimum = list(map(to_mp, rows["MIN"]))
            labdelta = list(map(to_mp, rows["DELTA"]))
            index = [(target[i] - minimum[i]) / labdelta[i] for i in range(3)]
            lower = [int(mp.floor(v)) for v in index]
            fraction = [index[i] - lower[i] for i in range(3)]

            @lru_cache(None)
            def potential(a, y, z):
                a = abs(a)
                r = mp.sqrt(a * a + y * y + z * z)
                return y * mp.log(z + r) + z * mp.log(y + r) - a * mp.atan(y * z / (a * r))

            @lru_cache(None)
            def kernel(axis, di, dj, dk):
                ds = [di * delta[0], dj * delta[1], dk * delta[2]]
                transverse = [i for i in range(3) if i != axis]
                y, z = [ds[i] for i in transverse]
                hy, hz = [delta[i] / 2 for i in transverse]
                hx = delta[axis] / 2

                def face(a):
                    return (
                        potential(a, y + hy, z + hz)
                        - potential(a, y - hy, z + hz)
                        - potential(a, y + hy, z - hz)
                        + potential(a, y - hy, z - hz)
                    )

                return (face(ds[axis] - hx) - face(ds[axis] + hx)) / vol

            answer = [mp.mpf(0), mp.mpf(0), mp.mpf(0)]
            # Reimplement trilinear gathering from the geometric node positions.
            for mask in range(8):
                bits = [(mask >> i) & 1 for i in range(3)]
                node = [lower[i] + bits[i] + 1 for i in range(3)]
                weight = mp.fprod(fraction[i] if bits[i] else 1 - fraction[i] for i in range(3))
                for source in rho:
                    difference = [node[i] - int(source[i]) for i in range(3)]
                    absolute = list(map(abs, difference))
                    for axis in range(3):
                        if difference[axis] == 0:
                            continue  # The self-plane's odd component integrates to exactly zero.
                        contribution = kernel(axis, *absolute)
                        sign = 1 if difference[axis] > 0 else -1
                        boost = to_mp(gamma) if axis != 2 else mp.mpf(1)
                        answer[axis] += weight * to_mp(source[3]) * sign * contribution * boost
            return [float(v) for v in answer]


    def sample(gamma, count, sites, radius, half_length, charge, mode):
        result = fortran(
            CLIENT, files={"lat.bmad": LATTICE},
            stdin=f"{gamma} {count} {sites} {radius:.17g} {half_length:.17g} {charge:.17g} {mode}\n",
        )
        rows = {key: [] for key in ("FACTORS", "MIN", "DELTA", "TARGET", "FIELD", "RHO")}
        for line in result.log.splitlines():
            words = line.split()
            if words and words[0] in rows:
                values = list(map(float, words[1:]))
                if words[0] == "RHO":
                    rows["RHO"].append(values)
                else:
                    rows[words[0]] = values
        if any(not values for values in rows.values()):
            raise RuntimeError("Missing native geometry, charge or field record")
        return rows


    def main():
        cases = [
            (2, 1000000, 5, 1e-3, 1e-3, 1e-9, 0),
            (10, 1000000, 5, 1e-4, 3e-4, 1e-10, 0),
            (40000, 125, 5, 3e-5, 1e-5, 1e-10, 0),
            (40000, 100000, 5, 3e-5, 1e-5, 1e-10, 0),
            (40000, 1000000, 5, 3e-5, 1e-5, 1e-10, 0),
            (1e8, 27, 3, 1e-3, 1e-11, 1e-12, 1),
        ]
        defects = []
        same_source = []
        for case in cases:
            rows = sample(*case)
            gamma, beta, mean, reconstructed, _mass, prefactor, deposited, _naive_charge_sum = rows["FACTORS"]
            # The native sum(q) is a naive million-term sum with its own accumulation
            # error. Compare deposited charge with the exact sum of assigned q bits.
            with mp.workdps(45):
                exact_charge = float(mp.mpf(-case[5] / case[1]) * case[1])
            np.testing.assert_allclose(deposited, exact_charge, rtol=2e-12, atol=0)
            expected = np.array(direct(rows, gamma)) * prefactor
            retained = np.array(rows["FIELD"][3:6])
            observed = np.array(rows["FIELD"][:3])
            # At high boosts, the transverse kernel has a separate far-cell cancellation
            # error. Use Ez for the finite-energy verdict and Ex on the extreme axis fixture.
            axis = 0 if case[6] else 2
            np.testing.assert_allclose(retained[axis], expected[axis], rtol=1e-9, atol=0)
            if math.isfinite(reconstructed):
                predicted = np.array(direct(rows, reconstructed)) * prefactor
                np.testing.assert_allclose(observed[axis], predicted[axis], rtol=1e-9, atol=0)
                mismatch = compare(
                    f"gamma={gamma:g}, N={case[1]}, Ez (V/m); independent Coulomb face integral",
                    observed[2], expected[2], atol=0, rtol=1e-9,
                )
                print(f"  gamma_from_mean={reconstructed:.17g}; beta={beta:.18g}; "
                      f"mean_beta={mean:.18g}; relative_Ez_error={observed[2]/expected[2]-1:+.10g}; "
                      "retained-gamma and reconstructed-gamma oracles pass rtol=1e-9")
                if gamma < 100:
                    np.testing.assert_allclose(observed[2], expected[2], rtol=4e-9, atol=0)
                if case[0] == 40000:
                    same_source.append(expected[[0, 2]])
                    defects.append(mismatch)
            else:
                finite = int(np.isfinite(observed).sum())
                print(f"gamma={gamma:g}, N={case[1]}: finite_E_components={finite}/3 expected=3/3; "
                      f"retained_Ex={retained[0]:.17g} expected_Coulomb_Ex={expected[0]:.17g} V/m")
                if finite not in (0, 3):
                    raise RuntimeError("Unexpected partially nonfinite callback")
                defects.append(finite == 0)
            if case[0] in (2, 40000) and case[1] == 1000000:
                refined = np.array(direct(rows, gamma, dps=65)) * prefactor
                np.testing.assert_allclose(refined[[0, 2]], expected[[0, 2]], rtol=1e-14, atol=0)
                print("  independent integral at 45/65 digits agrees rtol=1e-14")
        source_error = max(float(np.max(abs(v / same_source[0] - 1))) for v in same_source)
        np.testing.assert_allclose(same_source, np.tile(same_source[0], (3, 1)), rtol=1e-12, atol=0)
        print(f"same physical source, N=125/100000/1000000: relative_Coulomb_change={source_error:.4g} "
              "expected=0 within rtol=1e-12; charge controls pass rtol=2e-12")
        verdict(all(defects))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

