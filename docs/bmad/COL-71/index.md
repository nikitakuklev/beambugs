---
title: "COL-71: S-based FFT space charge uses the exit energy throughout energy-changing elements"
hide:
  - toc
---

# COL-71 — S-based FFT space charge uses the exit energy throughout energy-changing elements

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [collective-method contract, bmad/doc/methods.tex:624–650](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L624-L650)<br>[bmad/doc/elements.tex:3251–3253](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L3251-L3253)<br>[bmad/doc/multiparticle.tex:378–395](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/multiparticle.tex#L378-L395)<br>[bmad/doc/multiparticle.tex:488–502](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/multiparticle.tex#L488-L502)<br>[bmad/multiparticle/beam_mod.f90:283–319](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_mod.f90#L283-L319)<br>[bmad/space_charge/csr_and_space_charge_mod.f90:169–172](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L169-L172)<br>[bmad/space_charge/csr_and_space_charge_mod.f90:305–333](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L305-L333)<br>[bmad/space_charge/csr_and_space_charge_mod.f90:368–370](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L368-L370)<br>[bmad/space_charge/csr_and_space_charge_mod.f90:1695–1708](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1695-L1708)<br>[bmad/space_charge/csr_and_space_charge_mod.f90:1731–1741](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L1731-L1741)<br>[bmad/space_charge/open_spacecharge_mod.f90:373–389](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L373-L389)<br>[bmad/space_charge/open_spacecharge_mod.f90:858–908](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/open_spacecharge_mod.f90#L858-L908)<br>[bmad/doc/charged-tracking.tex:1162–1193](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/charged-tracking.tex#L1162-L1193)<br>[bmad/space_charge/space_charge_mod.f90:75–120](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/space_charge_mod.f90#L75-L120)<br>[bmad/space_charge/csr_and_space_charge_mod.f90:184–190](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/space_charge/csr_and_space_charge_mod.f90#L184-L190) |

With s-based FFT space charge inside an energy-changing cavity, Bmad boosts every current charge mesh with the exit-centroid Lorentz factor γ and uses that value for transverse electric–magnetic cancellation. Earlier kicks therefore use a future beam energy. For a weak electron bunch accelerated from γ = 4 to 16 in a 1 m cavity with 32 space-charge steps, the transverse impulse is 65.21% too small against direct Coulomb integration at the local energy. A decelerating γ = 16→4 case gives 2.829 times the local-energy impulse. Constant-energy tracking agrees with the independent calculation.

Run `python COL-71.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-71.py"

    [Download COL-71.py](COL-71.py)

    ```python
    """COL-71: S-based FFT space charge uses the exit energy throughout energy-changing elements

    With s-based FFT space charge inside an energy-changing cavity, Bmad boosts every current charge mesh with the exit-centroid Lorentz factor γ and uses that value for transverse electric–magnetic cancellation. Earlier kicks therefore use a future beam energy. For a weak electron bunch accelerated from γ = 4 to 16 in a 1 m cavity with 32 space-charge steps, the transverse impulse is 65.21% too small against direct Coulomb integration at the local energy. A decelerating γ = 16→4 case gives 2.829 times the local-energy impulse. Constant-energy tracking agrees with the independent calculation.

    Setup: run pinned to one CPU (`taskset -c <cpu>`) with `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` set to 1; the script refuses to run otherwise.

    The native client records current charge cells and applied fields without changing tracking, and a separate Gauss–Legendre face integral evaluates the same cells at the local energy without Bmad's Green function or FFT. Constant-energy, charge-scaling, endpoint-weight and native-versus-observation controls isolate the energy argument. The shared runner compiles inline clients and removes temporary files.
    """

    from __future__ import annotations

    import itertools
    import math
    import os

    import numpy as np
    from _bmad import compare, fortran, source, verdict
    from numpy.polynomial.legendre import leggauss

    MASS = 510998.95069
    # Match the existing COL-10 prefactor in the comparison. Using SI double
    # precision instead changes every force by the same known 32.269 ppb.
    COULOMB_NATIVE = float(np.float32(299792458.0) ** np.float32(2)) * 1.00000000055e-7

    DRIVER = r"""program confirmation
    use bmad
    use beam_mod, only: track1_bunch
    RECORD_USE
    implicit none
    type(lat_struct),target :: lat
    type(bunch_struct) bunch
    type(coord_struct) centroid(0:1)
    integer i,j,k,n,steps
    real(rp) q
    logical err
    read(*,*) steps,q
    call bmad_parser('case.bmad',lat,err_flag=err)
    if(err) error stop 'parser'
    call init_coord(centroid(0),lat%ele(0),downstream_end$)
    call track1(centroid(0),lat%ele(1),lat%param,centroid(1),err_flag=err)
    if(err.or.centroid(1)%state/=alive$) error stop 'centroid'
    space_charge_com%n_bin=8
    space_charge_com%space_charge_mesh_size=[12,12,12]
    space_charge_com%ds_track_step=1.0_rp/steps
    space_charge_com%mesh_growth_factor=0
    space_charge_com%mesh_shrink_factor=0
    bmad_com%csr_and_space_charge_on=.true.
    call reallocate_bunch(bunch,27)
    n=0
    do i=-1,1
     do j=-1,1
      do k=-1,1
       n=n+1
       call init_coord(bunch%particle(n), &
         [i*0.0008_rp,0.0_rp,j*0.0012_rp,0.0_rp,k*0.00015_rp,0.0_rp], &
         lat%ele(1),upstream_end$,electron$)
       bunch%particle(n)%charge=q/27
      enddo
     enddo
    enddo
    bunch%n_live=27; bunch%charge_tot=q; bunch%charge_live=q
    TRACK
    if(err.or.any(bunch%particle%state/=alive$)) error stop 'tracking'
    write(*,'(a,*(es25.16,1x))') 'RESULT ',bunch%particle(5)%p0c*bunch%particle(5)%vec(2), &
     centroid(0)%p0c,centroid(1)%p0c,real(bunch%n_live,rp)
    end program
    """


    def instrument() -> str:
        """Create an observation-only module without writing to the selected source tree."""
        text = (source() / "bmad/space_charge/csr_and_space_charge_mod.f90").read_text()
        text = text.replace("module csr_and_space_charge_mod", "module rm04_record_mod")
        anchor = "integer :: i, n, n_interp\n"
        assert text.count(anchor) == 1
        text = text.replace(anchor, anchor + "integer im,jm,km,iu\ninteger,save :: shot=-1\n")
        anchor = "  !$OMP PARALLEL DO PRIVATE(n, i, p, factor, pz0, ef, dpz, new_beta)"
        assert text.count(anchor) == 1
        diagnostic = r"""
      ! Only record current pre-kick data. No alternate force calculation or replay.
      shot=shot+1
      open(newunit=iu,file='mesh.dat',status='unknown',position='append')
      write(iu,'(a,i0,1x,*(es25.16,1x))') 'SHOT ',shot,csr%mesh3d%gamma,csr%actual_track_step, &
        particle(5)%p0c,particle(5)%beta,particle(5)%vec(6), &
        particle(5)%vec(1),particle(5)%vec(3),particle(5)%vec(5)-dct_ave*particle(5)%beta, &
        E_batch(1,5)
      write(iu,'(a,es25.16)') 'MEAN_GAMMA ',1/sqrt(1-(sum(particle%beta)/size(particle))**2)
      write(iu,'(a,*(es25.16,1x))') 'GEOM ',csr%mesh3d%min,csr%mesh3d%delta
      do km=1,size(csr%mesh3d%rho,3)
       do jm=1,size(csr%mesh3d%rho,2)
        do im=1,size(csr%mesh3d%rho,1)
         if(csr%mesh3d%rho(im,jm,km)==0) cycle
         write(iu,'(a,3(i0,1x),es25.16)') 'CELL ',im,jm,km,csr%mesh3d%rho(im,jm,km)
        enddo
       enddo
      enddo
      close(iu)
    """
        text = text.replace(anchor, diagnostic + "\n" + anchor)
        return text


    def run(record: bool, gi: float, gf: float, steps: int, q: float) -> tuple[np.ndarray, str]:
        """Compile and run each simulation in a fresh shared-runner process."""
        driver = DRIVER.replace(
            "RECORD_USE",
            "use rm04_record_mod, only: recorded_track => track1_bunch_csr" if record else "",
        ).replace(
            "TRACK",
            "call recorded_track(bunch,lat%ele(1),centroid,err)" if record else
            "call track1_bunch(bunch,lat%ele(1),err,centroid)",
        )
        lattice = f"""no_digested
    parameter[particle]=electron
    parameter[geometry]=open
    beginning[p0c]={MASS * math.sqrt(gi*gi-1):.17g}
    bmad_com[absolute_time_tracking]=F
    c: lcavity, l=1, voltage={MASS*(gf-gi):.17g}, rf_frequency=299792458, phi0=0, &
     cavity_type=traveling_wave, fringe_at=no_end, n_rf_steps=128, space_charge_method=fft_3d
    ln: line=(c)
    use,ln
    """
        result = fortran(
            (instrument() + "\n" if record else "") + driver,
            files={"case.bmad": lattice},
            outputs=["mesh.dat"] if record else [],
            stdin=f"{steps} {q:.17g}\n",
        )
        rows = [np.array(list(map(float, line.split()[1:]))) for line in result.log.splitlines()
                if line.startswith("RESULT ")]
        if len(rows) != 1 or rows[0].shape != (4,) or not np.isfinite(rows[0]).all():
            raise RuntimeError(f"Expected one finite RESULT record: {result.log}")
        return rows[0], result.outputs["mesh.dat"].decode() if record else ""


    def snapshots(text: str) -> list[dict]:
        shots: list[dict] = []
        cell_rows: list[list[float]] = []
        for line in text.splitlines():
            label, *values = line.split()
            row = list(map(float, values))
            if label == "SHOT":
                cell_rows = []
                shots.append({"data": np.array(row), "cells": cell_rows})
            elif label == "MEAN_GAMMA":
                shots[-1]["mean_gamma"] = row[0]
            elif label == "GEOM":
                shots[-1]["geom"] = np.array(row)
            elif label == "CELL":
                cell_rows.append(row)
            else:
                raise ValueError(label)
        for shot in shots:
            shot["cells"] = np.array(shot["cells"])
        return shots


    def direct_ex(shot: dict, gamma: float, order: int) -> float:
        """Integrate current uniform source cells, then gather to the witness.

        For each source cell and target mesh node, integrate the Coulomb kernel
        in the rest frame. Integrating r_x/r^3 along x first leaves the difference
        of two integrals of 1/r on the x faces. Numerical Gauss-Legendre integrates
        both faces over y,z. These nonsingular faces are half a cell from the node.
        The mesh stores charge per cell (C), despite its rho label. Thus
        rho_rest=Q_cell/(gamma*dx*dy*dz), E_x_lab=gamma*E_x_rest.
        The resulting lab E uses Q_cell/(dx*dy*dz) times the rest-frame cell integral.
        """
        data, geom, cells = shot["data"], shot["geom"], shot["cells"]
        minimum, delta = geom[:3], geom[3:]
        witness = data[6:9]
        u = (witness - minimum) / delta
        lower = np.floor(u).astype(int)
        fractions = u - lower
        cell_centers = cells[:, :3] - 1
        nodes, weights = leggauss(order)
        half_y = delta[1]/2
        half_z = gamma*delta[2]/2
        scale = np.array([delta[0], delta[1], gamma*delta[2]])
        field = 0.0
        # Integrate only occupied cells. Process all of them as one vector batch.
        for corners in itertools.product((0, 1), repeat=3):
            offset = np.array(corners)
            gather = float(np.prod(np.where(offset, fractions, 1-fractions)))
            if gather == 0:
                continue
            separation = (lower + offset - cell_centers)*scale
            xx = separation[:, 0, None, None]
            yy = separation[:, 1, None, None] + half_y*nodes[None, :, None]
            zz = separation[:, 2, None, None] + half_z*nodes[None, None, :]
            y2z2 = yy*yy + zz*zz
            left = 1/np.sqrt((xx-delta[0]/2)**2 + y2z2)
            right = 1/np.sqrt((xx+delta[0]/2)**2 + y2z2)
            integrals = np.einsum("ijk,j,k->i", left-right, weights, weights)*half_y*half_z
            field += gather*float(cells[:, 3] @ integrals)
        return COULOMB_NATIVE * field / float(np.prod(delta))


    def assert_close(actual, expected, rtol: float, label: str) -> None:
        if not np.allclose(actual, expected, rtol=rtol, atol=0):
            raise AssertionError(f"{label}: {actual} != {expected}; rtol={rtol}")


    def main() -> None:
        if len(os.sched_getaffinity(0)) != 1:
            raise RuntimeError("Run on one CPU using taskset.")
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert os.environ[var] == "1"
        defects = []
        for gi, gf, steps in ((4.0, 4.0, 16), (4.0, 16.0, 16), (16.0, 4.0, 16), (4.0, 16.0, 32)):
            production, _ = run(False, gi, gf, steps, 3e-22)
            recording, file = run(True, gi, gf, steps, 3e-22)
            assert_close(recording, production, 3e-12, "observation-only/native")
            assert production[3] == 27
            assert_close(production[1:3], MASS*np.sqrt(np.array([gi, gf])**2-1), 3e-14, "end momenta")
            shots = snapshots(file)
            assert len(shots) == steps+1
            actual, local, errors = [], [], []
            mesh_gammas, local_gammas = [], []
            convergence, gamma_errors, mean_gamma_errors, mean_local = [], [], [], []
            for j, shot in enumerate(shots):
                d = shot["data"]
                _, gm, ds, p0c, beta, delta_p, _, _, _, ex_native = d
                gamma = math.hypot(1, p0c*(1+delta_p)/MASS)
                assert_close(beta, math.sqrt(1-1/gamma**2), 1e-14, "local witness beta mass shell")
                # Half endpoint RF kicks and full interior kicks put the preceding
                # boundary RF kick before an aligned interior collective sample.
                analytic_gamma = gi+(gf-gi)*j/steps
                if j not in (0, steps):
                    analytic_gamma += (gf-gi)/(2*128)
                assert_close(gamma, analytic_gamma, 3e-13, "RF staircase energy")
                gamma_errors.append(abs(gamma/analytic_gamma-1))
                assert_close(shot["cells"][:, 3].sum(), 3e-22, 2e-14, "deposited total charge")
                ex_mesh = direct_ex(shot, gm, 48)
                ex_local = direct_ex(shot, gamma, 48)
                mean_gamma = shot["mean_gamma"]
                mean_gamma_errors.append(abs(mean_gamma/gamma-1))
                ex_mean = direct_ex(shot, mean_gamma, 48)
                mean_local.append(ex_mean*ds/(beta*mean_gamma**2))
                # A second quadrature resolution must converge, including the
                # smallest gamma at the entrance and after deceleration.
                ex_check = direct_ex(shot, gamma, 72)
                assert_close(ex_local, ex_check, 8e-13, "face quadrature convergence")
                convergence.append(abs(ex_local/ex_check-1))
                assert_close(ex_mesh, ex_native, 3e-12, "independent field/native FFT")
                errors.append(abs(ex_mesh/ex_native-1))
                # For a unit-charge comoving witness, magnetic cancellation gives
                # d(P_x c) = E_x,lab ds / (beta gamma^2), in eV.
                actual.append(ex_native*ds/(beta*gm**2))
                mesh_gammas.append(gm)
                local_gammas.append(gamma)
                local.append(ex_local*ds/(beta*gamma**2))
            assert_close(sum(actual), production[0], 4e-12, "impulse sum/native output")
            if gi == gf:
                assert_close(sum(local), production[0], 3e-12, "constant energy")
            endpoint_weights = np.ones(steps+1)
            endpoint_weights[[0, -1]] = 0.5
            small, _ = run(False, gi, gf, steps, 3e-23)
            assert_close(10*small[0], production[0], 4e-10, "weak-source linearity")
            mismatch = compare(
                f"gamma {gi:g}->{gf:g}, {steps} SC steps, physical impulse (eV/c); direct local Coulomb integral",
                production[0], sum(local), atol=0, rtol=3e-12,
            )
            if gi != gf:
                defects.append(mismatch)
            half_actual = float(endpoint_weights @ np.array(actual))
            half_local = float(endpoint_weights @ np.array(local))
            print(f"  mesh_gamma_range=({min(mesh_gammas):.8g}, {max(mesh_gammas):.8g}); "
                  f"local_gamma_range=({min(local_gammas):.8g}, {max(local_gammas):.8g})")
            print(f"  native/local={production[0]/sum(local):.16g}; "
                  f"half-endpoint impulse: observed={half_actual:.16e} expected={half_local:.16e} "
                  f"ratio={half_actual/half_local:.16g}; "
                  f"current-mean-speed expected={sum(mean_local):.16e}")
            print(f"  controls: 27 survivors; "
                  f"direct/FFT max_relative_error={max(errors):.4e}; "
                  f"quadrature_48/72={max(convergence):.4e}; "
                  f"RF_staircase={max(gamma_errors):.4e}; "
                  f"mean/reference_gamma={max(mean_gamma_errors):.4e}; "
                  f"record/native={abs(recording[0]/production[0]-1):.4e}; "
                  f"impulse_sum/native={abs(sum(actual)/production[0]-1):.4e}; "
                  f"tenfold_charge_scaling={abs(10*small[0]/production[0]-1):.4e}")
        verdict(all(defects))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

