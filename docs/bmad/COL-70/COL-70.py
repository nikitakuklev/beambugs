"""COL-70: Entrance-aperture rejects contribute to the first slice space-charge kick

With slice space charge enabled, a finite element applies its first half kick before rejecting particles at its entrance aperture. Surviving particles therefore interact with charge that should already have been removed. In a 2 mm drift at 6 MeV/c, a symmetric 10 pC electron bunch receives a central normalized horizontal kick of −5.450×10⁻⁹ instead of zero when one outlying source particle is rejected. Reflection symmetry supplies the expected zero force, and placing the same aperture on a separate entrance marker restores exact agreement with the prefiltered bunch. The erroneous contribution scales with the first half-step length.

Track a deterministic reflection-symmetric 175-electron grid plus one outlying source through an entrance aperture. Compare its central kick with the analytic zero-force invariant, a prefiltered bunch, and a separate entrance marker over four step counts. Disabled-collective and charge-scaling controls isolate source selection.
"""

import numpy as np
from _bmad import compare, fortran, verdict
from _slicing import close, record

LATTICE = """no_digested
parameter[particle]=electron
parameter[p0c]=6e6
a: marker,x_limit=0.002,aperture_at=entrance_end
d: drift,l=0.002,space_charge_method=slice,x_limit=0.002,aperture_at=entrance_end
l1: line=(a,d)
use,l1
"""

PROGRAM = """
! Mirror symmetry is the transverse-force oracle; an outlier is outside the entrance aperture.
program confirm_aperture
 use bmad
 use beam_mod
 implicit none
 type(lat_struct) lat
 type(bunch_struct) bunch
 type(coord_struct) centroid(0:3)
 integer mode,nstep,i,j,k,n,ie,witness,solver
 logical err
 real(rp) source_charge
 read(*,*) mode,nstep,solver,source_charge
 call bmad_parser('lat.bmad',lat)
 call init_coord(centroid(0),lat%ele(0),downstream_end$)
 do ie=1,3
   call track1(centroid(ie-1),lat%ele(ie),lat%param,centroid(ie))
 enddo
 space_charge_com%n_bin=12
 space_charge_com%particle_bin_span=2
 space_charge_com%n_shield_images=0
 space_charge_com%lsc_kick_transverse_dependence=.false.
 space_charge_com%ds_track_step=0.002_rp/nstep
 bmad_com%csr_and_space_charge_on=.true.
 if(solver==1) bmad_com%csr_and_space_charge_on=.false.
 if(mode/=2) then
   lat%ele(1)%value(x1_limit$)=0
   lat%ele(1)%value(x2_limit$)=0
 endif
 if(mode==3) then
   lat%ele(2)%value(x1_limit$)=0
   lat%ele(2)%value(x2_limit$)=0
 endif
 if(mode==4) lat%ele(2)%aperture_at=exit_end$
 call reallocate_bunch(bunch,176)
 n=0
 do i=-2,2
 do j=-2,2
 do k=-3,3
   n=n+1
   call init_coord(bunch%particle(n), &
      [i*0.0005_rp,0.0_rp,j*0.0005_rp,0.0_rp,k*0.0005_rp,0.0_rp], &
      lat%ele(1),upstream_end$,electron$)
   bunch%particle(n)%charge=source_charge/175
   if(i==0.and.j==0.and.k==0) witness=n
 enddo
 enddo
 enddo
 call init_coord(bunch%particle(176), &
    [0.006_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp],lat%ele(1),upstream_end$,electron$)
 bunch%particle(176)%charge=source_charge/175
 if(mode==0) bunch%particle(176)%state=lost$
 bunch%n_live=count(bunch%particle%state==alive$)
 bunch%charge_tot=176*source_charge/175
 bunch%charge_live=bunch%n_live*source_charge/175
 call track1_bunch(bunch,lat%ele(1),err,centroid)
 if(err) error stop 'marker failed'
 call track1_bunch(bunch,lat%ele(2),err,centroid)
 if(err) error stop 'drift failed'
 write(*,'(a,3i5,2es26.17e3)') 'LIVE ',count(bunch%particle%state==alive$), &
    bunch%particle(176)%state,witness,bunch%particle(witness)%p0c,bunch%particle(176)%s
 do n=1,175
   write(*,'(a,i4,6es26.17e3)') 'PART ',n,bunch%particle(n)%vec
 enddo
end program
"""


def run(mode, steps, collective_off=0, charge=1e-11):
    log = fortran(
        PROGRAM, files={"lat.bmad": LATTICE}, stdin=f"{mode} {steps} {collective_off} {charge}\n"
    ).log
    state = record(log, "LIVE", 5)
    rows = [line.split()[2:] for line in log.splitlines() if line.startswith("PART ")]
    if len(rows) != 175 or any(len(row) != 6 for row in rows):
        raise RuntimeError(f"Expected 175 complete particle records:\n{log}")
    particles = np.array(rows, dtype=float)
    if not np.isfinite(particles).all():
        raise RuntimeError("Nonfinite survivor coordinates")
    if state[0] != (176 if mode == 3 else 175) or state[2] != 88 or state[3] != 6e6:
        raise RuntimeError(f"Unexpected live count, central witness or reference momentum: {state}")
    if mode == 1 and (state[1] == 1 or state[4] != 0):
        raise RuntimeError(f"Outlier was not removed at the entrance: {state}")
    return particles


def main():
    discrepancies = []
    for steps in (1, 2, 4, 8):
        clean, entrance, marker, all_live = [run(mode, steps) for mode in range(4)]
        # Reflection in x and y fixes the central witness, so its transverse force is zero.
        close(clean[87, :4], 0, 1e-22, "Reflection-symmetry control")
        close(marker, clean, 0, "Every survivor: marker/prefilter equivalence")
        discrepancy = compare(
            f"{steps} steps central normalized px; reflection symmetry",
            entrance[87, 1],
            0,
            atol=1e-22,
        )
        discrepancies.append(discrepancy)
        print(
            f"{steps} steps controls: prefiltered_px={clean[87, 1]:.17g}, marker/prefilter_all_coordinates_max_error=0; 175 survivors, outlier lost at s=0, p0c=6000000 eV"
        )
        if discrepancy:
            # Only the initial half step sees the rejected charge: weight ds/2=L/(2*n).
            fraction = entrance[87, 1] / all_live[87, 1]
            close(fraction, 1 / (2 * steps), 1e-6, "Initial-half-step source signature")
            print(
                f"{steps} steps erroneous-source fraction={fraction:.17g} expected_first_half_step={1 / (2 * steps):.17g}"
            )
    for mode in (0, 1, 2):
        off = run(mode, 1, collective_off=1)
        close(off[87, :4], 0, 0, "Disabled-collective control")
    weak = run(1, 1, charge=1e-12)[87, 1]
    strong = run(1, 1, charge=1e-11)[87, 1]
    close(weak * 10, strong, 1e-14, "Linear charge scaling")
    print(
        f"charge scaling: 1 pC px={weak:.17g}, 10 pC px={strong:.17g}; three disabled-collective controls exactly zero"
    )
    verdict(any(discrepancies))


if __name__ == "__main__":
    main()
