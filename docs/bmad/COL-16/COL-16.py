"""COL-16: CSR3D momentum kicks leave the arrival-time coordinate inconsistent

When the experimental three-dimensional CSR wrapper changes a particle's speed, it updates β, the speed divided by the speed of light c, without rescaling the longitudinal coordinate z. Bmad defines z = βc(t₀ − t), where t₀ and t are reference and particle arrival times, so an instantaneous kick must preserve z/β. A 1 nC bunch at 10 MeV/c in a 1 cm bend leaves a 7.624 pm discrepancy between z and its stored arrival-time value. A zero-charge bunch satisfies the coordinate relation.

The script tracks 27 fixed particles through the complete CSR3D wrapper on a 4³ mesh at zero, 1
nC and 2 nC charge. At the exit it compares z with βc times the difference between reference and
stored particle times. BMAD_DIST selects the production build.
"""

from _bmad import compare, fortran, verdict
from _collective_contracts import rows

PROGRAM = """program clock
use csr_and_space_charge_mod
implicit none
type(lat_struct) lat
type(bunch_struct) bunch
type(coord_struct), allocatable :: centroid(:)
integer i,j,k,n
real(rp) charge
character(30) arg
logical err
call get_command_argument(1,arg)
read(arg,*) charge
call bmad_parser('lat.bmad',lat)
allocate(centroid(0:lat%n_ele_track))
call init_coord(centroid(0),lat%ele(0),upstream_end$,electron$)
call track_all(lat,centroid)
call reallocate_bunch(bunch,27)
space_charge_com%csr3d_mesh_size=[4,4,4]
n=0
do i=-1,1
 do j=-1,1
  do k=-1,1
   n=n+1
   call init_coord(bunch%particle(n),lat%ele(1),upstream_end$,electron$)
   bunch%particle(n)%vec(1)=i*0.001_rp
   bunch%particle(n)%vec(3)=j*0.002_rp
   bunch%particle(n)%vec(5)=k*0.001_rp+0.003_rp
   bunch%particle(n)%charge=charge/27
   bunch%particle(n)%t=lat%ele(0)%ref_time-bunch%particle(n)%vec(5)/bunch%particle(n)%beta/c_light
  enddo
 enddo
enddo
write(*,'(a)') 'READY: wrapper'
flush(6)
call track1_bunch_csr3d(bunch,lat%ele(1),centroid,err)
write(*,'(a,i0)') 'RESULT error ',merge(1,0,err)
write(*,'(a,6es25.16)') 'RESULT particle ',bunch%particle(1)%vec(5),bunch%particle(1)%beta, &
 bunch%particle(1)%vec(6),bunch%particle(1)%t, lat%ele(1)%ref_time, c_light
end program
"""
FILES = {
    "lat.bmad": """parameter[particle]=electron
parameter[p0c]=1e7
b: sbend, l=0.01, g=0.1, csr_ds_step=0.01
l: line=(b)
use, l
"""
}
defect = False
for charge in [0, 1e-9, 2e-9]:
    r = rows(fortran(PROGRAM, [str(charge)], files=FILES).log)
    if r["error"].tolist() != [0]:
        raise RuntimeError("CSR3D wrapper returned an error")
    z, beta, delta, time, ref_time, c = r["particle"]
    # Bmad defines z = beta*c*(t_ref-t). An instantaneous momentum kick
    # changes beta, while t and t_ref stay fixed; z must change with beta.
    expected = beta * c * (ref_time - time)
    mismatch = compare(
        f"charge={charge:g} C, exit z (m), beta*c*(t_ref-t)", z, expected, atol=2e-16
    )
    print(
        f"charge={charge:g} C: beta={beta:.16g}, delta={delta:.16g}, time error={(z / beta - c * (ref_time - time)) / c:.16g} s"
    )
    if charge == 0 and mismatch:
        raise RuntimeError("Zero-charge arrival-time control failed")
    defect |= mismatch
verdict(defect)
