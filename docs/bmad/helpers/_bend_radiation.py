"""Shared inclined-face radiation client and strict native-output parsing."""

import numpy as np
from _bmad import fortran

LOCAL_PROGRAM = r"""program local_probe
use bmad
use radiation_mod
use rad_6d_mod
implicit none
type(lat_struct), target :: lat
type(ele_struct), target :: slice
type(ele_struct), pointer :: e
type(coord_struct) p, q, a, b, zero, zout
type(rad_map_struct) rm
real(rp) v(6), loss, kd, h, yy, ee
integer n, i, j, axis, refmode, nn, ie
logical err
call bmad_parser('lat.bmad',lat,err_flag=err)
if(err) error stop 'parse'
read(*,*) h
bmad_com%radiation_damping_on=.false.
bmad_com%radiation_fluctuations_on=.false.
bmad_com%radiation_zero_average=.false.
kd=2.0_rp/3*classical_radius(electron$)*(lat%ele(1)%value(e_tot$)/m_electron)**3
write(*,'(a,es26.17)') 'KD ',kd
call init_coord(zero,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp],lat%ele(1),upstream_end$)
call make_mat6(lat%ele(1),lat%param,zero,zout,err)
if(err) error stop 'matrix'
call rad1_damp_and_stoc_mats(lat%ele(1),.false.,zero,zout,rm,1e-12_rp,1e-12_rp,err)
if(err) error stop 'direct rad1'
do i=1,6
write(*,'(a,i2,6es26.17)') 'DIRECT ',i,rm%damp_dmat(i,:)
enddo
! Reflection parity is an exact invariant, independent of the path quadrature.
! Re-expanding at the tested orbit must not remove first-order physical loss.
do nn=1,3
select case(nn)
case(1); n=1
case(2); n=8
case(3); n=64
end select
do refmode=0,2
do axis=1,3,2
do j=-1,1,2
v=0; v(axis)=j*h
call init_coord(p,v,lat%ele(1),upstream_end$)
loss=0
bmad_com%radiation_damping_on=.false.
do i=1,n
call element_slice_iterator(lat%ele(1),lat%param,i,n,slice)
e=>slice
! Refmode 0: zero-centered map. 1: explicit reference. 2: cached map orbit.
if(refmode==0) then
call init_coord(zero,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp],e,upstream_end$)
call make_mat6(e,lat%param,zero,zout,err)
else
call make_mat6(e,lat%param,p,q,err)
endif
if(err) error stop 'slice matrix'
if(refmode==1) then
call radiation_map_setup(e,err,p)
else
call radiation_map_setup(e,err)
endif
if(err) error stop 'setup'
call track1(p,e,lat%param,q,err_flag=err)
if(err.or.q%state/=alive$) error stop 'off track'
a=p; b=q
bmad_com%radiation_damping_on=.true.
call track1_radiation(a,e,start_edge$)
call track1_radiation(b,e,end_edge$)
if(a%state/=alive$.or.b%state/=alive$) error stop 'kick'
loss=loss+p%vec(6)-a%vec(6)+q%vec(6)-b%vec(6)
bmad_com%radiation_damping_on=.false.
p=q
enddo
write(*,'(a,4i5,es26.17)') 'LOSS ',n,refmode,axis,j,loss
enddo
enddo
enddo
enddo
end program
"""


def lattice(edge: float, method: str = "bmad_standard") -> str:
    """A pure, untilted 3 GeV/c dipole, with no cache files."""
    return f"""no_digested
parameter[geometry]=open
parameter[particle]=electron
parameter[p0c]=3e9
b:sbend,l=1,g=.2,e1={edge},e2={edge},fringe_type=full,fringe_at=both_ends,tracking_method={method},num_steps=128
beamline:line=(b)
use,beamline
"""


def local(edge: float, h: float = 0.001):
    """Return kd, the full-element damping matrix and all frozen endpoint losses.

    Each endpoint kick acts on a copy of its radiation-free trajectory. This
    removes feedback while retaining both physical half-map weights. Reference
    modes 0/1/2 mean centered, explicitly displaced, and cached displaced.
    """
    log = fortran(LOCAL_PROGRAM, files={"lat.bmad": lattice(edge)}, stdin=f"{h}\n").log
    rows = {}
    direct = {}
    coefficients = []
    for line in log.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "KD":
            coefficients.append(float(t[1]))
        elif t[0] == "LOSS":
            key = tuple(map(int, t[1:5]))
            if len(t) != 6 or key in rows:
                raise RuntimeError(f"Malformed or duplicate loss record: {line}")
            rows[key] = float(t[5])
        elif t[0] == "DIRECT":
            i = int(t[1])
            if len(t) != 8 or i in direct:
                raise RuntimeError(f"Malformed or duplicate matrix row: {line}")
            direct[i] = list(map(float, t[2:]))
    required = {
        (n, mode, axis, sign)
        for n in [1, 8, 64]
        for mode in range(3)
        for axis in [1, 3]
        for sign in [-1, 1]
    }
    if set(rows) != required or set(direct) != set(range(1, 7)) or len(coefficients) != 1:
        raise RuntimeError(f"Incomplete native output:\n{log}")
    kd = coefficients[0]
    matrix = np.array([direct[i] for i in range(1, 7)])
    if not (
        kd > 0
        and np.isfinite(kd)
        and np.isfinite(matrix).all()
        and np.isfinite(list(rows.values())).all()
    ):
        raise RuntimeError("Nonfinite native output or invalid damping coefficient")
    return kd, matrix, rows


def difference(rows, n: int, mode: int, axis: int) -> float:
    """Positive fractional momentum loss at +h minus that at -h."""
    return rows[n, mode, axis, 1] - rows[n, mode, axis, -1]


def slope(rows, kd: float, n: int, mode: int, h: float = 0.001) -> float:
    """Central x derivative of loss/kd, in m^-2."""
    return difference(rows, n, mode, 1) / (2 * h * kd)
