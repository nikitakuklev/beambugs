"""Inline Bmad drivers and analytic maps for the SRDT, wiggler and map reproducers."""

import math

import numpy as np
from _bmad import fortran

SRDT_PROGRAM = """program srdt_probe
use bmad
use srdt_mod
implicit none
type(lat_struct) lat
type(coord_struct), allocatable :: orb(:)
type(summation_rdt_struct) s
integer status
logical err
call bmad_parser('lat.bmad',lat,err_flag=err)
if(err) error stop 'Parsing failed'
call twiss_and_track(lat,orb,status)
if(status /= ok$) error stop 'Twiss failed'
call srdt_calc(lat,s,1)
write(*,'(a,5es25.16)') 'TWISS ',lat%ele(1)%a%beta,lat%ele(1)%b%beta, &
 lat%ele(1)%x%eta,lat%ele(1)%a%phi,lat%ele(1)%b%phi
write(*,'(a,8es25.16)') 'SRDT ',s%h21000,s%h10110,s%h11001,s%h00111
end program
"""

SRDT_RING = """parameter[geometry] = closed
parameter[taylor_order] = 3
beginning[e_tot] = 1e9
qf: quadrupole, l = 0.5, k1 = 0.8
qd: quadrupole, l = 0.5, k1 = -0.8
b: sbend, l = 2, angle = 2*pi/16
d: drift, l = 0.5
cell: line = (qf, d, b, d, qd, d, b, d)
s: {spec}
ring: line = (s, 8*cell)
use, ring
"""

LSQ_PROGRAM = """program probe

use bmad
use srdt_mod

implicit none

type (lat_struct) lat
type (coord_struct), allocatable :: orb(:)
type (summation_rdt_struct) s0, s1, tot1, tot2
type (summation_rdt_struct), allocatable :: pe1(:), pe2(:)
complex(rp), allocatable :: cache(:,:,:)
real(rp), allocatable :: ls(:), val(:)
real(rp) q0(18), q1(18), cx, cy
integer, allocatable :: vars(:)
integer i, j, n, status, iu
character(200) file, mode, arg

call get_command_argument(1, file)
call get_command_argument(2, mode)
call bmad_parser (file, lat)
call twiss_and_track (lat, orb, status)
if (status /= ok$) error stop 'Twiss failed'

n = 0
do i = 1, lat%n_ele_track
  if (lat%ele(i)%name(1:1) == 'S') n = n + 1
enddo
allocate (vars(n), val(n))
n = 0
do i = 1, lat%n_ele_track
  if (lat%ele(i)%name(1:1) == 'S') then
    n = n + 1
    vars(n) = i
  endif
enddo

select case (mode)
case ('columns')
  call srdt_calc (lat, s0, 1)
  call to_q(s0, q0)
  write (*, '(a, 18es24.15)') 'B ', q0
  do j = 1, n
    val = 0;  val(j) = 1
    call set_vars(val)
    call srdt_calc (lat, s1, 1)
    call to_q(s1, q1)
    write (*, '(a, i3, 18es24.15)') 'A ', j, q1 - q0
  enddo

case ('lsq')
  call get_command_argument(3, arg);  read (arg, *) cx
  call get_command_argument(4, arg);  read (arg, *) cy
  select case (mode)
  case ('lsq');  call srdt_lsq_solution (lat, vars, ls, chrom_set_x_opt = cx, chrom_set_y_opt = cy)
  end select
  write (*, '(a, 24es24.15)') 'SOLN ', ls
  call set_vars(ls)
  call evaluate()

case ('apply', 'perele')
  call get_command_argument(3, arg)
  open (newunit = iu, file = arg, status = 'old')
  read (iu, *) val
  close (iu)
  call set_vars(val)
  if (mode == 'apply') then
    call evaluate()
  else
    call srdt_calc (lat, s1, 2, per_ele_out = pe1)
    call srdt_calc_with_cache (lat, s0, 2, cache = cache, per_ele_out = pe2)
    tot1 = sum_ele(pe1);  tot2 = sum_ele(pe2)
    write (*, '(a)') 'term      srdt_sums (srdt_calc)               sum(per_ele_out) srdt_calc        sum(per_ele_out) with_cache'
    call row('h11001', s1%h11001, tot1%h11001, tot2%h11001)
    call row('h00111', s1%h00111, tot1%h00111, tot2%h00111)
    call row('h20001', s1%h20001, tot1%h20001, tot2%h20001)
    call row('h00201', s1%h00201, tot1%h00201, tot2%h00201)
    call row('h10002', s1%h10002, tot1%h10002, tot2%h10002)
    call row('h21000', s1%h21000, tot1%h21000, tot2%h21000)
    call row('h30000', s1%h30000, tot1%h30000, tot2%h30000)
    call row('h10110', s1%h10110, tot1%h10110, tot2%h10110)
    call row('h10020', s1%h10020, tot1%h10020, tot2%h10020)
    call row('h10200', s1%h10200, tot1%h10200, tot2%h10200)
    call row('h22000', s1%h22000, tot1%h22000, tot2%h22000)
    call row('h31000', s1%h31000, tot1%h31000, tot2%h31000)
    call row('h20020', s1%h20020, tot1%h20020, tot2%h20020)
    write (*, '(a, 2es24.15)') 'with_cache sums h22000 ', s0%h22000
  endif
end select

contains

subroutine set_vars(v)
real(rp) v(:)
character(60) str
logical err
integer k
do k = 1, n
  if (lat%ele(vars(k))%key == sextupole$) then
    write (str, '(a, es24.16)') 'K2 = ', v(k)
  else
    write (str, '(a, es24.16)') 'K2L = ', v(k)
  endif
  call set_ele_attribute (lat%ele(vars(k)), str, err)
  if (err) error stop 'set_ele_attribute failed'
enddo
call lattice_bookkeeper (lat)
call twiss_and_track (lat, orb, status)
if (status /= ok$) error stop 'Twiss failed'
end subroutine

subroutine to_q(s, q)
type (summation_rdt_struct) s
real(rp) q(18)
q = [real(s%h11001), real(s%h00111), real(s%h20001), aimag(s%h20001), real(s%h00201), aimag(s%h00201), &
     real(s%h10002), aimag(s%h10002), real(s%h21000), aimag(s%h21000), real(s%h30000), aimag(s%h30000), &
     real(s%h10110), aimag(s%h10110), real(s%h10020), aimag(s%h10020), real(s%h10200), aimag(s%h10200)]
end subroutine

subroutine evaluate()
real(rp) chrom_a, chrom_b, de
logical err
call srdt_calc (lat, s1, 1)
call to_q(s1, q1)
de = 1e-6
call chrom_calc (lat, de, chrom_a, chrom_b, err)
if (err) error stop 'Chromaticity failed'
write (*, '(a, 18es24.15)') 'Q ', q1
write (*, '(a, 2es24.15)') 'CHROM ', chrom_a, chrom_b
end subroutine

function sum_ele(pe) result (t)
type (summation_rdt_struct) pe(:), t
t%h11001 = sum(pe%h11001); t%h00111 = sum(pe%h00111); t%h20001 = sum(pe%h20001)
t%h00201 = sum(pe%h00201); t%h10002 = sum(pe%h10002); t%h21000 = sum(pe%h21000)
t%h30000 = sum(pe%h30000); t%h10110 = sum(pe%h10110); t%h10020 = sum(pe%h10020)
t%h10200 = sum(pe%h10200); t%h22000 = sum(pe%h22000); t%h31000 = sum(pe%h31000)
t%h20020 = sum(pe%h20020)
end function

subroutine row(name, a, b, c)
character(*) name
complex(rp) a, b, c
write (*, '(a8, 3(2es25.16, 4x))') name, a, b, c
end subroutine

end program"""

THICK_RING = """parameter[geometry] = closed
beginning[e_tot] = 1e9
qf: quadrupole, l = 0.5, k1 = 0.8
qd: quadrupole, l = 0.5, k1 = -0.8
b: sbend, l = 2, angle = 2*pi/16
d: drift, l = 0.5
da: drift, l = 0.2
sf1: sextupole, l = 0.1, k2 = 0
sd1: sextupole, l = 0.1, k2 = 0
sx1: sextupole, l = 0.1, k2 = 0
c1: line = (qf, da, sf1, da, b, d, qd, da, sd1, da, b, da, sx1, da)
sf2: sextupole, l = 0.1, k2 = 0
sd2: sextupole, l = 0.1, k2 = 0
sx2: sextupole, l = 0.1, k2 = 0
c2: line = (qf, da, sf2, da, b, d, qd, da, sd2, da, b, da, sx2, da)
sf3: sextupole, l = 0.1, k2 = 0
sd3: sextupole, l = 0.1, k2 = 0
sx3: sextupole, l = 0.1, k2 = 0
c3: line = (qf, da, sf3, da, b, d, qd, da, sd3, da, b, da, sx3, da)
sf4: sextupole, l = 0.1, k2 = 0
sd4: sextupole, l = 0.1, k2 = 0
sx4: sextupole, l = 0.1, k2 = 0
c4: line = (qf, da, sf4, da, b, d, qd, da, sd4, da, b, da, sx4, da)
sf5: sextupole, l = 0.1, k2 = 0
sd5: sextupole, l = 0.1, k2 = 0
sx5: sextupole, l = 0.1, k2 = 0
c5: line = (qf, da, sf5, da, b, d, qd, da, sd5, da, b, da, sx5, da)
sf6: sextupole, l = 0.1, k2 = 0
sd6: sextupole, l = 0.1, k2 = 0
sx6: sextupole, l = 0.1, k2 = 0
c6: line = (qf, da, sf6, da, b, d, qd, da, sd6, da, b, da, sx6, da)
sf7: sextupole, l = 0.1, k2 = 0
sd7: sextupole, l = 0.1, k2 = 0
sx7: sextupole, l = 0.1, k2 = 0
c7: line = (qf, da, sf7, da, b, d, qd, da, sd7, da, b, da, sx7, da)
sf8: sextupole, l = 0.1, k2 = 0
sd8: sextupole, l = 0.1, k2 = 0
sx8: sextupole, l = 0.1, k2 = 0
c8: line = (qf, da, sf8, da, b, d, qd, da, sd8, da, b, da, sx8, da)
ring: line = (c1, c2, c3, c4, c5, c6, c7, c8)
use, ring
"""

THIN_RING = """parameter[geometry] = closed
beginning[e_tot] = 1e9
qf: quadrupole, l = 0.5, k1 = 0.8
qd: quadrupole, l = 0.5, k1 = -0.8
b: sbend, l = 2, angle = 2*pi/16
d: drift, l = 0.5
da: drift, l = 0.2
db: drift, l = 0.3
sf1: multipole, k2l = 0
sd1: multipole, k2l = 0
sx1: multipole, k2l = 0
c1: line = (qf, da, sf1, db, b, d, qd, da, sd1, db, b, da, sx1, db)
sf2: multipole, k2l = 0
sd2: multipole, k2l = 0
sx2: multipole, k2l = 0
c2: line = (qf, da, sf2, db, b, d, qd, da, sd2, db, b, da, sx2, db)
sf3: multipole, k2l = 0
sd3: multipole, k2l = 0
sx3: multipole, k2l = 0
c3: line = (qf, da, sf3, db, b, d, qd, da, sd3, db, b, da, sx3, db)
sf4: multipole, k2l = 0
sd4: multipole, k2l = 0
sx4: multipole, k2l = 0
c4: line = (qf, da, sf4, db, b, d, qd, da, sd4, db, b, da, sx4, db)
sf5: multipole, k2l = 0
sd5: multipole, k2l = 0
sx5: multipole, k2l = 0
c5: line = (qf, da, sf5, db, b, d, qd, da, sd5, db, b, da, sx5, db)
sf6: multipole, k2l = 0
sd6: multipole, k2l = 0
sx6: multipole, k2l = 0
c6: line = (qf, da, sf6, db, b, d, qd, da, sd6, db, b, da, sx6, db)
sf7: multipole, k2l = 0
sd7: multipole, k2l = 0
sx7: multipole, k2l = 0
c7: line = (qf, da, sf7, db, b, d, qd, da, sd7, db, b, da, sx7, db)
sf8: multipole, k2l = 0
sd8: multipole, k2l = 0
sx8: multipole, k2l = 0
c8: line = (qf, da, sf8, db, b, d, qd, da, sd8, db, b, da, sx8, db)
ring: line = (c1, c2, c3, c4, c5, c6, c7, c8)
use, ring
"""

MAP_PROGRAM = """program maps
use bmad
implicit none
type(lat_struct), target :: lat
type(ele_struct), pointer :: e
type(coord_struct) :: a,b,plus,minus,bp,bm
real(rp) :: m(6,6),fd(6,6),bd(4,4),rot(4,4),expect(4,4),k,l,q,c,s
integer i,j
logical err
call bmad_parser('lat.bmad',lat,err_flag=err)
if(err) error stop 'Parsing failed'
  do i=1,3
    e=>lat%ele(i)
    call init_coord(a,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp],e,upstream_end$)
    call make_mat6_taylor(e,a,b,err)
    if(err) error stop 'Taylor matrix failed'
    m=e%mat6
    do j=1,6
      plus=a; minus=a
      plus%vec(j)=plus%vec(j)+2e-8_rp
      minus%vec(j)=minus%vec(j)-2e-8_rp
      call track1(plus,e,lat%param,bp,err_flag=err)
      if(err.or.bp%state/=alive$) error stop 'Stencil failed'
      call track1(minus,e,lat%param,bm,err_flag=err)
      if(err.or.bm%state/=alive$) error stop 'Stencil failed'
      fd(:,j)=(bp%vec-bm%vec)/4e-8_rp
    enddo
    k=e%value(k1$); l=e%value(l$); q=sqrt(k)
    bd=0
    bd(1,1)=cos(q*l); bd(1,2)=sin(q*l)/q
    bd(2,1)=-q*sin(q*l); bd(2,2)=cos(q*l)
    bd(3,3)=cosh(q*l); bd(3,4)=sinh(q*l)/q
    bd(4,3)=q*sinh(q*l); bd(4,4)=cosh(q*l)
    c=cos(e%value(tilt$)); s=sin(e%value(tilt$))
    rot=0
    rot(1,1)=c; rot(1,3)=s; rot(2,2)=c; rot(2,4)=s
    rot(3,1)=-s; rot(3,3)=c; rot(4,2)=-s; rot(4,4)=c
    expect=matmul(transpose(rot),matmul(bd,rot))
    print '(a,i3,5es25.16)', 'TAYLOR ',i,m(2,3),fd(2,3),expect(2,3), &
        maxval(abs(fd(1:4,1:4)-expect)),maxval(abs(m(1:4,1:4)-expect))
    if(maxval(abs(fd(1:4,1:4)-expect))>2e-13_rp) error stop 'quadrupole oracle failed'
    if(i>1.and.maxval(abs(m(1:4,1:4)-expect))>2e-13_rp) error stop 'Taylor control failed'
  enddo

end program
"""

MAP_LATTICE = """parameter[geometry]=open
beginning[beta_a]=10
beginning[beta_b]=10
parameter[particle]=proton
parameter[p0c]=1.7e9
parameter[taylor_order]=1
q1: quadrupole, l=0.8, k1=0.7, tilt=0.31, fringe_type=none, &
    tracking_method=taylor, mat6_calc_method=taylor, taylor_map_includes_offsets=F
q2: q1, taylor_map_includes_offsets=T
q3: q1, tilt=0
channel: line=(q1,q2,q3)
use,channel
"""

MATRIX_PROGRAM = """program verify
use bmad
implicit none
type(lat_struct), target :: lat
type(ele_struct), pointer :: e
type(coord_struct) :: a, b, d, q, w
real(rp) :: v(6), m(6,6), mt(6,6), mtfine(6,6), fd(6), h, vals(6,4), extra
integer :: id, ie, mode, shift, j, k, n, ios, default_method
integer, parameter :: offsets(4) = [-2,-1,1,2]
logical :: err
character(200) :: lattice, cases
call get_command_argument(1,lattice)
call get_command_argument(2,cases)
call bmad_parser(trim(lattice),lat)
bmad_com%radiation_damping_on = .false.
bmad_com%radiation_fluctuations_on = .false.
bmad_com%spin_tracking_on = .false.
bmad_com%d_orb = [1d-7,1d-7,1d-7,1d-7,1d-5,1d-4]
write(*,'(a,2es26.17)') 'CONSTANTS ',c_light,mass_of(electron$)
open(10,file=trim(cases),status='old')
do
  read(10,*,iostat=ios) id,ie,mode,shift,extra,v
  if (ios < 0) exit
  if (ios /= 0) error stop 'Invalid input record'
  e => lat%ele(ie)
  if (e%tracking_method /= bmad_standard$) error stop 'Unexpected tracking method'
  if (.not. valid_mat6_calc_method(e,lat%param%particle,bmad_standard$)) error stop 'Unsupported matrix method'
  e%symplectify = .false.
  bmad_com%absolute_time_tracking = (mode == 1)
  bmad_com%absolute_time_ref_shift = (shift == 1)
  call init_coord(a,v,e,upstream_end$,lat%param%particle)
  a%t = a%t + extra
  call track1(a,e,lat%param,b,err_flag=err)
  if (err .or. b%state /= alive$) error stop 'Direct track failed'
  default_method = e%mat6_calc_method
  if (default_method /= auto$) error stop 'Unexpected parsed matrix default'
  e%mat6_calc_method = auto$
  call make_mat6(e,lat%param,a,d,err)
  if (err .or. d%state /= alive$) error stop 'Auto matrix failed'
  m = e%mat6
  e%mat6_calc_method = bmad_standard$
  call make_mat6(e,lat%param,a,d,err)
  if (err .or. d%state /= alive$) error stop 'Native matrix failed'
  if (maxval(abs(m-e%mat6)) > 0) error stop 'Auto differs from native'
  if (maxval(abs(b%vec-d%vec)) > 1d-14) error stop 'Native matrix changes orbit'
  e%mat6_calc_method = tracking$
  call make_mat6(e,lat%param,a,d,err)
  if (err .or. d%state /= alive$) error stop 'Tracking matrix failed'
  mt = e%mat6
  bmad_com%d_orb(6) = 1d-6
  call make_mat6(e,lat%param,a,d,err)
  if (err .or. d%state /= alive$) error stop 'Fine tracking matrix failed'
  mtfine = e%mat6
  bmad_com%d_orb(6) = 1d-4
  e%mat6_calc_method = default_method
  write(*,'(a,5i7,21es26.17)') 'CASE ',id,ie,mode,shift,default_method, &
    a%vec,b%vec,a%beta,a%t,e%value(ref_time_start$),e%value(p0c$),e%value(l$), &
    e%value(b_max$),e%value(l_period$),e%value(t_offset$),real(e%field_calc,rp)
  do j=1,6
    write(*,'(a,2i7,18es26.17)') 'MAT ',id,j,m(j,:),mt(j,:),mtfine(j,:)
  enddo
  do k=1,2
    do j=1,6
      h = 4d-6 / 4**(k-1)
      if (j == 5) h = 4d-4 / 4**(k-1)
      if (j == 6) h = 4d-4 / 4**(k-1)
      do n=1,4
        q = a
        q%vec(j) = q%vec(j) + offsets(n)*h
        ! Preserve the particle's reference clock while varying phase space.
        call convert_pc_to(q%p0c*(1+q%vec(6)),q%species,beta=q%beta)
        q%t = a%t - q%vec(5)/(c_light*q%beta) + a%vec(5)/(c_light*a%beta)
        call track1(q,e,lat%param,w,err_flag=err)
        if (err .or. w%state /= alive$) error stop 'Stencil track failed'
        vals(:,n) = w%vec
      enddo
      fd = (vals(:,1)-8*vals(:,2)+8*vals(:,3)-vals(:,4))/(12*h)
      write(*,'(a,3i7,7es26.17)') 'FD ',id,k,j,h,fd
    enddo
  enddo
enddo
close(10)
end program
"""

RADIATION_PROGRAM = """program independent_review
use bmad
use twiss_and_track_mod
use radiation_mod, only: release_rad_int_cache
implicit none
type(lat_struct) lat
type(coord_struct), allocatable :: orb(:)
type(normal_modes_struct) modes
type(rad_int_all_ele_struct) by_ele
character(300) filename, action
integer status, i, choice, cache

call get_command_argument(1, filename)
call get_command_argument(2, action)
if (len_trim(filename) == 0) error stop 'Lattice required'
global_com%exit_on_error = .true.
call bmad_parser(trim(filename), lat)
bmad_com%radiation_damping_on = .false.
bmad_com%radiation_fluctuations_on = .false.
call twiss_and_track(lat, orb, status, use_particle_start=.true.)
if (status /= ok$) error stop 'Twiss or reference trajectory failed'
if (any(orb(0:lat%n_ele_track)%state /= alive$)) error stop 'Reference lost'
write(*,'(a,5es26.17)') 'META ', lat%ele(0)%value(p0c$), mass_of(lat%param%particle), &
  lat%ele(0)%value(e_tot$), c_light, classical_radius(lat%param%particle)
write(*,'(a,6es26.17)') 'START ', orb(0)%vec
write(*,'(a,6es26.17)') 'END ', orb(lat%n_ele_track)%vec


  do choice = 1, 3
    select case(choice)
    case(1); cache = -2
    case(2); cache = -1
    case(3); cache = 0
    end select
    call radiation_integrals(lat, orb, modes, ix_cache=cache, rad_int_by_ele=by_ele)
    write(*,'(a,2i5,7es26.17)') 'TOTAL ', choice, cache, modes%synch_int(0:3), &
       modes%lin%i2_E4, modes%lin%i3_E7, modes%lin%sig_E1
    do i = 1, lat%n_ele_track
      write(*,'(a,2i5,7es26.17)') 'ELEMENT ', choice, i, by_ele%branch(0)%ele(i)%i0, &
        by_ele%branch(0)%ele(i)%i1, by_ele%branch(0)%ele(i)%i2, by_ele%branch(0)%ele(i)%i3, &
        lat%ele(i)%value(l$), lat%ele(i)%value(g_max$), lat%ele(i)%value(num_steps$)
    enddo
    if (choice == 3) call release_rad_int_cache(cache)
  enddo
end program
"""


def records(log, label):
    """Require numeric records with the given first token."""
    rows = [
        list(map(float, line.split()[1:]))
        for line in log.splitlines()
        if line.startswith(label + " ")
    ]
    if not rows:
        raise RuntimeError(f"Missing {label} records:\n{log}")
    return np.asarray(rows)


def srdt_case(spec):
    result = fortran(SRDT_PROGRAM, files={"lat.bmad": SRDT_RING.format(spec=spec)})
    twiss = records(result.log, "TWISS")
    values = records(result.log, "SRDT")
    if twiss.shape != (1, 5) or values.shape != (1, 8):
        raise RuntimeError("Incomplete SRDT output")
    return twiss[0], values[0].reshape(4, 2)


def matrices(lattice, vectors):
    cases = "".join(
        f"{i} {ie} 0 1 0 " + " ".join(map(str, v)) + "\n" for i, (ie, v) in enumerate(vectors, 1)
    )
    log = fortran(
        MATRIX_PROGRAM, ["lat.bmad", "cases.txt"], files={"lat.bmad": lattice, "cases.txt": cases}
    ).log
    case, mat, fd = (records(log, label) for label in ("CASE", "MAT", "FD"))
    if (
        case.shape != (len(vectors), 26)
        or mat.shape != (6 * len(vectors), 20)
        or fd.shape != (12 * len(vectors), 10)
    ):
        raise RuntimeError("Incomplete matrix output")
    results = []
    for i in range(1, len(vectors) + 1):
        row = case[i - 1, 5:]
        m = mat[mat[:, 0] == i, 2:].reshape(6, 3, 6).transpose(1, 0, 2)
        differences = fd[fd[:, 0] == i, 4:].reshape(2, 6, 6).transpose(0, 2, 1)
        results.append((row, m, differences))
    return results


def split_map(v, length, period, field, p0, helical):
    """Two cubic half kicks around the averaged Hill rotation; p = P_transverse/P0."""
    r = 1 + v[5]
    kappa = 299792458.0 * field / (math.sqrt(2) * p0)
    theta = kappa * length / r
    coefficient = -length * kappa**2 * (2 * math.pi / period) ** 2 / (3 * r)
    out = np.array(v[:4], dtype=complex)
    for i in (0, 2):
        if i == 0 and not helical:
            out[i] += length * out[i + 1] / r
            continue
        x, p = out[i : i + 2]
        p += coefficient * x**3
        if field == 0:
            x += length * p / r
        else:
            x, p = (
                np.cos(theta) * x + np.sin(theta) * p / kappa,
                -kappa * np.sin(theta) * x + np.cos(theta) * p,
            )
        p += coefficient * x**3
        out[i : i + 2] = [x, p]
    return out


def split_jacobian(v, length, period, field, p0, helical):
    jac = np.empty((4, 6))
    for col in range(6):
        perturbed = np.array(v, dtype=complex)
        perturbed[col] += 1e-25j
        jac[:, col] = split_map(perturbed, length, period, field, p0, helical).imag / 1e-25
    return jac


def radiation(lattice):
    log = fortran(RADIATION_PROGRAM, ["lat.bmad"], files={"lat.bmad": lattice}).log
    meta, total, end = (records(log, label) for label in ("META", "TOTAL", "END"))
    if meta.shape != (1, 5) or total.shape != (3, 9) or end.shape != (1, 6):
        raise RuntimeError("Incomplete radiation output")
    np.testing.assert_array_equal(total[:, 2:], np.tile(total[0, 2:], (3, 1)))
    return meta[0], total[0, 2:], end[0]
