"""Inline field and photon fixtures for the s3-19 conversion reproductions."""

import cmath
import math

import numpy as np
from _bmad import fortran


def execute(program, args, files):
    output = fortran(program, args, files=files).log
    if "DONE" not in output.splitlines():
        raise RuntimeError(output)
    return output


def records(output, prefixes):
    result = {}
    for line in output.splitlines():
        tokens = line.split()
        if tokens and tokens[0].startswith(prefixes):
            if tokens[0] in result:
                raise RuntimeError(f"Duplicate record {tokens[0]}")
            values = np.array(list(map(float, tokens[1:])))
            if not np.isfinite(values).all():
                raise RuntimeError(f"Nonfinite record {line}")
            result[tokens[0]] = values
    if not result:
        raise RuntimeError(output)
    return result


def control(actual, expected, atol=2e-14):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (
        actual.shape != expected.shape
        or not actual.size
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
        or not math.isfinite(atol)
        or atol < 0
    ):
        raise AssertionError((actual, expected))
    error = float(np.max(np.abs(actual - expected)))
    if error > atol:
        raise AssertionError((error, atol, actual, expected))
    return error


def field_data(case):
    return records(execute(FIELD, [str(case)], {"verify.bmad": make_lattice()}), ("G_", "F_", "W_"))


def photon_data(case):
    return execute(PHOTON, [str(case)], {"checks.bmad": PHOTON_LATTICE})


def rotate(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def rows(output, label):
    result = [
        line.split()[1:]
        for line in output.splitlines()
        if line.split() and line.split()[0] == label
    ]
    if not result:
        raise RuntimeError(f"Missing {label} records")
    return result


def fresnel_stack(wavelength, chi1, chi2, d1, d2, count, angle, pol):
    """Match tangential E,H at each interface, recursively from a semi-infinite bottom layer.

    Y=kz/epsilon for pi and Y=kz for sigma. At interface i,j,
    E_ref/E_inc=(Y_i-Y_j)/(Y_i+Y_j). Each interior round trip adds exp(2 i kz d).
    This uses individual interfaces, independent of Bmad's analytic periodic-cell formula.
    """
    chi = [0, chi1, chi2]
    kz = [2 * math.pi / wavelength * cmath.sqrt(math.sin(angle) ** 2 + c) for c in chi]
    admittance = [kz[i] / (1 + chi[i]) if pol == "pi" else kz[i] for i in range(3)]
    media = [0] + [2, 1] * count
    thickness = [0, d1, d2]
    reflected = 0j
    for pos in range(len(media) - 2, -1, -1):
        i, j = media[pos : pos + 2]
        interface = (admittance[i] - admittance[j]) / (admittance[i] + admittance[j])
        from_deeper = reflected * cmath.exp(2j * kz[j] * thickness[j])
        reflected = (interface + from_deeper) / (1 + interface * from_deeper)
    return reflected


def make_lattice():
    def grid(geometry, spacing, points):
        return (
            "grid_field={geometry="
            + geometry
            + ",field_type=magnetic,interpolation_order=1,dr=("
            + spacing
            + "),{\n"
            + ",\n".join(points)
            + "}}"
        )

    a = grid(
        "xyz",
        "1,1,1",
        [f"{x} {y} {z}: 0.7 -0.4 0.2" for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (0, 1)],
    )
    b = grid(
        "rotationally_symmetric_rz",
        "1,1",
        [f"{r} {z}: 0 0 1.3" for r in (0, 1) for z in (0, 1)],
    )
    elements = [
        f"ga: quadrupole,l=1,field_calc=fieldmap,{a}",
        f"gb: quadrupole,l=1,field_calc=fieldmap,{b}",
        f"gab: quadrupole,l=1,field_calc=fieldmap,{a},{b}",
        f"gba: quadrupole,l=1,field_calc=fieldmap,{b},{a}",
    ]
    angles = ("0", "pi/6", "-pi/6")
    for group, attributes in (
        ("qe", "k1=0,b1_elec=1e5,scale_multipoles=F"),
        ("qk", "k1=0,hkick=.001"),
        ("qm", "k1=.5"),
    ):
        elements.extend(
            f"{group}{j}: quadrupole,l=1,tilt={angle},{attributes},fringe_type=none"
            for j, angle in enumerate(angles)
        )
    elements.extend(
        [
            "w0: wiggler,l=1,l_period=.2,b_max=1,kx=0,field_calc=planar_model",
            "w1: wiggler,l=1,l_period=.2,b_max=1,kx=13,field_calc=planar_model",
        ]
    )
    header = "parameter[geometry]=open\nparameter[particle]=electron\nparameter[p0c]=1e9\nbeginning[beta_a]=10\nbeginning[beta_b]=10\n"
    names = [element.split(":", 1)[0] for element in elements]
    content = header + "\n".join(elements) + "\nlat: line=(" + ",".join(names) + ")\nuse,lat\n"
    return content


FIELD = r"""program field_check
use bmad
use em_field_mod
implicit none
type(lat_struct), target :: lat
type(ele_struct), pointer :: e
type(coord_struct) o, ob
type(em_field_struct) f
integer n, i, j, k, m
real(rp) x, y, s, c, sn, xb, yb, h, sample_s
real(rp), parameter :: steps(4) = [-2.0_rp,-1.0_rp,1.0_rp,2.0_rp]
character(16) arg
character(100) tag, prefix
logical err
call get_command_argument(1,arg)
read(arg,*) n
call bmad_parser('verify.bmad',lat,err_flag=err)
if(err) error stop 'parse failed'
if(n == 1) then
  do j=0,2
    x=0
    if(j==1) x=1e-9_rp
    if(j==2) x=0.001_rp
    do i=1,4
      e=>lat%ele(i)
      call init_coord(o,[x,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp],e,upstream_end$)
      call em_field_calc(e,lat%param,0.37_rp,o,.true.,f,err_flag=err)
      if(err) error stop 'grid field error'
      write(tag,'(a,i0,a,i0)') 'G_',i,'_',j
      call emit(tag,f%b)
    enddo
  enddo
else if(n==4 .or. n==5) then
  x=0.003_rp; y=0.002_rp; s=0.37_rp
  do j=0,2
    i=5+j
    if(n==5) i=8+j
    e=>lat%ele(i)
    call init_coord(o,[x,0.0_rp,y,0.0_rp,0.0_rp,0.0_rp],e,upstream_end$)
    call em_field_calc(e,lat%param,s,o,.false.,f,err_flag=err)
    if(err) error stop 'lab field error'
    write(prefix,'(a,i0)') 'F_',j
    call emit(trim(prefix)//'_PARAM',[e%value(tilt_tot$),e%value(p0c$),e%value(l$),real(charge_of(e%ref_species),rp),c_light])
    call emit(trim(prefix)//'_LABE',f%e)
    call emit(trim(prefix)//'_LABB',f%b)
    c=cos(e%value(tilt_tot$)); sn=sin(e%value(tilt_tot$))
    xb=c*x+sn*y; yb=-sn*x+c*y
    ob=o; ob%vec(1)=xb; ob%vec(3)=yb
    call em_field_calc(e,lat%param,s,ob,.true.,f,err_flag=err)
    if(err) error stop 'body field error'
    call emit(trim(prefix)//'_BODYE',f%e)
    call emit(trim(prefix)//'_BODYB',f%b)
    ! Independent magnetic quadrupole control at exactly the same angle and point.
    e=>lat%ele(11+j)
    call em_field_calc(e,lat%param,s,o,.false.,f,err_flag=err)
    if(err) error stop 'magnetic control error'
    call emit(trim(prefix)//'_MAG',f%b)
  enddo
else if(n==6) then
  x=0.003_rp; y=0.002_rp; s=0.37_rp; h=1e-5_rp
  do i=14,15
    e=>lat%ele(i)
    call init_coord(o,[x,0.0_rp,y,0.0_rp,0.0_rp,0.0_rp],e,upstream_end$)
    call em_field_calc(e,lat%param,s,o,.true.,f,calc_dfield=.true.,err_flag=err)
    if(err) error stop 'wiggler error'
    write(prefix,'(a,i0)') 'W_',i-14
    call emit(trim(prefix)//'_PARAM',[e%value(kx$),e%value(l_period$),e%value(b_max$),e%value(l$),x,y,s,h])
    call emit(trim(prefix)//'_B',f%b)
    do j=1,3
      write(tag,'(a,a,i0)') trim(prefix),'_D_',j
      call emit(tag,f%db(:,j))
    enddo
    do j=1,3
      do k=1,4
        ob=o; sample_s=s
        if(j<3) then
          m=2*j-1
          ob%vec(m)=ob%vec(m)+steps(k)*h
        else
          sample_s=s+steps(k)*h
        endif
        call em_field_calc(e,lat%param,sample_s,ob,.true.,f,err_flag=err)
        if(err) error stop 'finite difference sample error'
        write(tag,'(a,a,i0,a,i0)') trim(prefix),'_S_',j,'_',k
        call emit(tag,f%b)
      enddo
    enddo
  enddo
else
  error stop 'unknown field case'
endif
print '(a)', 'DONE'
contains
subroutine emit(label,values)
character(*) label
real(rp) values(:)
write(*,'(a,1x,*(es25.16e3,1x))') trim(label),values
end subroutine
end program
"""

PHOTON = r"""program verify_photons
use bmad
use track1_photon_mod
use xraylib, only: crystal_struct, Crystal_GetCrystal
implicit none
type(lat_struct), target :: lat
type(ele_struct), pointer :: ele
type(coord_struct) :: ray, result_ray, surface_ray, origin
type(crystal_param_struct) :: cp
type(crystal_struct), pointer :: silicon
real(rp) :: z, dz(2), h, vals(4), fd, x, y, sx, sy, zc, ab, ph, amp, phase, p_factor
real(rp) :: w(3,3), wavelength, theta, xi_re, drp(3), drs(3), fields(2), phases(2), hbar(3), incoming(2)
real(rp) :: mean_re, mean_im, eta_delta
complex(rp) :: chi1, chi2
integer :: i, j, k, n, state, count_pos, count_neg
logical :: err
character(32) :: test_case
call get_command_argument(1,test_case)
call bmad_parser('checks.bmad',lat,err_flag=err)
if(err) error stop 'parse failed'
select case(trim(test_case))
case('1')
  ele=>lat%ele(8)
  wavelength=c_light*h_planck/9000.0_rp
  chi1=-conjg(ele%photon%material%f0_m1)*r_e*wavelength**2/(pi*ele%value(v1_unitcell$))
  chi2=-conjg(ele%photon%material%f0_m2)*r_e*wavelength**2/(pi*ele%value(v2_unitcell$))
  print '(a,7es26.17)', 'ML_CONFIG ',wavelength,chi1,chi2,ele%value(d1_thickness$),ele%value(d2_thickness$)
  do n=1,3
    ele%value(n_cell$)=real(2**n-1,rp)
    do j=1,3
      if(j==1) theta=.02_rp
      if(j==2) theta=.25_rp
      if(j==3) theta=pi/4
      call init_coord(ray,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,1.0_rp],ele,upstream_end$,photon$)
      ! Inverse of the entrance surface rotation: incidence is exactly theta.
      ray%vec(2)=sin(theta-ele%value(graze_angle$))
      ray%vec(6)=cos(theta-ele%value(graze_angle$))
      ray%field=1; ray%phase=0
      call track1(ray,ele,lat%param,result_ray)
      print '(a,2i4,5es26.17,i4)', 'ML ',nint(ele%value(n_cell$)),j,theta,result_ray%field,result_ray%phase,result_ray%state
    enddo
  enddo
case('2','6')
  ! Establish recognized material independently of the helper's error flag.
  silicon=>Crystal_GetCrystal('Si')
  if(.not.associated(silicon)) error stop 'silicon crystal lookup failed'
  if(silicon%volume<=0) error stop 'invalid silicon crystal volume'
  print '(a)', 'SILICON_RECOGNIZED'
  do i=5,7
    ele=>lat%ele(i)
    call photon_absorption_and_phase_shift(ele%wall3d(1)%section(1)%material,9000.0_rp,ab,ph,err)
    print '(a,i4,l2,2es26.17)', 'ABS ',i,err,ab,ph
    do j=1,2
      call init_coord(ray,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,1.0_rp],ele,upstream_end$,photon$)
      if(j==2) ray%vec(2:6:2)=[.6_rp,0.0_rp,.8_rp]
      ray%field=[.6_rp,.8_rp]; ray%phase=0
      surface_ray=ray
      call offset_photon(ele,surface_ray,set$)
      call track1(ray,ele,lat%param,result_ray)
      print '(a,3i4,5es26.17)', 'MASK ',i,j,result_ray%state,surface_ray%vec(6),result_ray%field,result_ray%phase
    enddo
  enddo
case('3')
  ele=>lat%ele(4)
  x=.093_rp; y=-.101_rp
  z=checked_z(x,y,dz)
  print '(a,5es26.17)', 'SEG ',x,y,z,dz
  associate(pt=>ele%photon%segmented%pt(1,-1))
    print '(a,5es26.17)', 'PLANE ',pt%x0,pt%y0,pt%z0,pt%dz_dx,pt%dz_dy
  end associate
  h=1e-5_rp
  sx=(checked_z(x+h,y)-checked_z(x-h,y))/(2*h)
  sy=(checked_z(x,y+h)-checked_z(x,y-h))/(2*h)
  print '(a,2es26.17)', 'SEG_FD ',sx,sy
  do i=-1,1,2
    do j=-1,1,2
      x=.08_rp+i*(.04_rp-1e-12_rp); y=-.12_rp+j*(.06_rp-1e-12_rp)
      z=checked_z(x,y)
      print '(a,3es26.17)', 'CORNER ',x,y,z
    enddo
  enddo
  call init_coord(ray,[.093_rp,0.0_rp,-.101_rp,0.0_rp,-.1_rp,1.0_rp],ele,upstream_end$,photon$)
  call track_to_surface(ele,ray,lat%param,w)
  print '(a,i4,6es26.17)', 'SEG_HIT ',ray%state,ray%vec
case('4')
  do i=1,3
    ele=>lat%ele(i)
    x=.17_rp; h=1e-4_rp
    z=checked_z(x,0.0_rp,dz)
    do j=1,4
      k=j-3
      if(j>2) k=j-2
      vals(j)=checked_z(x+k*h,0.0_rp)
    enddo
    fd=(vals(1)-8*vals(2)+8*vals(3)-vals(4))/(12*h)
    call init_coord(ray,[x,0.0_rp,0.0_rp,0.0_rp,0.0_rp,1.0_rp],ele,upstream_end$,photon$)
    call to_surface_coords(ray,ele,result_ray)
    if(result_ray%state/=alive$) error stop 'detector coordinate control lost photon'
    print '(a,i4,5es26.17)', 'DETECTOR ',i,x,z,dz(1),fd,result_ray%vec(1)
  enddo
case('5')
  ele=>lat%ele(9)
  cp%wavelength=c_light*h_planck/9000.0_rp
  cp%cap_gamma=r_e*cp%wavelength**2/(pi*ele%value(v_unitcell$))
  theta=asin(cp%wavelength/(2*ele%value(d_spacing$)))
  cp%old_vvec=[sin(theta),0.0_rp,cos(theta)]
  cp%new_vvec=[-sin(theta),0.0_rp,cos(theta)]
  cp%b_eff=1; cp%dtheta_sin_2theta=0
  lat%photon_type=coherent$
  count_pos=0; count_neg=0; mean_re=0; mean_im=0
  do i=1,128
    state=alive$
    call crystal_diffraction_field_calc(cp,ele,0.0_rp,lat%param,1.0_rp,.true.,amp,phase,state,drs)
    if(state/=alive$) error stop 'zero-thickness helper lost photon'
    print '(a,i4,2es26.17)', 'ZERO_SAMPLE ',i,amp,phase
    if(amp*cos(phase)>0) count_pos=count_pos+1
    if(amp*cos(phase)<0) count_neg=count_neg+1
    mean_re=mean_re+amp*cos(phase)/128
    mean_im=mean_im+amp*sin(phase)/128
  enddo
  print '(a,2i5,2es26.17)', 'ZERO_LIMIT ',count_pos,count_neg,mean_re,mean_im
  do i=1,12
    call init_coord(ray,[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,1.0_rp],ele,upstream_end$,photon$)
    ray%field=[0.0_rp,1.0_rp]; ray%phase=0
    call track1(ray,ele,lat%param,result_ray)
    print '(a,i3,3es26.17)', 'FINITE ',result_ray%state,result_ray%field(2),result_ray%phase(2),result_ray%field(2)*cos(result_ray%phase(2))
  enddo
  lat%photon_type=incoherent$
  state=alive$
  call crystal_diffraction_field_calc(cp,ele,0.0_rp,lat%param,1.0_rp,.true.,amp,phase,state,drs)
  if(state/=alive$) error stop 'incoherent helper lost photon'
  print '(a,2es26.17)', 'INCOHERENT_ZERO ',amp,phase
case('7')
  ele=>lat%ele(10)
  do k=1,3
    call init_coord(origin,[0.0_rp,real(k-2,rp)*2e-5_rp,0.0_rp,0.0_rp,0.0_rp,sqrt(1.0_rp-(real(k-2,rp)*2e-5_rp)**2)],ele,upstream_end$,photon$)
    origin%field=[0.0_rp,1.0_rp]; origin%phase=0
    surface_ray=origin
    call offset_photon(ele,origin,set$)
    cp%wavelength=c_light*h_planck/origin%p0c
    cp%cap_gamma=r_e*cp%wavelength**2/(pi*ele%value(v_unitcell$))
    hbar=ele%photon%material%h_norm*cp%wavelength/ele%value(d_spacing$)
    cp%old_vvec=origin%vec(2:6:2)
    cp%new_vvec=cp%old_vvec+hbar
    cp%new_vvec(3)=sqrt(1-sum(cp%new_vvec(1:2)**2))
    cp%b_eff=cp%old_vvec(3)/cp%new_vvec(3)
    cp%dtheta_sin_2theta=-dot_product(hbar+2*cp%old_vvec,hbar)/2
    p_factor=cos(ele%value(bragg_angle_in$)+ele%value(bragg_angle_out$))
    print '(a,i3,6es26.17)', 'CP ',k,cp%wavelength,cp%cap_gamma,cp%b_eff,cp%dtheta_sin_2theta,p_factor,ele%value(thickness$)
    print '(a,i3,6es26.17)', 'FACTORS ',k,ele%photon%material%f_0,ele%photon%material%f_hkl,ele%photon%material%f_hbar
    print '(a,i3,6es26.17)', 'VECTORS ',k,cp%old_vvec,cp%new_vvec
    state=alive$
    call crystal_diffraction_field_calc(cp,ele,ele%value(thickness$),lat%param,p_factor,.true.,fields(1),phases(1),state,drp)
    call crystal_diffraction_field_calc(cp,ele,ele%value(thickness$),lat%param,1.0_rp,.false.,fields(2),phases(2),state,drs)
    if(state/=alive$) error stop 'polarization helper lost photon'
    print '(a,i3,2es26.17)', 'RESPONSE ',k,fields
    print '(a,i3,6es26.17)', 'DR ',k,drp,drs
    do i=1,3
      incoming=0
      if(i==1) incoming=[1.0_rp,0.0_rp]
      if(i==2) incoming=[0.0_rp,1.0_rp]
      if(i==3) incoming=[.6_rp,.8_rp]
      ray=origin; ray%field=incoming
      call track1_crystal(ele,lat%param,ray)
      if(ray%state/=alive$) error stop 'polarization tracking lost photon'
      print '(a,2i3,5es26.17)', 'POL ',k,i,ray%vec(1:5:2)-origin%vec(1:5:2),ray%field
      if(i<3) then
        ray=surface_ray; ray%field=incoming
        call track1(ray,ele,lat%param,result_ray)
        if(result_ray%state/=alive$) error stop 'public polarization tracking lost photon'
        ray=origin; ray%field=incoming*fields; ray%phase=phases
        if(i==1) ray%vec(1:5:2)=ray%vec(1:5:2)+drp
        if(i==2) ray%vec(1:5:2)=ray%vec(1:5:2)+drs
        ray%vec(2:6:2)=cp%new_vvec
        call offset_photon(ele,ray,unset$)
        print '(a,2i3,2es26.17)', 'PUBLIC ',k,i,result_ray%vec(1),ray%vec(1)
      endif
    enddo
  enddo
case default
  error stop 'Unknown verification case'
end select
print '(a)', 'DONE'
contains
function checked_z(xx,yy,slopes) result(height)
real(rp) xx,yy,height
real(rp), optional :: slopes(2)
logical bad
height=z_at_surface(ele,xx,yy,bad,dz_dxy=slopes)
if(bad) error stop 'invalid surface evaluation'
end function
end program verify_photons
"""

PHOTON_LATTICE = """parameter[geometry] = open
parameter[particle] = photon
parameter[e_tot] = 9000
parameter[no_end_marker] = T
parameter[ran_seed] = 71439
no_digested
dp: detector, aperture_type=rectangular, curvature={x2y0=.7}
de: detector, aperture_type=rectangular, curvature={elliptical_x=.8, elliptical_z=1.3}
dm: detector, aperture_type=rectangular, curvature={x2y0=.7, elliptical_x=.8, elliptical_z=1.3}
sg: mirror, graze_angle=.2, curvature={x2y0=1.7, x0y2=.6}, segmented={dr=(.08,.12), ix_bounds=(-3,3), iy_bounds=(-3,3)}
si: mask, wall={section={type=clear, material=Si, thickness=2e-5, v(1)={0,0,.2}}}
ti: mask, wall={section={type=clear, material=Ti, thickness=2e-5, v(1)={0,0,.2}}}
tip: ti, y_pitch=acos(.8)
ml: multilayer_mirror, material_type="W:BORON_CARBIDE", n_cell=1, d1_thickness=1.2e-9, d2_thickness=1.8e-9
cf: crystal, crystal_type="Si(111)", b_param=1, thickness=1e-13, ref_orbit_follows=undefracted
cd: crystal, crystal_type="Si(111)", b_param=.55, thickness=2e-5
test: line=(dp,de,dm,sg,si,ti,tip,ml,cf,cd)
use,test
"""


def polarization_paths(k, crystal_parameters, factors, vectors, displacements):
    lam, gamma, b_eff, angular_delta, pi_factor, thick = crystal_parameters[k]
    f0_re, f0_im, fh_re, fh_im, fbar_re, fbar_im = factors[k]
    f0, fh, fbar = (
        complex(f0_re, f0_im),
        complex(fh_re, fh_im),
        complex(fbar_re, fbar_im),
    )
    old, new = np.asarray(vectors[k]).reshape(2, 3)
    independent_paths = []
    for pol_factor in [pi_factor, 1.0]:
        # Solve the two-beam dispersion equation as a 2x2 eigenproblem,
        # rather than Bmad's eta-based closed form. q=xi_a+xi_b;
        # xi_a*xi_b=-g^2, xi_H=g^2/(b*xi_0).
        q = b_eff * angular_delta + gamma * f0 * (1 - b_eff) / 2
        coupling = gamma * pol_factor * math.sqrt(b_eff) * fh / 2
        eigenvalues = np.linalg.eigvals(np.array([[q, coupling], [coupling, 0]], dtype=complex))
        branch_paths, weights = [], []
        for eigenvalue in eigenvalues:
            xi_h = coupling**2 / (b_eff * eigenvalue)
            delta_0 = 1 - gamma * f0.real / 2 + eigenvalue.real
            delta_h = 1 - gamma * f0.real / 2 + xi_h.real
            k0 = np.r_[old[:2], math.sqrt(delta_0**2 - sum(old[:2] ** 2))] / lam
            kh = np.r_[new[:2], math.sqrt(delta_h**2 - sum(new[:2] ** 2))] / lam
            field_ratio = -2 * eigenvalue / (pol_factor * gamma * fbar)
            energy_flow = k0 + abs(field_ratio) ** 2 * kh
            branch_paths.append(thick * energy_flow / energy_flow[2])
            attenuation_k = (delta_h / lam) ** 2 * (gamma * f0.imag / 2 - eigenvalue.imag) / k0[2]
            weights.append(math.exp(-2 * math.pi * attenuation_k * thick))
        independent_paths.append(np.average(branch_paths, axis=0, weights=weights))
    independent_paths = np.array(independent_paths)
    assert np.max(np.abs(independent_paths - np.asarray(displacements[k]).reshape(2, 3))) < 8e-20
    return independent_paths
