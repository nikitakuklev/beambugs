"""Deterministic dispersion and beam-moment probes, using the selected Bmad build."""

import re

import numpy as np
from _bmad import fortran


def records(log):
    """Read unique RESULT records and reject native diagnostics."""
    if re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b|ERROR STOP|Fortran runtime error", log):
        raise RuntimeError(log)
    result = {}
    for line in log.splitlines():
        if not line.startswith("RESULT "):
            continue
        _, key, *values = line.split()
        if key in result:
            raise RuntimeError(f"Duplicate record: {key}")
        result[key] = np.array([float(x) for x in values])
        if not result[key].size or not np.isfinite(result[key]).all():
            raise RuntimeError(f"Invalid numeric record: {line}")
    if not result:
        raise RuntimeError(f"No result records:\n{log}")
    return result


def execute(program, files=None, args=()):
    return records(fortran(program, files=files, args=args).log)


def require(actual, expected, *, atol=1e-15, rtol=3e-14):
    """Controls fail the run instead of supplying a defect verdict."""
    if not np.isfinite(np.asarray(actual)).all() or not np.isfinite(np.asarray(expected)).all():
        raise ValueError("Controls require finite observed and expected values")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)


def lattice(elements, *, p0c=1e6, species="electron", start=None, extra="", line="e", closed=False):
    start = np.zeros(6) if start is None else start
    return (
        f"parameter[geometry] = {'closed' if closed else 'open'}\n"
        f"parameter[particle] = {species}\nparameter[p0c] = {p0c:.17g}\n"
        "beginning[beta_a] = 3\nbeginning[alpha_a] = 0.4\nbeginning[beta_b] = 4\n"
        + "\n".join(
            f"particle_start[{k}] = {v:.17g}"
            for k, v in zip(("x", "px", "y", "py", "z", "pz"), start, strict=True)
        )
        + f"\n{elements}\n{extra}\nbl: line = ({line})\nuse, bl\n"
    )


def quad(k, length, momentum=1.0):
    root = np.sqrt(abs(k) / momentum)
    phase = root * length
    c, s = (np.cos(phase), np.sin(phase)) if k > 0 else (np.cosh(phase), np.sinh(phase))
    return np.array([[c, s / (momentum * root)], [-np.sign(k) * momentum * root * s, c]])


def fodo_matrix(delta=0.0, *, skew=False):
    momentum = 1 + delta
    drift = np.array([[1, 0.8 / momentum], [0, 1]])
    m = np.zeros((4, 4))
    for block, sign in [(slice(0, 2), 1), (slice(2, 4), -1)]:
        m[block, block] = (
            drift @ quad(-0.96 * sign, 0.2, momentum) @ drift @ quad(sign, 0.2, momentum)
        )
    if skew:
        kick = np.eye(4)
        kick[1, 2] = kick[3, 0] = -0.003
        m = kick @ m
    return m


PERIODIC = r"""
program dispersion_oracle
use bmad
implicit none
type(ele_struct) ele
real(rp) m(6,6), orb(6), growth
integer i, stat
logical stable, normalized
open(10, file='input.dat')
read(10,*) normalized
read(10,*) orb
do i=1,6
  read(10,*) m(i,:)
enddo
close(10)
bmad_com%normalize_twiss = normalized
call twiss_from_mat6(m, orb, ele, stable, growth, stat, .true.)
if (.not. stable .or. stat /= ok$) error stop 'unstable test map'
write(*,'(a,8es25.17)') 'RESULT dispersion ', ele%x%eta, ele%x%etap, ele%y%eta, ele%y%etap, &
  ele%x%deta_ds, ele%y%deta_ds, ele%a%deta_ds, ele%b%deta_ds
end program
"""


def periodic(normalized):
    orbit = np.array([0, 0.03, 0, -0.02, 0, 0.2])
    eta = np.array([0.8, 0.13, 0.4, -0.05])
    mat = np.eye(6)
    mat[:4, :4] = fodo_matrix(orbit[5])
    mat[:4, 5] = (np.eye(4) - mat[:4, :4]) @ eta
    # Symplectic completion of the longitudinal row for the imposed dispersion.
    j4 = np.kron(np.eye(2), [[0, 1], [-1, 0]])
    mat[4, :4] = mat[:4, 5] @ j4 @ mat[:4, :4]
    text = ("T" if normalized else "F") + "\n"
    text += " ".join(map(str, orbit)) + "\n"
    text += "\n".join(" ".join(map(str, row)) for row in mat) + "\n"
    values = execute(PERIODIC, {"input.dat": text})["dispersion"]
    return values, orbit, eta, mat


OPEN_LINE = r"""
program open_line
use bmad
implicit none
type(lat_struct) lat
type(coord_struct), allocatable :: orb(:)
integer stat
call bmad_parser('lat.bmad',lat)
call twiss_and_track(lat,orb,stat,use_particle_start=.true.)
if (stat /= ok$) error stop 'twiss_and_track failed'
write(*,'(a,4es25.17)') 'RESULT entrance ',lat%ele(0)%x%eta,lat%ele(0)%x%etap, &
 lat%ele(0)%y%eta,lat%ele(0)%y%etap
write(*,'(a,4es25.17)') 'RESULT exit ',lat%ele(1)%x%eta,lat%ele(1)%x%etap, &
 lat%ele(1)%y%eta,lat%ele(1)%y%etap
write(*,'(a,4es25.17)') 'RESULT slopes ',lat%ele(1)%x%deta_ds,lat%ele(1)%y%deta_ds, &
 lat%ele(1)%a%deta_ds,lat%ele(1)%b%deta_ds
write(*,'(a,6es25.17)') 'RESULT orbit ',orb(1)%vec
if (orb(1)%state /= alive$) error stop 'lost particle'
end program
"""


MOMENTS = r"""
program moments
use bmad
use beam_utils
implicit none
type(lat_struct) lat
type(ele_struct) ele
type(bunch_struct) bunch
type(bunch_params_struct) bp
type(coord_struct) before
real(rp) v(6), q, pbase, pnew, nm(6,6), invariant_error(4)
integer iu,nc,ic,n,i,j,species
logical err
character(32) species_label,tag
call bmad_parser('lat.bmad',lat)
ele=lat%ele(0)
open(newunit=iu,file='input.dat',status='old')
read(iu,*) nc
do ic=1,nc
  read(iu,*) n,species_label,pbase,pnew
  species=species_id(species_label)
  if (species == invalid$) error stop 'invalid species'
  ele%value(p0c$)=pbase
  ele%value(e_tot$)=sqrt(pbase**2+mass_of(species)**2)
  call reallocate_bunch(bunch,n)
  invariant_error=0
  do i=1,n
    read(iu,*) q,v
    call init_coord(bunch%particle(i),v,ele,downstream_end$,species,shift_vec6=.false.)
    bunch%particle(i)%charge=q
    before=bunch%particle(i)
    call orbit_reference_energy_correction(bunch%particle(i),pnew)
    invariant_error(1)=max(invariant_error(1),maxval(abs( &
      before%p0c*before%vec([2,4])-bunch%particle(i)%p0c*bunch%particle(i)%vec([2,4])))/pbase)
    invariant_error(2)=max(invariant_error(2),abs( &
      before%p0c*(1+before%vec(6))-bunch%particle(i)%p0c*(1+bunch%particle(i)%vec(6)))/pbase)
    invariant_error(3)=max(invariant_error(3),abs(before%t-bunch%particle(i)%t))
    invariant_error(4)=max(invariant_error(4),abs(before%beta-bunch%particle(i)%beta))
  enddo
  if (any(bunch%particle%state/=alive$)) error stop 'lost test particle'
  call calc_bunch_params(bunch,bp,err,n_mat=nm)
  if (err .or. .not.bp%twiss_valid) error stop 'moment calculation failed'
  write(tag,'(a,i0)') 'meta',ic
  call row(tag,[mass_of(species),invariant_error])
  write(tag,'(a,i0)') 'emit',ic
  call row(tag,[bp%x%emit,bp%x%norm_emit,bp%y%emit,bp%y%norm_emit,bp%a%norm_emit,bp%b%norm_emit])
  write(tag,'(a,i0)') 'disp',ic
  call row(tag,[bp%x%eta,bp%x%etap,bp%y%eta,bp%y%etap,bp%a%eta,bp%a%etap,bp%b%eta,bp%b%etap,bp%c%gamma])
  do j=1,6
    write(tag,'(a,i0,a,i0)') 'sigma',ic,'_',j
    call row(tag,bp%sigma(j,:))
    if (bp%sigma(5,5)>0 .and. bp%sigma(6,6)>0) then
      write(tag,'(a,i0,a,i0)') 'nmat',ic,'_',j
      call row(tag,nm(j,:))
    endif
  enddo
enddo
close(iu)
contains
subroutine row(name,values)
character(*) name
real(rp) values(:)
write(*,'(a,a,100(1x,es25.17))') 'RESULT ',trim(name),values
end subroutine
end program
"""


def twiss(beta, alpha, emit):
    return np.sqrt(emit) * np.array(
        [[np.sqrt(beta), 0], [-alpha / np.sqrt(beta), 1 / np.sqrt(beta)]]
    )


def moments(points, weights):
    weights = weights / weights.sum()
    mean = weights @ points
    offset = points - mean
    return mean, (offset.T * weights) @ offset


def moment_cases(cases, weights):
    lines = [str(len(cases))]
    for case in cases:
        lines.append(f"64 {case['species']} {case['pbase']:.17e} {case['pnew']:.17e}")
        lines.extend(
            " ".join(f"{x:.17e}" for x in [w, *p])
            for w, p in zip(weights, case["points"], strict=True)
        )
    return execute(
        MOMENTS, {"input.dat": "\n".join(lines) + "\n", "lat.bmad": lattice("e: drift, l=0.1")}
    )


RING = r"""
! 16-cell FODO ring of 32 identical sector bends (rho = 32/pi m), gamma = 1000 for every species.
! Strengths are geometric (angle, k1), so the optics do not depend on the species.
parameter[particle] = SPECIES
parameter[geometry] = GEOMETRY
beginning[e_tot] = 1000 * mass_of(SPECIES)
beginning[beta_a] = 10
beginning[beta_b] = 10
b: sbend, l = 2, angle = pi/16
qf: quadrupole, l = 0.5, k1 = 0.6
qd: quadrupole, l = 0.5, k1 = -0.6
d: drift, l = 1
cell: line = (qf, d, b, d, qd, d, b, d)
ring: line = (16*cell)
use, ring
"""


def radiation_ring(species="electron", closed=True, rf=False):
    text = RING.replace("SPECIES", species).replace("GEOMETRY", "closed" if closed else "open")
    if rf:
        text = text.replace(
            "ring: line = (16*cell)",
            "rf: rfcavity, l=0, voltage=1e4, harmon=100, phi0=0\nring: line = (8*cell,rf,8*cell,rf)",
        )
    return text


RADIATION = r"""
program radiation
use bmad
use rad_6d_mod
implicit none
type(lat_struct),target :: lat
type(coord_struct),allocatable :: orbit(:)
type(normal_modes_struct) mode,mode6
type(rad_map_struct) rmap
real(rp) sigma(6,6),mc2,energy,gamma,rc,cq,m(6,6),m65,nd(6,6)
integer stat,i,opening
logical err
character(32) tag
call bmad_parser('lat.bmad',lat)
if (lat%param%geometry == closed$) then
  call twiss_and_track(lat,orbit,stat)
  if (stat /= ok$) error stop 'twiss_and_track failed'
else
  call reallocate_coord(orbit,lat%n_ele_track)
  call init_coord(orbit(0),lat%ele(0),downstream_end$)
  call track_all(lat,orbit,err_flag=err)
  if (err) error stop 'open-line tracking failed'
  call twiss_propagate_all(lat,err_flag=err)
  if (err) error stop 'open-line Twiss failed'
endif
if (any(orbit%state/=alive$)) error stop 'orbit lost'
call radiation_integrals(lat,orbit,mode)
mc2=mass_of(lat%param%particle)
energy=lat%ele(0)%value(e_tot$)
gamma=energy/mc2
rc=classical_radius(lat%param%particle)
cq=55*h_bar_planck*c_light/(32*sqrt(3.0_rp)*mc2)
call row('constants',[mc2,energy,gamma,rc,cq,real(charge_of(lat%param%particle),rp)])
call row('integrals',mode%synch_int)
call row('loss',[mode%e_loss,mode%a%alpha_damp,mode%b%alpha_damp,mode%z%alpha_damp])
call row('partition',[mode%a%j_damp,mode%b%j_damp,mode%z%j_damp])
call row('linac',[mode%lin%sig_E1,mode%lin%i3_E7,mode%lin%a_emittance_end, &
 mode%lin%b_emittance_end,mode%lin%i5a_E6,mode%lin%i5b_E6])
call row('spread',[mode%sig_z,mode%sigE_E,mode%z%emittance])
if (lat%param%geometry == closed$) then
  call transfer_matrix_calc(lat,m)
  m65=0
  do i=1,lat%n_ele_track
    if (lat%ele(i)%key == rfcavity$) m65=m65+lat%ele(i)%mat6(6,5)
  enddo
  call row('m65',[m65,m(6,5)])
  do opening=0,1
    call emit_6d(lat%ele(0),opening==1,mode6,sigma,closed_orbit=orbit)
    write(tag,'(a,i0)') 'native',opening
    call row(tag,[mode6%z%emittance,mode6%z%emittance_no_vert,mode6%sig_z,mode6%sigE_E, &
                  mode6%e_loss,mode6%dpz_damp,sigma(5,5),sigma(6,6),mode6%a%alpha_damp])
  enddo
  call rad_damp_and_stoc_mats(lat%ele(0),lat%ele(0),.false.,rmap,mode6,nd,err,closed_orbit=orbit)
  if (err) error stop 'radiation map failed'
  call row('damped65',[rmap%xfer_damp_mat(6,5)])
endif
contains
subroutine row(name,values)
character(*) name
real(rp) values(:)
write(*,'(a,a,100(1x,es25.17))') 'RESULT ',trim(name),values
end subroutine
end program
"""
