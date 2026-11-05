"""Parsing and independent spin algebra for the RF and spin conversion chunk."""

import math

import numpy as np
from _bmad import compare


def rows(log, prefix="RESULT "):
    records = {}
    for line in log.splitlines():
        if line.startswith(prefix):
            key, *values = line[len(prefix) :].split()
            if key in records or not values:
                raise ValueError(f"Duplicate or empty result: {line}")
            records[key] = np.array([float(v.replace("D", "E")) for v in values])
    if not records:
        raise ValueError(f"Missing numerical records:\n{log}")
    return records


def control(label, observed, expected, *, atol, rtol=0.0):
    if compare(label, observed, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"Control failed: {label}")


def rotate(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def spin(q, initial):
    # Hamilton quaternion action; normalize the polynomial as Bmad's spin map requires.
    q = q / np.linalg.norm(q)
    return (
        (q[0] ** 2 - np.dot(q[1:], q[1:])) * initial
        + 2 * np.dot(q[1:], initial) * q[1:]
        + 2 * q[0] * np.cross(q[1:], initial)
    )


def evaluate(terms, x, reference):
    output = np.zeros(4)
    for component, coefficient, exponents in terms:
        output[component] += coefficient * math.prod(
            float(v) ** power for v, power in zip(x - reference, exponents, strict=True)
        )
    return output


def spin_data(log):
    records, terms = {}, {}
    for line in log.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "TERM":
            tag, component, coefficient = fields[1], int(fields[2]), float(fields[3])
            terms.setdefault(tag, []).append((component, coefficient, tuple(map(int, fields[4:]))))
        elif fields[0].startswith(("P_", "R_")):
            if fields[0] in records:
                raise ValueError(f"Duplicate result: {line}")
            records[fields[0]] = np.array(list(map(float, fields[1:])))
    return records, terms


TAYLOR_PROGRAM = """program spin_check
use bmad
use ptc_interface_mod
implicit none
type(lat_struct), target :: lat
type(ele_struct), pointer :: e
type(coord_struct) o, ending
integer n,i,j,k
real(rp) x
logical err
character(16) arg
character(80) prefix
call get_command_argument(1,arg)
read(arg,*) n
call bmad_parser('verify.bmad',lat,err_flag=err)
if(err) error stop 'parse'
bmad_com%spin_tracking_on=.true.
if(n==2) then
  call set_ptc(taylor_order=2)
  do i=1,2
    e=>lat%ele(i)
    call init_coord(o,[0.0012_rp,0.00008_rp,-0.0007_rp,-0.00003_rp,0.0_rp,0.0_rp],e,upstream_end$)
    o%spin=[0.3_rp,0.4_rp,sqrt(0.75_rp)]
    do j=0,1
      e%taylor_map_includes_offsets=(j==1)
      call kill_taylor(e%taylor)
      call kill_taylor(e%spin_taylor)
      e%tracking_method=bmad_standard$
      call ele_to_taylor(e)
      e%tracking_method=taylor$
      call track1(o,e,lat%param,ending,err_flag=err)
      if(err) error stop 'track'
      if(ending%state/=alive$) error stop 'spin Taylor particle lost'
      write(prefix,'(a,i0,a,i0)') 'P_',i-1,'_',j
      call emit(trim(prefix)//'_ANGLE',[e%value(tilt_tot$)])
      call emit(trim(prefix)//'_IN',o%vec)
      call emit(trim(prefix)//'_S0',o%spin)
      call emit(trim(prefix)//'_ACTUAL',ending%spin)
      call emit(trim(prefix)//'_ORBIT',ending%vec)
      call map_dump(prefix,e)
    enddo
  enddo
else if(n==3) then
  e=>lat%ele(3)
  do i=0,2
    x=0
    if(i==1) x=0.001_rp
    if(i==2) x=-0.001_rp
    call init_coord(o,[x,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp],e,upstream_end$)
    o%spin=[0.0_rp,0.0_rp,1.0_rp]
    do j=0,1
      call kill_taylor(e%spin_taylor)
      if(j==0) call sprint_spin_taylor_map(e)
      ! j=1 deliberately leaves the map absent: normal track1 creates it.
      call track1(o,e,lat%param,ending,err_flag=err)
      if(err) error stop 'track'
      if(ending%state/=alive$) error stop 'sprint particle lost'
      write(prefix,'(a,i0,a,i0)') 'R_',i,'_',j
      call emit(trim(prefix)//'_IN',o%vec)
      call emit(trim(prefix)//'_ACTUAL',ending%spin)
      call emit(trim(prefix)//'_PARAM',[e%value(k1$),e%value(l$),e%value(e_tot$)/mass_of(e%ref_species),anomalous_moment_of(e%ref_species)])
      call map_dump(prefix,e)
    enddo
  enddo
else
  error stop 'unknown spin case'
endif
contains
subroutine emit(label,values)
character(*) label
real(rp) values(:)
write(*,'(a,1x,*(es25.16e3,1x))') trim(label),values
end subroutine
subroutine map_dump(label,ele)
character(*) label
type(ele_struct) ele
integer a,b
call emit(trim(label)//'_ORBITREF',ele%taylor%ref)
call emit(trim(label)//'_SPINREF',ele%spin_taylor_ref_orb_in)
do a=0,3
  do b=1,size(ele%spin_taylor(a)%term)
    write(*,'(a,1x,a,1x,i0,1x,es25.16e3,1x,6(i0,1x))') 'TERM',trim(label),a,ele%spin_taylor(a)%term(b)%coef,ele%spin_taylor(a)%term(b)%expn
  enddo
enddo
end subroutine
end program
"""

TAYLOR_LATTICE = """no_digested
parameter[geometry]=open
parameter[particle]=electron
parameter[p0c]=1e9
beginning[beta_a]=10
beginning[beta_b]=10
p0: quadrupole,l=.6,k1=.65,fringe_type=none,spin_tracking_method=tracking
p1: quadrupole,l=.6,k1=.65,tilt=.43,fringe_type=none,spin_tracking_method=tracking
r: quadrupole,l=1,k1=.5,fringe_type=none,spin_tracking_method=sprint
lat: line=(p0,p1,r)
use,lat
"""


MAGNUS_PROGRAM = """program spin_check
use bmad
implicit none
type(lat_struct), target :: lat
type(ele_struct), pointer :: ele
type(coord_struct) :: a, fwd, returned, back, back_out
type(em_field_struct) :: field
real(rp) :: v(6), seed(3), deltas(3)
integer :: config, ie, isp, id, ispin, method, species
logical :: err
character(40) :: filename

bmad_com%spin_tracking_on = .true.
bmad_com%electric_dipole_moment = 0
bmad_com%radiation_damping_on = .false.
bmad_com%radiation_fluctuations_on = .false.
bmad_com%rel_tol_adaptive_tracking = 1e-12_rp
bmad_com%abs_tol_adaptive_tracking = 1e-14_rp
deltas = [-0.25_rp, 0.0_rp, 0.4_rp]
do config = 1, 4
  write(filename, '(a,i0,a)') 'spin_', config, '.bmad'
  call bmad_parser(trim(filename), lat, err_flag=err)
  if(err) error stop 'parse'
  do ie = 1, 3
    ele => lat%ele(ie)
    do isp = 1, 2
      species = electron$
      if (isp == 2) species = positron$
      do id = 1, 3
        v = 0
        v(6) = deltas(id)
        do ispin = 1, 2
          seed = [0.36_rp, 0.48_rp, 0.8_rp]
          if (ispin == 2) seed = [0.0_rp, 0.0_rp, 1.0_rp]
          do method = 1, 3
            ele%tracking_method = bmad_standard$
            ele%spin_tracking_method = tracking$
            if (method == 1) ele%spin_tracking_method = magnus$
            if (method == 3) ele%tracking_method = runge_kutta$
            call init_coord(a, v, ele, upstream_end$, particle=species)
            a%spin = seed
            call em_field_calc(ele, lat%param, ele%value(l$)/2, a, .true., field, err_flag=err)
            if (err) error stop 'field error'
            call track1(a, ele, lat%param, fwd, err_flag=err)
            if (err .or. fwd%state /= alive$) error stop 'forward error'
            fwd%time_dir = -1
            call track1(fwd, ele, lat%param, returned, err_flag=err)
            if (err .or. returned%state /= alive$) error stop 'return error'
            call init_coord(back, v, ele, downstream_end$, particle=species)
            back%time_dir = -1
            back%spin = seed
            call track1(back, ele, lat%param, back_out, err_flag=err)
            if (err .or. back_out%state /= alive$) error stop 'backward error'
            write(*,'(a,6i4,*(es25.16e3,1x))') 'SPIN', config,ie,isp,id,ispin,method, &
              real(charge_of(species),rp),real(charge_of(ele%ref_species),rp),anomalous_moment_of(species), &
              a%p0c*(1+v(6)),field%B(3),ele%value(l$),ele%value(ks$),seed, &
              fwd%spin,returned%spin,back_out%spin,maxval(abs(returned%vec-a%vec)),returned%t-a%t
          enddo
        enddo
      enddo
    enddo
  enddo
enddo
end program
"""


def magnus_records():
    """432 axial-solenoid records; independent physical-field and integration controls."""
    from _bmad import fortran

    files = {}
    for config, (species, p0c) in enumerate(
        ((species, p0c) for species in ("electron", "positron") for p0c in (3e6, 8e8)), 1
    ):
        lines = [
            "no_digested",
            f"parameter[particle]={species}",
            f"parameter[p0c]={p0c}",
            "parameter[geometry]=open",
            "beginning[beta_a]=10",
            "beginning[beta_b]=10",
        ]
        for index, (length, ks) in enumerate(((0.17, 0.24), (1.3, -1.1), (0.7, 0.0)), 1):
            lines.append(
                f"s{index}: solenoid, l={length}, ks={ks}, fringe_type=none, spin_tracking_method=magnus"
            )
        lines += ["lat: line=(s1,s2,s3)", "use,lat"]
        files[f"spin_{config}.bmad"] = "\n".join(lines) + "\n"
    log = fortran(MAGNUS_PROGRAM, files=files).log
    records = []
    maximum_orbit = maximum_clock = 0.0
    integration_error = {2: 0.0, 3: 0.0}
    endpoint_error = {2: 0.0, 3: 0.0}
    for line in log.splitlines():
        if not line.startswith("SPIN "):
            continue
        columns = line.split()
        config, ie, isp, idelta, ispin, method = map(int, columns[1:7])
        q, qref, anomaly, momentum, bz, length, ks, *values = map(float, columns[7:])
        seed, forward, returned, backward = np.array(values[:12]).reshape(4, 3)
        orbit_error, clock_error = values[12:]
        if not np.isfinite([q, qref, anomaly, momentum, bz, length, ks, *values]).all():
            raise ValueError("Nonfinite spin sweep output")
        p0c = 3e6 if config % 2 else 8e8
        if abs(bz - ks * p0c / (qref * 299792458.0)) > 1e-14:
            raise AssertionError("Solenoid field/rigidity control failed")
        # Axial T-BMT uses the physical B in tesla, charge in e units, and pc in eV.
        theta = -q * 299792458.0 * bz * (1 + anomaly) * length / momentum
        maximum_orbit = max(maximum_orbit, orbit_error)
        maximum_clock = max(maximum_clock, abs(clock_error))
        if method != 1:
            integration_error[method] = max(
                integration_error[method], float(np.max(np.abs(returned - seed)))
            )
            # The spline integrator omits 1e-11 m at each edge. RK starts 1e-12 m inside.
            excluded = 2e-11 if method == 2 else 1e-12
            integrated = theta * (1 - excluded / length)
            endpoint_error[method] = max(
                endpoint_error[method],
                float(np.max(np.abs(forward - rotate(seed, integrated)))),
                float(np.max(np.abs(backward - rotate(seed, -integrated)))),
            )
        if ie == 3 or ispin == 2:
            if (
                max(
                    np.max(np.abs(forward - seed)),
                    np.max(np.abs(returned - seed)),
                    np.max(np.abs(backward - seed)),
                )
                > 3e-15
            ):
                raise AssertionError("Zero-field or longitudinal-spin control failed")
        records.append(
            (
                (config, ie, isp, idelta, ispin, method),
                q == qref,
                theta,
                seed,
                forward,
                returned,
                backward,
            )
        )
    if len(records) != 432 or len({r[0] for r in records}) != 432:
        raise ValueError(f"Expected 432 unique spin records, got {len(records)}")
    control("orbital round-trip maximum residual", maximum_orbit, 0, atol=1e-15)
    control("clock round-trip maximum residual (s)", maximum_clock, 0, atol=2e-23)
    for method in (2, 3):
        control(
            f"integration method {method} spin round-trip maximum",
            integration_error[method],
            0,
            atol=3e-15,
        )
        control(
            f"integration method {method} endpoint-adjusted T-BMT maximum",
            endpoint_error[method],
            0,
            atol=3e-15,
        )
    return records
