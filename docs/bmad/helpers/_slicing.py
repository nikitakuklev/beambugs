"""Shared clients and independent linear Lorentz–BMT oracle for slicing checks.

All native cases use _bmad's isolated compiler/runtime scratch and a fresh process.
Diagnostic helper variants append modified source to the client translation unit;
the selected reference source and libraries remain read-only.
"""

import numpy as np
from _bmad import fortran, source
from scipy.linalg import expm

LENGTH = 0.6
KS = -1.1
PC = 6e6
LATTICE = """no_digested
parameter[particle]=electron
parameter[p0c]=6e6
s: solenoid,l=0.6,ks=-1.1,spin_tracking_method=tracking,spin_fringe_on=T
l1: line=(s)
use,l1
"""

SPIN_PROGRAM = """
! Independent client: analytic linear-spin checks and direct boundary-distance diagnostics.
program confirm_spin
 use bmad
 implicit none
 type(lat_struct) lat
 type(ele_struct) piece
 type(coord_struct) p,q,test
 type(fringe_field_info_struct) fringe
 integer method,n,i,field_mode,fringe_on,axis
 logical err
 real(rp) amp,edge,vec(6)
 read(*,*) method,n,field_mode,fringe_on,axis,amp
 call bmad_parser('lat.bmad',lat)
 bmad_com%spin_tracking_on=.true.
 bmad_com%rel_tol_adaptive_tracking=1e-13_rp
 bmad_com%abs_tol_adaptive_tracking=1e-19_rp
 bmad_com%init_ds_adaptive_tracking=0.001_rp
 select case(method)
 case(0); lat%ele(1)%tracking_method=bmad_standard$
 case(1); lat%ele(1)%tracking_method=runge_kutta$
 case(2); lat%ele(1)%tracking_method=fixed_step_runge_kutta$
 case(3); lat%ele(1)%tracking_method=time_runge_kutta$
 case(4); lat%ele(1)%tracking_method=fixed_step_time_runge_kutta$
 end select
 lat%ele(1)%value(ds_step$)=0.001_rp
 lat%ele(1)%value(spin_fringe_on$)=fringe_on
 vec=amp*[2e-6_rp,0.3e-6_rp,-1e-6_rp,-0.2e-6_rp,0.0_rp,0.0_rp]
 if(axis==1) vec=0
 call init_coord(p,vec,lat%ele(1),upstream_end$,electron$)
 p%spin=[0.0_rp,0.0_rp,1.0_rp]
 if(axis==1) p%spin=[1.0_rp,0.0_rp,0.0_rp]
 write(*,'(a,5es26.17e3)') 'CONST ',m_electron,anomalous_moment_of(electron$), &
   lat%ele(1)%value(bs_field$),bmad_com%significant_length,lat%ele(1)%value(ks$)
 if(n==0) then
   call track1(p,lat%ele(1),lat%param,q,err_flag=err)
   if(err.or.q%state/=alive$) error stop 'unsliced tracking failed'
   p=q
 else
   do i=1,n
     call element_slice_iterator(lat%ele(1),lat%param,i,n,piece)
     if(field_mode==1) piece%field_calc=bmad_standard$
     if(method==0) then
       test=p
       test%location=upstream_end$
       call calc_next_fringe_edge(piece,edge,fringe,test,.true.)
       if(fringe%has_fringe) then
         write(*,'(a,3i4,4es26.17e3)') 'EDGE ',i,1,fringe%particle_at, &
           test%s,piece%s_start,edge,fringe%ds_edge
       else
         write(*,'(a,3i4)') 'EDGE ',i,0,fringe%particle_at
       endif
     endif
     call track1(p,piece,lat%param,q,err_flag=err)
     if(err.or.q%state/=alive$) error stop 'slice tracking failed'
     p=q
   enddo
 endif
 write(*,'(a,9es26.17e3)') 'OUT ',p%vec,p%spin
end program
"""

SCOPE_PROGRAM = """
! Track the parser's actual branch, including marker-created super slaves.
program confirm_scope
 use bmad
 implicit none
 type(lat_struct) lat
 type(coord_struct) p,q,test
 type(fringe_field_info_struct) fringe
 integer i
 logical err
 real(rp) edge
 call bmad_parser('lat.bmad',lat)
 bmad_com%spin_tracking_on=.true.
 bmad_com%rel_tol_adaptive_tracking=1e-13_rp
 bmad_com%abs_tol_adaptive_tracking=1e-19_rp
 call init_coord(p,[2e-6_rp,0.3e-6_rp,-1e-6_rp,-0.2e-6_rp,0.0_rp,0.0_rp], &
   lat%ele(1),upstream_end$,electron$)
 p%spin=[0.0_rp,0.0_rp,1.0_rp]
 do i=1,lat%n_ele_track
   test=p
   test%location=upstream_end$
   call calc_next_fringe_edge(lat%ele(i),edge,fringe,test,.true.)
   if(fringe%has_fringe) write(*,'(a,i3,1x,a,6i5,2es26.17e3)') 'STRUCT ',i,trim(lat%ele(i)%name), &
       lat%ele(i)%slave_status,lat%ele(i)%field_calc,lat%ele(i)%tracking_method,lat%ele(i)%key,lat%ele(i)%spin_tracking_method,1,edge,fringe%ds_edge
   if(.not.fringe%has_fringe) write(*,'(a,i3,1x,a,6i5)') 'STRUCT ',i,trim(lat%ele(i)%name), &
       lat%ele(i)%slave_status,lat%ele(i)%field_calc,lat%ele(i)%tracking_method,lat%ele(i)%key,lat%ele(i)%spin_tracking_method,0
   call track1(p,lat%ele(i),lat%param,q,err_flag=err)
   if(err.or.q%state/=alive$) error stop 'branch tracking failed'
   p=q
 enddo
 write(*,'(a,9es26.17e3)') 'OUT ',p%vec,p%spin
end program
"""


def record(log, tag, size):
    rows = [line.split()[1:] for line in log.splitlines() if line.startswith(tag + " ")]
    if len(rows) != 1 or len(rows[0]) != size:
        raise RuntimeError(f"Expected one complete {tag} record:\n{log}")
    values = np.array(rows[0], dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Nonfinite {tag} record: {values}")
    return values


def close(actual, expected, tolerance, label):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    if not np.isfinite(error) or error > tolerance:
        raise RuntimeError(f"{label}: error={error:.17g}, tolerance={tolerance:.17g}")
    return error


def spin(method, n, direct=0, fringe=1, axis=0, amp=1, variant=""):
    program = SPIN_PROGRAM + diagnostic_sources(variant)
    log = fortran(
        program,
        files={"lat.bmad": LATTICE},
        stdin=f"{method} {n} {direct} {fringe} {axis} {amp}\n",
    ).log
    edges = [line.split()[1:] for line in log.splitlines() if line.startswith("EDGE ")]
    if method == 0 and len(edges) != n:
        raise RuntimeError(f"Expected {n} boundary diagnostics:\n{log}")
    return record(log, "OUT", 9), record(log, "CONST", 5), edges


def analytic(constants, fringe=True):
    mass, anomaly = constants[:2]
    # State (x,y,u,v,Sx,Sy): u,v are kinetic momenta/P0; Sz=1+O(amplitude^2).
    # Signed ks=q*c*Bs/pc; n'=n cross K and S'=Omega cross S are linearized.
    gamma = np.sqrt(1 + (PC / mass) ** 2)
    a = anomaly * (gamma - 1) * KS
    c = (1 + anomaly) * KS
    body = np.zeros((6, 6))
    body[0, 2] = body[1, 3] = 1
    body[2, 3], body[3, 2] = KS, -KS
    body[4, 3], body[4, 5] = a, c
    body[5, 2], body[5, 4] = -a, -c
    # Maxwell div B=0 gives integral K_transverse ds=-(x,y)*Delta(ks)/2.
    # Integrating Lorentz/BMT gives the entrance matrix; its inverse is the exit.
    entry = np.eye(6)
    entry[2, 1], entry[3, 0] = KS / 2, -KS / 2
    if fringe:
        h = (1 + anomaly * gamma) * KS / 2
        entry[4, 1], entry[5, 0] = h, -h
    exit_map = 2 * np.eye(6) - entry  # The off-diagonal edge generator squares to zero.
    initial = np.array([2e-6, -1e-6, 0.3e-6, -0.2e-6, 0, 0])
    return exit_map @ expm(body * LENGTH) @ entry @ initial


def diagnostic_sources(variant):
    if not variant:
        return ""
    if variant not in {"distance", "both"}:
        raise ValueError(f"Unknown diagnostic variant: {variant}")
    root = source()
    calc = (root / "bmad/low_level/calc_next_fringe_edge.f90").read_text()
    old = "s_edge_body + s_off - s_orb"
    if calc.count(old) != 1:
        raise RuntimeError("Boundary-distance diagnostic source anchor changed")
    result = "\n" + calc.replace(old, "s_edge_body - s_orb")
    if variant == "both":
        init = (root / "bmad/low_level/init_fringe_info.f90").read_text()
        old = "if (ele%key /= rfcavity$ .and. ele%key /= lcavity$ .and. ele%key /= crab_cavity$ .and. &"
        if init.count(old) != 1:
            raise RuntimeError("Boundary-initialization diagnostic source anchor changed")
        result += "\n" + init.replace(old, "if (ele%key /= solenoid$ .and. " + old[4:])
    return result


def branch(method, split, kind="solenoid", spin_method="tracking"):
    strength = "ks=-1.1" if kind == "solenoid" else "ks=-1.1,k1=0"
    lattice = f"""no_digested
parameter[particle]=electron
parameter[p0c]=6e6
s: {kind},l=0.6,{strength},tracking_method={method},spin_tracking_method={spin_method},spin_fringe_on=T
"""
    if split:
        lattice += """m1: marker,superimpose,ref=s,offset=-0.15
m2: marker,superimpose,ref=s
m3: marker,superimpose,ref=s,offset=0.15
"""
    lattice += "l1: line=(s)\nuse,l1\n"
    log = fortran(SCOPE_PROGRAM, files={"lat.bmad": lattice}).log
    structures = [line.split()[1:] for line in log.splitlines() if line.startswith("STRUCT ")]
    if len(structures) != (8 if split else 2):
        raise RuntimeError(f"Unexpected parser branch structure:\n{log}")
    return record(log, "OUT", 9), structures
