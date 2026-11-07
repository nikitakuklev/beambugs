"""TRK-67: Intrinsic-spin lookup returns zero for helion and antihelion

Requesting intrinsic spin for a helion or antihelion through `spin_of` returns zero instead of one half in units of ℏ. Both species are covered by the helper's explicit subatomic-particle contract, and an independent helium-3 isotope table establishes the expected spin. Supplying the optional non-subatomic default does not alter these wrong table values. No internal tracking routine consumes this accessor, and direct charged helion and antihelion spin tracking agrees with the magnetic-precession expectation.

The script uses BMAD_DIST and the shared runner to query intrinsic spin with and without the optional non-subatomic default. The independently tabulated helium-3 nuclear spin supplies the exact expectation, and other subatomic species and an isotope spelling check the helper contract. A separate native Runge–Kutta client checks charged helion and antihelion precession against the Thomas–BMT equation.
"""

from __future__ import annotations

import math

from _bmad import fortran, verdict

PROGRAM = r"""
program confirmation
use bmad
implicit none
type(lat_struct) :: lat
type(coord_struct) :: a,b
type(em_field_struct) :: field
real(rp) :: v(6), omega(3)
integer :: i,j,sid,ids(8)
character(100) :: action
logical :: err
call get_command_argument(1,action)
if(trim(action)=="lookup") then
 ids=[proton$,antiproton$,deuteron$,anti_deuteron$,helion$,anti_helion$,neutron$,anti_neutron$]
 do i=1,size(ids)
  sid=ids(i)
  write(*,'(a,i4,5es26.17)') 'LOOKUP ',sid,spin_of(sid),spin_of(sid,17.0_rp), &
       anomalous_moment_of(sid),mass_of(sid),real(charge_of(sid),rp)
 enddo
 sid=species_id('#3He+2')
 write(*,'(a,2es26.17)') 'NON_SUBATOMIC ',spin_of(sid),spin_of(sid,17.0_rp)
else if(trim(action)=="spin") then
 call bmad_parser('spin.bmad',lat,err_flag=err)
 if(err) error stop 'spin parse'
 ids(1:2)=[helion$,anti_helion$]
 do i=1,2
  sid=ids(i)
  a=coord_struct()
  a%species=sid; a%beta=0; a%vec=0
  field=em_field_struct(); field%B=[0.0_rp,0.0_rp,1.0_rp]
  omega=spin_omega(field,a,1,.false.)
  write(*,'(a,i4,5es26.17)') 'REST ',sid,omega,mass_of(sid),anomalous_moment_of(sid)
  do j=0,1
   lat%ele(1)%is_on=(j==1)
   v=0; v(6)=0.75_rp*mass_of(sid)/lat%ele(1)%value(p0c$)-1
   call init_coord(a,v,lat%ele(1),upstream_end$,particle=sid)
   call convert_pc_to(0.75_rp*mass_of(sid),sid,beta=a%beta)
   a%spin=[1.0_rp,0.0_rp,0.0_rp]
   call track1(a,lat%ele(1),lat%param,b,err_flag=err)
   if(err.or.b%state/=alive$) error stop 'spin tracking'
   write(*,'(a,2i4,5es26.17)') 'SPIN ',sid,j,a%beta,b%spin,b%t-a%t
  enddo
 enddo
else
 error stop "unknown action"
endif
end program
"""

LATTICE = """no_digested
parameter[particle]=proton
parameter[p0c]=1e9
parameter[geometry]=open
bmad_com[spin_tracking_on]=T
bmad_com[significant_length]=1e-14
bmad_com[rel_tol_adaptive_tracking]=1e-13
bmad_com[abs_tol_adaptive_tracking]=1e-15
s: solenoid,l=0.4,bs_field=0.2,field_master=T,fringe_type=none,tracking_method=runge_kutta,spin_tracking_method=tracking
ln: line=(s)
use,ln
"""

def records(log: str, tag: str, count: int) -> list[list[float]]:
    if any(marker in log for marker in ("[ERROR", "[FATAL", "[ABORT")):
        raise RuntimeError(log)
    result = [[float(x) for x in line.split()[1:]] for line in log.splitlines()
              if line.startswith(tag + " ")]
    if not result or any(len(row) != count or not all(map(math.isfinite, row)) for row in result):
        raise RuntimeError(f"Missing or invalid {tag} records:\n{log}")
    return result


def close(actual: list[float], expected: list[float], tolerance: float, label: str) -> bool:
    if len(actual) != len(expected) or not actual:
        raise RuntimeError(f"{label}: shape mismatch")
    return all(math.isclose(a, e, abs_tol=tolerance, rel_tol=0)
               for a, e in zip(actual, expected, strict=True))


def require(actual: list[float], expected: list[float], tolerance: float, label: str) -> None:
    if not close(actual, expected, tolerance, label):
        raise RuntimeError(f"{label}: observed={actual}, expected={expected}, atol={tolerance}")


def main() -> None:
    log = fortran(PROGRAM, ["lookup"]).log
    rows = records(log, "LOOKUP", 6)
    if len(rows) != 8:
        raise RuntimeError("Expected eight subatomic lookups")
    wrong = False
    for row in rows:
        sid = int(row[0])
        expected = 1.0 if abs(sid) == 5 else 0.5
        # NIST's helium isotope table gives helium-3 nuclear spin exactly 1/2.
        # Charge conjugation preserves intrinsic spin magnitude.
        target = [expected, expected]
        if abs(sid) == 8:
            wrong |= not close(row[1:3], target, 0, "Helium-3 nuclear spin")
        else:
            require(row[1:3], target, 0, "Other subatomic spins")
        print(f"species={sid}: spin/hbar={row[1]:g} with_default_17={row[2]:g} "
              f"expected_intrinsic={expected:g}; exact subatomic-spin contract")
    require(records(log, "NON_SUBATOMIC", 2)[0], [0, 17], 0, "Isotope optional default")
    print("#3He+2 non-subatomic control: observed=[0, 17] expected_optional_defaults=[0, 17]")
    spinlog = fortran(PROGRAM, ["spin"], files={"spin.bmad": LATTICE}).log
    constants = {int(r[0]): r[1:] for r in records(spinlog, "REST", 6)}
    spins = records(spinlog, "SPIN", 7)
    if len(constants) != 2 or len(spins) != 4:
        raise RuntimeError("Missing charged helion spin controls")
    for row in spins:
        sid, on = map(int, row[:2])
        beta, sx, sy, sz, dt = row[2:]
        _, _, _, mass, anomaly = constants[sid]
        # Longitudinal-field Thomas–BMT rotation at gamma=5/4, beta=3/5:
        # theta=-(1+a)*Q*c^2/M * B*L/(gamma*beta*c), M in eV.
        charge = math.copysign(2, sid)
        theta = -(1+anomaly)*charge*299792458**2/mass*(0.2*on)*0.4/(1.25*0.6*299792458)
        expected_spin = [math.cos(theta), math.sin(theta), 0.0]
        require([beta], [0.6], 2e-16, "Charged helion beta")
        require([dt], [0.4/(0.6*299792458)], 8e-23, "Charged helion flight time")
        require([sx, sy, sz], expected_spin, 5e-14, "Charged helion BMT control")
        print(f"charged species={sid} field_on={on}: spin_y={sy:.17g} "
              f"expected_BMT={expected_spin[1]:.17g}; spin/time controls pass")
    verdict(wrong)


if __name__ == "__main__":
    main()
