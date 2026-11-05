"""COL-45: Potential-well routines omit the ion charge state

Calling the potential-well matrix or Vlasov bunch-length routines for multiply charged ions
gives unit-charge results. The matrix kick requires Q² at fixed particle count, while the
Vlasov equation requires Q times the RF and wake voltages, where Q is the ion charge in
elementary-charge units. For carbon ions with Q = 6, the matrix slope is −5.745 × 10⁻⁵ m⁻¹
instead of −2.068 × 10⁻³ m⁻¹; the no-wake bunch length is 17.92 mm instead of 7.304 mm. The
public routines are affected; the ibs_ring application disables potential-well distortion.

The script calls pwd_mat for protons and two carbon charge states at fixed particle count,
energy, inductance and bunch length. It compares the matrix slope with the inductive-voltage
and energy-kick law; the unit-charge species are controls. The companion script checks the
Vlasov bunch-length routine.
"""

from _bmad import compare, fortran, verdict

PROGRAM = """! Usage: probe <lattice file>
! pwd_mat's potential-well kick slope Mpwd(6,5) at fixed n_part, energy, inductance and bunch
! length, against the inductive-impedance expectation: the bunch current is n_part*|Q|*e*c*lambda(z),
! V = -L dI/dt, and the energy kick of a particle of charge Q*e is Q*e*V, so
! Mpwd(6,5) = -Q^2 * L * n_part * e * c^2 / (sqrt(2 pi) sigma_z^3 E0[eV]).
program probe
use bmad
use longitudinal_profile_mod
implicit none
type (lat_struct), target :: lat
real(rp) :: t6(6,6), t6_pwd(6,6), expected
real(rp), parameter :: inductance = 1e-9_rp, sig_z = 1e-3_rp, n_part = 1e10_rp
integer :: i, q
character(200) :: lat_file

call get_command_argument(1, lat_file)
call bmad_parser(lat_file, lat)
lat%param%n_part = n_part
t6 = 0
do i = 1, 6
  t6(i,i) = 1
enddo
t6_pwd = pwd_mat(lat, t6, inductance, sig_z)
q = charge_of(lat%param%particle)
expected = -q**2 * inductance * n_part * e_charge * c_light**2 / sqrt(twopi) / sig_z**3 / lat%ele(0)%value(e_tot$)
write(*, '(a,3es25.16)') 'RESULT ', real(q,rp), t6_pwd(6,5), expected
end program
"""
defect = False
for species in ("proton", "#12C+1", "#12C+6"):
    lattice = f"parameter[particle]={species}\nparameter[e_tot]=1e12\nparameter[geometry]=open\nd: drift,l=1\nl: line=(d)\nuse,l\n"
    log = fortran(PROGRAM, ["lat.bmad"], files={"lat.bmad": lattice}).log
    rows = [line.split()[1:] for line in log.splitlines() if line.startswith("RESULT ")]
    if len(rows) != 1:
        raise RuntimeError(log)
    q, observed, expected = map(float, rows[0])
    bad = compare(
        f"{species} Mpwd(6,5), m^-1; -Q^2 L N e c^2/(sqrt(2pi) sigma_z^3 E)",
        observed,
        expected,
        atol=0,
        rtol=2e-14,
    )
    if abs(q) == 1 and bad:
        raise RuntimeError("Unit-charge matrix control failed")
    defect |= bad
verdict(defect)
