"""COL-47: Tail-cut Coulomb logarithms exceed the classical scattering bound

Selecting a tail-cut Coulomb logarithm with a long radiation-damping time lets the
intrabeam-scattering calculation include collisions beyond its classical scattering limit.
Options 3 and 4 omit the documented maximum of the classical and damping-time minimum impact
parameters; option 2 likewise fails to bound the maximum momentum transfer. At a damping time
of 10⁸ s in the proton example, the respective rate excesses are 14.27%, 10.69% and 12.58%.
Strongly damped cases keep the tail cut inside the classical bound.

The script sweeps the radiation-damping time of a proton element and compares options 3 and 4
with the manual’s lower-impact-parameter bound. The classic log and strongly damped cases are
controls. The companion script checks the Raubenheimer option against its classical momentum-
transfer bound.
"""

import math

from _bmad import compare, fortran, verdict

PROGRAM = """! multi_coulomb_log options 1-4 for one proton element (gamma = 100, beta_a = beta_b = 10 m,
! emit_a = emit_b = 1e-8 m, sigma_p = 1e-3, no dispersion, line density 1e12 /m) while the
! damping time tau_a grows. Columns: tau_a, then the four logs.
program probe
use bmad
use ibs_mod
implicit none
type (lat_struct), target :: lat
type (ele_struct), pointer :: ele
type (ibs_sim_param_struct) :: cfg
real(rp) :: clog(4)
real(rp), parameter :: lambda = 1e12_rp
integer :: i, k

call bmad_parser('proton.bmad', lat)
ele => lat%ele(1)
ele%a%beta = 10;  ele%a%alpha = 0;  ele%a%gamma = 0.1_rp;  ele%a%eta = 0;  ele%a%etap = 0
ele%b%beta = 10;  ele%b%alpha = 0;  ele%b%gamma = 0.1_rp;  ele%b%eta = 0;  ele%b%etap = 0
ele%a%emit = 1e-8_rp;  ele%b%emit = 1e-8_rp;  ele%z%sigma_p = 1e-3_rp
write(*, '(a,es25.16)') 'RADIUS ', classical_radius(proton$)
write(*, '(a)') '#      tau_a(s)            clog1                 clog2                 clog3                 clog4'
do k = -2, 12, 2
  cfg%tau_a = 10.0_rp**k
  do i = 1, 4
    cfg%clog_to_use = i
    call multi_coulomb_log(cfg, ele, clog(i), lambda)
  enddo
  write(*, '(es14.6, 4f22.15)') cfg%tau_a, clog
enddo
end program
"""
log = fortran(
    PROGRAM,
    files={
        "proton.bmad": """! One proton drift at gamma = 100; the probe sets the Twiss and beam sizes.
parameter[particle] = proton
parameter[geometry] = open
beginning[e_tot] = 100 * mass_of(proton)
d: drift, l = 1
l1: line = (d)
use, l1
"""
    },
).log
radius_rows = [line.split()[1] for line in log.splitlines() if line.startswith("RADIUS ")]
rows = [
    list(map(float, line.split()))
    for line in log.splitlines()
    if len(line.split()) == 5 and line.lstrip()[0].isdigit()
]
if len(radius_rows) != 1 or len(rows) != 8:
    raise RuntimeError(log)
radius = float(radius_rows[0])
c, gamma, beta, emit, sig_p, density = 299792458.0, 100.0, 10.0, 1e-8, 1e-3, 1e12
sigma = math.sqrt(beta * emit)
volume = (4 * math.pi) ** 1.5 * sigma**2 * gamma
bmax = min(sigma, (volume / density) ** (1 / 3))
classic = math.log(gamma**2 * sigma * emit / (radius * beta))
defect = False
for tau, log1, log2, log3, log4 in rows:
    if compare(f"tau={tau:g} classic control", log1, classic, atol=2e-13):
        raise RuntimeError("Classic logarithm control failed")
    for option, angle, observed in [
        (3, math.sqrt(emit / beta), log3),
        (4, math.sqrt(2 * emit / beta + (sig_p / gamma) ** 2), log4),
    ]:
        tail = math.sqrt(volume / density / math.pi / (tau / gamma)) / math.sqrt(c * gamma * angle)
        # The documented pi/2 scattering bound is the larger minimum impact parameter.
        minimum = radius / (angle * gamma) ** 2
        expected = math.log(bmax / max(tail, minimum))
        bad = compare(
            f"option {option} tau={tau:g} s; ln(bmax/max(classical,tail))",
            observed,
            expected,
            atol=2e-13,
        )
        if tau <= 1e4 and bad:
            raise RuntimeError("Strong-damping control failed")
        defect |= bad
verdict(defect)
