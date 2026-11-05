"""COL-47: Raubenheimer tail-cut bound

The script independently integrates the Raubenheimer expression over the production integration interval and caps its maximum momentum transfer at the classical π/2 boundary. It checks dispersive and dispersion-free elements over eight damping times, retaining the separately measured quadrature residual.
"""

import math

from _bmad import compare, fortran, verdict
from scipy.integrate import quad

PROGRAM = """! multi_coulomb_log options 1 and 2 (Raubenheimer tail cut) for one proton element at gamma 100
! while the damping time tau_a grows from 1e-2 s to 1e12 s, line density 1e12 /m:
!   nodisp -- the COL-47 element: beta_a = beta_b = 10 m, emit_a = emit_b = 1e-8 m, sigma_p = 1e-3;
!   disp   -- the same with alpha_a = 0.5, eta_a = 1 m, etap_a = 0.05.
! Columns: label, beta_a, alpha_a, eta_a, etap_a, beta_b, emit_a, emit_b, sigma_p, lambda, tau_a,
! clog1, clog2. The Python oracle rebuilds option 2 from Raubenheimer's Eqs. (12) and (15).
program probe
use bmad
use ibs_mod
implicit none
type (lat_struct), target :: lat
type (ele_struct), pointer :: ele
type (ibs_sim_param_struct) :: cfg
integer :: k

call bmad_parser('proton.bmad', lat)
write(*, '(a,es25.16)') 'RADIUS ', classical_radius(proton$)
ele => lat%ele(1)
do k = -2, 12, 2
  call one('nodisp', 10.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 10.0_rp**k)
enddo
do k = -2, 12, 2
  call one('disp', 10.0_rp, 0.5_rp, 1.0_rp, 0.05_rp, 10.0_rp**k)
enddo

contains

subroutine one(label, beta_a, alpha_a, eta_a, etap_a, tau_a)
character(*) :: label
real(rp) :: beta_a, alpha_a, eta_a, etap_a, tau_a, clog(2)
real(rp), parameter :: lambda = 1e12_rp
integer :: i
ele%a%beta = beta_a;  ele%a%alpha = alpha_a;  ele%a%gamma = (1 + alpha_a**2) / beta_a
ele%a%eta = eta_a;  ele%a%etap = etap_a
ele%b%beta = 10;  ele%b%alpha = 0;  ele%b%gamma = 0.1_rp;  ele%b%eta = 0;  ele%b%etap = 0
ele%a%emit = 1e-8_rp;  ele%b%emit = 1e-8_rp;  ele%z%sigma_p = 1e-3_rp
cfg%tau_a = tau_a
do i = 1, 2
  cfg%clog_to_use = i
  call multi_coulomb_log(cfg, ele, clog(i), lambda)
enddo
write(*, '(a, 1x, 10es24.16, 2f22.15)') label, beta_a, alpha_a, eta_a, etap_a, ele%b%beta, ele%a%emit, &
                                      ele%b%emit, ele%z%sigma_p, lambda, tau_a, clog
end subroutine

end program
"""
log = fortran(
    PROGRAM,
    files={
        "proton.bmad": """! One proton drift; the probe sets the Twiss and beam sizes at gamma 100.
parameter[particle] = proton
parameter[geometry] = open
beginning[e_tot] = 100 * mass_of(proton)
d: drift, l = 1
l1: line = (d)
use, l1
"""
    },
).log
radius = [float(line.split()[1]) for line in log.splitlines() if line.startswith("RADIUS ")]
rows = [line.split() for line in log.splitlines() if line.startswith(("nodisp ", "disp "))]
if len(radius) != 1 or len(rows) != 16:
    raise RuntimeError(log)
r, gamma, c = radius[0], 100.0, 299792458.0
defect = False
for row in rows:
    label = row[0]
    ba, aa, eta, etap, bb, ea, eb, sp, density, tau, log1, observed = map(float, row[1:])
    g2, sp2 = gamma**2, sp**2
    ha = (eta**2 + (ba * etap + aa * eta) ** 2) / ba
    u = g2 * (ha / ea + 1 / sp2 + ba / g2 / ea + bb / g2 / eb)
    v = g2 * (
        ha * bb / ea / eb + ba * bb / g2 / ea / eb + ba / sp2 / ea + bb / sp2 / eb + eta**2 / ea**2
    )
    w = g2 * (eta**2 * bb / ea**2 / eb + ba * bb / sp2 / ea / eb)

    # Raubenheimer Eq. (15), x=exp(2y). Scaling the integrand prevents a tiny
    # absolute tolerance from terminating integration before relative accuracy is reached.
    def integrand(y, u=u, v=v, w=w):
        return (
            2e10
            * math.exp(y)
            / math.sqrt(math.exp(6 * y) + u * math.exp(4 * y) + v * math.exp(2 * y) + w)
        )

    integral, error = quad(integrand, 0, 40, epsabs=0, epsrel=2e-12)
    if error > abs(integral) * 1e-10:
        raise RuntimeError("Independent quadrature did not converge")
    bbar = gamma * math.sqrt((1 + aa**2) / ba * ea + eb / bb + sp2 / g2)
    qmin = 2 * r / math.sqrt(bb * eb) / bbar
    qstar = math.sqrt(
        tau * density * c * r * r / (4 * math.pi * g2 * ea * eb * sp) * integral / 1e10
    )
    reconstructed = math.log(qstar / qmin)
    cap = math.log(bbar / qmin)
    # The existing quadrature has a separately established <8e-5 log offset.
    # Bound violations are O(1); controls below the cap isolate that offset.
    if abs(observed - reconstructed) > 8e-5:
        raise RuntimeError("Option-2 result differs from its independently integrated formula")
    expected = min(reconstructed, cap)
    print(
        f"{label} tau={tau:g} s: independent integral={integral / 1e10:.16g} quadrature log residual={observed - reconstructed:.8g}"
    )
    bad = compare(
        f"{label} option 2 tau={tau:g} s; ln(min(qstar,Bbar)/qmin)",
        observed,
        expected,
        # Once capped, the expected log is algebraic and has no quadrature error.
        atol=2e-13 if qstar >= bbar else 8e-5,
    )
    if tau <= 1e4 and bad:
        raise RuntimeError("Strong-damping option-2 control failed")
    defect |= bad
verdict(defect)
