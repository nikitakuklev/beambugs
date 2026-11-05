"""COL-45: Vlasov bunch-length charge scaling

This script evaluates the charge-explicit profile equation with DOP853 and a normalization root solve, and compares six production-library cases. A closed-form no-wake profile and per-charge inputs provide independent controls.
"""

import math
import struct

from _bmad import compare, fortran, verdict
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

PROGRAM = """! bl_via_vlassov bunch lengths for protons and 12C6+ at gamma 100 in one ring (C = 3000 m,
! alpha = 1e-3, sigma_p = 1e-4, Vrf = 10 MV at 500 MHz), N = 1e11 particles, current = N*Q*e/T0.
!   nat -- R = L = 0, U0 = 0 (no potential-well distortion);
!   pwd -- R = 50 Ohm, L = 5 nH, U0 = 200 keV per particle.
! "particle" passes the per-particle beam energy and energy loss in eV, as the header describes;
! "per_charge" passes both divided by the charge state Q.
! Columns: label, Q, Energy (eV), U0 (eV), current (A), R (Ohm), L (H), sigma_z (m).
program probe
use bmad
use ibs_mod
implicit none
real(rp), parameter :: circ = 3000, alpha = 1e-3_rp, sigma_p = 1e-4_rp, vrf = 1e7_rp
real(rp), parameter :: omega = twopi * 500e6_rp, n_part = 1e11_rp, gamma = 100
real(rp) :: t0, e_p, e_c
integer :: c6

t0 = circ / c_light
c6 = species_id('#12C+6')
e_p = gamma * mass_of(proton$)
e_c = gamma * mass_of(c6)
call one('nat_proton',       1, e_p,     0.0_rp,      0.0_rp,  0.0_rp)
call one('nat_C6_particle',  6, e_c,     0.0_rp,      0.0_rp,  0.0_rp)
call one('nat_C6_per_charge',6, e_c / 6, 0.0_rp,      0.0_rp,  0.0_rp)
call one('pwd_proton',       1, e_p,     2e5_rp,      50.0_rp, 5e-9_rp)
call one('pwd_C6_particle',  6, e_c,     2e5_rp,      50.0_rp, 5e-9_rp)
call one('pwd_C6_per_charge',6, e_c / 6, 2e5_rp / 6,  50.0_rp, 5e-9_rp)

contains

subroutine one(label, q, energy, u0, r, l)
character(*) :: label
integer :: q
real(rp) :: energy, u0, r, l, current, sigma_z
current = n_part * q * e_charge / t0
call bl_via_vlassov(current, alpha, energy, sigma_p, vrf, omega, u0, circ, r, l, sigma_z)
write(*, '(a, 1x, i2, 6es24.16)') label, q, energy, u0, current, r, l, sigma_z
end subroutine

end program
"""
c = 2.99792458e8
circ, alpha, sigma_p, vrf, omega = 3000.0, 1e-3, 1e-4, 1e7, 2 * math.pi * 500e6
t0 = circ / c
bound = 500e-12
fwhm_per_sigma = 2 * math.sqrt(2 * math.log(2))
# get_bl_from_fwhm's literal 2.354820045030949 has no _rp kind, so it is rounded to single
# precision; dividing the ratio by this factor removes that 1.36e-8 offset.
literal = struct.unpack("f", struct.pack("f", 2.354820045030949))[0]
literal_factor = fwhm_per_sigma / literal


def bunch_length(q, e0, u0, current, r, ind):
    a = e0 / ((sigma_p * e0) ** 2 * alpha * t0)
    qb = current * t0
    phi = -math.acos(u0 / (q * vrf))

    def rhs(t, y):
        psi = y[0]
        dpsi = -a * psi * (q * vrf * math.cos(omega * t + phi) + q * qb * r * psi - u0)
        return [dpsi / (1 + a * q * qb * ind * psi), psi]

    def solve(psi0, t_end):
        result = solve_ivp(
            rhs,
            (0, t_end),
            [psi0, 0],
            method="DOP853",
            rtol=1e-12,
            atol=[1e-30, 1e-30],
            dense_output=True,
        )
        if not result.success:
            raise RuntimeError(result.message)
        return result

    def norm_error(psi0):
        return solve(psi0, bound).y[1, -1] - solve(psi0, -bound).y[1, -1] - 1

    psi0 = brentq(norm_error, 1e8, 1e12, xtol=1e-6, rtol=1e-14)
    neg, pos = solve(psi0, -bound).sol, solve(psi0, bound).sol

    def psi(t):
        return (neg if t < 0 else pos)(t)[0]

    t_max = brentq(
        lambda t: q * vrf * math.cos(omega * t + phi) + q * qb * r * psi(t) - u0,
        -bound,
        0,
        xtol=1e-22,
        rtol=1e-15,
    )
    half = psi(t_max) / 2
    t_lo = brentq(lambda t: psi(t) - half, -bound, t_max, xtol=1e-22, rtol=1e-15)
    t_hi = brentq(lambda t: psi(t) - half, t_max, bound, xtol=1e-22, rtol=1e-15)
    return c * (t_hi - t_lo) / fwhm_per_sigma


log = fortran(PROGRAM).log
rows = [line.split() for line in log.splitlines() if line.startswith(("nat_", "pwd_"))]
if len(rows) != 6:
    raise RuntimeError(log)
defect = False
for row in rows:
    label, q = row[0], int(row[1])
    energy, u0, current, resistance, inductance, observed = map(float, row[2:])
    scale = q if label.endswith("per_charge") else 1
    e0, loss = energy * scale, u0 * scale
    expected = bunch_length(q, e0, loss, current, resistance, inductance)
    if label.startswith("nat_"):
        k = q * vrf / (sigma_p**2 * e0 * alpha * t0 * omega)
        closed = c * 2 * math.acos(1 - math.log(2) / k) / omega / fwhm_per_sigma
        if compare(
            label + " independent ODE vs closed-form FWHM, m", expected, closed, atol=0, rtol=2e-10
        ):
            raise RuntimeError("Independent no-wake solver control failed")
    # Account explicitly for Bmad's single-precision FWHM divisor; this isolates charge scaling.
    bad = compare(
        label + " sigma_z, m; charge-explicit profile equation",
        observed / literal_factor,
        expected,
        atol=0,
        rtol=3e-10,
    )
    if (q == 1 or scale == q) and bad:
        raise RuntimeError("Unit-charge or per-charge Vlasov control failed")
    defect |= bad
verdict(defect)
