---
title: "COL-45: Potential-well routines omit the ion charge state"
hide:
  - toc
---

# COL-45 — Potential-well routines omit the ion charge state

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [pwd_mat and its particle-count/inductance contract](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/longitudinal_profile_mod.f90#L658-L699)<br>[set_pwd_ele residual](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/longitudinal_profile_mod.f90#L611-L625)<br>[bl_via_vlassov input contract and parameter construction](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/ibs_mod.f90#L832-L872)<br>[psi_prime voltage and per-particle loss convention](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/longitudinal_profile_mod.f90#L27-L66)<br>[ibs_ring PWD disablement](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bsim/ibs_ring/ibs_ring.f90#L109-L113) |

Calling the potential-well matrix or Vlasov bunch-length routines for multiply charged ions
gives unit-charge results. The matrix kick requires Q² at fixed particle count, while the
Vlasov equation requires Q times the RF and wake voltages, where Q is the ion charge in
elementary-charge units. For carbon ions with Q = 6, the matrix slope is −5.745 × 10⁻⁵ m⁻¹
instead of −2.068 × 10⁻³ m⁻¹; the no-wake bunch length is 17.92 mm instead of 7.304 mm. The
public routines are affected; the ibs_ring application disables potential-well distortion.

Run `python COL-45.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-45.py"

    [Download COL-45.py](COL-45.py)

    ```python
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

    ```

??? example "Reproducer: COL-45-vlassov.py"

    [Download COL-45-vlassov.py](COL-45-vlassov.py)

    ```python
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

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-24 | ibs-ion-classical-radius (`886607535af8`), source, production, CPU, double, serial |
| Confirmed<br><small>pwd_mat charge-state scaling</small> | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed<br><small>Vlasov bunch-length charge-state scaling</small> | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |

