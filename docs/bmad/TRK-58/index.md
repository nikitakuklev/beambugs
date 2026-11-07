---
title: "TRK-58: Kinematic conversion helpers omit charge from magnetic rigidity"
hide:
  - toc
---

# TRK-58 — Kinematic conversion helpers omit charge from magnetic rigidity

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad 20261031-0 (`c4c629f`, committed 2026-10-31) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`convert_pc_to`, bmad/code/convert_pc_to.f90:4–47](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/convert_pc_to.f90#L4-L47)<br>[`convert_total_energy_to`, bmad/code/convert_total_energy_to.f90:4–53](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/convert_total_energy_to.f90#L4-L53)<br>[public subroutine list, bmad/doc/subroutines.tex:2358–2368](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/subroutines.tex#L2358-L2368)<br>[magnetic-rigidity definition, bmad/doc/electromagnetic-fields.tex:27–36](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/electromagnetic-fields.tex#L27-L36)<br>[bend-curvature relation, bmad/doc/elements.tex:715–724](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L715-L724)<br>[species and whole-particle input contract, bmad/doc/lattice-file.tex:812–854](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/lattice-file.tex#L812-L854)<br>[species table, sim_utils/interfaces/particle_species_mod.f90:57–91](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/sim_utils/interfaces/particle_species_mod.f90#L57-L91)<br>[`charge_of`, sim_utils/interfaces/particle_species_mod.f90:1486–1526](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/sim_utils/interfaces/particle_species_mod.f90#L1486-L1526)<br>[field-to-strength bookkeeping, bmad/code/attribute_bookkeeper.f90:210–231](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/attribute_bookkeeper.f90#L210-L231)<br>[the inverse conversion, bmad/code/attribute_bookkeeper.f90:246–271](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/attribute_bookkeeper.f90#L246-L271) |

Requesting magnetic rigidity from either kinematic conversion helper gives a positive unit-charge value even when the specified particle has another charge. The documented field times bend radius is momentum divided by signed particle charge. At 1 GeV/c, a helion with charge +2 receives 3.336 T m instead of 1.668 T m; an electron receives +3.336 T m instead of −3.336 T m. An independent SI-unit calculation and field-specified bends establish the discrepancy, while ordinary energy, momentum and speed outputs pass their controls.

Run `python TRK-58.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: TRK-58.py"

    [Download TRK-58.py](TRK-58.py)

    ```python
    """TRK-58: Kinematic conversion helpers omit charge from magnetic rigidity

    Requesting magnetic rigidity from either kinematic conversion helper gives a positive unit-charge value even when the specified particle has another charge. The documented field times bend radius is momentum divided by signed particle charge. At 1 GeV/c, a helion with charge +2 receives 3.336 T m instead of 1.668 T m; an electron receives +3.336 T m instead of −3.336 T m. An independent SI-unit calculation and field-specified bends establish the discrepancy, while ordinary energy, momentum and speed outputs pass their controls.

    The script uses BMAD_DIST and the shared runner to compile a Fortran client for nine charged species at two independent momentum/energy inputs. A 60-digit SI mass-shell calculation supplies the expected signed rigidity and ordinary kinematic outputs, including positive-unit-charge controls. Fresh-process field-specified bends check Bmad's field-times-radius contract, and each helper is called with full keyword outputs, rigidity alone and the seventh positional argument.
    """

    from __future__ import annotations

    import math
    from decimal import Decimal, localcontext

    from _bmad import fortran, verdict

    PROGRAM = r"""
    program independent_rigidity
    use bmad
    implicit none
    character(100) :: name, mode
    integer :: species_code
    real(rp) :: p, energy, ep, pe, bp, be, gp, ge, kp, ke, b1p, b1e
    real(rp) :: rp_full, re_full, rp_only, re_only, rp_pos, re_pos, a, b, c, d
    logical :: errp, erre
    type(lat_struct) :: lat
    call get_command_argument(1, name)
    call get_command_argument(2, mode)
    species_code = species_id(trim(name))
    if (species_code == invalid$) error stop 'Invalid species'
    if (trim(mode) == 'meta') then
      write(*,'(a,1x,i0,1x,i0,1x,es26.17)') 'META', species_code, charge_of(species_code), mass_of(species_code)
      stop
    endif
    if (trim(mode) == 'bend') then
      call bmad_parser('field.bmad', lat, err_flag=errp)
      if (errp) error stop 'Bend parse failed'
      if (lat%param%particle /= species_code) error stop 'Bend species mismatch'
      write(*,'(a,3(1x,es26.17))') 'BEND', lat%ele(1)%value(p0c$), &
          lat%ele(1)%value(g$), lat%ele(1)%value(b_field$)
      stop
    endif
    read(*,*) p, energy
    errp = .true.
    erre = .true.
    call convert_pc_to(p, species_code, E_tot=ep, gamma=gp, kinetic=kp, beta=bp, &
        brho=rp_full, beta1=b1p, err_flag=errp)
    call convert_total_energy_to(energy, species_code, gamma=ge, kinetic=ke, beta=be, &
        pc=pe, brho=re_full, beta1=b1e, err_flag=erre)
    if (errp .or. erre) error stop 'Valid input rejected'
    call convert_pc_to(p, species_code, brho=rp_only)
    call convert_total_energy_to(energy, species_code, brho=re_only)
    call convert_pc_to(p, species_code, a, b, c, d, rp_pos)
    call convert_total_energy_to(energy, species_code, a, b, c, d, re_pos)
    write(*,'(a,20(1x,es26.17))') 'VALUES', p, energy, rp_full, re_full, rp_only, &
        re_only, rp_pos, re_pos, ep, pe, bp, be, gp, ge, kp, ke, b1p, b1e
    ! A below-rest-energy input retains its existing explicit error contract.
    errp = .false.
    call convert_total_energy_to(0.5_rp*mass_of(species_code), species_code, err_flag=errp, print_err=.false.)
    if (.not. errp) error stop 'Invalid energy accepted'
    end program
    """

    SPECIES = (
        ("positron", 1),
        ("electron", -1),
        ("proton", 1),
        ("antiproton", -1),
        ("helion", 2),
        ("anti_helion", -2),
        ("#12C+", 1),
        ("#12C++", 2),
        ("#12C+6", 6),
    )


    def row(log: str, tag: str, count: int) -> list[float]:
        rows = [line.split()[1:] for line in log.splitlines() if line.startswith(tag + " ")]
        if len(rows) != 1 or len(rows[0]) != count:
            raise RuntimeError(f"Expected one complete {tag} record:\n{log}")
        values = [float(value) for value in rows[0]]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Nonfinite {tag} record: {values}")
        return values


    def equal(actual: float, expected: float, label: str, rtol: float = 3e-15) -> None:
        if not math.isclose(actual, expected, rel_tol=rtol, abs_tol=0):
            raise RuntimeError(f"{label}: observed={actual:.17g} expected={expected:.17g}")


    def main() -> None:
        discrepancies = 0
        for name, charge in SPECIES:
            _, q, mass = row(fortran(PROGRAM, [name, "meta"]).log, "META", 3)
            equal(q, charge, "Species charge", rtol=0)
            with localcontext() as context:
                context.prec = 60
                # Exact SI definitions: an eV is e joules and c is in metres/second.
                # The rest-energy table is an explicit input, avoiding a mass convention change.
                elementary_charge = Decimal("1.602176634e-19")
                speed = Decimal(299792458)
                rest_ev = Decimal.from_float(mass)
                for regime in ("gamma_1.25", "pc_1_GeV"):
                    # The 3-4-5 mass-shell triangle independently sets pc=3M/4 and E=5M/4.
                    target_pc = Decimal("0.75") * rest_ev if regime == "gamma_1.25" else Decimal(10**9)
                    target_energy = (
                        Decimal("1.25") * rest_ev
                        if regime == "gamma_1.25"
                        else (target_pc**2 + rest_ev**2).sqrt()
                    )
                    p, energy = float(target_pc), float(target_energy)
                    values = row(
                        fortran(PROGRAM, [name, "probe"], stdin=f"{p:.17g} {energy:.17g}\n").log,
                        "VALUES",
                        18,
                    )
                    (
                        p_in,
                        e_in,
                        rp,
                        re,
                        rp_only,
                        re_only,
                        rp_pos,
                        re_pos,
                        ep,
                        pe,
                        bp,
                        be,
                        gp,
                        ge,
                        kp,
                        ke,
                        b1p,
                        b1e,
                    ) = values
                    equal(p_in, p, "Momentum input", rtol=0)
                    equal(e_in, energy, "Energy input", rtol=0)
                    p_si = Decimal.from_float(p_in) * elementary_charge / speed
                    energy_j = Decimal.from_float(e_in) * elementary_charge
                    rest_j = rest_ev * elementary_charge
                    # Lorentz force gives B*rho=p_SI/(Q*e). Energy input uses the SI mass shell.
                    pe_si = ((energy_j - rest_j) * (energy_j + rest_j)).sqrt() / speed
                    expected_p = float(p_si / (Decimal(charge) * elementary_charge))
                    expected_e = float(pe_si / (Decimal(charge) * elementary_charge))
                    expected_ep = (Decimal.from_float(p_in) ** 2 + rest_ev**2).sqrt()
                    expected_pe = pe_si * speed / elementary_charge
                    expectations = (
                        (ep, expected_ep, "E(p)"),
                        (pe, expected_pe, "p(E)"),
                        (bp, Decimal.from_float(p_in) / expected_ep, "beta(p)"),
                        (be, expected_pe / Decimal.from_float(e_in), "beta(E)"),
                        (gp, expected_ep / rest_ev, "gamma(p)"),
                        (ge, Decimal.from_float(e_in) / rest_ev, "gamma(E)"),
                        (kp, expected_ep - rest_ev, "kinetic(p)"),
                        (ke, Decimal.from_float(e_in) - rest_ev, "kinetic(E)"),
                        (b1p, 1 - Decimal.from_float(p_in) / expected_ep, "1-beta(p)"),
                        (b1e, 1 - expected_pe / Decimal.from_float(e_in), "1-beta(E)"),
                    )
                    control_rtol = 3e-15 if regime == "gamma_1.25" else 5e-14
                    for actual, expected, label in expectations:
                        equal(actual, float(expected), f"{name} {regime} {label}", control_rtol)
                    for variant, full in ((rp_only, rp), (rp_pos, rp), (re_only, re), (re_pos, re)):
                        equal(variant, full, "Optional/positional call equivalence", rtol=0)
                    for actual, expected in ((rp, expected_p), (re, expected_e)):
                        mismatch = not math.isclose(actual, expected, rel_tol=3e-15, abs_tol=0)
                        if charge == 1:
                            equal(actual, expected, "Positive unit-charge control")
                        elif mismatch:
                            equal(actual / expected, charge, "Missing-charge factor")
                            discrepancies += 3  # full keyword, rigidity-only, positional
                    residual = max(
                        abs(actual / float(expected) - 1) for actual, expected, _ in expectations
                    )
                    print(
                        f"{name} {regime} Q={charge:+d}: "
                        f"brho_p={rp:.17g} expected_SI_p={expected_p:.17g} "
                        f"brho_E={re:.17g} expected_SI_E={expected_e:.17g} T m; "
                        f"p_SI/(Q e), kinematic_max_rel={residual:.3g}, call_modes_equal"
                    )
            # Specify physical field rather than curvature; parsing starts in a fresh process.
            lattice = (
                "no_digested\n"
                f"parameter[particle]={name}\nparameter[p0c]=7.5e8\n"
                "parameter[geometry]=open\n"
                "b: sbend, l=0.3, b_field=-0.37, field_master=T\n"
                "ln: line=(b)\nuse, ln\n"
            )
            bend_p, bend_g, bend_b = row(
                fortran(PROGRAM, [name, "bend"], files={"field.bmad": lattice}).log, "BEND", 3
            )
            equal(bend_p, 7.5e8, "Specified bend momentum", rtol=0)
            equal(bend_b, -0.37, "Specified bend field", rtol=0)
            expected_g = charge * 299792458 * (-0.37) / 7.5e8
            expected_r = 7.5e8 / (charge * 299792458)
            equal(bend_g, expected_g, "Field-specified curvature")
            equal(bend_b / bend_g, expected_r, "Field/curvature rigidity")
            print(
                f"{name} field_master control: g={bend_g:.17g} expected_QcB/pc={expected_g:.17g} m^-1; "
                f"B/g={bend_b / bend_g:.17g} expected_SI={expected_r:.17g} T m"
            )
        print(
            "18 input pairs, 108 rigidity outputs, 180 kinematic controls, "
            "18 invalid-energy controls, 9 field-specified bend controls; "
            f"{discrepancies} rigidity discrepancies"
        )
        verdict(discrepancies > 0)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

