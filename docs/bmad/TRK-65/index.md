---
title: "TRK-65: PTC retains the first initialized species across lattice parses"
hide:
  - toc
---

# TRK-65 — PTC retains the first initialized species across lattice parses

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [initialization contract, bmad/ptc/set_ptc.f90:7–9](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/set_ptc.f90#L7-L9)<br>[species argument, bmad/ptc/set_ptc.f90:19–21](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/set_ptc.f90#L19-L21)<br>[species-selection guard, bmad/ptc/set_ptc.f90:79–99](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/set_ptc.f90#L79-L99)<br>[later energy changes, bmad/ptc/set_ptc.f90:123–134](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/set_ptc.f90#L123-L134)<br>[explicit force_init, bmad/ptc/set_ptc.f90:61–61](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/set_ptc.f90#L61-L61)<br>[parser handoff, bmad/parsing/bmad_parser.f90:1151–1157](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/bmad_parser.f90#L1151-L1157)<br>[the FOREST constructor, forest/code/Sn_mad_like.f90:3291–3296](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/forest/code/Sn_mad_like.f90#L3291-L3296)<br>[element conversion, bmad/ptc/ele_to_fibre.f90:698–707](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/ele_to_fibre.f90#L698-L707)<br>[PTC element tracking, bmad/low_level/track1_symp_lie_ptc.f90:55–55](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track1_symp_lie_ptc.f90#L55-L55)<br>[spin transport, bmad/low_level/track1_symp_lie_ptc.f90:85–89](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track1_symp_lie_ptc.f90#L85-L89)<br>[absolute-time reconstruction, bmad/low_level/track1_symp_lie_ptc.f90:109–113](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track1_symp_lie_ptc.f90#L109-L113)<br>[PTC for higher-order Taylor maps, bmad/doc/ptc-programming.tex:25–27](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/ptc-programming.tex#L25-L27)<br>[parser initialization, bmad/doc/ptc-programming.tex:65–73](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/ptc-programming.tex#L65-L73)<br>[universe loop, tao/code/tao_init_lattice.f90:108–115](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_init_lattice.f90#L108-L115)<br>[the ordinary lattice parser, tao/code/tao_init_lattice.f90:199–204](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_init_lattice.f90#L199-L204) |

Parsing lattices with different particle species in one process leaves newly created PTC fibres using the first species' mass and magnetic anomaly. This affects PTC tracking and Taylor-map generation, including normal multi-universe initialization in Tao. An electron at 200.0 MeV/c in a 0.8000 m solenoid ends with vertical spin −0.9736 after a proton parse instead of −0.4623. A subsequent 3.700 m drift gives a linear longitudinal map coefficient of 3.539 m instead of 2.415×10⁻⁵ m, as independently established from the relativistic mass shell. Native orbital and spin tracking are history independent. Explicit PTC reinitialization restores the fresh-process results only when the first species is an electron, positron, proton or antiproton; after a muon or deuteron it resets the anomaly but keeps that species' mass.

Run `python TRK-65.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: TRK-65.py"

    [Download TRK-65.py](TRK-65.py)

    ```python
    """TRK-65: PTC retains the first initialized species across lattice parses

    Parsing lattices with different particle species in one process leaves newly created PTC fibres using the first species' mass and magnetic anomaly. This affects PTC tracking and Taylor-map generation, including normal multi-universe initialization in Tao. An electron at 200.0 MeV/c in a 0.8000 m solenoid ends with vertical spin −0.9736 after a proton parse instead of −0.4623. A subsequent 3.700 m drift gives a linear longitudinal map coefficient of 3.539 m instead of 2.415×10⁻⁵ m, as independently established from the relativistic mass shell. Native orbital and spin tracking are history independent. Explicit PTC reinitialization restores the fresh-process results only when the first species is an electron, positron, proton or antiproton; after a muon or deuteron it resets the anomaly but keeps that species' mass.

    The script uses BMAD_DIST and the shared runner to compile independent Bmad and Tao clients, starting each lattice history in a fresh native process. An 80-digit relativistic mass-shell calculation and on-axis Thomas–BMT rotation supply expected drift, Taylor and spin values. Native tracking, a fresh electron process and explicit PTC reinitialization provide controls, while a two-universe Tao run exercises the normal application path.
    """

    from __future__ import annotations

    import math
    from decimal import Decimal, localcontext

    from _bmad import fortran, verdict

    PROGRAM = r"""
    program confirmation
    use bmad
    use madx_ptc_module, only: bmadl
    implicit none
    type(lat_struct) :: lat, first_lat
    type(coord_struct) :: a,b
    real(rp) :: v(6), coef
    integer :: method
    character(100) :: action,history,arg
    logical :: err
    call get_command_argument(1,action)
    select case(trim(action))
    case('ptc')
     call get_command_argument(2,history)
     call get_command_argument(3,arg)
     method=symp_lie_ptc$
     if(trim(arg)=='native') method=bmad_standard$
     if(trim(history)/='fresh') then
      call bmad_parser('prior.bmad',first_lat,err_flag=err)
      if(err) error stop 'prior parse'
     endif
     call bmad_parser('electron.bmad',lat,err_flag=err)
     if(err) error stop 'electron parse'
     if(trim(history)=='reset') call set_ptc(lat%ele(0)%value(e_tot$),electron$,force_init=.true.)
     lat%ele(1)%tracking_method=method
     lat%ele(2)%tracking_method=method
     v=0
     call init_coord(a,v,lat%ele(1),upstream_end$)
     a%spin=[1.0_rp,0.0_rp,0.0_rp]
     call track1(a,lat%ele(1),lat%param,b,err_flag=err)
     if(err.or.b%state/=alive$) error stop 'solenoid tracking'
     write(*,'(a,6es26.17)') 'PTCSPIN ',b%spin,lat%ele(1)%value(bs_field$),mass_of(electron$),anomalous_moment_of(electron$)
     v=0; v(6)=-0.03_rp
     call init_coord(a,v,lat%ele(2),upstream_end$)
     call track1(a,lat%ele(2),lat%param,b,err_flag=err)
     if(err.or.b%state/=alive$) error stop 'drift tracking'
     write(*,'(a,4es26.17)') 'DRIFT ',b%vec(5),b%t-a%t,a%beta,lat%ele(2)%value(delta_ref_time$)
     call ele_to_taylor(lat%ele(2))
     coef=taylor_coef(lat%ele(2)%taylor(5),[0,0,0,0,0,1])
     write(*,'(a,3es26.17)') 'TAYLOR ',coef,bmadl%end%mass*1e9_rp,bmadl%end%ag
    case default
     error stop 'unknown action'
    end select
    end program
    """

    TAO_PROGRAM = r"""
    program tao_confirmation
    use tao_interface
    use madx_ptc_module, only: bmadl
    implicit none
    type(lat_struct), pointer :: lat
    type(coord_struct) :: a,b
    integer :: u,errcode
    real(rp) :: v(6)
    logical :: err
    character(40) :: mode
    call get_command_argument(1,mode)
    if(trim(mode)=='single') then
     call tao_top_level(command='-noinit -lat electron.bmad -noplot -nostartup',errcode=errcode)
    else
     call tao_top_level(command='-noinit -lat prior.bmad|electron.bmad -noplot -nostartup',errcode=errcode)
    endif
    if(errcode/=0) error stop 'Tao initialization'
    u=size(s%u)
    lat=>s%u(u)%model%lat
    if(lat%param%particle/=electron$) error stop 'final universe is not electron'
    v=0
    call init_coord(a,v,lat%ele(1),upstream_end$)
    a%spin=[1.0_rp,0.0_rp,0.0_rp]
    call track1(a,lat%ele(1),lat%param,b,err_flag=err)
    if(err.or.b%state/=alive$) error stop 'spin tracking'
    write(*,'(a,i4,5es26.17)') 'TAOSPIN ',u,b%spin,bmadl%end%mass*1e9_rp,bmadl%end%ag
    v=0; v(6)=-0.03_rp
    call init_coord(a,v,lat%ele(2),upstream_end$)
    call track1(a,lat%ele(2),lat%param,b,err_flag=err)
    if(err.or.b%state/=alive$) error stop 'drift tracking'
    write(*,'(a,2es26.17)') 'TAODRIFT ',b%vec(5),b%t-a%t
    end program
    """

    FILES = {
        "prior.bmad": """no_digested
    parameter[particle]=proton
    parameter[p0c]=200e6
    parameter[geometry]=open
    beginning[beta_a]=10
    beginning[beta_b]=10
    d: drift,l=1
    ln: line=(d)
    use,ln
    """,
        "electron.bmad": """no_digested
    parameter[particle]=electron
    parameter[p0c]=200e6
    parameter[taylor_order]=3
    parameter[geometry]=open
    beginning[beta_a]=10
    beginning[beta_b]=10
    bmad_com[spin_tracking_on]=T
    bmad_com[significant_length]=1e-14
    s: solenoid,l=0.8,ks=0.6,tracking_method=symp_lie_ptc,spin_tracking_method=tracking,integrator_order=6,ds_step=0.02
    d: drift,l=3.7,tracking_method=symp_lie_ptc
    ln: line=(s,d)
    use,ln
    """,
    }

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


    def mass_shell(mass: float) -> tuple[float, float, float]:
        # Bmad z=beta*c*(t_ref-t). At fixed reference momentum, the exact drift
        # gives z=L*(beta(P)/beta(P0)-1) and dz/ddelta=L*M^2/(P0^2+M^2).
        with localcontext() as context:
            context.prec = 80
            rest, p, length = Decimal(str(mass)), Decimal(200000000), Decimal("3.7")
            tracked = p * Decimal("0.97")
            beta0 = p / (p*p + rest*rest).sqrt()
            beta = tracked / (tracked*tracked + rest*rest).sqrt()
            return (float(length*(beta/beta0-1)),
                    float(length/(beta*Decimal(299792458))),
                    float(length*rest*rest/(p*p+rest*rest)))


    def main() -> None:
        runs: dict[tuple[str, str], dict[str, list[float]]] = {}
        for history in ("fresh", "history", "reset"):
            for method in ("ptc", "native"):
                log = fortran(PROGRAM, ["ptc", history, method], files=FILES).log
                result = {}
                for tag, count in (("PTCSPIN", 6), ("DRIFT", 4), ("TAYLOR", 3)):
                    rows = records(log, tag, count)
                    if len(rows) != 1:
                        raise RuntimeError(f"Expected one {tag} record")
                    result[tag] = rows[0]
                runs[history, method] = result
        fresh = runs["fresh", "ptc"]
        wrong = False
        for (history, method), result in runs.items():
            spin, drift, taylor = (result[k] for k in ("PTCSPIN", "DRIFT", "TAYLOR"))
            # FOREST and Bmad use slightly different constant tables. Each oracle uses
            # the constants of its own transport library to isolate species history.
            mass, anomaly = (taylor[1:3] if method == "ptc" else spin[4:6])
            angle = -0.8*0.6*(1+anomaly)
            require(spin[:3], [math.cos(angle), math.sin(angle), 0.0], 4e-15,
                    "BMT with actual transport anomaly")
            z, dt, _ = mass_shell(mass)
            require(drift[:1], [z], 5e-16 if method == "ptc" else 4e-15,
                    "Drift with actual transport mass")
            require(taylor[:1], [mass_shell(taylor[1])[2]], 4e-15,
                    "Taylor with actual fibre mass")
            if method == "native":
                require(drift[1:2], [dt], 3e-23, "Native mass-shell flight time")
                baseline = runs["fresh", "native"]
                for tag in ("PTCSPIN", "DRIFT"):
                    require(result[tag], baseline[tag], 0, "Native history independence")
            else:
                baseline = fresh
                # PTC supplies z using its fibre mass, but Bmad reconstructs time with
                # its own reference clock and coordinate beta. Keep both conventions.
                require(drift[1:2], [drift[3]-z/(drift[2]*299792458)], 3e-23,
                        "PTC flight time from Bmad clock and mass-shell z")
            if history == "reset":
                for tag in ("PTCSPIN", "DRIFT", "TAYLOR"):
                    require(result[tag], runs["fresh", method][tag], 0, "Explicit reset control")
            target_z, _, target_map = mass_shell(fresh["TAYLOR"][1])
            target_angle = -0.8*0.6*(1+fresh["TAYLOR"][2])
            target_spin = [math.cos(target_angle), math.sin(target_angle), 0.0]
            if method == "ptc":
                wrong |= not close(spin[:3], target_spin, 4e-15, "Fresh-species spin")
                wrong |= not close(drift[:1], [target_z], 5e-16, "Fresh-species drift")
            wrong |= not close(taylor[:1], [target_map], 4e-15, "Fresh-species Taylor")
            expected_anomaly = baseline["PTCSPIN"][5] if method == "native" else fresh["TAYLOR"][2]
            expected_sy = math.sin(-0.8*0.6*(1+expected_anomaly))
            print(f"{history} {method}: spin_y={spin[1]:.17g} "
                  f"expected_BMT={expected_sy:.17g}; "
                  f"z={drift[0]:.17g} expected_mass_shell="
                  f"{mass_shell(baseline['PTCSPIN'][4] if method == 'native' else fresh['TAYLOR'][1])[0]:.17g} m; "
                  f"dz/ddelta={taylor[0]:.17g} expected_mass_shell={target_map:.17g} m; "
                  f"fibre_mass={taylor[1]:.17g} eV anomaly={taylor[2]:.17g}")
        for mode, history in (("single", "fresh"), ("multi", "history")):
            log = fortran(TAO_PROGRAM, [mode], tao=True, files=FILES).log
            spin = records(log, "TAOSPIN", 6)[0]
            drift = records(log, "TAODRIFT", 2)[0]
            require(spin[:1], [1.0 if mode == "single" else 2.0], 0, "Universe count")
            reference = runs[history, "ptc"]
            require(spin[1:4], reference["PTCSPIN"][:3], 0, "Tao/direct spin equivalence")
            require(drift, reference["DRIFT"][:2], 0, "Tao/direct drift equivalence")
            wrong |= not close(spin[1:4], fresh["PTCSPIN"][:3], 4e-15, "Tao fresh species")
            print(f"Tao {mode}: spin_y={spin[2]:.17g} expected_fresh={fresh['PTCSPIN'][1]:.17g}; "
                  f"z={drift[0]:.17g} expected_mass_shell={mass_shell(fresh['TAYLOR'][1])[0]:.17g} m")
        excess = (runs["history", "ptc"]["DRIFT"][1]-fresh["DRIFT"][1])*1e12
        print(f"PTC proton-to-electron flight-time excess={excess:.17g} ps; "
              "native orbital/spin and explicit-reset controls pass")
        verdict(wrong)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `claude-opus-5-5` | high | 2026-11-07 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

