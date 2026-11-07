---
title: "OPT-40: Higher-order PTC compaction and slip series omit particle-speed conversion"
hide:
  - toc
---

# OPT-40 — Higher-order PTC compaction and slip series omit particle-speed conversion

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Discussed in | [[gh:https://github.com/bmad-sim/bmad-ecosystem/issues/325]] |
| Source | [ptc_map_to_normal_form, bmad/ptc/ptc_layout_mod.f90:825–839](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/ptc_layout_mod.f90#L825-L839)<br>[RF-on averaged branch, bmad/ptc/ptc_layout_mod.f90:848–851](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/ptc_layout_mod.f90#L848-L851)<br>[momentum-compaction accessor, tao/code/tao_evaluate_a_datum.f90:1777–1783](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_evaluate_a_datum.f90#L1777-L1783)<br>[slip-factor accessor, tao/code/tao_evaluate_a_datum.f90:2830–2836](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_evaluate_a_datum.f90#L2830-L2836)<br>[show chromaticity, tao/code/tao_show_this.f90:863–874](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_show_this.f90#L863-L874)<br>[ΔL(δ)/L₀, tao/doc/data.tex:1339–1356](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/data.tex#L1339-L1356)<br>[ΔT(δ)/T₀, tao/doc/data.tex:1787–1803](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/doc/data.tex#L1787-L1803)<br>[ordinary longitudinal-coordinate contract, bmad/doc/phase-space.tex:81–94](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/phase-space.tex#L81-L94) |

With RF off, the PTC normal-form momentum-compaction and slip-factor series (Tao's `momentum_compaction_ptc.N`, `slip_factor_ptc.N` and `show chromaticity`) convert the one-turn z advance to path length and flight time as if z used the reference speed β₀. Bmad's z = −βc(t − t₀) uses the particle's speed β, so coefficients of second and higher order in the fractional momentum deviation δ are wrong. For a straight proton cell at Lorentz factor γ₀ = 2, the δ² path coefficient is −0.06250 instead of zero and the δ² time coefficient is 0.2812 instead of 0.3437, from a relativistic mass-shell expansion of the returned PTC map. The PTC map itself is correct, and first-order coefficients are correct when the on-momentum orbit has zero net z advance.

Run `python OPT-40.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: OPT-40.py"

    [Download OPT-40.py](OPT-40.py)

    ```python
    """OPT-40: Higher-order PTC compaction and slip series omit particle-speed conversion

    With RF off, the PTC normal-form momentum-compaction and slip-factor series (Tao's `momentum_compaction_ptc.N`, `slip_factor_ptc.N` and `show chromaticity`) convert the one-turn z advance to path length and flight time as if z used the reference speed β₀. Bmad's z = −βc(t − t₀) uses the particle's speed β, so coefficients of second and higher order in the fractional momentum deviation δ are wrong. For a straight proton cell at Lorentz factor γ₀ = 2, the δ² path coefficient is −0.06250 instead of zero and the δ² time coefficient is 0.2812 instead of 0.3437, from a relativistic mass-shell expansion of the returned PTC map. The PTC map itself is correct, and first-order coefficients are correct when the on-momentum orbit has zero net z advance.

    Build straight periodic proton cells at Lorentz factors 2 and 3 with RF off. Compare normal-form coefficients, raw maps and finite-offset tracking with a 70-digit Decimal mass-shell oracle, preserving the separate Bmad and FOREST masses. Check the public Tao display and first-order controls using BMAD_DIST and the shared native runner.
    """

    import re
    from decimal import Decimal, localcontext

    import numpy as np
    from _bmad import compare, fortran, run, verdict

    PROGRAM = r"""
    program confirm
    use bmad
    use ptc_layout_mod
    use pointer_lattice
    implicit none
    type(lat_struct),target::lattice
    type(branch_struct),pointer::br
    type(coord_struct),allocatable::orb(:)
    type(ptc_normal_form_struct)::nf
    type(c_damap)::raw_map
    real(rp)::vtrack(6),dlt,mf
    integer j,ex(6),status
    logical err
    call bmad_parser('lat.bmad',lattice,err_flag=err)
    if(err) error stop 'parse'
    br=>lattice%branch(0)
    call twiss_and_track(lattice,orb,status)
    if(status/=ok$) error stop 'optics'
    call lat_to_ptc_layout(lattice)
    mf=br%ele(1)%ptc_fibre%mass*1e9_rp
    call ptc_one_turn_map_at_ele(br%ele(0),nf%orb0,nf%one_turn_map,nf%state,pz=0.0_rp,rf_on=no$)
    call alloc(nf%normal_form);call alloc(nf%phase);call alloc(nf%spin_tune)
    call alloc(nf%path_length);call alloc(nf%u_normal_form);call alloc(nf%u_path_length);call alloc(nf%isf)
    call ptc_map_to_normal_form(nf%one_turn_map,nf%state%nocavity,br,nf)
    call alloc(raw_map);raw_map=nf%one_turn_map
    write(*,'(a,5es25.16)') 'BASE ',br%param%total_length,br%ele(0)%value(p0c$),mass_of(proton$),mf, &
     momentum_compaction(br)
    write(*,*) 'STATE ',nf%state%time,nf%state%nocavity,use_bmad_units
    do j=1,3
     ex=0;ex(6)=j
     write(*,'(a,i2,3es25.16)') 'COEFF ',j,real(nf%phase(3).sub.ex)/br%param%total_length, &
     real(raw_map%v(5).sub.ex)/br%param%total_length, &
     real(nf%path_length.sub.ex)/br%param%total_length
    enddo
    do j=-2,2
     dlt=j*0.01_rp;vtrack=0;vtrack(6)=dlt
     call track_probe_x(vtrack,nf%state,fibre1=br%ele(0)%ptc_fibre%next)
     write(*,'(a,3es25.16)') 'RAW_TRACK ',dlt,vtrack(5),vtrack(6)
    enddo
    end program
    """

    LAT = """no_digested
    parameter[geometry]=closed
    parameter[particle]=proton
    parameter[p0c]=mass_of(proton)*sqrt({gamma}^2-1)
    parameter[taylor_order]=3
    qf:quadrupole,l=.18,k1=2.1,tracking_method=symp_lie_ptc,num_steps=8,fringe_type=none
    qd:quadrupole,l=.18,k1=-2.1,tracking_method=symp_lie_ptc,num_steps=8,fringe_type=none
    d:drift,l=.8,tracking_method=symp_lie_ptc
    cell:line=(qf,d,qd,d)
    ring:line=(6*cell)
    use,ring
    """


    def shell_coefficients(mass, momentum):
        """Coefficients of beta(delta)/beta(0), with delta=pc/P0c-1."""
        m, p = Decimal(str(mass)), Decimal(str(momentum))
        a = m**2 / (m**2 + p**2)
        return [a, -Decimal("1.5") * a * (1 - a), a * (1 - a) * (4 - 5 * a) / 2]


    def expected_series(p, mb, mf):
        with localcontext() as ctx:
            ctx.prec = 70
            b, f = shell_coefficients(mb, p), shell_coefficients(mf, p)
            # The physical Bmad clock uses beta_B; FOREST's map uses beta_F.
            # C/L = r_B - Z, T/T0 = 1 - Z/r_B, and analytically Z = r_F-1.
            path = [b[i] - f[i] for i in range(3)]
            time = [-f[0], -f[1] + f[0] * b[0], -f[2] + f[1] * b[0] - f[0] * (b[0] ** 2 - b[1])]
            return (
                np.array(f, dtype=float),
                np.array(path, dtype=float),
                np.array(time, dtype=float),
            )


    def main():
        defects = []
        for gamma in (2, 3):
            lattice = LAT.format(gamma=gamma)
            raw = fortran(PROGRAM, files={"lat.bmad": lattice}).log
            base = [x for x in raw.splitlines() if x.startswith("BASE ")]
            if len(base) != 1 or not re.search(r"STATE\s+T\s+T\s+T", raw):
                raise RuntimeError(raw)
            length, p, mb, mf, compaction = map(float, base[0].split()[1:])
            coeff = np.array(
                [[float(x) for x in s.split()[2:]] for s in raw.splitlines() if s.startswith("COEFF ")]
            )
            z, path, time = expected_series(p, mb, mf)
            print(f"gamma={gamma}: length={length:.17g} P0c={p:.17g} mass_B={mb:.17g} mass_F={mf:.17g}")
            for label, value, target in (
                ("normal phase z/L", coeff[:, 0], z),
                ("raw map z/L", coeff[:, 1], z),
                ("native alpha1", compaction, path[0]),
                ("PTC alpha1", coeff[0, 2], path[0]),
            ):
                if compare(
                    f"gamma={gamma} {label}: Decimal mass-shell oracle", value, target, atol=3e-13
                ):
                    raise AssertionError("analytic control failed")
            defects.append(
                compare(
                    f"gamma={gamma} physical path coefficients: mass-shell oracle",
                    coeff[:, 2],
                    path,
                    atol=3e-13,
                )
            )
            for s in raw.splitlines():
                if not s.startswith("RAW_TRACK "):
                    continue
                delta, tracked_z, tracked_delta = map(float, s.split()[1:])
                with localcontext() as ctx:
                    ctx.prec = 70
                    q, pm, mass = Decimal(str(delta)), Decimal(str(p)), Decimal(str(mf))
                    beta0 = pm / (pm**2 + mass**2).sqrt()
                    beta = pm * (1 + q) / ((pm * (1 + q)) ** 2 + mass**2).sqrt()
                    expected_z = float(Decimal(str(length)) * (beta / beta0 - 1))
                if compare(
                    f"gamma={gamma} delta={delta} returned Bmad z [m]",
                    tracked_z,
                    expected_z,
                    atol=1e-14,
                ):
                    raise AssertionError("finite-offset units check failed")
                if compare(f"gamma={gamma} delta identity", tracked_delta, delta, atol=3e-15):
                    raise AssertionError("momentum variable changed")
            commands = "set universe 1 one_turn_map_calc = T\nshow chromaticity\nquit\n"
            tao = run(
                "tao",
                ["-noplot", "-lat", "lat.bmad", "-noinit", "-startup", "run.tao"],
                files={"lat.bmad": lattice, "run.tao": commands},
            ).log
            if "slip_factor_ptc.N" not in tao or "Tao: quit" not in tao or "[ERROR" in tao:
                raise RuntimeError(tao)
            table = tao.split("slip_factor_ptc.N   momentum_compaction_ptc.N", 1)[1]
            displayed = np.array(
                [
                    [float(x) for x in line.split()[1:]]
                    for line in table.splitlines()
                    if re.match(r"^\s+[123]\s+[-+0-9]", line)
                ]
            )
            # Tao prints eight significant figures; check the public display
            # against the full-precision diagnostics, then against physical series.
            if compare(
                f"gamma={gamma} Tao path series versus native coefficients",
                displayed[:, 1],
                coeff[:, 2],
                atol=5e-9,
            ):
                raise AssertionError("Tao and native diagnostics differ")
            if compare(
                f"gamma={gamma} Tao eta1: mass-shell oracle", displayed[0, 0], time[0], atol=5e-9
            ):
                raise AssertionError("first-order time control failed")
            defects.append(
                compare(
                    f"gamma={gamma} Tao [time,path] series: mass-shell expectation",
                    displayed,
                    np.column_stack((time, path)),
                    atol=5e-9,
                )
            )
        verdict(all(defects))


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `claude-opus-5-5` | high | 2026-11-07 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

