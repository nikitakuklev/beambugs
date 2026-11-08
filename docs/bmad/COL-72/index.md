---
title: "COL-72: Wake evaluation uses the reference frame instead of the placed structure frame"
hide:
  - toc
---

# COL-72 — Wake evaluation uses the reference frame instead of the placed structure frame

| | |
|---|---|
| Code | [Bmad](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [bunch dispatch, bmad/multiparticle/beam_mod.f90:325–329](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_mod.f90#L325-L329)<br>[first-half tracking and wake calls, bmad/multiparticle/beam_utils.f90:97–141](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L97-L141)<br>[exit transform, bmad/low_level/track_a_lcavity.f90:204–204](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_lcavity.f90#L204-L204)<br>[coordinate-transform contract, bmad/code/offset_particle.f90:10–19](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/offset_particle.f90#L10-L19)<br>[short-range longitudinal factors, bmad/multiparticle/wake_mod.f90:343–380](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L343-L380)<br>[short-range transverse factors, bmad/multiparticle/wake_mod.f90:440–483](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L440-L483)<br>[long-range multipoles and deposits, bmad/multiparticle/wake_mod.f90:214–276](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L214-L276)<br>[offsets, bmad/doc/attributes.tex:563–579](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L563-L579)<br>[tilt, bmad/doc/attributes.tex:608–609](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L608-L609)<br>[transverse wake equation, bmad/doc/wakefields.tex:56–65](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/wakefields.tex#L56-L65)<br>[pseudo-mode syntax, bmad/doc/attributes.tex:3036–3064](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L3036-L3064)<br>[long-range wake equations, bmad/doc/wakefields.tex:95–146](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/wakefields.tex#L95-L146)<br>[tabulated deposition and gathering, bmad/multiparticle/wake_mod.f90:563–607](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L563-L607)<br>[longitudinal scale, bmad/multiparticle/wake_mod.f90:324–325](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L324-L325)<br>[transverse scale, bmad/multiparticle/wake_mod.f90:420–421](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/wake_mod.f90#L420-L421)<br>[single-particle contract, bmad/doc/attributes.tex:2974–2977](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L2974-L2977) |

When bunch tracking applies wakes in an offset or tilted cavity, the wake calculation uses particle coordinates and kick directions in the local reference frame instead of the structure frame. Translating a cavity and its on-axis bunch horizontally by 4.000 mm produces a normalized transverse momentum kick of −3.360 × 10⁻⁶ instead of zero. Independent constant-wake products and rigid rotations establish the error. Short-range source- and witness-dependent modes, transverse-dependent longitudinal modes and long-range dipole excitation are affected; centered, untilted controls give the expected kicks.

Run `python COL-72.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: COL-72.py"

    [Download COL-72.py](COL-72.py)

    ```python
    """COL-72: Wake evaluation uses the reference frame instead of the placed structure frame

    When bunch tracking applies wakes in an offset or tilted cavity, the wake calculation uses particle coordinates and kick directions in the local reference frame instead of the structure frame. Translating a cavity and its on-axis bunch horizontally by 4.000 mm produces a normalized transverse momentum kick of −3.360 × 10⁻⁶ instead of zero. Independent constant-wake products and rigid rotations establish the error. Short-range source- and witness-dependent modes, transverse-dependent longitudinal modes and long-range dipole excitation are affected; centered, untilted controls give the expected kicks.

    A charged electron source and a zero-charge electron witness cross a field-free traveling-wave cavity in fresh native processes. Constant short-range modes reduce the expectation to one analytic product; a separate-bunch dipole uses a quarter-period delay. Centered, disabled-wake and three body-tracker controls accompany translations and a quarter-turn roll, using BMAD_DIST for the production build.
    """

    from itertools import product
    from pathlib import Path

    import numpy as np
    from _bmad import _case, _execute, _recipe, _scratch, compare, verdict

    PROGRAM = r"""
    program confirmation
    use bmad
    use beam_mod
    implicit none
    type(lat_struct), target :: lat
    type(bunch_struct) bunch
    real(rp) source(6), witness(6), delay
    integer kind,j
    logical err
    open(21,file='input.dat',status='old'); read(21,*) kind,delay; read(21,*) source; read(21,*) witness; close(21)
    call bmad_parser('lat.bmad',lat,err_flag=err)
    if(err) error stop 'lattice rejected'
    bmad_com%sr_wakes_on=(kind/=3 .and. kind>0)
    bmad_com%lr_wakes_on=(kind==3)
    bmad_com%radiation_damping_on=.false.
    bmad_com%radiation_fluctuations_on=.false.
    bmad_com%csr_and_space_charge_on=.false.
    if(kind/=3) then
      call reallocate_bunch(bunch,2)
      call init_coord(bunch%particle(1),source,lat%ele(0),downstream_end$)
      call init_coord(bunch%particle(2),witness,lat%ele(0),downstream_end$)
      bunch%particle(1)%charge=-1.2e-9_rp
      bunch%particle(2)%charge=0
      bunch%charge_tot=-1.2e-9_rp
      call track1_bunch(bunch,lat%ele(1),err)
    else
      call reallocate_bunch(bunch,1)
      call init_coord(bunch%particle(1),source,lat%ele(0),downstream_end$)
      bunch%particle(1)%charge=-1.2e-9_rp
      bunch%charge_tot=-1.2e-9_rp
      call track1_bunch(bunch,lat%ele(1),err)
      if(err) error stop 'source failed'
      write(*,'(a,4es26.17)') 'STORED ',lat%ele(1)%wake%lr%mode(1)%b_sin, &
        lat%ele(1)%wake%lr%mode(1)%b_cos,lat%ele(1)%wake%lr%mode(1)%a_sin,lat%ele(1)%wake%lr%mode(1)%a_cos
      call init_coord(bunch%particle(1),witness,lat%ele(0),downstream_end$)
      bunch%particle(1)%t=bunch%particle(1)%t+delay
      bunch%particle(1)%charge=0
      bunch%charge_tot=0
      call track1_bunch(bunch,lat%ele(1),err)
    endif
    if(err .or. any(bunch%particle%state/=alive$)) error stop 'tracking failed'
    j=size(bunch%particle)
    write(*,'(a,6es26.17)') 'RESULT ',bunch%particle(j)%vec
    write(*,'(a,2es26.17)') 'PLACEMENT ',lat%ele(1)%value(x_offset_tot$),lat%ele(1)%value(y_offset_tot$)
    end program
    """

    SOURCE = np.array([0.0008, -0.0006])
    WITNESS = np.array([-0.0013, 0.0009])
    DISPLACEMENT = np.array([0.004, -0.0025])
    # Integrated constant wake: K = A_bar |q_source| / (P0 c).
    COUPLING = 7e17 * 1.2e-9 / 1e12


    def lattice(
        mode: str, d: np.ndarray, length: float, angle: float = 0, method: str = "bmad_standard"
    ) -> str:
        return f"""no_digested
    parameter[geometry]=open
    parameter[particle]=electron
    beginning[p0c]=1e12
    e: lcavity,l={length},voltage=0,rf_frequency=1e9,cavity_type=traveling_wave,fringe_type=none, &
     tracking_method={method},tilt={angle:.17g},x_offset={d[0]:.17g},y_offset={d[1]:.17g}, &
     {mode}
    lat: line=(e)
    use,lat
    """


    def measure(
        work: Path,
        number: int,
        lat: str,
        source: np.ndarray,
        witness: np.ndarray,
        kind: int,
        delay: float = 0,
    ) -> np.ndarray:
        data = f"{kind} {delay:.17g}\n"
        for xy, z in ((source, 0 if kind == 3 else 0.002), (witness, 0 if kind == 3 else -0.001)):
            data += " ".join(map(str, [xy[0], 0, xy[1], 0, z, 0])) + "\n"
        result = _case(
            [str(work / "probe")], work / str(number), {"lat.bmad": lat, "input.dat": data}, (), "", 290
        )
        lines = [line for line in result.log.splitlines() if line.startswith("RESULT ")]
        if len(lines) != 1:
            raise RuntimeError(result.log)
        values = np.array(list(map(float, lines[0].split()[1:])))
        if values.shape != (6,) or not np.isfinite(values).all():
            raise RuntimeError(result.log)
        return values[[1, 3, 5]]


    def main() -> None:
        defect = False
        cases = 0
        failures = 0
        with _scratch() as tmp:
            work = Path(tmp)
            (work / "probe.f90").write_text(PROGRAM)
            comp, link = _recipe(False)
            _execute(comp, work, "", 290)
            _execute(link, work, "", 290)
            # A constant pseudo-mode at phase 1/4 turn has no recurrence or binning oracle.
            # Each mode probes asymmetric centered particles, both translation signs, and
            # a translated on-axis bunch at both zero and finite integrated-wake lengths.
            geometries = [
                ("center", SOURCE, WITNESS, np.zeros(2), 0.6),
                ("minus", SOURCE, WITNESS, -DISPLACEMENT, 0.6),
                ("axis-zero", np.zeros(2), np.zeros(2), DISPLACEMENT, 0.0),
                ("axis-finite", np.zeros(2), np.zeros(2), DISPLACEMENT, 0.6),
            ]
            for kind, dep, plane, (name, src, wit, d, length) in product(
                ("transverse", "longitudinal"), ("leading", "trailing"), (0, 1), geometries
            ):
                axis = ("x", "y")[plane]
                options = f"{axis}_axis,{dep}" if kind == "transverse" else f"{axis}_{dep}"
                mode = f"sr_wake={{scale_with_length=F,{kind}={{7e17,0,0,.25,{options}}}}}"
                observed = measure(work, cases, lattice(mode, d, length), src + d, wit + d, 1)
                cases += 1
                expected = np.zeros(3)
                index = plane if kind == "transverse" else 2
                coordinate = (src if dep == "leading" else wit)[plane]
                expected[index] = -COUPLING * coordinate
                bad = compare(
                    f"SR {kind} {axis}-{dep} {name} L={length:g}; -K*u_body",
                    observed,
                    expected,
                    atol=2e-20,
                    rtol=5e-13,
                )
                if name == "center" and bad:
                    raise RuntimeError("centered short-range control failed")
                failures += int(bad)
                defect |= bad

            # Separate bunches avoid same-bunch LR semantics. At omega*delay=pi/2,
            # the documented unpolarized dipole gives +c |q| (R/Q) r_source / (P0 c).
            coefficient = 299792458 * 1.2e-9 * 1e6 / 1e12
            for name, src, d in (
                ("center-axis", np.zeros(2), np.zeros(2)),
                ("center-asymmetric", SOURCE, np.zeros(2)),
                ("translated-axis+", np.zeros(2), DISPLACEMENT),
                ("translated-axis-", np.zeros(2), -DISPLACEMENT),
            ):
                mode = "lr_self_wake_on=F,lr_wake={mode={1e9,1e6,0,0,1,unpolarized}}"
                observed = measure(work, cases, lattice(mode, d, 0.6), src + d, d, 3, 2.5e-10)[:2]
                cases += 1
                expected = coefficient * src
                bad = compare(
                    f"LR {name}; c*|q|*(R/Q)*r_body/(P0*c)", observed, expected, atol=2e-25, rtol=3e-13
                )
                if name.startswith("center") and bad:
                    raise RuntimeError("centered long-range control failed")
                failures += int(bad)
                defect |= bad

            # A position-independent polarized kick must still rotate with the structure.
            # A quarter-turn maps its body x component into the laboratory y component.
            for method, angle, enabled in product(
                ("bmad_standard", "runge_kutta", "linear"), (0, np.pi / 2), (False, True)
            ):
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                mode = "sr_wake={scale_with_length=F,transverse={7e17,0,0,.25,x_axis,none}}"
                observed = measure(
                    work,
                    cases,
                    lattice(mode, np.zeros(2), 0.6, angle, method),
                    rotation @ SOURCE,
                    rotation @ WITNESS,
                    1 if enabled else -1,
                )[:2]
                cases += 1
                expected = rotation @ np.array([-COUPLING, 0]) if enabled else np.zeros(2)
                bad = compare(
                    f"roll {method} quarter_turn={int(angle != 0)} on={int(enabled)}; R*(-K,0)",
                    observed,
                    expected,
                    atol=3e-20,
                    rtol=5e-13,
                )
                if (not enabled or angle == 0) and bad:
                    raise RuntimeError("disabled or unrotated wake control failed")
                failures += int(bad)
                defect |= bad
        print(
            f"{cases} cases; {failures} structure-frame discrepancies; {cases - failures} controls pass"
        )
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

