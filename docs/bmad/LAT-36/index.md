---
title: "LAT-36: Initial PTC lcavity parsing reaches tracking before step bookkeeping"
hide:
  - toc
---

# LAT-36 — Initial PTC lcavity parsing reaches tracking before step bookkeeping

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [integration-step contract, bmad/doc/methods.tex:675–700](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L675-L700)<br>[lcavity defaults, bmad/parsing/set_ele_defaults.f90:204–215](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/set_ele_defaults.f90#L204-L215)<br>[the reference-momentum guard, bmad/code/attribute_bookkeeper.f90:289–299](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/attribute_bookkeeper.f90#L289-L299)<br>[count-to-step resolution, bmad/code/attribute_bookkeeper.f90:387–400](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/attribute_bookkeeper.f90#L387-L400)<br>[sets the reference and calls autoscaling without bookkeeping, bmad/low_level/compute_reference_energy.f90:615–660](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/compute_reference_energy.f90#L615-L660)<br>[tracks the reference when required, lines 664–680](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/compute_reference_energy.f90#L664-L680)<br>[the constant-energy branch, lines 774–784](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/compute_reference_energy.f90#L774-L784)<br>[PTC step conversion, bmad/ptc/ele_to_fibre.f90:189–205](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/ptc/ele_to_fibre.f90#L189-L205) |

Parsing a nonzero-length lcavity with PTC orbital tracking aborts when ds_step is omitted, even when num_steps is explicitly supplied as the manual permits. Reference energy and time bookkeeping invokes PTC before resolving the integration step from the element length and step count. A 1.000 m cavity with num_steps=10 terminates with “DS_STEP IS ZERO” instead of selecting a 0.1000 m step. An explicit ds_step=0.1000 succeeds, as does resolving the same count after reference setup. Standing-wave, traveling-wave, and PTC-standard cavity models are affected; equivalent rfcavity defaults parse successfully.

Run `python LAT-36.py` with [_bmad.py](../helpers/_bmad.py) in the same directory.

??? example "Reproducer: LAT-36.py"

    [Download LAT-36.py](LAT-36.py)

    ```python
    """LAT-36: Initial PTC lcavity parsing reaches tracking before step bookkeeping

    Parsing a nonzero-length lcavity with PTC orbital tracking aborts when ds_step is omitted, even when num_steps is explicitly supplied as the manual permits. Reference energy and time bookkeeping invokes PTC before resolving the integration step from the element length and step count. A 1.000 m cavity with num_steps=10 terminates with “DS_STEP IS ZERO” instead of selecting a 0.1000 m step. An explicit ds_step=0.1000 succeeds, as does resolving the same count after reference setup. Standing-wave, traveling-wave, and PTC-standard cavity models are affected; equivalent rfcavity defaults parse successfully.

    Parse fresh one-element positron lattices for all three PTC cavity models, using default steps, an explicit count and an explicit step size. Compare lcavities with rfcavities and with a control that resolves the documented count after native reference setup before tracking through PTC. Recognize only the exact zero-step fatal diagnostic as the defect and require successful parsing and tracking controls;
    """

    from __future__ import annotations

    import math
    import resource
    import signal

    from _bmad import RunFailure, fortran, verdict

    PROGRAM = r"""
    program parse_step_order
    use bmad
    implicit none
    type(lat_struct), target :: lat
    type(ele_struct), pointer :: ele
    type(coord_struct) :: start, finish
    character(16) :: mode
    logical :: err
    call get_command_argument(1, mode)
    call bmad_parser('lat.bmad', lat, err_flag=err)
    if (err) error stop 'parse failed'
    ele => lat%ele(1)
    if (trim(mode) == 'late_bookkeeping') then
      ! Resolve the documented count only after the reference momentum is available.
      ! This calls the production bookkeeper without modifying any library routine.
      ele%tracking_method = symp_lie_ptc$
      ele%value(ds_step$) = 0
      ele%value(num_steps$) = 10
      call attribute_bookkeeper(ele, .true.)
    endif
    write(*, '(a,2es26.17)') 'DATA_STEP ', ele%value(ds_step$), ele%value(num_steps$)
    write(*, '(a)') 'DATA_PARSED'
    if (trim(mode) == 'late_bookkeeping') then
      call init_coord(start, [0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp, 0.0_rp], &
                      ele, upstream_end$)
      call track1(start, ele, lat%param, finish, err_flag=err)
      if (err .or. finish%state /= alive$) error stop 'tracking failed'
      write(*, '(a)') 'DATA_TRACKED'
    endif
    end program
    """

    GAMMA = 10.0
    P0C = 510998.95069 * math.sqrt(GAMMA**2 - 1)
    DIAGNOSTIC = "DS_STEP IS ZERO FOR ELEMENT: C"


    def lattice(key: str, wave: str, control: str, *, autoscale: bool) -> str:
        method = "bmad_standard" if control == "late_bookkeeping" else "symp_lie_ptc"
        step = {"num_steps": ", num_steps=10", "ds_step": ", ds_step=0.1"}.get(control, "")
        scaling = "" if autoscale else ", autoscale_amplitude=F, autoscale_phase=F"
        return f"""no_digested
    parameter[geometry]=open
    parameter[particle]=positron
    parameter[p0c]={P0C:.17g}
    parameter[no_end_marker]=T
    c: {key}, l=1, voltage=10, rf_frequency=1.3e9, &
     phi0={0 if key == "lcavity" else 0.25}, cavity_type={wave}, &
     tracking_method={method}, fringe_at=entrance_end{step}{scaling}
    ln: line=(c)
    use, ln
    """


    def check_case(key: str, wave: str, control: str, *, autoscale: bool = False) -> bool:
        label = f"{key}/{wave}/{control}" + ("/autoscale" if autoscale else "")
        # The lcavity default is one slice; explicit counts and the other controls
        # use ten. The manual permits ds_step = abs(L) / num_steps in PTC tracking.
        count = 1 if key == "lcavity" and control == "default" else 10
        expected_step = 1.0 / count
        expected = f"parse OK, ds_step={expected_step:.4f} m = 1.000 m / {count}"
        try:
            result = fortran(
                PROGRAM,
                args=[control],
                files={"lat.bmad": lattice(key, wave, control, autoscale=autoscale)},
            )
        except RunFailure as failure:
            affected = key == "lcavity" and control in ("default", "num_steps")
            if not (
                affected
                and failure.status == -signal.SIGILL
                and DIAGNOSTIC in failure.log
                and "DATA_PARSED" not in failure.log
            ):
                raise
            print(f"{label}: observed=SIGILL ({DIAGNOSTIC}); expected={expected}")
            return True
        if any(marker in result.log for marker in ("[ERROR", "[FATAL", "[ABORT")):
            raise RuntimeError(result.log)
        rows = [line.split()[1:] for line in result.log.splitlines() if line.startswith("DATA_STEP ")]
        if len(rows) != 1 or result.log.splitlines().count("DATA_PARSED") != 1:
            raise RuntimeError(f"Missing or repeated parse records:\n{result.log}")
        step, steps = map(float, rows[0])
        if not (math.isfinite(step) and abs(step - expected_step) <= 1e-16 and steps == count):
            raise RuntimeError(f"Unexpected step resolution: {step=}, {steps=}; {expected}")
        tracked = result.log.splitlines().count("DATA_TRACKED")
        if tracked != int(control == "late_bookkeeping"):
            raise RuntimeError(f"Missing or repeated tracking records:\n{result.log}")
        suffix = "; PTC tracking OK" if tracked else ""
        print(
            f"{label}: observed=parse OK, ds_step={step:.4f} m, num_steps={steps:g}{suffix}; "
            f"expected={expected}"
        )
        return False


    def main() -> None:
        # err_exit deliberately raises SIGILL; suppress core files for expected aborts.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        defect = False
        for key in ("lcavity", "rfcavity"):
            for wave in ("traveling_wave", "standing_wave", "ptc_standard"):
                for control in ("default", "num_steps", "ds_step", "late_bookkeeping"):
                    defect |= check_case(key, wave, control)
        defect |= check_case("lcavity", "traveling_wave", "default", autoscale=True)
        verdict(defect)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | 20261031-0 (`c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3`), source, production, GNU Fortran 15.3.0, CPU, double, OpenMP one thread, MPI disabled |

