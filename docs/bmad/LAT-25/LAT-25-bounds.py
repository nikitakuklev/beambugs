"""LAT-25: Identify the twenty-first overlay target with bounds checking

The script recompiles the unchanged bookkeeper module from BMAD_SOURCE with bounds checking, links it ahead of the BMAD_DIST production library, and parses overlays with twenty and twenty-one valid targets. It checks all twenty control values before identifying the out-of-bounds pointer-buffer access.
"""

from pathlib import Path

from _bmad import RunFailure, _execute, _recipe, _scratch, source, verdict
from _multipass_slicing import HEADER, control, rows

PROGRAM = r"""program probe
use bmad
implicit none
type(lat_struct) lat
logical err
integer n
character(20) arg
call get_command_argument(1,arg)
read(arg,*) n
write(*,'(a,i0)') 'CHECKPOINT overlay attribute count ',n
flush(6)
call bmad_parser('lat.bmad',lat,err_flag=err)
if (err) stop 1
if (.not. associated(lat%ele(1)%b_pole)) stop 2
write(*,'(a,30es25.16)') 'RESULT assigned ',lat%ele(1)%b_pole(0:n-1)
call deallocate_lat_pointers(lat)
end program
"""


def lattice(count):
    targets = ", ".join(f"q[b{i}]:v" for i in range(count))
    return (
        HEADER
        + f"q:quadrupole,l=1,k1=0\no:overlay={{{targets}}},var={{v}},v=0.001\nbl:line=(q)\nuse,bl\n"
    )


compile_argv, link_argv = _recipe(False)
with _scratch() as tmp:
    work = Path(tmp)
    module = source() / "bmad/modules/bookkeeper_mod.f90"
    # Override optimization and enable bounds checking only for this diagnostic object.
    module_compile = [
        str(module) if a == "probe.f90" else "bookkeeper_bounds.o" if a == "probe.o" else a
        for a in compile_argv
    ]
    _execute([*module_compile, "-g", "-O0", "-fcheck=bounds"], work, "", 290)
    (work / "probe.f90").write_text(PROGRAM)
    _execute(compile_argv, work, "", 290)
    link_argv.insert(link_argv.index("probe.o") + 1, "bookkeeper_bounds.o")
    _execute(link_argv, work, "", 290)
    for count in (20, 21):
        run_dir = work / f"case{count}"
        run_dir.mkdir()
        (run_dir / "lat.bmad").write_text(lattice(count))
        try:
            log = _execute([str(work / "probe"), str(count)], run_dir, "", 290)
        except RunFailure as exc:
            diagnostic = "Index '21' of dimension 1 of array 'ptr_attrib' above upper bound of 20"
            if (
                count != 21
                or exc.status != 2
                or diagnostic not in exc.log
                or "CHECKPOINT overlay attribute count 21" not in exc.log
            ):
                raise
            print(
                f"bounds-checked twenty-first target: observed={diagnostic} expected=successful assignment"
            )
            defect = True
        else:
            control(
                f"bounds-checked {count} assigned coefficients",
                rows(log)["assigned"],
                [0.001] * count,
                atol=1e-17,
            )
            if count == 21:
                defect = False
verdict(defect)
