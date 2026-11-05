"""Small parsers and source readers for the collective-contract reproductions."""

import numpy as np
from _bmad import fortran, source


def source_text(path: str) -> str:
    """Read the explicitly supplied upstream source tree."""
    return (source() / path).read_text()


def rows(log: str) -> dict[str, np.ndarray]:
    """Require unique labelled numeric results; preserve native failures."""
    result = {}
    for line in log.splitlines():
        if not line.startswith("RESULT "):
            continue
        _, key, *values = line.split()
        if key in result or not values:
            raise RuntimeError(f"Invalid result record: {line}")
        result[key] = np.array([float(v.replace("D", "E")) for v in values])
    if not result:
        raise RuntimeError(f"No RESULT records:\n{log}")
    return result


def parse_lattice(element: str, extra: str = "") -> tuple[bool, str]:
    """Exercise the production parser, retaining its explicit error return."""
    result = fortran(
        """program parse
use bmad
implicit none
type(lat_struct) lat
logical err
global_com%exit_on_error = .false.
call bmad_parser('lat.bmad', lat, err_flag=err)
write(*,'(a,i0)') 'RESULT error ', merge(1,0,err)
if (.not. err) then
  write(*,'(a,i0)') 'RESULT csr_method ', lat%ele(1)%csr_method
endif
end program
""",
        files={
            "lat.bmad": "parameter[particle]=electron\nparameter[p0c]=1e9\n"
            + extra
            + "e: "
            + element
            + "\nl: line=(e)\nuse, l\n"
        },
    )
    data = rows(result.log)
    if data["error"].shape != (1,):
        raise RuntimeError(result.log)
    return bool(data["error"][0] == 0), result.log


def section(text: str, start: str, end: str) -> str:
    """Select one explicit source passage; refuse missing or ambiguous anchors."""
    if text.count(start) != 1:
        raise RuntimeError(f"Expected one source anchor: {start}")
    tail = text.split(start, 1)[1]
    if end not in tail:
        raise RuntimeError(f"Missing source endpoint: {end}")
    return start + tail.split(end, 1)[0] + end


def bounds_checked(program: str, args: list[str]) -> str:
    """Recompile exact upstream source with bounds checks, using the build's recipe."""
    from pathlib import Path

    from _bmad import TIMEOUT, _execute, _recipe, _scratch

    compile_command, link_command = _recipe(False)
    # Last compiler options take precedence over production optimization settings.
    compile_command.extend(["-O0", "-fcheck=bounds"])
    with _scratch() as tmp:
        work = Path(tmp)
        (work / "probe.f90").write_text(program)
        _execute(compile_command, work, "", TIMEOUT)
        _execute(link_command, work, "", TIMEOUT)
        return _execute([str(work / "probe"), *args], work, "", TIMEOUT)
