"""Isolated translation and tracking fixtures for the SAD/multipole conversion chunk."""

import cmath
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from _bmad import RunFailure, _case, _execute, _scratch, fortran, run, source

HEADER = """parameter[geometry]=open
parameter[particle]=electron
parameter[p0c]=1e9
beginning[beta_a]=10
beginning[beta_b]=10
"""
PROGRAM = """program check_lattice
use bmad
implicit none
type(lat_struct) :: lat
type(coord_struct) :: a,b
real(rp) :: v(6)
integer :: i
logical :: err
global_com%exit_on_error=.false.
call bmad_parser('case.bmad',lat,err_flag=err)
if(err) then
  print '(a)', 'PARSE_REJECTED'
  stop
endif
open(11,file='start.txt'); read(11,*) v; close(11)
call init_coord(a,v,lat%ele(0),downstream_end$)
do i=1,lat%n_ele_track
  call track1(a,lat%ele(i),lat%param,b)
  if(b%state/=alive$) error stop 'Particle lost'
  if(i==1) print '(a,6es25.16)', 'FIRST ',b%vec
  a=b
enddo
print '(a,6es25.16)', 'ORBIT ',b%vec
print '(a,3es25.16)', 'FLOOR ',lat%ele(lat%n_ele_track)%floor%r
print '(a,5es25.16)', 'ATTR ',lat%ele(1)%value(l$),lat%ele(1)%value(voltage$), &
  lat%ele(1)%value(gradient$),lat%ele(1)%value(p0c_start$),lat%ele(1)%value(p0c$)
print '(a)', 'COMPLETED'
end program
"""


def lattice(element: str, tail: str = "", line: str = "m") -> str:
    return HEADER + f"m: {element}\n{tail}\nlat: line=({line})\nuse,lat\n"


def track(text: str, start=(0.001, 0, 0.002, 0, 0, 0), *, allow_diagnostics=False):
    """Return named native records; parser rejection carries its exact diagnostic log."""
    result = fortran(PROGRAM, files={"case.bmad": text, "start.txt": " ".join(map(str, start))})
    if not allow_diagnostics and re.search(r"\[(?:ERROR|FATAL|ABORT)|ERROR IN ", result.log):
        raise RuntimeError(result.log)
    if "PARSE_REJECTED" in result.log:
        return {}, result.log
    if "COMPLETED" not in result.log:
        raise RuntimeError(result.log)
    records = {}
    for line in result.log.splitlines():
        tokens = line.split()
        if tokens and tokens[0] in ("FIRST", "ORBIT", "FLOOR", "ATTR"):
            records[tokens[0]] = np.array([float(x) for x in tokens[1:]])
    if set(records) != {"FIRST", "ORBIT", "FLOOR", "ATTR"}:
        raise RuntimeError(result.log)
    if not all(np.isfinite(x).all() for x in records.values()):
        raise RuntimeError("Nonfinite native tracking records")
    return records, result.log


def export(text: str, kind: str) -> str:
    suffix = {"madx": "madx", "mad8": "mad8", "sad": "sad", "elegant": "lte"}[kind]
    result = run(
        "bmad_to_mad_sad_elegant",
        [f"-{kind}", "case.bmad"],
        files={"case.bmad": text},
        outputs=[f"case.{suffix}"],
    )
    if "ERROR IN" in result.log or "FATAL" in result.log:
        raise RuntimeError(result.log)
    return result.outputs[f"case.{suffix}"].decode()


def translate_sad(deck: str) -> str:
    """Execute the upstream importer selected by BMAD_SOURCE with inline parameters."""
    params = f"""sad_lattice_file = "case.sad"
bmad_lattice_file = "case.bmad"
lattice_geometry = "open"
patch_for_fshift = "FALSE"
sad_to_bmad_postprocess_exe = "sad_to_bmad_postprocess"
calc_fshift_for = "bmad"
ignore_marker_offsets = False
header_lines = {HEADER!r}
footer_lines = ""
"""
    with _scratch() as tmp:
        result = _case(
            [
                sys.executable,
                "-I",
                str(source() / "util_programs/sad_to_bmad/sad_to_bmad.py"),
                "case.params",
                "case.sad",
            ],
            Path(tmp) / "run",
            {"case.params": params, "case.sad": deck},
            ["case.bmad"],
            "",
            290,
        )
    return result.outputs["case.bmad"].decode()


def sad_deck(element: str) -> str:
    return f"MOMENTUM = 1 GEV;\nDRIFT D = (L = 1);\n{element}\nLINE RING = (M D);\nUSE RING;\n"


def sad_kick(n: int, k: float, rotate: float, x=0.001, y=0.002):
    # SAD TENT rotates lab coordinates by +ROTATE. Its thin kick uses
    # Kn/n!*(xb-i*yb)^n; TEXIT undoes that frame rotation for the momenta.
    c, s = math.cos(rotate), math.sin(rotate)
    w = complex(c * x - s * y, -(s * x + c * y)) ** n * k / math.factorial(n)
    return np.array([-c * w.real - s * w.imag, s * w.real - c * w.imag])


def bmad_kick(n: int, kn: float, tilt: float, skew=0.0):
    # Bmad: -dpx+i*dpy=(KnL+i*KnSL)/n! * exp[-i(n+1)T]*(x+i*y)^n.
    g = (
        complex(kn, skew)
        / math.factorial(n)
        * cmath.exp(-1j * (n + 1) * tilt)
        * complex(0.001, 0.002) ** n
    )
    return np.array([-g.real, g.imag])


def madx_track(text: str, *, survey: bool = False, ptc: bool = False):
    from cpymad.madx import Madx

    with _scratch() as tmp:
        m = Madx(stdout=False, cwd=tmp)
        try:
            m.input(text)
            if survey:
                table = m.survey()
                result = np.array([table.x[-1], table.y[-1]])
            elif ptc:
                m.input(
                    "ptc_create_universe; ptc_create_layout, model=2, method=6, nst=10, exact=true; ptc_start,x=.001,y=.002; ptc_track,icase=4,element_by_element,onetable,dump=false; ptc_track_end; ptc_end;"
                )
                result = np.array([m.table.trackone.px[-1], m.table.trackone.py[-1]])
            else:
                m.input("track,onepass,onetable; start,x=.001,y=.002; run,turns=1; endtrack;")
                result = np.array([m.table.trackone.px[-1], m.table.trackone.py[-1]])
        finally:
            m.quit()
    if not np.isfinite(result).all():
        raise RuntimeError("MAD-X returned nonfinite results")
    return result


def statements(text: str):
    content = "\n".join(line for line in text.splitlines() if not line.startswith("!"))
    return [" ".join(s.split()) for s in content.replace("&\n", " ").splitlines() if s.strip()]


def undefined_names(text: str):
    items = statements(text)
    defined = {s.split(":")[0].strip().upper() for s in items if ":" in s}
    beamline = next(s for s in items if s.upper().startswith("LINE_1:"))
    match = re.search(r"\((.*)\)", beamline)
    if match is None:
        raise RuntimeError("No exported beamline")
    used = match[1].replace(" ", "").split(",")
    return sorted({n for n in used if n.upper() not in defined}), beamline


def elegant_track(text: str, *, x=0.001, y=0.002, rejected: str | None = None):
    """Track one particle, or recognize exactly the expected undefined-name rejection."""
    import pysdds

    executable = os.environ.get("ELEGANT_BIN")
    rpn = os.environ.get("RPN_DEFNS")
    if not executable or not rpn:
        raise RuntimeError(
            "Set ELEGANT_BIN to the Elegant executable and RPN_DEFNS to its definitions"
        )
    p0 = 1957.0
    with _scratch() as tmp:
        work = Path(tmp)
        (work / "lattice.lte").write_text(text)
        beam = pysdds.SDDSFile(add_data_nm=True)
        beam.mode = "binary"
        beam.n_pages = 1
        for name, val, unit in (
            ("x", x, "m"),
            ("xp", 0.0, ""),
            ("y", y, "m"),
            ("yp", 0.0, ""),
            ("t", 0.0, "s"),
            ("p", p0, "m$be$nc"),
        ):
            beam.add_column(name, "double", units=unit, data=[np.array([val], dtype=float)])
        pysdds.write(beam, work / "beam.sdds")
        (work / "run.ele").write_text(
            f"&run_setup lattice=lattice.lte, use_beamline=LAT, p_central={p0}, output=out.sdds, default_order=2 &end\n&run_control n_steps=1,n_passes=1 &end\n&sdds_beam input=beam.sdds &end\n&track &end\n"
        )
        try:
            _execute([executable, "run.ele", f"-rpnDefns={rpn}"], work, "", 290)
        except RunFailure as error:
            if rejected and f"no expansion for entity {rejected}" in error.log:
                print(
                    f"Elegant rejected undefined name {rejected}: expected a complete beamline definition"
                )
                return None
            raise
        if rejected:
            # If a future writer defines every name, successful tracking is a valid result.
            print(f"Elegant accepted beamline (previously undefined name {rejected})")
        data = pysdds.read(work / "out.sdds")
        cols = [np.asarray(data[0, c]) for c in ("x", "xp", "y", "yp", "p")]
        if any(c.shape != (1,) for c in cols):
            raise RuntimeError("Expected exactly one surviving particle")
        xx, xp, yy, yp, p = [float(c[0]) for c in cols]
        # Elegant stores slopes; Bmad uses canonical momenta normalized by P0.
        scale = p / p0 / math.sqrt(1 + xp * xp + yp * yp)
        result = np.array([xx, xp * scale, yy, yp * scale])
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite Elegant orbit")
        return result
