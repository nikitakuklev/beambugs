"""Inline lattice and translator runners for the lattice import/export cases."""

import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from _bmad import Result, fortran, run, source

HEADER = """no_digested
parameter[geometry] = open
parameter[particle] = electron
parameter[p0c] = 3e9
beginning[beta_a] = 10
beginning[beta_b] = 10
"""
PROBE = """program probe
use bmad
implicit none
type(lat_struct) lat
type(coord_struct), allocatable :: orbit(:)
character(20) fmt, mode
integer i, status, ix_pole
real(rp) aa(0:n_pole_maxx), bb(0:n_pole_maxx)
logical err
call get_command_argument(1, fmt)
call get_command_argument(2, mode)
global_com%exit_on_error = .false.
call bmad_parser('lat.bmad', lat, err_flag=err)
print *, 'PARSE_ERROR ', err
if (err) stop
global_com%exit_on_error = .true.
! Imported LINE definitions default to closed geometry; track each fixture once as a transfer line.
lat%branch(0)%param%geometry = open$
! Importers omit initial Twiss parameters; give those fixtures explicit finite optical initial conditions.
if (lat%ele(0)%a%beta == 0) lat%ele(0)%a%beta = 10
if (lat%ele(0)%b%beta == 0) lat%ele(0)%b%beta = 10
if (mode == 'noorbit') then
  call write_lattice_foreign_format(trim(fmt), 'out', lat, err=err)
else
  call twiss_and_track(lat, orbit, status, orb_start=lat%particle_start)
  if (status /= ok$) stop 2
  do i=0,lat%n_ele_track
    print '(a,i0,1x,a,1x,a)', 'ELEMENT ', i, trim(lat%ele(i)%name), trim(key_name(lat%ele(i)%key))
    print '(a,6es25.16)', 'ORBIT ', orbit(i)%vec
    print '(a,4es25.16)', 'DISP ', lat%ele(i)%a%eta,lat%ele(i)%a%etap,lat%ele(i)%b%eta,lat%ele(i)%b%etap
    print '(a,es25.16)', 'P0C ',lat%ele(i)%value(p0c$)
    if (lat%ele(i)%key == sbend$) then
      call multipole_ele_to_ab(lat%ele(i), .false., ix_pole, aa, bb, magnetic$, include_kicks$)
      print '(a,es25.16)', 'FIELD ',aa(0)
    endif
    if (associated(lat%ele(i)%a_pole)) print '(a,es25.16)', 'A0 ',lat%ele(i)%a_pole(0)
  enddo
  if (len_trim(fmt) > 0) call write_lattice_foreign_format(trim(fmt), 'out', lat, orbit, err=err)
endif
if (len_trim(fmt) > 0 .and. err) stop 3
print *, 'COMPLETE'
end program
"""


def inspect(lattice: str, fmt: str = "", *, noorbit: bool = False) -> Result:
    return fortran(
        PROBE,
        [fmt, "noorbit" if noorbit else "orbit"],
        files={"lat.bmad": lattice},
        outputs=["out"] if fmt else [],
    )


def valid(result: Result) -> None:
    if "PARSE_ERROR  F" not in result.log or "COMPLETE" not in result.log:
        raise RuntimeError(result.log)
    if re.search(r"\[(?:ERROR|FATAL|ABORT)", result.log):
        raise RuntimeError(result.log)


def export(lattice: str, fmt: str = "madx", *, noaperture: bool = False) -> str:
    suffix = {"madx": "madx", "elegant": "lte", "sad": "sad"}[fmt]
    args = [f"-{fmt}"] + (["-noaperture"] if noaperture else []) + ["lat.bmad"]
    result = run(
        "bmad_to_mad_sad_elegant", args, files={"lat.bmad": lattice}, outputs=[f"lat.{suffix}"]
    )
    if "Written" not in result.log or re.search(r"\[(?:ERROR|FATAL|ABORT)", result.log):
        raise RuntimeError(result.log)
    return result.outputs[f"lat.{suffix}"].decode()


def translate(lattice: str, kind: str = "madx") -> tuple[int, str, str]:
    relative, suffix = {
        "madx": ("mad_to_bmad/madx_to_bmad.py", "madx"),
        "elegant": ("elegant_to_bmad/elegant_to_bmad.py", "lte"),
    }[kind]
    script = source() / "util_programs" / relative
    if not script.is_file():
        raise RuntimeError(f"Missing translator: {script}")
    parent = Path(tempfile.gettempdir()) / "bmad-scratch"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lattice-import-export-", dir=parent) as tmp:
        work = Path(tmp)
        (work / f"lat.{suffix}").write_text(lattice)
        result = subprocess.run(
            [sys.executable, str(script), f"lat.{suffix}"],
            cwd=work,
            env=dict(
                os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=290,
            check=False,
        )
        output = (work / "lat.bmad").read_text() if result.returncode == 0 else ""
        return result.returncode, result.stdout, output


def definition(text: str, name: str) -> str:
    flat = re.sub(r"&\s*\n\s*", " ", text)
    flat = re.sub(r",\s*\n\s*", ", ", flat)
    match = re.search(
        rf"^\s*{re.escape(name)}\s*:\s*(.*?)(?:;|$)", flat, re.MULTILINE | re.IGNORECASE
    )
    if match is None:
        raise RuntimeError(f"Missing definition {name}:\n{text}")
    return match[1]


def number(text: str, key: str) -> float:
    match = re.search(rf"\b{re.escape(key)}\s*=\s*([^,;\s}})]+)", text, re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"Missing explicit {key}: {text}")
    return float(match[1])


def records(result: Result, label: str) -> list[list[float]]:
    rows = [
        [float(x) for x in line.split()[1:]]
        for line in result.log.splitlines()
        if line.startswith(label + " ")
    ]
    if not rows:
        raise RuntimeError(f"Missing {label}: {result.log}")
    return rows


@contextmanager
def madx():
    """Keep MAD-X/PTC diagnostic files inside an automatically removed scratch directory."""
    from cpymad.madx import Madx

    parent = Path(tempfile.gettempdir()) / "bmad-scratch"
    parent.mkdir(exist_ok=True)
    with (
        tempfile.TemporaryDirectory(prefix="lattice-import-export-madx-", dir=parent) as tmp,
        Madx(stdout=False, cwd=tmp) as instance,
    ):
        yield instance
