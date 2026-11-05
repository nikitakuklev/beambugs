"""COL-65: The long-range damping example omits a factor of two pi

Copying the long-range wake example gives a damping rate smaller by a factor of 2π than the manual’s quality-factor relation requires. For frequency 1.650 GHz and quality factor 70,000, the example supplies 1.179 × 10⁴ s⁻¹ instead of 7.405 × 10⁴ s⁻¹. The parser stores that value directly and tracking uses it as the exponential damping rate, making the decay time 6.283 times too long.

The script extracts the example’s first mode and constants from BMAD_SOURCE and parses them on BMAD_DIST. It measures the decay of a seeded mode and checks a control with damping πf/Q. The expected decay follows the manual’s quality-factor equation and the supplied rate is checked independently against the tracking result.
"""

import re

import numpy as np
from _bmad import compare, fortran, source, verdict
from _wake_tables_and_bbu import require, rows

PROGRAM = r"""
program damping_example
use bmad
use wake_mod
implicit none
type(lat_struct) lat
type(bunch_struct) bunch
logical err
call bmad_parser('lat.bmad',lat,err_flag=err)
if (err) error stop 'Parser failed'
write(*,'(a,2es25.16)') 'RESULT mode ',lat%ele(1)%wake%lr%mode(1)%freq,lat%ele(1)%wake%lr%mode(1)%damp
lat%ele(1)%wake%lr%mode(1)%b_sin=1
lat%ele(1)%wake%lr%self_wake_on=.false.
call reallocate_bunch(bunch,1)
call init_coord(bunch%particle(1),[0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp], &
 lat%ele(1),upstream_end$,electron$)
bunch%particle(1)%charge=0
bunch%particle(1)%t=1e-5_rp
call order_particles_in_z(bunch)
call track1_lr_wake(bunch,lat%ele(1))
write(*,'(a,es25.16)') 'RESULT decay ',hypot(lat%ele(1)%wake%lr%mode(1)%b_sin,lat%ele(1)%wake%lr%mode(1)%b_cos)
end program
"""
manual = (source() / "bmad/doc/attributes.tex").read_text()
examples = re.findall(r"\\begin\{example\}(.*?)\\end\{example\}", manual, re.DOTALL)
example = next(x for x in examples if "f1 = 1.65e9" in x and "cav9:" in x)
constants = re.search(r"f1\s*=\s*([\d.eE+\-]+);\s*q1\s*=\s*([\d.eE+\-]+)", example)
mode = re.search(r"mode\s*=\s*\\\{([^}]+)\\\}", example)
if constants is None or mode is None:
    raise RuntimeError("Cannot extract long-range example")
frequency, quality = map(float, constants.groups())
components = [part.strip() for part in mode[1].split(",")]
if len(components) != 6:
    raise RuntimeError("Unexpected example mode layout")
expected_rate = np.pi * frequency / quality
expected_decay = np.exp(-expected_rate * 1e-5)


def track(parts):
    lattice = (
        "parameter[particle]=electron\nparameter[p0c]=1e9\n"
        f"f1={frequency:.17g};q1={quality:.17g}\n"
        "w: lcavity,l=1,lr_wake={mode={" + ",".join(parts) + "}}\nl: line=(w)\nuse,l\n"
    )
    return rows(fortran(PROGRAM, files={"lat.bmad": lattice}).log)


corrected = components.copy()
corrected[2] = "pi*f1/q1"
control = track(corrected)
require(
    "quality-factor control damping (s^-1); omega/(2Q)",
    control["mode"][1],
    expected_rate,
    rtol=2e-15,
)
require("quality-factor control decay after 10 us", control["decay"], [expected_decay], rtol=3e-14)
actual = track(components)
require("example frequency (Hz)", actual["mode"][0], frequency, rtol=2e-15)
require(
    "example decay; exponential of its supplied rate",
    actual["decay"],
    [np.exp(-actual["mode"][1] * 1e-5)],
    rtol=3e-14,
)
defect = compare(
    "example damping (s^-1); omega/(2Q)", actual["mode"][1], expected_rate, rtol=2e-15, atol=0
)
defect |= compare(
    "example decay after 10 us; quality-factor relation",
    actual["decay"],
    [expected_decay],
    rtol=3e-14,
    atol=0,
)
verdict(defect)
