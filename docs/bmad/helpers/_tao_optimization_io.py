"""Embedded Tao optimization/I/O cases and checked command-output parsing."""

import re
from collections.abc import Mapping, Sequence

import numpy as np
from _bmad import Result, run


def tao(
    commands: Sequence[str],
    *,
    lattice: str,
    init: str | None = None,
    files: Mapping[str, str | bytes] | None = None,
    outputs: Sequence[str] = (),
    allow_errors: bool = False,
    allow_aborted_script: bool = False,
) -> Result:
    """Run one fresh Tao process; callers accepting diagnostics must identify them exactly."""
    inputs: dict[str, str | bytes] = {
        "line.bmad": lattice,
        "lat.bmad": lattice,
        "run.tao": "\n".join([*commands, "quit", ""]),
        **(files or {}),
    }
    if init is None:
        options = ["-noinit", "-lat", "lat.bmad"]
    else:
        inputs["case.init"] = init
        options = ["-init", "case.init"]
    result = run("tao", ["-noplot", *options, "-startup", "run.tao"], files=inputs, outputs=outputs)
    if not allow_aborted_script and result.log.count("Tao: quit\n") != 1:
        raise RuntimeError(f"Tao did not complete its command file:\n{result.log}")
    if not allow_errors and diagnostics(result.log):
        raise RuntimeError(result.log)
    return result


def diagnostics(log: str) -> list[tuple[str, tuple[str, ...]]]:
    """Parse complete Tao errors and reject unmatched fatal/runtime diagnostics."""
    pattern = re.compile(
        r"^(?:Tao> )?\[(?:ERROR|FATAL|ABORT|SEVERE)\b[^\n]*\]\s+(\w+):\n"
        r"((?:[ \t]+[^\n]*\n)+)",
        re.MULTILINE,
    )
    remainder = pattern.sub("", log)
    if re.search(r"\[(?:ERROR|FATAL|ABORT|SEVERE)\b|Fortran runtime error|SIGSEGV", remainder):
        raise RuntimeError(log)
    return [
        (m[1], tuple(line.strip() for line in m[2].splitlines())) for m in pattern.finditer(log)
    ]


def block(log: str, command: str) -> str:
    marker = f"Tao: {command}\n"
    if log.count(marker) != 1:
        raise RuntimeError(f"Expected one completed command {command!r}:\n{log}")
    return log.split(marker, 1)[1].split("\nTao:", 1)[0].strip()


def records(text: str) -> list[list[str]]:
    return [[part.strip() for part in line.split(";")] for line in text.splitlines() if ";" in line]


def parameters(text: str) -> dict[str, str]:
    rows = records(text)
    if not rows or any(len(row) < 4 for row in rows):
        raise ValueError(f"Expected typed parameter records: {text}")
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate parameters: {text}")
    return {row[0]: row[3] for row in rows}


def numbers(text: str) -> np.ndarray:
    rows = [line.replace(";", " ").split() for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Expected numeric rows: {text}")
    return np.array(rows, dtype=float)


def data_values(log: str, command: str) -> np.ndarray:
    rows = records(block(log, command))
    if len(rows) != 1 or rows[0][0] != "1":
        raise ValueError(f"Expected one data array in universe 1: {rows}")
    return np.array(rows[0][1:], dtype=float)


def control(label: str, observed, expected, *, atol: float) -> None:
    from _bmad import compare

    if compare(label, observed, expected, atol=atol):
        raise RuntimeError(f"Control failed: {label}")


LINE = """no_digested
parameter[geometry] = open
parameter[particle] = electron
beginning[p0c] = 1e9
beginning[beta_a] = 2
beginning[beta_b] = 3
beginning[dbeta_dpz_a] = 11
beginning[dbeta_dpz_b] = 22
beginning[dalpha_dpz_a] = 33
beginning[dalpha_dpz_b] = 44
particle_start[x] = 0.003
particle_start[px] = 0.001
d1: drift, l = 1
q1: quadrupole, l = 0.2, k1 = 0.3
d2: drift, l = 1
q2: quadrupole, l = 0.2, k1 = -0.2
fin: marker
lin: line = (d1, q1, d2, q2, fin)
use, lin
"""

LINEAR_INIT = """&tao_design_lattice
  design_lattice(1)%file = 'line.bmad'
/
&tao_params
  global%plot_on = F
  global%optimizer = 'svd'
  global%n_opti_cycles = 1
  global%opti_write_var_file = F
/
&tao_d2_data
  d2_data%name = 'linear'
  n_d1_data = 1
/
&tao_d1_data
  ix_d1_data = 1
  d1_data%name = 'k'
  datum(1)%data_type = 'expression:ele::q1[k1]'
  datum(1)%meas = 0
  datum(1)%weight = 1
  datum(2)%data_type = 'expression:ele::q2[k1]'
  datum(2)%meas = 0
  datum(2)%weight = 1
/
&tao_var
  v1_var%name = 'v'
  default_step = 1e-5
  default_universe = '1'
  default_attribute = 'k1'
  ix_min_var = 1
  ix_max_var = 2
  var(1)%ele_name = 'q1'
  var(1)%weight = 0
  var(2)%ele_name = 'q2'
  var(2)%merit_type = 'target'
  var(2)%meas = 0.1
  var(2)%weight = 1
/
"""

WEIGHT_INIT = """&tao_design_lattice
  design_lattice(1)%file = 'line.bmad'
/
&tao_params
  global%plot_on = F
  global%optimizer = 'geodesic_lm'
  global%opti_write_var_file = F
/
&tao_d2_data
  d2_data%name = 'linear'
  n_d1_data = 1
/
&tao_d1_data
  ix_d1_data = 1
  d1_data%name = 'k'
  datum(1)%data_type = 'expression:ele::q2[k1]'
  datum(1)%meas = 0
  datum(1)%weight = 1
/
&tao_var
  v1_var%name = 'v'
  default_step = 1e-5
  default_universe = '1'
  default_attribute = 'k1'
  ix_min_var = 1
  ix_max_var = 1
  var(1)%ele_name = 'q2'
  var(1)%merit_type = 'target'
  var(1)%meas = 0.1
  var(1)%weight = 100
/
"""

RMS_INIT = """&tao_design_lattice
  design_lattice(1)%file = 'line.bmad'
/
&tao_params
  global%plot_on = F
/
&tao_d2_data
  d2_data%name = 'range'
  n_d1_data = 1
/
&tao_d1_data
  ix_d1_data = 1
  d1_data%name = 'x'
  datum(1) = 'orbit.x' '' 'beginning' 'd1' 'rms' 0.0 1.0
  datum(2) = 'orbit.x' '' 'beginning' 'd1' 'average' 0.0 1.0
  datum(3) = 'orbit.x' '' 'beginning' 'd1' 'integral' 0.0 1.0
/
"""

ABS_INIT = """&tao_design_lattice
  design_lattice(1)%file = 'line.bmad'
/
&tao_params
  global%plot_on = F
/
&tao_d2_data
  d2_data%name = 'dat'
  n_d1_data = 1
/
&tao_d1_data
  ix_d1_data = 1
  d1_data%name = 'x'
  datum(1) = 'orbit.x' 'beginning' '' 'd1' 'abs_max' 0.0 1.0
  datum(2) = 'orbit.x' 'beginning' '' 'd1' 'target' 0.0 1.0
  datum(3) = 'orbit.x' 'beginning' 'd1' 'd1' 'abs_min' 0.0 1.0
/
"""

LATTICE = """parameter[geometry] = open
parameter[particle] = electron
parameter[p0c] = 1e6
beginning[beta_a] = 3
beginning[alpha_a] = 0.2
beginning[beta_b] = 4
particle_start[x] = 0.001
particle_start[y] = -0.002
particle_start[z] = 0.03
e: drift, l=1.7
lat: line=(e)
use, lat
"""

RING_LATTICE = """parameter[geometry] = closed
parameter[particle] = electron
parameter[p0c] = 1e9
qf: quadrupole, l=0.2, k1=1, fringe_type=none
qd: quadrupole, l=0.2, k1=-0.96, fringe_type=none
d: drift, l=0.8
lat: line=(qf,d,qd,d)
use, lat
"""
