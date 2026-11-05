"""Shared code for the lattice-translator reproducers: Bmad export, Tao records, MAD-X, Elegant.

export() runs the production converter bmad_to_mad_sad_elegant, the program users run to write
MAD-X, MAD8, SAD and Elegant files; import_madx() runs util_programs/mad_to_bmad/madx_to_bmad.py
of the BMAD_SOURCE tree. tao() collects `pipe` records of a Bmad lattice. madx() runs MAD-X through
cpymad. elegant_track() runs serial `elegant`, which must be on PATH with RPN_DEFNS naming the
defns.rpn file of SDDS. Every case runs in fresh scratch with one numerical thread.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from _bmad import TIMEOUT, RunFailure, run, source
from cpymad.madx import Madx

SUFFIX = {"madx": "madx", "mad8": "mad8", "sad": "sad", "elegant": "lte"}


def lattice(
    elements: str,
    line: str,
    *,
    p0c: float = 1e9,
    particle: str = "electron",
    beta: tuple[float, float] = (10.0, 10.0),
    start: Sequence[float] | None = None,
    extra: str = "",
) -> str:
    """Open-geometry Bmad lattice: element definitions, beamline `lat` and the initial optics."""
    lines = [
        "no_digested",
        "parameter[geometry] = open",
        f"parameter[particle] = {particle}",
        f"parameter[p0c] = {p0c!r}",
        f"beginning[beta_a] = {beta[0]!r}",
        f"beginning[beta_b] = {beta[1]!r}",
    ]
    for name, value in zip(("x", "px", "y", "py", "z", "pz"), start or (), strict=False):
        lines.append(f"particle_start[{name}] = {value!r}")
    lines += [extra, elements, f"lat: line = ({line})", "use, lat", ""]
    return "\n".join(lines)


def bend_exit(g: float, g_tot: float, angle: float) -> np.ndarray:
    """Exit (x, px) of an on-axis particle at reference momentum in a sector bend, edges ignored.

    The reference orbit is a circle of radius 1/g about the origin, from (1/g, 0) through the
    bend angle; the particle moves on a circle of radius 1/g_tot tangent to it at the entrance.
    Bmad and MAD-X take x outward (away from the centre) and px = sin of the angle between the
    particle and the reference direction at the exit plane.
    """
    rho, rho_p = 1 / g, 1 / g_tot
    # Particle at (rho - rho_p + rho_p cos(psi), rho_p sin(psi)); the exit plane is the ray from
    # the origin at the bend angle: -X sin(angle) + Y cos(angle) = 0, which gives
    # rho_p sin(psi - angle) = (rho - rho_p) sin(angle).
    psi = angle + np.arcsin((rho - rho_p) * np.sin(angle) / rho_p)
    big_x, big_y = rho - rho_p + rho_p * np.cos(psi), rho_p * np.sin(psi)
    x = big_x * np.cos(angle) + big_y * np.sin(angle) - rho
    return np.array([x, -np.sin(psi - angle)])


def tao(lat: str, commands: Sequence[str]) -> dict[str, list[str]]:
    """Run Tao on a lattice and return the output lines of each command, keyed by command.

    Every command must echo and print at least one record, and the run must reach `quit`.
    """
    startup = "\n".join([*commands, "quit", ""])
    log = run(
        "tao",
        ["-noplot", "-noinit", "-lat", "lat.bmad", "-startup", "run.tao"],
        files={"lat.bmad": lat, "run.tao": startup},
    ).log
    if re.search(r"\[(?:ERROR|FATAL|ABORT)|ERROR IN ", log):
        raise RuntimeError(f"Tao initialization or command failed:\n{log[-3000:]}")
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for text in log.splitlines():
        if text.startswith("Tao: "):
            current = sections.setdefault(text[5:].strip(), [])
        elif current is not None and text.strip():
            current.append(text)
    if "quit" not in sections:
        raise RuntimeError(f"Tao did not reach quit:\n{log[-3000:]}")
    for command in commands:
        if not sections.get(command):
            raise RuntimeError(f"Tao printed no records for {command!r}:\n{log[-3000:]}")
        if any("ERROR" in text for text in sections[command]):
            raise RuntimeError(f"Tao reported an error for {command!r}:\n{log[-3000:]}")
    return sections


def record(lines: Sequence[str]) -> dict[str, str]:
    """Parse `name;type;settable;value` pipe records into {name: value text}."""
    values = {}
    for text in lines:
        fields = text.split(";")
        if len(fields) >= 4:
            values[fields[0].strip()] = fields[3].strip()
    if not values:
        raise RuntimeError(f"No pipe records in {lines!r}")
    return values


def numbers(lines: Sequence[str], keys: Sequence[str]) -> np.ndarray:
    """Named real values of one pipe record."""
    values = record(lines)
    return np.array([float(values[key]) for key in keys])


def export(lat: str, fmt: str) -> str:
    """Translate a Bmad lattice with bmad_to_mad_sad_elegant -<fmt>; return the written file."""
    name = "lat." + SUFFIX[fmt]
    result = run(
        "bmad_to_mad_sad_elegant", [f"-{fmt}", "lat.bmad"], files={"lat.bmad": lat}, outputs=[name]
    )
    if "PROBLEM TRACKING" in result.log or "ERROR" in result.log:
        raise RuntimeError(f"Export to {fmt} reported a problem:\n{result.log[-3000:]}")
    return result.outputs[name].decode()


def statements(text: str, fmt: str) -> dict[str, str]:
    """Element definitions of a written file by element name, continuation lines joined.

    MAD-X statements end at `;`; MAD8 and Elegant continue a line ending in `&`.
    """
    if fmt == "madx":
        joined = [" ".join(s.split()) for s in text.replace("\n", " ").split(";")]
    else:
        joined = [" ".join(s.split()) for s in text.replace("&\n", " ").splitlines()]
    found = {}
    for statement in joined:
        match = re.match(r"([A-Za-z_][\w.]*)\s*:\s*(.*)$", statement)
        if match and not statement.startswith(("!", "//")):
            found[match[1].upper()] = match[2]
    return found


def value(statement: str, key: str, default: float | None = None) -> float:
    """Number assigned to `key` in one element statement; `default` when it is absent."""
    match = re.search(rf"\b{key}\s*=\s*([^,;\s]+)", statement)
    if match is None:
        if default is None:
            raise KeyError(f"{key} is not set in: {statement}")
        return default
    return float(match[1])


def element_type(text: str, fmt: str, name: str) -> str:
    """Element class keyword that a written file gives to the named element."""
    return statements(text, fmt)[name.upper()].split(",")[0].strip().lower()


@contextlib.contextmanager
def madx(deck: str) -> Iterator[Madx]:
    """A MAD-X process (cpymad) that has read the deck, in a scratch directory."""
    with tempfile.TemporaryDirectory(prefix="madx-") as tmp:
        Path(tmp, "deck.madx").write_text(deck)
        with Madx(stdout=False, cwd=tmp) as mad:
            mad.call("deck.madx")
            yield mad


def madx_track(mad: Madx, start: Sequence[float], *, sequence: str = "lat") -> np.ndarray:
    """One-pass MAD-X TRACK of one particle; returns (x, px, y, py, t, pt) at the end."""
    names = ("x", "px", "y", "py", "t", "pt")
    begin = ", ".join(f"{k} = {float(v)!r}" for k, v in zip(names, start, strict=True))
    mad.input(
        f"use, period = {sequence}; track, onepass, onetable; start, {begin};"
        "run, turns = 1, maxaper = {1, 1, 1, 1, 1, 1}; endtrack;"
    )
    table = mad.table.trackone
    last = table.turn == 1
    if np.count_nonzero(last) != 1:
        raise RuntimeError("MAD-X TRACK lost the particle")
    return np.array([table[k][last][0] for k in names])


def import_madx(deck: str) -> str:
    """Translate a MAD-X deck with BMAD_SOURCE's madx_to_bmad.py; return the Bmad text."""
    script = source() / "util_programs/mad_to_bmad/madx_to_bmad.py"
    with tempfile.TemporaryDirectory(prefix="madx-import-") as tmp:
        Path(tmp, "source.madx").write_text(deck)
        done = subprocess.run(
            [sys.executable, "-I", str(script), "source.madx"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
        if done.returncode:
            raise RunFailure("madx_to_bmad.py", done.returncode, done.stdout + done.stderr)
        return Path(tmp, "source.bmad").read_text()


def elegant_track(
    lte: str, p_central_mev: float, start: Sequence[float], *, beamline: str = "lat"
) -> np.ndarray:
    """Track one particle (x, xp, y, yp) through an Elegant lattice; return the final centroid.

    Returns (s, x, xp, y, yp) at the end of the beamline, from the centroid output.
    """
    import pysdds

    if "RPN_DEFNS" not in os.environ or shutil.which("elegant") is None:
        raise RuntimeError("Put elegant on PATH and set RPN_DEFNS to the defns.rpn file of SDDS.")
    centroid = ", ".join(f"centroid[{i}] = {float(v)!r}" for i, v in enumerate(start))
    commands = (
        f'&run_setup lattice = "lat.lte", use_beamline = "{beamline}", '
        f'p_central_mev = {p_central_mev!r}, centroid = "out.cen", output = "out.sdds" &end\n'
        "&run_control n_steps = 1 &end\n"
        f"&bunched_beam n_particles_per_bunch = 1, {centroid} &end\n"
        "&track &end\n"
    )
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="elegant-") as tmp:
        Path(tmp, "lat.lte").write_text(lte)
        Path(tmp, "run.ele").write_text(commands)
        done = subprocess.run(
            ["elegant", "run.ele"],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
        if done.returncode:
            raise RunFailure("elegant", done.returncode, done.stdout + done.stderr)
        particles = pysdds.read(Path(tmp, "out.sdds"))
        phase = [np.asarray(particles[0, name]) for name in ("x", "xp", "y", "yp", "t", "p")]
        if any(values.shape != (1,) or not np.isfinite(values).all() for values in phase):
            raise RuntimeError("Elegant must return exactly one finite surviving particle")
        out = pysdds.read(Path(tmp, "out.cen"))
        columns = [np.asarray(out[0, name]) for name in ("s", "Cx", "Cxp", "Cy", "Cyp")]
    if any(
        column.ndim != 1 or not len(column) or not np.isfinite(column).all() for column in columns
    ):
        raise RuntimeError("Elegant returned incomplete or nonfinite centroid records")
    result = np.array([column[-1] for column in columns])
    if not np.allclose(result[1:], [values[0] for values in phase[:4]], rtol=1e-12, atol=1e-15):
        raise RuntimeError("Elegant final particle and centroid disagree")
    return result
