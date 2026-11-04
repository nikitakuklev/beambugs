"""Shared helpers for the reproducers of SDDS library defects: run the SDDS utility `sddsprocess`
on inline tables and compile small C programs against the SDDS libraries.

`sddsprocess` must be on PATH, as for the Elegant reproducers; the SDDS build tree (its `lib`
directory with `libmdbmth.a` and `libmdblib.a`) is found from the location of that executable.
Compiling needs `gcc`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pysdds

TIMEOUT = 120  # seconds per native run


def sdds_lib() -> Path:
    """The `lib/Linux-x86_64` directory of the SDDS build that provides `sddsprocess`."""
    tool = shutil.which("sddsprocess")
    if tool is None:
        sys.exit("sddsprocess must be on PATH (see the elegant installation).")
    lib = Path(tool).resolve().parents[2] / "lib" / "Linux-x86_64"
    if not (lib / "libmdbmth.a").exists():
        sys.exit(f"{lib} has no libmdbmth.a: sddsprocess is not in an SDDS build tree.")
    return lib


def execute(argv: list[str], cwd: Path) -> str:
    """Run a native command; a nonzero exit status or a timeout raises RuntimeError."""
    done = subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT, check=False
    )
    if done.returncode:
        raise RuntimeError(
            f"{argv[0]} exited with status {done.returncode}:\n{done.stderr[-2000:]}"
        )
    return done.stdout


def c_check(source: str) -> list[list[str]]:
    """Compile a C program against the SDDS math libraries, run it, and return the stdout words.

    The program prints one result per line; each line comes back as a list of words.
    """
    lib = sdds_lib()
    with tempfile.TemporaryDirectory(prefix="sdds-repro-") as tmp:
        work = Path(tmp)
        (work / "check.c").write_text(source)
        execute(
            [
                "gcc",
                "-O2",
                "check.c",
                str(lib / "libmdbmth.a"),
                str(lib / "libmdblib.a"),
                "-fopenmp",
                "-lm",
                "-o",
                "check",
            ],
            work,
        )
        return [line.split() for line in execute([str(work / "check")], work).splitlines()]


def process(columns: dict[str, np.ndarray], *options: str) -> dict[str, np.ndarray]:
    """Run `sddsprocess` on a one-page table of double columns.

    Returns every parameter and column of the output page as a NumPy array (a parameter has
    one element). The input is written as binary SDDS, which carries the doubles exactly.
    """
    table = pysdds.SDDSFile(add_data_nm=True)
    table.mode = "binary"
    table.n_pages = 1
    for name, values in columns.items():
        table.add_column(name, "double", data=[np.asarray(values, dtype=np.float64)])
    with tempfile.TemporaryDirectory(prefix="sdds-repro-") as tmp:
        work = Path(tmp)
        pysdds.write(table, work / "input.sdds", overwrite=True)
        execute(["sddsprocess", "input.sdds", "output.sdds", *options], work)
        out = pysdds.read(work / "output.sdds")
        if out.n_pages != 1:
            raise RuntimeError(f"sddsprocess wrote {out.n_pages} pages")
        result = {p.name: np.atleast_1d(np.asarray(out[p.name].data[0])) for p in out.parameters}
        result.update({c.name: np.asarray(out[0, c.name]) for c in out.columns})
        return result
